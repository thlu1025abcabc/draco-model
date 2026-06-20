from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Iterator

import polars as pl

from draco_model.core import Model, Node, resolve_node_names
from draco_model.data.source import SourceCatalog
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.data.universe import UniverseCatalog
from draco_model.market.schema import DAILY_KEY_COLUMNS
from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.runtime.execution import (
    can_collect,
    EvalContext,
    FrameInfo,
    TraceStep,
    format_factor_output,
    get_executor,
    get_info_builder,
)
from draco_model.runtime.profiling import DEFAULT_EXCLUDE_CACHE_OPS, PlanProfile, Profiler, profile_plan


logger = logging.getLogger(__name__)

BatchCacheKey = tuple[str | None, str, str]


@dataclass
class _BatchScope:
    candidates: set[BatchCacheKey]
    frames: dict[BatchCacheKey, pl.DataFrame] = field(default_factory=dict)
    lazy_frames: dict[BatchCacheKey, pl.LazyFrame] = field(default_factory=dict)


class Engine:
    """Evaluate model DAGs against local data catalogs."""

    def __init__(
        self,
        data_root: str | Path = "data",
        *,
        minute_calendar: MinuteCalendar | None = None,
        trading_calendar: TradingCalendar | None = None,
    ) -> None:
        """Create an engine bound to one data root and optional calendars."""
        self.data_root = Path(data_root)
        self.minute_calendar = minute_calendar or MinuteCalendar()
        self.sources = SourceCatalog(self.data_root, self.minute_calendar)
        self.universes = UniverseCatalog(self.data_root)
        self.trading_calendar = trading_calendar
        self._memory: dict[tuple[str | None, str, str], pl.LazyFrame] = {}
        self._info_memo: dict[tuple[str | None, str, str], FrameInfo] = {}
        self._grid_memory: dict[tuple[str, str, tuple[int, ...]], pl.DataFrame] = {}
        self._profiler: Profiler | None = None

    def collect(
        self,
        model: Model,
        dates: list[str] | tuple[str, ...],
    ) -> pl.DataFrame:
        """Evaluate a model output for dates and collect daily factor rows."""
        _require_model_universe(model, "Engine.collect")
        with self._profile_span(
            "collect",
            model=model.name,
            universe=model.universe,
            dates=tuple(_normalize_date(date) for date in dates),
        ):
            return self._collect(model, dates)

    def collect_many(
        self,
        models: list[Model] | tuple[Model, ...],
        dates: list[str] | tuple[str, ...],
        *,
        min_cache_ref_count: int = 2,
        exclude_cache_ops: Iterable[str] = DEFAULT_EXCLUDE_CACHE_OPS,
    ) -> pl.DataFrame:
        """Evaluate multiple models with batch-scoped materialized cache reuse."""
        model_list = list(models)
        if not model_list:
            raise ValueError("Engine.collect_many requires at least one model.")
        if not dates:
            raise ValueError("Engine.collect_many requires at least one date.")
        _validate_unique_model_names(model_list)
        _require_models_universe(model_list, "Engine.collect_many")
        normalized_dates = tuple(_normalize_date(date) for date in dates)
        excluded_ops = tuple(exclude_cache_ops)
        with self._profile_span(
            "collect_many",
            models=tuple(model.name for model in model_list),
            dates=normalized_dates,
            min_cache_ref_count=min_cache_ref_count,
            exclude_cache_ops=excluded_ops,
        ):
            return self._collect_many(
                model_list,
                normalized_dates,
                min_cache_ref_count,
                excluded_ops,
            )

    def _collect(
        self,
        model: Model,
        dates: list[str] | tuple[str, ...],
    ) -> pl.DataFrame:
        if not dates:
            raise ValueError("Engine.collect requires at least one date.")
        t0 = perf_counter()
        normalized_dates = [_normalize_date(date) for date in dates]
        logger.info(
            "collect.start model=%s universe=%s dates=%s",
            model.name,
            model.universe,
            normalized_dates,
        )
        self._ensure_calendar()
        outputs = []
        for date in normalized_dates:
            logger.debug("collect.date.start model=%s date=%s", model.name, date)
            self._grid_memory.clear()
            universe = self._output_universe(model.universe, date)
            for output_name, output_node in model.outputs:
                info = self._infer_info(model, output_node, date)
                value_column = _validate_collect_info(info, model.name, output_name)
                frame = self.evaluate(model, output_node, date)
                outputs.append(
                    format_factor_output(
                        frame,
                        _factor_name(model, output_name),
                        date,
                        value_column,
                        universe,
                    )
                )
            logger.debug("collect.date.done model=%s date=%s", model.name, date)
        with self._profile_span(
            "collect.materialize",
            model=model.name,
            universe=model.universe,
            dates=tuple(normalized_dates),
        ):
            result = pl.concat(outputs, how="vertical").collect()
        logger.info(
            "collect.done model=%s universe=%s dates=%s rows=%d elapsed=%.3fs",
            model.name,
            model.universe,
            normalized_dates,
            result.height,
            perf_counter() - t0,
        )
        return result

    def evaluate(self, model: Model, node: Node, eval_date: str) -> pl.LazyFrame:
        """Evaluate any node in a model for one date and return a LazyFrame."""
        self._ensure_calendar()
        normalized = _normalize_date(eval_date)
        logger.debug(
            "evaluate model=%s universe=%s date=%s node_id=%s op=%s",
            model.name,
            model.universe,
            normalized,
            node.id,
            node.op,
        )
        with self._profile_span(
            "evaluate",
            model=model.name,
            universe=model.universe,
            date=normalized,
            node_id=node.id,
            op=node.op,
        ):
            return self._eval(model, node, normalized)

    def evaluate_outputs(self, model: Model, eval_date: str) -> dict[str, pl.LazyFrame]:
        """Evaluate all named model outputs for one date without formatting their grain."""
        self._ensure_calendar()
        normalized = _normalize_date(eval_date)
        self._grid_memory.clear()
        logger.debug(
            "evaluate_outputs model=%s universe=%s date=%s outputs=%s",
            model.name,
            model.universe,
            normalized,
            [name for name, _ in model.outputs],
        )
        with self._profile_span(
            "evaluate_outputs",
            model=model.name,
            universe=model.universe,
            date=normalized,
            outputs=tuple(name for name, _ in model.outputs),
        ):
            return {name: self._eval(model, node, normalized) for name, node in model.outputs}

    def trace(self, model: Model, date: str) -> list[TraceStep]:
        """Evaluate frame nodes one by one and return their materialized outputs."""
        eval_date = _normalize_date(date)
        self._ensure_calendar()
        self._grid_memory.clear()
        logger.info("trace.start model=%s universe=%s date=%s", model.name, model.universe, eval_date)

        materialized: dict[str, pl.DataFrame] = {}
        steps: list[TraceStep] = []
        names = resolve_node_names(model.nodes())

        def evaluate(parent: Node) -> pl.LazyFrame:
            try:
                return materialized[parent.id].lazy()
            except KeyError:
                raise ValueError(f"Trace parent node {parent.id!r} has not been materialized.") from None

        for node in model.nodes():
            if node.kind != "frame":
                continue

            context = self._context(model, eval_date, evaluate=evaluate)
            frame = get_executor(node.op)(node, context).collect()
            materialized[node.id] = frame
            steps.append(TraceStep(index=len(steps), resolved_name=names[node.id], node=node, frame=frame))
            logger.debug(
                "trace.step index=%d name=%s node_id=%s op=%s rows=%d cols=%d",
                len(steps) - 1,
                names[node.id],
                node.id,
                node.op,
                frame.height,
                len(frame.columns),
            )

        logger.info("trace.done model=%s universe=%s date=%s steps=%d", model.name, model.universe, eval_date, len(steps))
        return steps

    def profile_plan(
        self,
        models: list[Model] | tuple[Model, ...],
        *,
        min_cache_ref_count: int = 2,
        exclude_cache_ops: Iterable[str] = DEFAULT_EXCLUDE_CACHE_OPS,
    ) -> PlanProfile:
        """Return a static shared-node profile for a group of models."""
        return profile_plan(
            models,
            min_cache_ref_count=min_cache_ref_count,
            exclude_cache_ops=exclude_cache_ops,
        )

    @contextmanager
    def profiler(self) -> Iterator[Profiler]:
        """Collect runtime profile events for Engine calls inside the context."""
        profiler = Profiler()
        previous = self._profiler
        self._profiler = profiler
        try:
            yield profiler
        finally:
            self._profiler = previous

    def _ensure_calendar(self) -> None:
        if self.trading_calendar is None:
            logger.debug("calendar.load data_root=%s", self.data_root)
            self.trading_calendar = TradingCalendar.from_data_root(self.data_root)

    def _collect_many(
        self,
        models: list[Model],
        dates: tuple[str, ...],
        min_cache_ref_count: int,
        exclude_cache_ops: tuple[str, ...],
    ) -> pl.DataFrame:
        t0 = perf_counter()
        self._ensure_calendar()
        models_by_universe = _models_by_universe(models)
        scope = _BatchScope(
            candidates=_batch_cache_candidates(
                models_by_universe,
                dates,
                min_cache_ref_count=min_cache_ref_count,
                exclude_cache_ops=exclude_cache_ops,
            )
        )
        self._profile_record(
            "collect_many.plan",
            model_count=len(models),
            date_count=len(dates),
            candidate_count=len(scope.candidates),
            min_cache_ref_count=min_cache_ref_count,
            exclude_cache_ops=exclude_cache_ops,
        )
        logger.info(
            "collect_many.start models=%d dates=%s cache_candidates=%d",
            len(models),
            list(dates),
            len(scope.candidates),
        )
        outputs: list[pl.LazyFrame] = []
        universe_cache: dict[tuple[str, str], pl.LazyFrame] = {}
        for date in dates:
            self._grid_memory.clear()
            for model in models:
                logger.debug("collect_many.model.start model=%s date=%s", model.name, date)
                universe_key = (model.universe, date)
                if universe_key not in universe_cache:
                    universe_cache[universe_key] = self._output_universe(model.universe, date)
                universe = universe_cache[universe_key]
                for output_name, output_node in model.outputs:
                    info = self._infer_info(model, output_node, date)
                    value_column = _validate_collect_info(info, model.name, output_name)
                    frame = self._eval_batch(model, output_node, date, scope)
                    outputs.append(
                        format_factor_output(
                            frame,
                            _factor_name(model, output_name),
                            date,
                            value_column,
                            universe,
                        )
                    )
                logger.debug("collect_many.model.done model=%s date=%s", model.name, date)
        with self._profile_span(
            "collect_many.materialize",
            models=tuple(model.name for model in models),
            dates=dates,
        ):
            result = pl.concat(outputs, how="vertical").collect()
        logger.info(
            "collect_many.done models=%d dates=%s rows=%d elapsed=%.3fs",
            len(models),
            list(dates),
            result.height,
            perf_counter() - t0,
        )
        return result

    def _output_universe(self, universe: str, date: str) -> pl.LazyFrame:
        return (
            self.universes.scan(universe, date)
            .with_columns(pl.lit(date).alias("date"))
            .select(list(DAILY_KEY_COLUMNS))
        )

    def _context(self, model: Model, eval_date: str, *, evaluate: Callable[[Node], pl.LazyFrame] | None = None) -> EvalContext:
        assert self.trading_calendar is not None
        return EvalContext(
            model=model,
            eval_date=eval_date,
            sources=self.sources,
            universes=self.universes,
            minute_calendar=self.minute_calendar,
            trading_calendar=self.trading_calendar,
            evaluate=evaluate or (lambda parent: self._eval(model, parent, eval_date)),
            infer_info=lambda parent: self._infer_info(model, parent, eval_date),
            grid_cache=self._grid_memory,
        )

    def _eval(self, model: Model, node: Node, eval_date: str) -> pl.LazyFrame:
        key = (model.universe, node.id, eval_date)
        if key in self._memory:
            logger.debug("eval.cache_hit universe=%s date=%s node_id=%s op=%s", model.universe, eval_date, node.id, node.op)
            self._profile_record(
                "eval.cache_hit",
                model=model.name,
                universe=model.universe,
                date=eval_date,
                node_id=node.id,
                op=node.op,
            )
            return self._memory[key]

        logger.debug("eval.cache_miss universe=%s date=%s node_id=%s op=%s", model.universe, eval_date, node.id, node.op)
        self._profile_record(
            "eval.cache_miss",
            model=model.name,
            universe=model.universe,
            date=eval_date,
            node_id=node.id,
            op=node.op,
        )
        with self._profile_span(
            "eval",
            model=model.name,
            universe=model.universe,
            date=eval_date,
            node_id=node.id,
            op=node.op,
        ):
            out = get_executor(node.op)(node, self._context(model, eval_date))

        self._memory[key] = out
        return out

    def _eval_batch(self, model: Model, node: Node, eval_date: str, scope: _BatchScope) -> pl.LazyFrame:
        key = (model.universe, node.id, eval_date)
        if key in scope.frames:
            frame = scope.frames[key]
            self._profile_record(
                "batch_cache.hit",
                model=model.name,
                universe=model.universe,
                date=eval_date,
                node_id=node.id,
                op=node.op,
                rows=frame.height,
                columns=len(frame.columns),
            )
            return frame.lazy()
        if key in scope.lazy_frames and key not in scope.candidates:
            self._profile_record(
                "eval.cache_hit",
                model=model.name,
                universe=model.universe,
                date=eval_date,
                node_id=node.id,
                op=node.op,
            )
            return scope.lazy_frames[key]

        if key in scope.candidates:
            self._profile_record(
                "batch_cache.miss",
                model=model.name,
                universe=model.universe,
                date=eval_date,
                node_id=node.id,
                op=node.op,
            )
            lazy = self._execute_node_batch(model, node, eval_date, scope)
            start = perf_counter()
            frame = lazy.collect()
            self._profile_record(
                "batch_cache.materialize",
                elapsed_ms=(perf_counter() - start) * 1000,
                model=model.name,
                universe=model.universe,
                date=eval_date,
                node_id=node.id,
                op=node.op,
                rows=frame.height,
                columns=len(frame.columns),
            )
            scope.frames[key] = frame
            return frame.lazy()

        # Reaching here means key is not a cache candidate; cache its lazy plan.
        lazy = self._execute_node_batch(model, node, eval_date, scope)
        scope.lazy_frames[key] = lazy
        return lazy

    def _execute_node_batch(self, model: Model, node: Node, eval_date: str, scope: _BatchScope) -> pl.LazyFrame:
        self._profile_record(
            "eval.cache_miss",
            model=model.name,
            universe=model.universe,
            date=eval_date,
            node_id=node.id,
            op=node.op,
        )
        with self._profile_span(
            "eval",
            model=model.name,
            universe=model.universe,
            date=eval_date,
            node_id=node.id,
            op=node.op,
        ):
            return get_executor(node.op)(
                node,
                self._context(
                    model,
                    eval_date,
                    evaluate=lambda parent: self._eval_batch(model, parent, eval_date, scope),
                ),
            )

    def _infer_info(self, model: Model, node: Node, eval_date: str) -> FrameInfo:
        key = (model.universe, node.id, eval_date)
        if key in self._info_memo:
            logger.debug("info.cache_hit universe=%s date=%s node_id=%s op=%s", model.universe, eval_date, node.id, node.op)
            self._profile_record(
                "infer_info.cache_hit",
                model=model.name,
                universe=model.universe,
                date=eval_date,
                node_id=node.id,
                op=node.op,
            )
            return self._info_memo[key]

        logger.debug("info.cache_miss universe=%s date=%s node_id=%s op=%s", model.universe, eval_date, node.id, node.op)
        self._profile_record(
            "infer_info.cache_miss",
            model=model.name,
            universe=model.universe,
            date=eval_date,
            node_id=node.id,
            op=node.op,
        )
        parent_infos = {
            input_name: self._infer_info(model, parent, eval_date)
            for input_name, parent in node.inputs.items()
            if parent.kind == "frame"
        }
        context = self._context(model, eval_date)
        builder = get_info_builder(node.op)
        if builder is None:
            raise ValueError(
                f"Node op {node.op!r} has no registered frame-info builder; register one with register_info."
            )
        with self._profile_span(
            "infer_info",
            model=model.name,
            universe=model.universe,
            date=eval_date,
            node_id=node.id,
            op=node.op,
        ):
            info = builder(node, parent_infos, context)
        self._info_memo[key] = info
        return info

    def _profile_record(self, event: str, **fields: object) -> None:
        if self._profiler is not None:
            self._profiler.record(event, **fields)

    @contextmanager
    def _profile_span(self, event: str, **fields: object) -> Iterator[None]:
        if self._profiler is None:
            yield
            return
        with self._profiler.span(event, **fields):
            yield


def _normalize_date(value: object) -> str:
    return str(value).replace("-", "")


def _factor_name(model: Model, output_name: str) -> str:
    if len(model.outputs) == 1 and output_name == "value":
        return model.name
    return f"{model.name}__{output_name}"


def _validate_unique_model_names(models: list[Model]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for model in models:
        if model.name in seen:
            duplicates.append(model.name)
        seen.add(model.name)
    if duplicates:
        raise ValueError(f"Engine.collect_many requires unique model names, got duplicates: {duplicates}.")


def _require_model_universe(model: Model, operation: str) -> str:
    if model.universe is None:
        raise ValueError(
            f"{operation} requires Model.universe; "
            "use Engine.evaluate_outputs() for universe-independent model outputs."
        )
    return model.universe


def _require_models_universe(models: list[Model], operation: str) -> None:
    missing = [model.name for model in models if model.universe is None]
    if missing:
        raise ValueError(
            f"{operation} requires Model.universe for every model; "
            f"missing for models: {missing}. "
            "Use Engine.evaluate_outputs() for universe-independent model outputs."
        )


def _models_by_universe(models: list[Model]) -> dict[str | None, list[Model]]:
    out: dict[str | None, list[Model]] = defaultdict(list)
    for model in models:
        out[model.universe].append(model)
    return dict(out)


def _batch_cache_candidates(
    models_by_universe: dict[str | None, list[Model]],
    dates: tuple[str, ...],
    *,
    min_cache_ref_count: int,
    exclude_cache_ops: Iterable[str],
) -> set[BatchCacheKey]:
    out: set[BatchCacheKey] = set()
    for universe, models in models_by_universe.items():
        plan = profile_plan(
            models,
            min_cache_ref_count=min_cache_ref_count,
            exclude_cache_ops=exclude_cache_ops,
        )
        for node in plan.cache_candidates():
            for date in dates:
                out.add((universe, node.node_id, date))
    return out


def _validate_collect_info(info: FrameInfo, model_name: str, output_name: str) -> str:
    if not can_collect(info):
        raise ValueError(
            "Engine.collect requires a daily output with date/secu_code keys and exactly one public value column; "
            f"model={model_name!r}, output={output_name!r}, "
            f"got grain={info.grain!r}, keys={info.keys!r}, "
            f"available={info.value_columns()!r}."
        )
    return info.value_columns()[0]
