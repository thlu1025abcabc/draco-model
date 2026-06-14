from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

import polars as pl

from draco_model.core import Model, Node, resolve_node_names
from draco_model.data.source import SourceCatalog
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.data.universe import UniverseCatalog
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
from draco_model.runtime.profiling import PlanProfile, Profiler, profile_plan


logger = logging.getLogger(__name__)


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
        self._memory: dict[tuple[str, str, str], pl.LazyFrame] = {}
        self._info_memo: dict[tuple[str, str, str], FrameInfo] = {}
        self._grid_memory: dict[tuple[str, str, tuple[int, ...]], pl.DataFrame] = {}
        self._profiler: Profiler | None = None

    def collect(self, model: Model, dates: list[str] | tuple[str, ...]) -> pl.DataFrame:
        """Evaluate a model output for dates and collect daily factor rows."""
        with self._profile_span(
            "collect",
            model=model.name,
            universe=model.universe,
            dates=tuple(_normalize_date(date) for date in dates),
        ):
            return self._collect(model, dates)

    def _collect(self, model: Model, dates: list[str] | tuple[str, ...]) -> pl.DataFrame:
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
            info = self._infer_info(model, model.output, date)
            _validate_collect_info(info)
            frame = self.evaluate(model, model.output, date)
            outputs.append(format_factor_output(frame, model.name, date))
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

    def profile_plan(self, models: list[Model] | tuple[Model, ...]) -> PlanProfile:
        """Return a static shared-node profile for a group of models."""
        return profile_plan(models)

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


def _validate_collect_info(info: FrameInfo) -> None:
    if not can_collect(info):
        raise ValueError(
            "Engine.collect requires a daily output with date/secu_code keys; "
            f"got grain={info.grain!r}, keys={info.keys!r}."
        )
