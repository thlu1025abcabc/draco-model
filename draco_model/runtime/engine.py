from __future__ import annotations

from pathlib import Path

import polars as pl

from draco_model.core import Model, Node, resolve_node_names
from draco_model.data.source import SourceCatalog
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.data.universe import UniverseCatalog
from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.market.schema import DAILY_KEY_COLUMNS
from draco_model.runtime.execution import (
    EvalContext,
    FrameSchema,
    TraceStep,
    format_factor_output,
    get_executor,
    get_plan_builder,
    get_schema_inferer,
)


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
        self._grid_memory: dict[tuple[str, str], pl.DataFrame] = {}

    def collect(self, model: Model, dates: list[str] | tuple[str, ...]) -> pl.DataFrame:
        """Evaluate a model output for dates and collect daily factor rows."""
        if not dates:
            raise ValueError("Engine.collect requires at least one date.")
        self._ensure_calendar()
        outputs = []
        for date in [_normalize_date(date) for date in dates]:
            self._grid_memory.clear()
            schema = self._infer_schema(model, model.output, date)
            _validate_collect_schema(schema)
            frame = self.evaluate(model, model.output, date)
            outputs.append(format_factor_output(frame, model.name, date))
        return pl.concat(outputs, how="vertical").collect()

    def evaluate(self, model: Model, node: Node, eval_date: str) -> pl.LazyFrame:
        """Evaluate any node in a model for one date and return a LazyFrame."""
        self._ensure_calendar()
        return self._eval(model, node, _normalize_date(eval_date))

    def trace(self, model: Model, date: str) -> list[TraceStep]:
        """Evaluate frame nodes one by one and return their materialized outputs."""
        eval_date = _normalize_date(date)
        self._ensure_calendar()
        self._grid_memory.clear()

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

            assert self.trading_calendar is not None
            context = EvalContext(
                model=model,
                eval_date=eval_date,
                sources=self.sources,
                universes=self.universes,
                minute_calendar=self.minute_calendar,
                trading_calendar=self.trading_calendar,
                evaluate=evaluate,
                infer_schema=lambda parent: self._infer_schema(model, parent, eval_date),
                grid_cache=self._grid_memory,
            )
            frame = get_executor(node.op)(node, context).collect()
            materialized[node.id] = frame
            steps.append(TraceStep(index=len(steps), resolved_name=names[node.id], node=node, frame=frame))

        return steps

    def _ensure_calendar(self) -> None:
        if self.trading_calendar is None:
            self.trading_calendar = TradingCalendar.from_data_root(self.data_root)

    def _eval(self, model: Model, node: Node, eval_date: str) -> pl.LazyFrame:
        key = (model.universe, node.id, eval_date)
        if key in self._memory:
            return self._memory[key]

        assert self.trading_calendar is not None
        context = EvalContext(
            model=model,
            eval_date=eval_date,
            sources=self.sources,
            universes=self.universes,
            minute_calendar=self.minute_calendar,
            trading_calendar=self.trading_calendar,
            evaluate=lambda parent: self._eval(model, parent, eval_date),
            infer_schema=lambda parent: self._infer_schema(model, parent, eval_date),
            grid_cache=self._grid_memory,
        )
        out = get_executor(node.op)(node, context)

        self._memory[key] = out
        return out

    def _infer_schema(self, model: Model, node: Node, eval_date: str) -> FrameSchema:
        parent_schemas = {
            input_name: self._infer_schema(model, parent, eval_date)
            for input_name, parent in node.inputs.items()
            if parent.kind == "frame"
        }
        assert self.trading_calendar is not None
        context = EvalContext(
            model=model,
            eval_date=eval_date,
            sources=self.sources,
            universes=self.universes,
            minute_calendar=self.minute_calendar,
            trading_calendar=self.trading_calendar,
            evaluate=lambda parent: self._eval(model, parent, eval_date),
            infer_schema=lambda parent: self._infer_schema(model, parent, eval_date),
            grid_cache=self._grid_memory,
        )
        planner = get_plan_builder(node.op)
        if planner is not None:
            return planner(node, parent_schemas, context).schema()
        inferer = get_schema_inferer(node.op)
        if inferer is not None:
            return inferer(node, parent_schemas, context)
        return FrameSchema(tuple(self._eval(model, node, eval_date).collect_schema().names()))


def _normalize_date(value: object) -> str:
    return str(value).replace("-", "")


def _validate_collect_schema(schema: FrameSchema) -> None:
    if schema.keys != DAILY_KEY_COLUMNS or schema.grain != "daily":
        raise ValueError(
            "Engine.collect requires a daily output with date/secu_code keys; "
            f"got grain={schema.grain!r}, keys={schema.keys!r}."
        )
    if "value" not in schema.value_columns():
        raise ValueError("Engine.collect requires a public 'value' column.")
