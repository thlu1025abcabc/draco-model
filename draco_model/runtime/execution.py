from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, MutableMapping

import polars as pl

from draco_model.core import Model, Node
from draco_model.data.source import SourceCatalog
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.data.universe import UniverseCatalog
from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS


Executor = Callable[[Node, "EvalContext"], pl.LazyFrame]
PlanBuilder = Callable[[Node, dict[str, "FrameSchema"], "EvalContext"], "FramePlan"]

_EXECUTORS: dict[str, Executor] = {}
_PLAN_BUILDERS: dict[str, PlanBuilder] = {}


@dataclass(frozen=True)
class FrameSchema:
    """Logical and physical column contract for a frame node."""

    columns: tuple[str, ...]
    keys: tuple[str, ...] = ()
    grain: str = "unknown"
    fields: dict[str, "FieldInfo"] = field(default_factory=dict)

    def value_columns(self) -> list[str]:
        """Return public value columns, excluding keys and internal payload."""
        if self.fields:
            return [info.column for info in self.fields.values()]
        keys = set(self.keys)
        return [column for column in self.columns if column not in keys and not column.startswith("__")]


@dataclass(frozen=True)
class FieldInfo:
    """Metadata for one public field inside a frame."""

    name: str
    column: str
    operator: str = "identity"
    components: tuple[str, ...] = ()
    source: str | None = None
    lookback_days: int = 1
    component_agg: bool = False


@dataclass(frozen=True)
class FramePlan:
    """Single source of truth for a frame node's output layout."""

    columns: tuple[str, ...]
    keys: tuple[str, ...] = ()
    grain: str = "unknown"
    fields: dict[str, FieldInfo] = field(default_factory=dict)

    @classmethod
    def from_schema(cls, schema: FrameSchema) -> "FramePlan":
        """Create a plan that preserves an existing schema layout."""
        return cls(
            columns=schema.columns,
            keys=schema.keys,
            grain=schema.grain,
            fields=schema.fields,
        )

    def schema(self) -> FrameSchema:
        """Return this plan's public schema contract."""
        return FrameSchema(
            columns=self.columns,
            keys=self.keys,
            grain=self.grain,
            fields=self.fields,
        )


@dataclass(frozen=True)
class TraceStep:
    """Materialized output for one traced frame node."""

    index: int
    resolved_name: str
    node: Node
    frame: pl.DataFrame


def register_executor(op: str) -> Callable[[Executor], Executor]:
    """Register the function that evaluates nodes with a given op name."""

    def decorator(executor: Executor) -> Executor:
        _EXECUTORS[op] = executor
        return executor

    return decorator


def register_plan(op: str) -> Callable[[PlanBuilder], PlanBuilder]:
    """Register the output layout planner for nodes with a given op name."""

    def decorator(builder: PlanBuilder) -> PlanBuilder:
        _PLAN_BUILDERS[op] = builder
        return builder

    return decorator


def get_executor(op: str) -> Executor:
    """Return the registered executor for an op or raise a clear error."""
    try:
        return _EXECUTORS[op]
    except KeyError:
        raise ValueError(f"Unsupported node op {op!r}.") from None


def get_plan_builder(op: str) -> PlanBuilder | None:
    """Return the registered frame plan builder for an op, if one exists."""
    return _PLAN_BUILDERS.get(op)


@dataclass(frozen=True)
class EvalContext:
    """Runtime services shared by node executors during one evaluation."""

    model: Model
    eval_date: str
    sources: SourceCatalog
    universes: UniverseCatalog
    minute_calendar: MinuteCalendar
    trading_calendar: TradingCalendar
    evaluate: Callable[[Node], pl.LazyFrame]
    infer_schema: Callable[[Node], FrameSchema]
    grid_cache: MutableMapping[tuple[str, str], pl.DataFrame]

    def intraday_grid(self, universe: str, dates: list[str]) -> pl.LazyFrame:
        """Build or reuse the universe-by-minute grid for intraday inputs."""
        key = (universe, self.eval_date)
        if key not in self.grid_cache:
            universe_frame = self.universes.scan(universe, self.eval_date).select("secu_code")
            minutes = pl.DataFrame({"minute": self.minute_calendar.minbars()}).lazy()
            self.grid_cache[key] = universe_frame.join(minutes, how="cross").collect()
        frames = [
            self.grid_cache[key]
            .lazy()
            .with_columns(pl.lit(date).alias("date"))
            .select(list(KEY_COLUMNS))
            for date in dates
        ]
        return pl.concat(frames, how="vertical")


def format_factor_output(frame: pl.LazyFrame, factor_name: str, eval_date: str) -> pl.LazyFrame:
    """Normalize a daily frame into the public factor output schema."""
    columns = frame.collect_schema().names()
    if "value" not in columns:
        raise ValueError("Model output must contain a value column.")
    return (
        frame.filter(pl.col("date") == eval_date)
        .select([*DAILY_KEY_COLUMNS, "value"])
        .with_columns(pl.lit(factor_name).alias("factor_name"))
        .select(["date", "secu_code", "factor_name", "value"])
    )
