from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, MutableMapping

import polars as pl

from draco_model.core import Model, Node
from draco_model.data.source import SourceCatalog
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.data.universe import UniverseCatalog
from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS


Executor = Callable[[Node, "EvalContext"], pl.LazyFrame]
SchemaInferer = Callable[[Node, dict[str, "FrameSchema"], "EvalContext"], "FrameSchema"]

_EXECUTORS: dict[str, Executor] = {}
_SCHEMA_INFERERS: dict[str, SchemaInferer] = {}


@dataclass(frozen=True)
class FrameSchema:
    """Column contract for a frame node."""

    columns: tuple[str, ...]


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


def register_schema(op: str) -> Callable[[SchemaInferer], SchemaInferer]:
    """Register the function that infers output columns for nodes with an op."""

    def decorator(inferer: SchemaInferer) -> SchemaInferer:
        _SCHEMA_INFERERS[op] = inferer
        return inferer

    return decorator


def get_executor(op: str) -> Executor:
    """Return the registered executor for an op or raise a clear error."""
    try:
        return _EXECUTORS[op]
    except KeyError:
        raise ValueError(f"Unsupported node op {op!r}.") from None


def get_schema_inferer(op: str) -> SchemaInferer | None:
    """Return the registered schema inferer for an op, if one exists."""
    return _SCHEMA_INFERERS.get(op)


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
