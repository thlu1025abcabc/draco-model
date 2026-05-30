from __future__ import annotations

import polars as pl

from draco_model.core import Node
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FrameSchema, register_executor, register_schema


class Source:
    """Factory for raw source frame nodes."""

    def __new__(cls, source: str, *, lookback_days: int = 1, name: str | None = None) -> Node:
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1.")
        return Node(
            kind="frame",
            op="source",
            params={"source": source, "lookback_days": lookback_days},
            name=name,
        )


@register_executor("source")
def _source(node: Node, context: EvalContext) -> pl.LazyFrame:
    dates = context.trading_calendar.previous_sessions(context.eval_date, int(node.params.get("lookback_days", 1)))
    return context.sources.scan(str(node.params["source"]), dates)


@register_schema("source")
def _source_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    dates = context.trading_calendar.previous_sessions(context.eval_date, int(node.params.get("lookback_days", 1)))
    frame = context.sources.scan(str(node.params["source"]), dates)
    columns = tuple(frame.collect_schema().names())
    if all(column in columns for column in KEY_COLUMNS):
        keys = KEY_COLUMNS
        grain = "raw"
    elif all(column in columns for column in DAILY_KEY_COLUMNS):
        keys = DAILY_KEY_COLUMNS
        grain = "daily"
    else:
        keys = ()
        grain = "raw"
    return FrameSchema(columns=columns, keys=keys, grain=grain)
