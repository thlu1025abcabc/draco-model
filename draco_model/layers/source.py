from __future__ import annotations

import polars as pl

from draco_model.core import Node
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FramePlan, FrameSchema, register_executor, register_plan


def Source(source: str, *, lookback_days: int = 1, name: str | None = None) -> Node:
    """Create a raw source frame node."""
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
    source = str(node.params["source"])
    columns = context.sources.schema(source, dates)
    return context.sources.scan(source, dates).select(list(columns))


@register_plan("source")
def _source_plan(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FramePlan:
    dates = context.trading_calendar.previous_sessions(context.eval_date, int(node.params.get("lookback_days", 1)))
    columns = context.sources.schema(str(node.params["source"]), dates)
    if all(column in columns for column in KEY_COLUMNS):
        keys = KEY_COLUMNS
        grain = "raw"
    elif all(column in columns for column in DAILY_KEY_COLUMNS):
        keys = DAILY_KEY_COLUMNS
        grain = "daily"
    else:
        keys = ()
        grain = "raw"
    return FramePlan(columns=columns, keys=keys, grain=grain)
