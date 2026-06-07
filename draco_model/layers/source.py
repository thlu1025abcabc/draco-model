from __future__ import annotations

import logging

import polars as pl

from draco_model.core import Node
from draco_model.runtime.execution import EvalContext, FrameInfo, register_executor, register_info


logger = logging.getLogger(__name__)


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
    logger.debug("source.evaluate source=%s dates=%s columns=%d", source, dates, len(columns))
    # Fixed source contracts intentionally narrow extra normalized columns.
    return context.sources.scan(source, dates).select(list(columns))


@register_info("source")
def _source_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    source = str(node.params["source"])
    lookback_days = int(node.params.get("lookback_days", 1))
    dates = context.trading_calendar.previous_sessions(context.eval_date, lookback_days)
    columns = context.sources.schema(source, dates)
    identity_keys = context.sources.identity_keys(source, dates)
    return FrameInfo.from_columns(
        columns,
        identity_keys=identity_keys,
        source=source,
        lookback_days=lookback_days,
    )
