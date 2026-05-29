from __future__ import annotations

import polars as pl

from draco_model.core import Node
from draco_model.runtime.execution import EvalContext, FrameSchema, register_executor, register_schema


class Input:
    """Factory for raw source input nodes."""

    def __new__(cls, *, source: str, lookback_days: int = 1, name: str | None = None) -> Node:
        """Create an input node that scans one source."""
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1.")
        return Node(
            kind="frame",
            op="input",
            params={
                "source": source,
                "lookback_days": lookback_days,
            },
            name=name,
        )


@register_executor("input")
def _input(node: Node, context: EvalContext) -> pl.LazyFrame:
    params = node.params
    dates = context.trading_calendar.previous_sessions(context.eval_date, int(params.get("lookback_days", 1)))
    return context.sources.scan(str(params["source"]), dates)


@register_schema("input")
def _input_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    params = node.params
    dates = context.trading_calendar.previous_sessions(context.eval_date, int(params.get("lookback_days", 1)))
    frame = context.sources.scan(str(params["source"]), dates)
    return FrameSchema(tuple(frame.collect_schema().names()))
