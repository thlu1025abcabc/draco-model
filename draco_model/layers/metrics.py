from __future__ import annotations

import polars as pl

from draco_model.core import Node
from draco_model.layers.aggregate import Aggregate
from draco_model.layers.filters import Flag, Side, Where
from draco_model.layers.names import validate_public_alias
from draco_model.layers.operators import Col
from draco_model.market.schema import KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FieldInfo, FramePlan, FrameSchema, register_executor, register_plan


def Metric(name: str, source: Node, *, alias: str | None = None) -> Node:
    """Expand a named metric into its semantic DAG recipe."""
    output = alias or name
    validate_public_alias(output)
    if name == "volume":
        return Aggregate("1m", "sum", value_col="volume", alias=output)(Col("volume")(source))
    if name == "no":
        return Aggregate("1m", "sum", value_col="no", alias=output)(Col("no")(source))
    if name == "amount":
        row = (Col("price") * Col("volume")).alias("amount")(source)
        return Aggregate("1m", "sum", value_col="amount", alias=output)(row)
    if name == "buyamount":
        row = (Col("price") * Col("volume")).alias("amount")(Where(Side("buy"))(source))
        return Aggregate("1m", "sum", value_col="amount", alias=output)(row)
    if name == "sellamount":
        row = (Col("price") * Col("volume")).alias("amount")(Where(Side("sell"))(source))
        return Aggregate("1m", "sum", value_col="amount", alias=output)(row)
    if name == "vwap":
        return (Metric("amount", source) / Metric("volume", source)).alias(output)
    if name == "close":
        return Aggregate("1m", "last", value_col="close", alias=output)(
            Col("price").alias("close")(Where(Flag("is_last"))(source))
        )
    if name == "open":
        return Aggregate("1m", "first", value_col="open", alias=output)(
            Col("price").alias("open")(Where(Flag("is_first"))(source))
        )
    if name == "high":
        return Aggregate("1m", "max", value_col="high", alias=output)(Col("price").alias("high")(source))
    if name == "low":
        return Aggregate("1m", "min", value_col="low", alias=output)(Col("price").alias("low")(source))
    if name == "preclose":
        return Node(kind="frame", op="metric_reserved", params={"name": "preclose", "alias": output}, inputs={"input": source})
    raise ValueError(f"Unsupported metric {name!r}.")


@register_executor("metric_reserved")
def _metric_reserved(node: Node, context: EvalContext) -> pl.LazyFrame:
    raise ValueError(
        "preclose metric is reserved; use FillNull('state')(Metric('preclose', Source(...))) "
        "to derive it from close_state."
    )


@register_plan("metric_reserved")
def _metric_reserved_plan(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FramePlan:
    alias = str(node.params["alias"])
    source = _source_name(node.inputs["input"])
    lookback = int(node.inputs["input"].params.get("lookback_days", 1)) if node.inputs["input"].op == "source" else 1
    return FramePlan(
        columns=(*KEY_COLUMNS, alias),
        keys=KEY_COLUMNS,
        grain="minute",
        fields={
            alias: FieldInfo(
                name=alias,
                column=alias,
                operator="preclose",
                source=source,
                lookback_days=lookback,
            )
        },
    )


def _source_name(node: Node) -> str | None:
    if node.op == "source":
        return str(node.params["source"])
    return None
