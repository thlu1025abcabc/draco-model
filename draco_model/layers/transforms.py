from __future__ import annotations

from dataclasses import replace

import polars as pl

from draco_model.core import Layer, Node
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FieldInfo, FrameSchema, register_executor, register_schema


class FillNull(Layer):
    """Fill nulls in a single public field."""

    op = "fill_null"

    def __init__(self, value: int | float | str = "state", *, name: str | None = None) -> None:
        super().__init__(name=name, value=value)

    def __call__(self, frame: Node) -> Node:
        inputs = {"input": frame}
        if self.params.get("value") == "state":
            source = _find_unique_source(frame)
            if source is not None:
                source_name, lookback = source
                inputs["close_state"] = _build_close_state_subtree(source_name, lookback, _aggregate_transforms(frame))
        return Node(kind="frame", op=self.op, params=dict(self.params), inputs=inputs, name=self.name)


@register_executor("fill_null")
def _fill_null(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    schema = context.infer_schema(parent)
    value = node.params.get("value", "state")
    value_col = _single_value_column(schema)
    info = _field_for_column(schema, value_col)

    if _is_numeric_fill(value):
        return context.evaluate(parent).with_columns(pl.col(value_col).fill_null(value).alias(value_col))
    if value == "ffill":
        return (
            context.evaluate(parent)
            .sort(list(schema.keys))
            .with_columns(pl.col(value_col).forward_fill().over(list(DAILY_KEY_COLUMNS)).alias(value_col))
        )
    if value != "state":
        raise ValueError("FillNull supports numeric literals, 'ffill', or 'state'.")

    close_state = _close_state_from_node(node, info, context)
    if info.operator == "preclose":
        return _preclose_from_state(close_state, info.column)
    return (
        context.evaluate(parent)
        .join(close_state.select([*KEY_COLUMNS, "__close_state"]), on=list(KEY_COLUMNS), how="left")
        .with_columns(pl.col(value_col).fill_null(pl.col("__close_state")).alias(value_col))
        .drop("__close_state")
    )


@register_schema("fill_null")
def _fill_null_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    parent = parent_schemas["input"]
    value_col = _single_value_column(parent)
    info = _field_for_column(parent, value_col)
    return FrameSchema(
        columns=parent.columns,
        keys=parent.keys,
        grain=parent.grain,
        fields={value_col: replace(info, name=value_col, column=value_col, components=(), component_agg=False)},
    )


def _close_state_from_node(node: Node, info: FieldInfo, context: EvalContext) -> pl.LazyFrame:
    if "close_state" in node.inputs:
        close_node = node.inputs["close_state"]
    else:
        if info.source is None:
            raise ValueError("FillNull('state') requires source lineage.")
        close_node = _build_close_state_subtree(info.source, info.lookback_days, ())
    close = context.evaluate(close_node)
    dates = context.trading_calendar.previous_sessions(context.eval_date, info.lookback_days)
    daily_preclose = _daily_preclose_frame(context, dates)
    return (
        close.join(daily_preclose, on=list(DAILY_KEY_COLUMNS), how="left")
        .sort(list(KEY_COLUMNS))
        .with_columns(
            pl.col("close")
            .forward_fill()
            .over(list(DAILY_KEY_COLUMNS))
            .fill_null(pl.col("__daily_preclose"))
            .alias("__close_state")
        )
        .select([*KEY_COLUMNS, "__close_state", "__daily_preclose"])
    )


def _preclose_from_state(state: pl.LazyFrame, output_column: str) -> pl.LazyFrame:
    return (
        state.sort(list(KEY_COLUMNS))
        .with_columns(
            pl.col("__close_state")
            .shift(1)
            .over(list(DAILY_KEY_COLUMNS))
            .fill_null(pl.col("__daily_preclose"))
            .alias(output_column)
        )
        .select([*KEY_COLUMNS, output_column])
    )


def _daily_preclose_frame(context: EvalContext, dates: list[str]) -> pl.LazyFrame:
    frame = context.sources.scan("daily_k", dates)
    columns = frame.collect_schema().names()
    missing = [column for column in [*DAILY_KEY_COLUMNS, "preclose"] if column not in columns]
    if missing:
        raise ValueError(f"daily_k preclose source is missing columns: {missing}.")
    return frame.select([*DAILY_KEY_COLUMNS, pl.col("preclose").alias("__daily_preclose")])


def _build_close_state_subtree(source_name: str, lookback_days: int, transforms: tuple[dict, ...]) -> Node:
    from draco_model.layers.aggregate import Aggregate
    from draco_model.layers.metrics import Metric
    from draco_model.layers.source import Source

    node = Metric("close", Source(source_name, lookback_days=lookback_days))
    for params in transforms:
        node = Aggregate(
            str(params["frequency"]),
            "last",
            auction=str(params.get("auction", "keep")),
            apply_to="field",
        )(node)
    return node


def _aggregate_transforms(node: Node) -> tuple[dict, ...]:
    transforms: list[dict] = []
    current = node
    while current.op == "aggregate":
        transforms.append(dict(current.params))
        current = current.inputs["input"]
    return tuple(reversed(transforms))


def _find_unique_source(node: Node) -> tuple[str, int] | None:
    found: set[tuple[str, int]] = set()

    def visit(item: Node) -> None:
        if item.op == "source":
            found.add((str(item.params["source"]), int(item.params.get("lookback_days", 1))))
            return
        for parent in item.inputs.values():
            if parent.kind == "frame":
                visit(parent)

    visit(node)
    return next(iter(found)) if len(found) == 1 else None


def _single_value_column(schema: FrameSchema) -> str:
    values = schema.value_columns()
    if len(values) != 1:
        raise ValueError(f"FillNull requires exactly one public value column, got {values}.")
    return values[0]


def _field_for_column(schema: FrameSchema, column: str) -> FieldInfo:
    for info in schema.fields.values():
        if info.column == column:
            return info
    return FieldInfo(column, column)


def _is_numeric_fill(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
