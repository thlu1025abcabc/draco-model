from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from draco_model.core import Layer, Node
from draco_model.layers.aggregate import (
    APPLY_TO_COMPONENTS,
    APPLY_TO_FIELD,
    aggregate_frame,
    aggregate_input_columns,
    aggregate_value_columns,
    parse_minute_frequency,
    ratio_payload_columns,
    ratio_payloads,
    resample_columns,
    resample_frame,
)
from draco_model.layers.inputs.field import Field
from draco_model.layers.inputs.input import Input
from draco_model.market.minute_calendar import AUCTION_MINUTES
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FrameSchema, register_executor, register_schema


class Fill(Layer):
    """Fill missing values with an explicit rule."""

    op = "fill"

    def __init__(self, value: int | float | str = "state", *, name: str | None = None) -> None:
        super().__init__(name=name, value=value)

    def __call__(self, inputs: Node) -> Node:
        """Build a fill node, including explicit close_state dependency for state fill."""
        if self.params.get("value") != "state":
            return super().__call__(inputs)
        lineage = _parse_state_fill_lineage(inputs)
        return Node(
            kind=self.output_kind,
            op=self.op,
            params=dict(self.params),
            inputs={"input": inputs, "close_state": _build_close_state_subtree(lineage)},
            name=self.name,
        )


class Auction(Layer):
    """Handle auction minutes without relying on input policies."""

    op = "auction"

    def __init__(
        self,
        mode: str,
        agg: str | None = None,
        *,
        apply_to: str = APPLY_TO_COMPONENTS,
        name: str | None = None,
    ) -> None:
        if mode not in {"drop", "merge"}:
            raise ValueError("auction mode must be 'drop' or 'merge'.")
        if apply_to not in {APPLY_TO_COMPONENTS, APPLY_TO_FIELD}:
            raise ValueError("apply_to must be 'components' or 'field'.")
        super().__init__(
            name=name,
            mode=mode,
            agg=agg,
            apply_to=None if apply_to == APPLY_TO_COMPONENTS else apply_to,
        )


class Resample(Layer):
    """Resample minute bars using one explicit aggregation method."""

    op = "resample"

    def __init__(
        self,
        frequency: str,
        agg: str,
        *,
        apply_to: str = APPLY_TO_COMPONENTS,
        name: str | None = None,
    ) -> None:
        if apply_to not in {APPLY_TO_COMPONENTS, APPLY_TO_FIELD}:
            raise ValueError("apply_to must be 'components' or 'field'.")
        super().__init__(
            name=name,
            frequency=frequency,
            agg=agg,
            apply_to=None if apply_to == APPLY_TO_COMPONENTS else apply_to,
        )


@register_executor("fill")
def _fill_executor(node: Node, context: EvalContext) -> pl.LazyFrame:
    value = node.params.get("value", "state")
    parent = node.inputs["input"]
    parent_columns = list(context.infer_schema(parent).columns)
    if _is_numeric_fill(value):
        return _fill_literal(context.evaluate(parent), parent_columns, value)
    if value == "ffill":
        return _fill_forward(context.evaluate(parent), parent_columns)
    if value != "state":
        raise ValueError("Fill supports numeric literals, 'ffill', or 'state'.")

    lineage = _parse_state_fill_lineage(parent)
    try:
        close_state_node = node.inputs["close_state"]
    except KeyError:
        raise ValueError("Fill('state') node is missing its explicit close_state input.") from None
    state = _close_state_from_close(context.evaluate(close_state_node), lineage, context)

    if lineage.field == "preclose":
        return _preclose_from_state(state, lineage.output_column)

    frame = context.evaluate(parent)
    value_columns = aggregate_value_columns(parent_columns, KEY_COLUMNS)
    if len(value_columns) != 1:
        raise ValueError(f"Fill('state') requires exactly one value column, got {value_columns}.")
    value_column = value_columns[0]
    return (
        frame.join(state.select([*KEY_COLUMNS, "__close_state"]), on=list(KEY_COLUMNS), how="left")
        .with_columns(pl.col(value_column).fill_null(pl.col("__close_state")).alias(value_column))
        .drop("__close_state")
        .drop(ratio_payload_columns(ratio_payloads([value_column], parent_columns)))
    )


def _fill_literal(frame: pl.LazyFrame, columns: list[str], value: int | float) -> pl.LazyFrame:
    value_column = _single_value_column(columns, f"Fill({value!r})")
    payloads = ratio_payloads([value_column], columns)
    return frame.with_columns(pl.col(value_column).fill_null(value).alias(value_column)).drop(
        ratio_payload_columns(payloads)
    )


def _fill_forward(frame: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    value_column = _single_value_column(columns, "Fill('ffill')")
    payloads = ratio_payloads([value_column], columns)
    frame = frame.sort(list(KEY_COLUMNS))
    if value_column not in payloads:
        return frame.with_columns(pl.col(value_column).forward_fill().over(list(DAILY_KEY_COLUMNS)).alias(value_column))
    numerator, denominator = payloads[value_column]
    return frame.with_columns(
        [
            pl.col(numerator).forward_fill().over(list(DAILY_KEY_COLUMNS)).alias(numerator),
            pl.col(denominator).forward_fill().over(list(DAILY_KEY_COLUMNS)).alias(denominator),
        ]
    ).with_columns(_ratio_expr(numerator, denominator).alias(value_column))


@register_executor("auction")
def _auction_executor(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    return _auction_frame(
        context.evaluate(parent),
        str(node.params["mode"]),
        node.params.get("agg"),
        list(context.infer_schema(parent).columns),
        str(node.params.get("apply_to", APPLY_TO_COMPONENTS)),
    )


def _auction_frame(
    frame: pl.LazyFrame,
    mode: str,
    agg: object | None = None,
    columns: list[str] | None = None,
    apply_to: str = APPLY_TO_COMPONENTS,
) -> pl.LazyFrame:
    columns = columns or frame.collect_schema().names()
    if mode == "drop":
        return frame.filter(~pl.col("minute").is_in(AUCTION_MINUTES))
    if mode == "merge":
        if agg not in {"first", "last", "sum"}:
            raise ValueError("Auction('merge') requires agg to be 'first', 'last', or 'sum'.")
        value_columns = aggregate_value_columns(columns, KEY_COLUMNS)
        if not value_columns:
            raise ValueError("Auction('merge') input has no value columns to aggregate.")
        pass_columns = aggregate_input_columns(columns, value_columns, apply_to)
        remapped = frame.select([*KEY_COLUMNS, *pass_columns]).with_columns(
            [
                pl.col("minute").alias("__order_minute"),
                pl.when(pl.col("minute") == 925)
                .then(930)
                .when(pl.col("minute") == 1456)
                .then(1500)
                .otherwise(pl.col("minute"))
                .alias("minute"),
            ]
        )
        return aggregate_frame(
            remapped,
            columns,
            value_columns,
            str(agg),
            group_keys=KEY_COLUMNS,
            apply_to=apply_to,
            order_col="__order_minute",
            keep_payload=apply_to == APPLY_TO_COMPONENTS,
        )
    raise ValueError("auction mode must be 'drop' or 'merge'.")


@register_executor("resample")
def _resample_executor(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    return resample_frame(
        context.evaluate(parent),
        list(context.infer_schema(parent).columns),
        str(node.params["frequency"]),
        str(node.params["agg"]),
        context,
        apply_to=str(node.params.get("apply_to", APPLY_TO_COMPONENTS)),
    )


@register_schema("fill")
def _fill_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    columns = list(parent_schemas["input"].columns)
    value = node.params.get("value", "state")
    if value == "ffill":
        return FrameSchema(tuple(columns))
    value_column = _single_value_column(columns, f"Fill({value!r})")
    payload_columns = set(ratio_payload_columns(ratio_payloads([value_column], columns)))
    return FrameSchema(tuple(column for column in columns if column not in payload_columns))


@register_schema("auction")
def _auction_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    if node.params.get("mode") != "merge" or node.params.get("apply_to", APPLY_TO_COMPONENTS) == APPLY_TO_COMPONENTS:
        return parent_schemas["input"]
    columns = list(parent_schemas["input"].columns)
    return FrameSchema(tuple([*KEY_COLUMNS, *aggregate_value_columns(columns, KEY_COLUMNS)]))


@register_schema("resample")
def _resample_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    frequency = str(node.params["frequency"])
    if parse_minute_frequency(frequency) == 1:
        return parent_schemas["input"]
    return FrameSchema(
        tuple(
            resample_columns(
                list(parent_schemas["input"].columns),
                str(node.params.get("apply_to", APPLY_TO_COMPONENTS)),
                None,
                None,
            )
        )
    )


@dataclass(frozen=True)
class _StateFillLineage:
    source: str
    lookback_days: int
    field: str
    output_column: str
    transforms: tuple[Node, ...]


def _parse_state_fill_lineage(node: Node) -> _StateFillLineage:
    transforms: list[Node] = []
    current = node
    while current.op in {"auction", "resample"}:
        transforms.append(current)
        current = current.inputs["input"]

    if current.op == "field":
        field = str(current.params["name"])
        output_column = str(current.params.get("alias") or field)
    elif current.op == "ratio_field":
        output_column = str(current.params["alias"])
        field = output_column
    else:
        raise ValueError("Fill('state') requires a Field(...) or RatioField(...) input chain.")
    source_node = current.inputs["input"]
    if source_node.op != "input":
        raise ValueError("Fill('state') requires Field(...) to read from Input(source=...).")
    return _StateFillLineage(
        source=str(source_node.params["source"]),
        lookback_days=int(source_node.params.get("lookback_days", 1)),
        field=field,
        output_column=output_column,
        transforms=tuple(reversed(transforms)),
    )


def _build_close_state_subtree(lineage: _StateFillLineage) -> Node:
    node = Field("close")(Input(source=lineage.source, lookback_days=lineage.lookback_days))
    for transform in lineage.transforms:
        if transform.op == "auction":
            mode = str(transform.params["mode"])
            node = Auction(mode, agg="last" if mode == "merge" else None)(node)
        elif transform.op == "resample":
            node = Resample(str(transform.params["frequency"]), "last")(node)
        else:
            raise ValueError(f"Fill('state') cannot replay transform {transform.op!r}.")
    return node


def _close_state_from_close(close: pl.LazyFrame, lineage: _StateFillLineage, context: EvalContext) -> pl.LazyFrame:
    dates = context.trading_calendar.previous_sessions(context.eval_date, lineage.lookback_days)
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


def _single_value_column(columns: list[str], subject: str) -> str:
    value_columns = aggregate_value_columns(columns, KEY_COLUMNS)
    if len(value_columns) != 1:
        raise ValueError(f"{subject} requires exactly one value column, got {value_columns}.")
    return value_columns[0]


def _is_numeric_fill(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ratio_expr(numerator: str, denominator: str) -> pl.Expr:
    return pl.when(pl.col(denominator) == 0).then(None).otherwise(pl.col(numerator) / pl.col(denominator))
