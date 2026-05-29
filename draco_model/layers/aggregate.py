from __future__ import annotations

import polars as pl

from draco_model.core import Layer, Node
from draco_model.layers.expressions import sum_or_null
from draco_model.market.minute_calendar import AUCTION_MINUTES
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FrameSchema, register_executor, register_schema


APPLY_TO_COMPONENTS = "components"
APPLY_TO_FIELD = "field"
DAILY_FREQUENCIES = {"1d", "daily"}


class Aggregate(Layer):
    """Aggregate frame values to another time frequency."""

    op = "aggregate"

    def __init__(
        self,
        frequency: str,
        agg: str,
        *,
        apply_to: str = APPLY_TO_FIELD,
        value_col: str | None = None,
        alias: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create a frequency-aware aggregation node."""
        apply_to = _normalize_apply_to(apply_to)
        super().__init__(
            name=name,
            frequency=frequency,
            agg=agg,
            apply_to=apply_to,
            value_col=value_col,
            alias=alias,
        )


class DailyAgg(Layer):
    """Aggregate an intraday value column into daily factor values."""

    op = "daily_agg"

    def __init__(
        self,
        *,
        value_col: str,
        agg: str,
        apply_to: str = APPLY_TO_FIELD,
        name: str | None = None,
    ) -> None:
        """Create a daily aggregation node for one input value column."""
        apply_to = _normalize_apply_to(apply_to)
        super().__init__(
            name=name,
            value_col=value_col,
            agg=agg,
            apply_to=None if apply_to == APPLY_TO_FIELD else apply_to,
        )


@register_executor("aggregate")
def _aggregate(node: Node, context: EvalContext) -> pl.LazyFrame:
    frequency = str(node.params["frequency"]).strip().lower()
    frame = context.evaluate(node.inputs["input"])
    columns = list(context.infer_schema(node.inputs["input"]).columns)
    apply_to = _normalize_apply_to(str(node.params.get("apply_to", APPLY_TO_FIELD)))
    if frequency in DAILY_FREQUENCIES:
        return _daily_aggregate(
            frame,
            columns,
            str(node.params["agg"]),
            apply_to,
            node.params.get("value_col"),
            node.params.get("alias"),
        )
    return resample_frame(
        frame,
        columns,
        frequency,
        str(node.params["agg"]),
        context,
        apply_to=apply_to,
        value_col=node.params.get("value_col"),
        alias=node.params.get("alias"),
    )


@register_schema("aggregate")
def _aggregate_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    frequency = str(node.params["frequency"]).strip().lower()
    parent_columns = list(parent_schemas["input"].columns)
    apply_to = _normalize_apply_to(str(node.params.get("apply_to", APPLY_TO_FIELD)))
    if frequency in DAILY_FREQUENCIES:
        value_col = _aggregate_value_col(parent_columns, node.params.get("value_col"), "Aggregate")
        alias = str(node.params.get("alias") or value_col)
        return FrameSchema((*DAILY_KEY_COLUMNS, alias))
    if parse_minute_frequency(frequency) == 1:
        return parent_schemas["input"]
    return FrameSchema(
        tuple(resample_columns(parent_columns, apply_to, node.params.get("value_col"), node.params.get("alias")))
    )


@register_executor("daily_agg")
def _daily_agg(node: Node, context: EvalContext) -> pl.LazyFrame:
    frame = context.evaluate(node.inputs["input"])
    col = str(node.params["value_col"])
    columns = list(context.infer_schema(node.inputs["input"]).columns)
    return _daily_aggregate(
        frame,
        columns,
        str(node.params["agg"]),
        _normalize_apply_to(str(node.params.get("apply_to", APPLY_TO_FIELD))),
        col,
        "value",
    )


@register_schema("daily_agg")
def _daily_agg_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    columns = parent_schemas["input"].columns
    col = str(node.params["value_col"])
    if col not in columns:
        raise ValueError(f"DailyAgg value_col {col!r} is not available.")
    return FrameSchema((*DAILY_KEY_COLUMNS, "value"))


def aggregate_frame(
    frame: pl.LazyFrame,
    columns: list[str],
    value_columns: list[str],
    agg: str,
    *,
    group_keys: tuple[str, ...],
    apply_to: str,
    order_col: str | None = None,
    keep_payload: bool,
    aliases: dict[str, str] | None = None,
) -> pl.LazyFrame:
    """Aggregate public fields or their components to the requested key grain."""
    apply_to = _normalize_apply_to(apply_to)
    aliases = aliases or {}
    payloads = ratio_payloads(value_columns, columns)
    exprs: list[pl.Expr] = []
    output_columns: list[str] = []
    ratio_exprs: list[pl.Expr] = []
    for column in value_columns:
        output = aliases.get(column, column)
        if apply_to == APPLY_TO_COMPONENTS and column in payloads:
            numerator, denominator = payloads[column]
            exprs.extend(
                [
                    _agg_expr(numerator, agg, order_col=order_col).alias(numerator),
                    _agg_expr(denominator, agg, order_col=order_col).alias(denominator),
                ]
            )
            ratio_exprs.append(_ratio_expr(numerator, denominator).alias(output))
            output_columns.append(output)
            if keep_payload:
                output_columns.extend([numerator, denominator])
        else:
            exprs.append(_agg_expr(column, agg, order_col=order_col).alias(output))
            output_columns.append(output)
    grouped = frame.group_by(list(group_keys)).agg(exprs)
    if ratio_exprs:
        grouped = grouped.with_columns(ratio_exprs)
    return grouped.select([*group_keys, *output_columns])


def resample_frame(
    frame: pl.LazyFrame,
    columns: list[str],
    frequency: str,
    agg: str,
    context: EvalContext,
    *,
    apply_to: str,
    value_col: object | None = None,
    alias: object | None = None,
) -> pl.LazyFrame:
    """Aggregate an intraday frame to a coarser minute frequency."""
    interval = parse_minute_frequency(frequency)
    if interval == 1:
        return frame
    value_columns = _selected_value_columns(columns, KEY_COLUMNS, value_col, "Aggregate")
    if not value_columns:
        raise ValueError("Resample input has no value columns to aggregate.")
    if alias is not None and apply_to == APPLY_TO_COMPONENTS:
        raise ValueError("Intraday component aggregation cannot rename output columns; project or aggregate by field first.")
    if alias is not None and len(value_columns) != 1:
        raise ValueError("Intraday aggregation alias requires exactly one value column.")
    aliases = {value_columns[0]: str(alias)} if alias is not None and len(value_columns) == 1 else {}
    pass_columns = aggregate_input_columns(columns, value_columns, apply_to)
    bucket_map = context.minute_calendar.bucket_map(interval)
    auctions = frame.filter(pl.col("minute").is_in(AUCTION_MINUTES)).select([*KEY_COLUMNS, *pass_columns])
    continuous = frame.filter(~pl.col("minute").is_in(AUCTION_MINUTES)).select([*KEY_COLUMNS, *pass_columns])
    bucketed = (
        continuous.with_columns(pl.col("minute").alias("__order_minute"))
        .join(bucket_map, on="minute", how="inner")
        .with_columns(pl.col("__bucket_minute").alias("minute"))
        .drop("__bucket_minute")
    )
    resampled = aggregate_frame(
        bucketed,
        columns,
        value_columns,
        agg,
        group_keys=KEY_COLUMNS,
        apply_to=apply_to,
        order_col="__order_minute",
        keep_payload=apply_to == APPLY_TO_COMPONENTS,
        aliases=aliases,
    )
    return pl.concat([resampled, auctions], how="diagonal_relaxed").sort(list(KEY_COLUMNS))


def aggregate_value_columns(columns: list[str], key_columns: tuple[str, ...]) -> list[str]:
    """Return public value columns, excluding keys and internal component columns."""
    return [column for column in columns if column not in key_columns and not column.startswith("__")]


def parse_minute_frequency(frequency: str) -> int:
    """Parse strings like '1m', '5m', and '15m' into minute intervals."""
    text = frequency.strip().lower()
    if not text.endswith("m"):
        raise ValueError("frequency must look like '1m', '5m', '15m', or 'daily'.")
    value = int(text[:-1])
    if value < 1:
        raise ValueError("frequency interval must be >= 1.")
    return value


def aggregate_input_columns(columns: list[str], value_columns: list[str], apply_to: str) -> list[str]:
    """Return physical columns needed to aggregate the requested public fields."""
    if _normalize_apply_to(apply_to) == APPLY_TO_FIELD:
        return value_columns
    payloads = ratio_payloads(value_columns, columns)
    return [*value_columns, *ratio_payload_columns(payloads)]


def ratio_payloads(value_columns: list[str], columns: list[str]) -> dict[str, tuple[str, str]]:
    """Return current ratio component payload columns keyed by public output column."""
    column_set = set(columns)
    payloads = {}
    for column in value_columns:
        numerator = f"__ratio_{column}_num"
        denominator = f"__ratio_{column}_den"
        if numerator in column_set and denominator in column_set:
            payloads[column] = (numerator, denominator)
    return payloads


def ratio_payload_columns(payloads: dict[str, tuple[str, str]]) -> list[str]:
    """Flatten ratio payload column pairs."""
    columns = []
    for numerator, denominator in payloads.values():
        columns.extend([numerator, denominator])
    return columns


def _daily_aggregate(
    frame: pl.LazyFrame,
    columns: list[str],
    agg: str,
    apply_to: str,
    value_col: object,
    alias: object | None,
) -> pl.LazyFrame:
    value_col = _aggregate_value_col(columns, value_col, "Daily aggregate")
    output = str(alias or value_col)
    return aggregate_frame(
        frame,
        columns,
        [value_col],
        agg,
        group_keys=DAILY_KEY_COLUMNS,
        apply_to=apply_to,
        order_col="minute" if "minute" in columns else None,
        keep_payload=False,
        aliases={value_col: output},
    )


def resample_columns(
    columns: list[str],
    apply_to: str,
    value_col: object | None,
    alias: object | None,
) -> list[str]:
    """Return the output schema columns for an intraday aggregation."""
    apply_to = _normalize_apply_to(apply_to)
    value_columns = _selected_value_columns(columns, KEY_COLUMNS, value_col, "Aggregate")
    if apply_to == APPLY_TO_FIELD:
        if alias is not None:
            if len(value_columns) != 1:
                raise ValueError("Intraday aggregation alias requires exactly one value column.")
            value_columns = [str(alias)]
        return [*KEY_COLUMNS, *value_columns]
    if alias is not None:
        raise ValueError("Intraday component aggregation cannot rename output columns; project or aggregate by field first.")
    payloads = ratio_payloads(value_columns, columns)
    return [*KEY_COLUMNS, *value_columns, *ratio_payload_columns(payloads)]


def _selected_value_columns(
    columns: list[str],
    key_columns: tuple[str, ...],
    value_col: object | None,
    subject: str,
) -> list[str]:
    if value_col is not None:
        col = str(value_col)
        if col not in columns:
            raise ValueError(f"{subject} value_col {col!r} is not available.")
        return [col]
    return aggregate_value_columns(columns, key_columns)


def _aggregate_value_col(columns: list[str] | tuple[str, ...], value_col: object, subject: str) -> str:
    if value_col is not None:
        col = str(value_col)
        if col not in columns:
            raise ValueError(f"{subject} value_col {col!r} is not available.")
        return col
    values = aggregate_value_columns(list(columns), KEY_COLUMNS)
    if len(values) != 1:
        raise ValueError(f"{subject} requires exactly one value column when value_col is omitted, got {values}.")
    return values[0]


def _normalize_apply_to(value: str) -> str:
    value = value.strip().lower()
    if value not in {APPLY_TO_COMPONENTS, APPLY_TO_FIELD}:
        raise ValueError("apply_to must be 'components' or 'field'.")
    return value


def _agg_expr(column: str | pl.Expr, method: str, *, order_col: str | None = None) -> pl.Expr:
    method = method.lower()
    expr = pl.col(column) if isinstance(column, str) else column
    if order_col is not None and method in {"first", "last"}:
        expr = expr.sort_by(order_col)
    if method == "sum":
        return sum_or_null(expr)
    if method == "mean":
        return expr.mean()
    if method == "max":
        return expr.max()
    if method == "min":
        return expr.min()
    if method == "std":
        return expr.std()
    if method == "median":
        return expr.median()
    if method == "first":
        return expr.drop_nulls().first()
    if method == "last":
        return expr.drop_nulls().last()
    raise ValueError(f"Unsupported aggregation {method!r}.")


def _ratio_expr(numerator: str, denominator: str) -> pl.Expr:
    return pl.when(pl.col(denominator) == 0).then(None).otherwise(pl.col(numerator) / pl.col(denominator))
