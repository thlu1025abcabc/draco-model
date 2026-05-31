from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from draco_model.core import Layer, Node
from draco_model.layers.expressions import sum_or_null
from draco_model.layers.names import validate_public_alias
from draco_model.layers.operators import ARITHMETIC_OPS
from draco_model.market.minute_calendar import AUCTION_MINUTES
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FieldInfo, FramePlan, FrameSchema, register_executor, register_plan


APPLY_TO_COMPONENTS = "components"
APPLY_TO_FIELD = "field"
DAILY_FREQUENCIES = {"1d", "daily"}


@dataclass(frozen=True)
class AggregateValueSpec:
    """One public field aggregation described by the aggregate layout plan."""

    source: str
    output: str
    info: FieldInfo
    component_sources: tuple[str, ...] = ()
    component_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class AggregatePlan(FramePlan):
    """Output layout and expression specs for an aggregate node."""

    value_specs: tuple[AggregateValueSpec, ...] = ()
    payloads: tuple[str, ...] = ()


class Aggregate(Layer):
    """Aggregate raw, minute, or daily fields to a target frequency."""

    op = "aggregate"

    def __init__(
        self,
        frequency: str,
        agg: str,
        *,
        apply_to: str = APPLY_TO_FIELD,
        value_col: str | None = None,
        alias: str | None = None,
        auction: str = "keep",
        name: str | None = None,
    ) -> None:
        if apply_to not in {APPLY_TO_COMPONENTS, APPLY_TO_FIELD}:
            raise ValueError("apply_to must be 'components' or 'field'.")
        if auction not in {"keep", "drop", "merge"}:
            raise ValueError("auction must be 'keep', 'drop', or 'merge'.")
        if alias is not None:
            validate_public_alias(alias)
        super().__init__(
            name=name,
            frequency=frequency,
            agg=agg,
            apply_to=apply_to,
            value_col=value_col,
            alias=alias,
            auction=None if auction == "keep" else auction,
        )


@register_executor("aggregate")
def _aggregate(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    schema = context.infer_schema(parent)
    frame = context.evaluate(parent)
    frequency = str(node.params["frequency"]).strip().lower()
    agg = str(node.params["agg"])
    apply_to = str(node.params.get("apply_to", APPLY_TO_FIELD))
    value_col = node.params.get("value_col")
    alias = node.params.get("alias")
    auction = str(node.params.get("auction", "keep"))

    if frequency in DAILY_FREQUENCIES:
        if "minute" in schema.columns:
            frame = frame.with_columns(pl.col("minute").alias("__order_minute"))
            frame = _apply_auction(frame, auction, context, 1)
            if auction == "merge":
                frame, schema = _merge_auction_frame(frame, schema, agg, apply_to, value_col, alias)
                if alias is not None:
                    value_col = alias
        grouped, _ = _aggregate_values(
            frame,
            schema,
            DAILY_KEY_COLUMNS,
            agg,
            apply_to,
            value_col,
            alias,
            order_col="minute" if "minute" in schema.columns else None,
        )
        return grouped

    interval = parse_minute_frequency(frequency)
    if "minute" in schema.columns:
        frame = frame.with_columns(pl.col("minute").alias("__order_minute"))
    frame = _apply_auction(frame, auction, context, interval)
    if interval == 1:
        grouped, _ = _aggregate_values(
            frame,
            schema,
            KEY_COLUMNS,
            agg,
            apply_to,
            value_col,
            alias,
            order_col="__order_minute" if "minute" in schema.columns else None,
        )
        return grouped.sort(list(KEY_COLUMNS))

    bucketed = _bucket_minutes(frame, interval, context)
    grouped, _ = _aggregate_values(
        bucketed,
        schema,
        KEY_COLUMNS,
        agg,
        apply_to,
        value_col,
        alias,
        order_col="__order_minute",
    )
    return grouped.sort(list(KEY_COLUMNS))


@register_plan("aggregate")
def _aggregate_plan(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FramePlan:
    parent = parent_schemas["input"]
    frequency = str(node.params["frequency"]).strip().lower()
    apply_to = str(node.params.get("apply_to", APPLY_TO_FIELD))
    value_col = node.params.get("value_col")
    alias = node.params.get("alias")
    if frequency in DAILY_FREQUENCIES:
        keys = DAILY_KEY_COLUMNS
        grain = "daily"
    else:
        parse_minute_frequency(frequency)
        keys = KEY_COLUMNS
        grain = "minute"
    return _aggregate_plan_from_schema(parent, keys, grain, apply_to, value_col, alias)


def aggregate_value_columns(schema: FrameSchema, value_col: object | None = None) -> list[str]:
    """Return public value columns selected for aggregation."""
    if value_col is not None:
        col = str(value_col)
        if col not in schema.columns:
            raise ValueError(f"Aggregate value_col {col!r} is not available.")
        return [col]
    values = schema.value_columns()
    if not values:
        keys = set(schema.keys)
        values = [column for column in schema.columns if column not in keys and not column.startswith("__")]
    if not values:
        raise ValueError("Aggregate input has no public value columns.")
    return values


def parse_minute_frequency(frequency: str) -> int:
    text = frequency.strip().lower()
    if not text.endswith("m"):
        raise ValueError("frequency must look like '1m', '5m', '15m', '1d', or 'daily'.")
    value = int(text[:-1])
    if value < 1:
        raise ValueError("frequency interval must be >= 1.")
    return value


def _aggregate_values(
    frame: pl.LazyFrame,
    schema: FrameSchema,
    group_keys: tuple[str, ...],
    agg: str,
    apply_to: str,
    value_col: object | None,
    alias: object | None,
    *,
    order_col: str | None,
) -> tuple[pl.LazyFrame, AggregatePlan]:
    grain = "daily" if group_keys == DAILY_KEY_COLUMNS else "minute"
    plan = _aggregate_plan_from_schema(schema, group_keys, grain, apply_to, value_col, alias)
    exprs: list[pl.Expr] = []
    recompute: list[pl.Expr] = []

    for spec in plan.value_specs:
        if spec.component_sources:
            for component, out_component in zip(spec.component_sources, spec.component_outputs):
                exprs.append(_agg_expr(component, agg, order_col).alias(out_component))
            recompute.append(_operator_expr(spec.info.operator, spec.component_outputs).alias(spec.output))
        else:
            exprs.append(_agg_expr(spec.source, agg, order_col).alias(spec.output))
    for payload in plan.payloads:
        exprs.append(_agg_expr(payload, agg, order_col).alias(payload))
    grouped = frame.group_by(list(group_keys)).agg(exprs)
    if recompute:
        grouped = grouped.with_columns(recompute)
    return grouped.select(list(plan.columns)), plan


def _aggregate_plan_from_schema(
    parent: FrameSchema,
    keys: tuple[str, ...],
    grain: str,
    apply_to: str,
    value_col: object | None,
    alias: object | None,
) -> AggregatePlan:
    values = aggregate_value_columns(parent, value_col)
    if alias is not None and len(values) != 1:
        raise ValueError("Aggregate alias requires exactly one value column.")
    columns: list[str] = list(keys)
    fields: dict[str, FieldInfo] = {}
    specs: list[AggregateValueSpec] = []
    consumed_payloads: set[str] = set()
    for column in values:
        info = _field_for_column(parent, column)
        output = str(alias) if alias is not None else column
        columns.append(output)
        components: tuple[str, ...] = ()
        operator = "identity"
        component_agg = False
        if apply_to == APPLY_TO_COMPONENTS and info.component_agg and info.components:
            components = tuple(f"__op_{output}_{idx}" for idx, _ in enumerate(info.components))
            columns.extend(components)
            operator = info.operator
            component_agg = True
            consumed_payloads.update(info.components)
        fields[output] = FieldInfo(
            name=output,
            column=output,
            operator=operator,
            components=components,
            source=info.source,
            lookback_days=info.lookback_days,
            component_agg=component_agg,
        )
        specs.append(
            AggregateValueSpec(
                source=column,
                output=output,
                info=info,
                component_sources=info.components if components else (),
                component_outputs=components,
            )
        )
    output_columns = set(columns) - set(keys)
    payloads = tuple(
        payload
        for payload in _payload_columns(parent)
        if payload not in consumed_payloads and payload not in output_columns
    )
    columns.extend(payloads)
    return AggregatePlan(
        columns=tuple(columns),
        keys=keys,
        grain=grain,
        fields=fields,
        value_specs=tuple(specs),
        payloads=payloads,
    )


def _apply_auction(frame: pl.LazyFrame, auction: str, context: EvalContext, interval: int) -> pl.LazyFrame:
    if auction == "drop":
        return frame.filter(~pl.col("minute").is_in(AUCTION_MINUTES))
    if auction == "merge":
        opening_auction, closing_auction = AUCTION_MINUTES
        first_continuous, last_continuous = _auction_merge_targets(context, interval)
        return frame.with_columns(
            pl.when(pl.col("minute") == opening_auction)
            .then(first_continuous)
            .when(pl.col("minute") == closing_auction)
            .then(last_continuous)
            .otherwise(pl.col("minute"))
            .alias("minute")
        )
    return frame


def _auction_merge_targets(context: EvalContext, interval: int) -> tuple[int, int]:
    continuous = [minute for minute in context.minute_calendar.minbars() if minute not in AUCTION_MINUTES]
    if not continuous:
        raise ValueError("Minute calendar must contain at least one non-auction minute for auction='merge'.")
    if interval == 1:
        return continuous[0], continuous[-1]
    bucket_map = context.minute_calendar.bucket_map(interval).collect()
    buckets = bucket_map["__bucket_minute"].to_list()
    return int(buckets[0]), int(buckets[-1])


def _merge_auction_frame(
    frame: pl.LazyFrame,
    schema: FrameSchema,
    agg: str,
    apply_to: str,
    value_col: object | None,
    alias: object | None,
) -> tuple[pl.LazyFrame, FrameSchema]:
    merged, plan = _aggregate_values(
        frame,
        schema,
        KEY_COLUMNS,
        agg,
        apply_to,
        value_col,
        alias,
        order_col="__order_minute",
    )
    return merged, FrameSchema(
        columns=plan.columns,
        keys=KEY_COLUMNS,
        grain="minute",
        fields=plan.fields,
    )


def _bucket_minutes(frame: pl.LazyFrame, interval: int, context: EvalContext) -> pl.LazyFrame:
    auctions = frame.filter(pl.col("minute").is_in(AUCTION_MINUTES))
    continuous = frame.filter(~pl.col("minute").is_in(AUCTION_MINUTES))
    bucketed = (
        continuous.join(context.minute_calendar.bucket_map(interval), on="minute", how="inner")
        .with_columns(pl.col("__bucket_minute").alias("minute"))
        .drop("__bucket_minute")
    )
    return pl.concat([bucketed, auctions], how="diagonal_relaxed")


def _agg_expr(column: str, method: str, order_col: str | None) -> pl.Expr:
    method = method.lower()
    expr = pl.col(column)
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


def _operator_expr(operator: str, components: tuple[str, ...]) -> pl.Expr:
    if len(components) != 2:
        raise ValueError(f"Component aggregation for {operator!r} requires two components.")
    left = pl.col(components[0])
    right = pl.col(components[1])
    if operator == "add":
        return left + right
    if operator == "sub":
        return left - right
    if operator == "mul":
        return left * right
    if operator == "div":
        return pl.when(right == 0).then(None).otherwise(left / right)
    raise ValueError(f"Unsupported component aggregation operator {operator!r}.")


def _field_for_column(schema: FrameSchema, column: str) -> FieldInfo:
    for info in schema.fields.values():
        if info.column == column:
            return info
    return FieldInfo(name=column, column=column)


def _payload_columns(schema: FrameSchema) -> list[str]:
    return [
        column
        for column in schema.columns
        if column not in schema.keys and column.startswith("__")
    ]
