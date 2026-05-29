from __future__ import annotations

from collections.abc import Callable

import polars as pl

from draco_model.core import Layer, Node
from draco_model.layers.expressions import sum_or_null
from draco_model.market.schema import KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FrameSchema, register_executor, register_schema


FieldBuilder = Callable[[pl.LazyFrame, list[str]], pl.LazyFrame]

FIELD_BUILDERS: dict[str, FieldBuilder] = {}


class Field(Layer):
    """Build one value field from a raw source frame."""

    op = "field"

    def __init__(self, field: str, *, alias: str | None = None, name: str | None = None) -> None:
        _validate_public_alias(alias)
        super().__init__(name=name)
        self.params = {"name": field}
        if alias is not None:
            self.params["alias"] = alias


class RatioField(Layer):
    """Build a ratio field that keeps numerator and denominator payloads."""

    op = "ratio_field"

    def __init__(
        self,
        numerator: str,
        denominator: str,
        *,
        alias: str | None = None,
        name: str | None = None,
    ) -> None:
        output = alias or f"{numerator}_over_{denominator}"
        _validate_public_alias(output)
        super().__init__(name=name, numerator=numerator, denominator=denominator, alias=output)


def register_field(name: str) -> Callable[[FieldBuilder], FieldBuilder]:
    """Register a field builder by output field name."""

    def decorator(builder: FieldBuilder) -> FieldBuilder:
        if name in FIELD_BUILDERS:
            raise ValueError(f"Duplicate field builder {name!r}.")
        FIELD_BUILDERS[name] = builder
        return builder

    return decorator


def get_field_builder(name: str) -> FieldBuilder:
    """Return a field builder or raise a clear unsupported-field error."""
    try:
        return FIELD_BUILDERS[name]
    except KeyError:
        available = ", ".join(sorted(FIELD_BUILDERS))
        raise ValueError(f"Unsupported field {name!r}. Available fields: {available}.") from None


@register_executor("field")
def _field_executor(node: Node, context: EvalContext) -> pl.LazyFrame:
    name = str(node.params["name"])
    alias = node.params.get("alias")
    raw = context.evaluate(node.inputs["input"])
    frame = get_field_builder(name)(raw, list(context.infer_schema(node.inputs["input"]).columns))
    if alias is not None:
        frame = frame.rename({name: str(alias)})
    return frame


@register_executor("ratio_field")
def _ratio_field_executor(node: Node, context: EvalContext) -> pl.LazyFrame:
    numerator = str(node.params["numerator"])
    denominator = str(node.params["denominator"])
    alias = str(node.params["alias"])
    raw = context.evaluate(node.inputs["input"])
    columns = list(context.infer_schema(node.inputs["input"]).columns)
    numerator_column = f"__ratio_{alias}_num"
    denominator_column = f"__ratio_{alias}_den"
    grouped = raw.group_by(list(KEY_COLUMNS)).agg(
        [
            _component_expr(numerator, columns).alias(numerator_column),
            _component_expr(denominator, columns).alias(denominator_column),
        ]
    )
    return grouped.with_columns(_ratio_expr(numerator_column, denominator_column).alias(alias)).select(
        [*KEY_COLUMNS, alias, numerator_column, denominator_column]
    )


@register_schema("field")
def _field_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    name = str(node.params["name"])
    alias = str(node.params.get("alias") or name)
    return FrameSchema((*KEY_COLUMNS, alias))


@register_schema("ratio_field")
def _ratio_field_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    alias = str(node.params["alias"])
    return FrameSchema((*KEY_COLUMNS, alias, f"__ratio_{alias}_num", f"__ratio_{alias}_den"))


@register_field("close")
def _close_field(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    _require_columns(columns, [*KEY_COLUMNS, "price", "is_last"])
    return raw.group_by(list(KEY_COLUMNS)).agg(_last_trade_price("close")).select([*KEY_COLUMNS, "close"])


@register_field("open")
def _open_field(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    _require_columns(columns, [*KEY_COLUMNS, "price", "is_first"])
    return raw.group_by(list(KEY_COLUMNS)).agg(_first_trade_price("open")).select([*KEY_COLUMNS, "open"])


@register_field("high")
def _high_field(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    _require_columns(columns, [*KEY_COLUMNS, "price"])
    return raw.group_by(list(KEY_COLUMNS)).agg(pl.col("price").max().alias("high")).select([*KEY_COLUMNS, "high"])


@register_field("low")
def _low_field(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    _require_columns(columns, [*KEY_COLUMNS, "price"])
    return raw.group_by(list(KEY_COLUMNS)).agg(pl.col("price").min().alias("low")).select([*KEY_COLUMNS, "low"])


@register_field("volume")
def _volume_field(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    _require_columns(columns, [*KEY_COLUMNS, "volume"])
    return raw.group_by(list(KEY_COLUMNS)).agg(sum_or_null(pl.col("volume")).alias("volume")).select(
        [*KEY_COLUMNS, "volume"]
    )


@register_field("no")
def _no_field(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    _require_columns(columns, [*KEY_COLUMNS, "no"])
    return raw.group_by(list(KEY_COLUMNS)).agg(sum_or_null(pl.col("no")).alias("no")).select([*KEY_COLUMNS, "no"])


@register_field("amount")
def _amount_field(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    return raw.group_by(list(KEY_COLUMNS)).agg(_component_expr("amount", columns).alias("amount")).select(
        [*KEY_COLUMNS, "amount"]
    )


@register_field("preclose")
def _preclose_field(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    raise ValueError(
        "preclose field is reserved; minute preclose should be close lag(1) "
        "with daily_k.preclose filling the first missing bar."
    )


def _first_trade_price(alias: str) -> pl.Expr:
    return pl.col("price").filter(pl.col("is_first")).drop_nulls().first().alias(alias)


def _last_trade_price(alias: str) -> pl.Expr:
    return pl.col("price").filter(pl.col("is_last")).drop_nulls().last().alias(alias)


def _require_columns(
    columns: list[str],
    required: list[str] | tuple[str, ...],
    *,
    subject: str = "Field input",
) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{subject} is missing columns: {missing}.")


def _ratio_expr(numerator: str, denominator: str) -> pl.Expr:
    return pl.when(pl.col(denominator) == 0).then(None).otherwise(pl.col(numerator) / pl.col(denominator))


def _component_expr(name: str, columns: list[str]) -> pl.Expr:
    _require_columns(columns, list(KEY_COLUMNS))
    if name == "amount" and "amount" not in columns:
        _require_columns(columns, ["price", "volume"])
        return sum_or_null(pl.col("price") * pl.col("volume"))
    _require_columns(columns, [name])
    return sum_or_null(pl.col(name))


def _validate_public_alias(alias: str | None) -> None:
    if alias is not None and alias.startswith("__"):
        raise ValueError("Field alias must not start with '__'; this prefix is reserved for internal payload columns.")
