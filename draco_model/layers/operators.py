from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import polars as pl

from draco_model.core import Node
from draco_model.layers.names import validate_public_alias
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import (
    EvalContext,
    FieldInfo,
    FrameInfo,
    merge_source_contexts,
    register_executor,
    register_info,
)


ARITHMETIC_OPS = {"add", "sub", "mul", "div"}
WINDOW_OPS = {"rolling_corr", "rolling_beta", "rolling_alpha"}


@dataclass(frozen=True)
class FieldExpr:
    """Lazy field expression that can be applied to a frame node."""

    alias_name: str | None = None

    def alias(self, name: str) -> "FieldExpr":
        validate_public_alias(name)
        return replace(self, alias_name=name)

    def __add__(self, other: Any) -> "OpExpr":
        return _expr_op("add", self, other)

    def __radd__(self, other: Any) -> "OpExpr":
        return _expr_op("add", other, self)

    def __sub__(self, other: Any) -> "OpExpr":
        return _expr_op("sub", self, other)

    def __rsub__(self, other: Any) -> "OpExpr":
        return _expr_op("sub", other, self)

    def __mul__(self, other: Any) -> "OpExpr":
        return _expr_op("mul", self, other)

    def __rmul__(self, other: Any) -> "OpExpr":
        return _expr_op("mul", other, self)

    def __truediv__(self, other: Any) -> "OpExpr":
        return _expr_op("div", self, other)

    def __rtruediv__(self, other: Any) -> "OpExpr":
        return _expr_op("div", other, self)


@dataclass(frozen=True)
class Col(FieldExpr):
    """Reference a raw column inside the frame passed to the expression."""

    name: str = ""

    def __init__(self, name: str, alias_name: str | None = None) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "alias_name", alias_name)

    def __call__(self, frame: Node) -> Node:
        alias = self.alias_name or self.name
        validate_public_alias(alias)
        return Node(
            kind="frame",
            op="column",
            params={"column": self.name, "alias": alias},
            inputs={"input": frame},
        )


@dataclass(frozen=True)
class LiteralExpr(FieldExpr):
    """Literal scalar operand in an operator expression."""

    value: int | float | bool = 0

    def __init__(self, value: int | float | bool, alias_name: str | None = None) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "alias_name", alias_name)


@dataclass(frozen=True)
class OpExpr(FieldExpr):
    """Operator expression that becomes a row-level op when called on a frame."""

    name: str = ""
    operands: tuple[FieldExpr, ...] = ()
    params: dict[str, Any] | None = None

    def __call__(self, frame: Node) -> Node:
        alias = self.alias_name or _default_alias(self.name)
        validate_public_alias(alias)
        return Node(
            kind="frame",
            op="op",
            params={
                "name": self.name,
                "mode": "row",
                "alias": alias,
                "operands": [_serialize_expr_operand(operand) for operand in self.operands],
                **(self.params or {}),
            },
            inputs={"input": frame},
        )


def Op(name: str, *operands: Any, alias: str | None = None, **params: Any) -> Node | OpExpr:
    """Create a generic operator node or row-level expression."""
    if name not in ARITHMETIC_OPS | WINDOW_OPS:
        raise ValueError(f"Unsupported operator {name!r}.")
    normalized = [_normalize_operand(operand) for operand in operands]
    if name in WINDOW_OPS:
        _validate_window_params(name, params)
        if len(normalized) != 2 or not all(isinstance(operand, Node) for operand in normalized):
            raise ValueError(f"{name} requires exactly two frame (Node) operands.")
    if any(isinstance(operand, Node) for operand in normalized):
        if any(isinstance(operand, (Col, OpExpr)) for operand in normalized):
            raise ValueError("Frame-level Op cannot mix Node operands with Col expressions.")
        return _frame_op(name, normalized, alias, params)
    return OpExpr(name=name, operands=tuple(normalized), alias_name=alias, params=params)


def alias_node(node: Node, alias: str) -> Node:
    """Return a node that exposes a single public field under a new alias."""
    validate_public_alias(alias)
    if node.op in {"op", "aggregate", "column", "metric_reserved"}:
        params = dict(node.params)
        params["alias"] = alias
        return Node(kind=node.kind, op=node.op, params=params, inputs=dict(node.inputs), name=node.name)
    return Node(kind="frame", op="rename", params={"alias": alias}, inputs={"input": node})


@register_executor("column")
def _column(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent_schema = context.infer_info(node.inputs["input"])
    info = _column_info_from_parent(node, parent_schema)
    frame = context.evaluate(node.inputs["input"])
    column = str(node.params["column"])
    alias = str(node.params["alias"])
    _require_columns(list(parent_schema.columns), [column])
    return frame.with_columns(pl.col(column).alias(alias)).select(list(info.columns))


@register_info("column")
def _column_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    return _column_info_from_parent(node, parent_infos["input"])


def _column_info_from_parent(node: Node, parent: FrameInfo) -> FrameInfo:
    column = str(node.params["column"])
    alias = str(node.params["alias"])
    _require_columns(list(parent.columns), [column])
    columns = _append_column(parent.columns, alias)
    source, lookback, grain_path = _source_context_for_column(parent, column)
    return FrameInfo.from_columns(
        columns,
        identity_keys=parent.keys,
        fields={
            **parent.fields,
            alias: FieldInfo(
                alias,
                alias,
                source=source,
                lookback_days=lookback,
                grain_path=grain_path,
            ),
        },
    )


@register_executor("op")
def _op(node: Node, context: EvalContext) -> pl.LazyFrame:
    mode = str(node.params.get("mode", "frame"))
    if mode == "row":
        return _row_op(node, context)
    return _frame_op_executor(node, context)


@register_info("op")
def _op_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    mode = str(node.params.get("mode", "frame"))
    if mode == "row":
        parent = parent_infos["input"]
        alias = str(node.params["alias"])
        source, lookback, grain_path = parent.merged_source_context()
        return FrameInfo.from_columns(
            _append_column(parent.columns, alias),
            identity_keys=parent.keys,
            fields={
                **parent.fields,
                alias: FieldInfo(
                    name=alias,
                    column=alias,
                    operator=str(node.params["name"]),
                    source=source,
                    lookback_days=lookback,
                    component_agg=False,
                    grain_path=grain_path,
                ),
            },
        )

    alias = str(node.params["alias"])
    name = str(node.params["name"])
    input_schemas = [schema for key, schema in parent_infos.items() if key.startswith("operand")]
    keys = _common_keys(input_schemas)
    components = tuple(f"__op_{alias}_{idx}" for idx, _ in enumerate(input_schemas))
    specs = [dict(spec) for spec in node.params["operands"]]
    can_component_agg = (
        name in ARITHMETIC_OPS
        and len(components) == 2
        and all(spec["kind"] == "input" for spec in specs)
    )
    payloads = tuple(
        renamed
        for input_name, schema in parent_infos.items()
        if input_name.startswith("operand")
        for renamed in _payload_renames(input_name, schema).values()
    )
    source, lookback, grain_path = merge_source_contexts(input_schemas)
    columns = (*keys, alias, *components, *payloads)
    fields: dict[str, FieldInfo] = {
        alias: FieldInfo(
            name=alias,
            column=alias,
            operator=name,
            components=components,
            source=source,
            lookback_days=lookback,
            component_agg=can_component_agg,
            grain_path=grain_path,
        )
    }
    for payload in payloads:
        fields[payload] = FieldInfo(payload, payload, is_public=False, is_payload=True)
    return FrameInfo.from_columns(
        columns,
        identity_keys=keys,
        fields=fields,
    )


@register_executor("rename")
def _rename(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    schema = context.infer_info(parent)
    info = _rename_info_from_parent(node, schema)
    value = schema.single_value_column()
    alias = str(node.params["alias"])
    return context.evaluate(parent).rename({value: alias}).select(list(info.columns))


@register_info("rename")
def _rename_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    return _rename_info_from_parent(node, parent_infos["input"])


def _rename_info_from_parent(node: Node, parent: FrameInfo) -> FrameInfo:
    value = parent.single_value_column()
    alias = str(node.params["alias"])
    columns = tuple(alias if column == value else column for column in parent.columns)
    info = parent.field_for(value)
    fields = {alias: replace(info, name=alias, column=alias)}
    return FrameInfo.from_columns(
        columns,
        identity_keys=parent.keys,
        fields=fields,
    )


def _frame_op(name: str, operands: list[Any], alias: str | None, params: dict[str, Any]) -> Node:
    alias = alias or _default_alias(name)
    validate_public_alias(alias)
    inputs: dict[str, Node] = {}
    specs: list[dict[str, Any]] = []
    input_index = 0
    for operand in operands:
        if isinstance(operand, Node):
            input_name = f"operand{input_index}"
            input_index += 1
            inputs[input_name] = operand
            specs.append({"kind": "input", "name": input_name})
        else:
            specs.append({"kind": "literal", "value": operand.value})
    return Node(
        kind="frame",
        op="op",
        params={"name": name, "mode": "frame", "alias": alias, "operands": specs, **params},
        inputs=inputs,
    )


def _row_op(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    schema = context.infer_info(parent)
    info = _op_info(node, {"input": schema}, context)
    columns = list(schema.columns)
    alias = str(node.params["alias"])
    expr = _expr_from_spec(dict(node.params["operands"][0]), columns)
    for spec in list(node.params["operands"])[1:]:
        expr = _combine_expr(str(node.params["name"]), expr, _expr_from_spec(dict(spec), columns))
    return context.evaluate(parent).with_columns(expr.alias(alias)).select(list(info.columns))


def _frame_op_executor(node: Node, context: EvalContext) -> pl.LazyFrame:
    alias = str(node.params["alias"])
    operator = str(node.params["name"])
    specs = list(node.params["operands"])
    parent_schemas = {
        input_name: context.infer_info(parent)
        for input_name, parent in node.inputs.items()
        if input_name.startswith("operand")
    }
    info = _op_info(node, parent_schemas, context)
    frames: list[pl.LazyFrame] = []
    keys: tuple[str, ...] | None = None
    expr_operands: list[pl.Expr] = []
    component_columns: list[str] = []
    payload_columns: list[str] = []
    input_counter = 0
    for idx, spec in enumerate(specs):
        spec = dict(spec)
        if spec["kind"] == "literal":
            expr_operands.append(pl.lit(spec["value"]))
            continue
        input_name = str(spec["name"])
        schema = parent_schemas[input_name]
        value = schema.single_value_column()
        if keys is None:
            keys = schema.keys
        elif keys != schema.keys:
            raise ValueError("Operator inputs must use the same key columns.")
        component = f"__op_{alias}_{input_counter}"
        input_counter += 1
        payload_renames = _payload_renames(input_name, schema)
        frames.append(
            context.evaluate(node.inputs[input_name]).select(
                [
                    *schema.keys,
                    pl.col(value).alias(component),
                    *[pl.col(source).alias(target) for source, target in payload_renames.items()],
                ]
            )
        )
        component_columns.append(component)
        payload_columns.extend(payload_renames.values())
        expr_operands.append(pl.col(component))

    if keys is None:
        raise ValueError("Frame-level operator requires at least one frame operand.")
    frame = frames[0]
    for right in frames[1:]:
        frame = frame.join(right, on=list(keys), how="inner")
    if operator in WINDOW_OPS:
        return _window_op(
            frame,
            keys,
            operator,
            expr_operands,
            alias,
            int(node.params["window"]),
            _cross_day_param(node),
            [*component_columns, *payload_columns],
        )
    out_expr = _combine_many(operator, expr_operands).alias(alias)
    return frame.with_columns(out_expr).select(list(info.columns))


def _window_op(
    frame: pl.LazyFrame,
    keys: tuple[str, ...],
    operator: str,
    operands: list[pl.Expr],
    alias: str,
    window: int,
    cross_day: bool,
    payload_columns: list[str],
) -> pl.LazyFrame:
    if keys == KEY_COLUMNS:
        group_keys = ["secu_code"] if cross_day else list(DAILY_KEY_COLUMNS)
        order_keys = list(KEY_COLUMNS)
    elif keys == DAILY_KEY_COLUMNS:
        group_keys = ["secu_code"]
        order_keys = list(DAILY_KEY_COLUMNS)
    else:
        raise ValueError("Rolling operators require minute or daily key columns.")
    if len(operands) != 2:
        raise ValueError(f"{operator} requires exactly two frame operands.")
    y, x = operands[0], operands[1]
    y_mean = y.rolling_mean(window).over(group_keys)
    x_mean = x.rolling_mean(window).over(group_keys)
    xy_mean = (y * x).rolling_mean(window).over(group_keys)
    cov = xy_mean - y_mean * x_mean
    x_var = (x * x).rolling_mean(window).over(group_keys) - x_mean * x_mean
    y_var = (y * y).rolling_mean(window).over(group_keys) - y_mean * y_mean
    # Variances from E[x^2] - E[x]^2 can go slightly negative from float error.
    if operator == "rolling_corr":
        expr = (
            pl.when((x_var <= 0) | (y_var <= 0))
            .then(None)
            .otherwise((cov / (x_var.sqrt() * y_var.sqrt())).clip(-1.0, 1.0))
        )
    elif operator == "rolling_beta":
        expr = pl.when(x_var <= 0).then(None).otherwise(cov / x_var)
    elif operator == "rolling_alpha":
        beta = pl.when(x_var <= 0).then(None).otherwise(cov / x_var)
        expr = y_mean - beta * x_mean
    else:
        raise ValueError(f"Unsupported rolling operator {operator!r}.")
    return frame.sort(order_keys).with_columns(expr.alias(alias)).select([*keys, alias, *payload_columns])


def _cross_day_param(node: Node) -> bool:
    value = node.params.get("cross_day", False)
    if not isinstance(value, bool):
        raise ValueError("Rolling operator cross_day must be a boolean.")
    return value


def _validate_window_params(name: str, params: dict[str, Any]) -> None:
    window = params.get("window")
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError(f"{name} requires a positive integer window.")


def _expr_op(name: str, left: Any, right: Any) -> OpExpr:
    return OpExpr(name=name, operands=(_normalize_expr_operand(left), _normalize_expr_operand(right)))


def _normalize_operand(value: Any) -> Any:
    if isinstance(value, (Node, FieldExpr)):
        return value
    if _is_literal(value):
        return LiteralExpr(value)
    raise TypeError(f"Unsupported operator operand {value!r}.")


def _normalize_expr_operand(value: Any) -> FieldExpr:
    if isinstance(value, FieldExpr):
        return value
    if _is_literal(value):
        return LiteralExpr(value)
    raise TypeError("Row-level arithmetic requires Col(...) or scalar operands.")


def _serialize_expr_operand(operand: FieldExpr) -> dict[str, Any]:
    if isinstance(operand, Col):
        return {"kind": "column", "name": operand.name}
    if isinstance(operand, LiteralExpr):
        return {"kind": "literal", "value": operand.value}
    if isinstance(operand, OpExpr):
        return {
            "kind": "op",
            "name": operand.name,
            "operands": [_serialize_expr_operand(item) for item in operand.operands],
            **(operand.params or {}),
        }
    raise TypeError(f"Unsupported row operand {operand!r}.")


def _expr_from_spec(spec: dict[str, Any], columns: list[str]) -> pl.Expr:
    kind = spec["kind"]
    if kind == "column":
        name = str(spec["name"])
        _require_columns(columns, [name])
        return pl.col(name)
    if kind == "literal":
        return pl.lit(spec["value"])
    if kind == "op":
        operands = [_expr_from_spec(dict(item), columns) for item in spec["operands"]]
        return _combine_many(str(spec["name"]), operands)
    raise ValueError(f"Unsupported operand kind {kind!r}.")


def _combine_many(operator: str, operands: list[pl.Expr]) -> pl.Expr:
    if not operands:
        raise ValueError(f"Operator {operator!r} requires operands.")
    out = operands[0]
    for expr in operands[1:]:
        out = _combine_expr(operator, out, expr)
    return out


def _combine_expr(operator: str, left: pl.Expr, right: pl.Expr) -> pl.Expr:
    if operator == "add":
        return left + right
    if operator == "sub":
        return left - right
    if operator == "mul":
        return left * right
    if operator == "div":
        return pl.when(right == 0).then(None).otherwise(left / right)
    raise ValueError(f"Unsupported arithmetic operator {operator!r}.")


def _payload_renames(input_name: str, schema: FrameInfo) -> dict[str, str]:
    payloads = schema.payload_columns()
    return {column: f"__{input_name}_{column.lstrip('_')}" for column in payloads}


def _source_context_for_column(schema: FrameInfo, column: str) -> tuple[str | None, int, tuple[tuple[str, str], ...]]:
    if column in schema.fields:
        info = schema.fields[column]
        return info.source, info.lookback_days, info.grain_path
    return schema.merged_source_context()


def _common_keys(schemas: list[FrameInfo]) -> tuple[str, ...]:
    if not schemas:
        raise ValueError("Operator requires at least one input schema.")
    keys = schemas[0].keys
    if any(schema.keys != keys for schema in schemas):
        raise ValueError("Operator inputs must use the same key columns.")
    return keys


def _append_column(columns: tuple[str, ...], column: str) -> tuple[str, ...]:
    return columns if column in columns else (*columns, column)


def _default_alias(operator: str) -> str:
    return f"{operator}_value"


def _require_columns(columns: list[str], required: list[str]) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Input frame is missing columns: {missing}.")


def _is_literal(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, str)
