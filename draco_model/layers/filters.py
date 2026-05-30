from __future__ import annotations

from typing import Any, Callable

import polars as pl

from draco_model.core import Condition, Layer, Node
from draco_model.runtime.execution import EvalContext, FramePlan, FrameSchema, register_executor, register_plan


ConditionExpression = Callable[[Node], pl.Expr]

_CONDITIONS: dict[str, ConditionExpression] = {}


def _register_condition(op: str) -> Callable[[ConditionExpression], ConditionExpression]:
    def decorator(condition: ConditionExpression) -> ConditionExpression:
        _CONDITIONS[op] = condition
        return condition

    return decorator


class Side(Condition):
    """Semantic side condition."""

    def __init__(self, side: str) -> None:
        if side not in {"buy", "sell"}:
            raise ValueError("Side must be 'buy' or 'sell'.")
        super().__init__("side", {"side": side})


class Flag(Condition):
    """Boolean column condition, used by metric recipes."""

    def __init__(self, column: str) -> None:
        super().__init__("flag", {"column": column})


class Threshold(Condition):
    """Compare one column with a literal threshold value."""

    def __init__(self, column: str, *, op: str = ">", value: Any) -> None:
        if op not in {">", ">=", "<", "<=", "==", "=", "!=", "<>"}:
            raise ValueError("Unsupported threshold op.")
        super().__init__("threshold", {"column": column, "op": op, "value": value})


class TopQuantile(Condition):
    """Keep rows whose column value is at or above a group quantile."""

    def __init__(self, column: str, *, q: float, over: list[str] | tuple[str, ...]) -> None:
        if not 0 <= q <= 1:
            raise ValueError("q must be in [0, 1].")
        super().__init__("top_quantile", {"column": column, "q": float(q), "over": list(over)})


class Where(Layer):
    """Filter a frame with a condition node."""

    op = "where"

    def __init__(self, condition: Condition, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.condition = condition

    def __call__(self, frame: Node) -> Node:
        return Node(
            kind="frame",
            op=self.op,
            inputs={"frame": frame, "condition": self.condition.to_node(frame)},
            name=self.name,
        )


@register_executor("where")
def _where(node: Node, context: EvalContext) -> pl.LazyFrame:
    return context.evaluate(node.inputs["frame"]).filter(_condition_expr(node.inputs["condition"]))


@register_plan("where")
def _where_plan(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FramePlan:
    return FramePlan.from_schema(parent_schemas["frame"])


def _condition_expr(node: Node) -> pl.Expr:
    try:
        return _CONDITIONS[node.op](node)
    except KeyError:
        raise ValueError(f"Unsupported condition {node.op!r}.") from None


@_register_condition("side")
def _side_expr(node: Node) -> pl.Expr:
    side = str(node.params["side"])
    code = {"buy": 0, "sell": 1}[side]
    return pl.col("side") == code


@_register_condition("flag")
def _flag_expr(node: Node) -> pl.Expr:
    return pl.col(str(node.params["column"])).fill_null(False)


@_register_condition("threshold")
def _threshold_expr(node: Node) -> pl.Expr:
    params = node.params
    col = pl.col(str(params["column"]))
    value = params["value"]
    op = str(params["op"])
    if op == ">":
        return col > value
    if op == ">=":
        return col >= value
    if op == "<":
        return col < value
    if op == "<=":
        return col <= value
    if op in {"==", "="}:
        return col == value
    if op in {"!=", "<>"}:
        return col != value
    raise ValueError(f"Unsupported threshold op {op!r}.")


@_register_condition("top_quantile")
def _top_quantile_expr(node: Node) -> pl.Expr:
    params = node.params
    column = str(params["column"])
    over = list(params["over"])
    threshold = pl.col(column).quantile(float(params["q"])).over(over)
    return pl.col(column) >= threshold
