from __future__ import annotations

from typing import Any, Callable

import polars as pl

from draco_model.core import Condition, Layer, Node
from draco_model.runtime.execution import EvalContext, FrameSchema, register_executor, register_schema


ConditionExpression = Callable[[Node], pl.Expr]

_CONDITIONS: dict[str, ConditionExpression] = {}


def _register_condition(op: str) -> Callable[[ConditionExpression], ConditionExpression]:
    def decorator(condition: ConditionExpression) -> ConditionExpression:
        _CONDITIONS[op] = condition
        return condition

    return decorator


def _condition_expr(node: Node) -> pl.Expr:
    try:
        return _CONDITIONS[node.op](node)
    except KeyError:
        raise ValueError(f"Unsupported condition {node.op!r}.") from None


class Threshold(Condition):
    """Compare one column with a literal threshold value."""

    def __init__(self, column: str, *, op: str = ">", value: Any) -> None:
        """Create a column comparison condition."""
        if op not in {">", ">=", "<", "<=", "==", "=", "!=", "<>"}:
            raise ValueError("Unsupported threshold op.")
        super().__init__("threshold", {"column": column, "op": op, "value": value})


class TopQuantile(Condition):
    """Keep rows whose column value is at or above a group quantile."""

    def __init__(self, column: str, *, q: float, over: list[str] | tuple[str, ...]) -> None:
        """Create a quantile condition grouped by the over columns."""
        if not 0 <= q <= 1:
            raise ValueError("q must be in [0, 1].")
        super().__init__("top_quantile", {"column": column, "q": float(q), "over": list(over)})


class Filter(Layer):
    """Filter a frame with a condition node."""

    op = "filter"

    def __init__(self, condition: Condition, *, name: str | None = None) -> None:
        """Store the condition to attach when the layer is called."""
        super().__init__(name=name)
        self.condition = condition

    def __call__(self, frame: Node) -> Node:
        """Build a filter node with both frame and condition dependencies."""
        condition_node = self.condition.to_node(frame)
        return Node(
            kind="frame",
            op=self.op,
            inputs={"frame": frame, "condition": condition_node},
            name=self.name,
        )


@register_executor("filter")
def _filter(node: Node, context: EvalContext) -> pl.LazyFrame:
    frame = context.evaluate(node.inputs["frame"])
    condition = node.inputs["condition"]
    return frame.filter(_condition_expr(condition))


@register_schema("filter")
def _filter_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    return parent_schemas["frame"]


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
