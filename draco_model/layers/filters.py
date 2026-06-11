from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import polars as pl

from draco_model.core import Layer, Node
from draco_model.runtime.execution import EvalContext, FrameInfo, register_executor, register_info


@dataclass(frozen=True)
class FilterSpec:
    """Boolean expression descriptor used by Where."""

    op: str
    params: dict[str, Any]

    def to_params(self) -> dict[str, Any]:
        return {"op": self.op, "params": dict(self.params)}


FilterExpression = Callable[[dict[str, Any]], pl.Expr]

_FILTERS: dict[str, FilterExpression] = {}


def _register_filter(op: str) -> Callable[[FilterExpression], FilterExpression]:
    def decorator(filter_expr: FilterExpression) -> FilterExpression:
        _FILTERS[op] = filter_expr
        return filter_expr

    return decorator


class Side(FilterSpec):
    """Semantic side condition."""

    def __init__(self, side: str) -> None:
        if side not in {"buy", "sell"}:
            raise ValueError("Side must be 'buy' or 'sell'.")
        super().__init__("side", {"side": side})


class Flag(FilterSpec):
    """Boolean column condition, used by metric recipes."""

    def __init__(self, column: str) -> None:
        super().__init__("flag", {"column": column})


class Threshold(FilterSpec):
    """Compare one column with a literal threshold value."""

    def __init__(self, column: str, *, op: str = ">", value: Any) -> None:
        if op not in {">", ">=", "<", "<=", "==", "=", "!=", "<>"}:
            raise ValueError("Unsupported threshold op.")
        super().__init__("threshold", {"column": column, "op": op, "value": value})


class TopQuantile(FilterSpec):
    """Keep rows whose column value is at or above a group quantile."""

    def __init__(self, column: str, *, q: float, over: list[str] | tuple[str, ...]) -> None:
        if not 0 <= q <= 1:
            raise ValueError("q must be in [0, 1].")
        super().__init__("top_quantile", {"column": column, "q": float(q), "over": list(over)})


class Where(Layer):
    """Filter a frame with a filter condition."""

    op = "where"

    def __init__(self, condition: FilterSpec, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.condition = condition

    def __call__(self, frame: Node) -> Node:
        return Node(
            kind="frame",
            op=self.op,
            params={"condition": self.condition.to_params()},
            inputs={"frame": frame},
            name=self.name,
        )


@register_executor("where")
def _where(node: Node, context: EvalContext) -> pl.LazyFrame:
    return context.evaluate(node.inputs["frame"]).filter(_condition_expr(dict(node.params["condition"])))


@register_info("where")
def _where_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    return parent_infos["frame"]


def _condition_expr(condition: dict[str, Any]) -> pl.Expr:
    op = str(condition["op"])
    try:
        return _FILTERS[op](dict(condition["params"]))
    except KeyError:
        raise ValueError(f"Unsupported condition {op!r}.") from None


@_register_filter("side")
def _side_expr(params: dict[str, Any]) -> pl.Expr:
    side = str(params["side"])
    code = {"buy": 0, "sell": 1}[side]
    return pl.col("side") == code


@_register_filter("flag")
def _flag_expr(params: dict[str, Any]) -> pl.Expr:
    return pl.col(str(params["column"])).fill_null(False)


@_register_filter("threshold")
def _threshold_expr(params: dict[str, Any]) -> pl.Expr:
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


@_register_filter("top_quantile")
def _top_quantile_expr(params: dict[str, Any]) -> pl.Expr:
    column = str(params["column"])
    over = list(params["over"])
    threshold = pl.col(column).quantile(float(params["q"])).over(over)
    return pl.col(column) >= threshold
