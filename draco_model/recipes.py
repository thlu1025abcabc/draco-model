from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from draco_model.core import Model, Node
from draco_model.layers.aggregate import Aggregate
from draco_model.layers.filters import Flag, Side, Threshold, Where
from draco_model.layers.names import validate_public_alias
from draco_model.layers.operators import Col


SUPPORTED_METRICS = {
    "amount",
    "buyamount",
    "close",
    "high",
    "low",
    "no",
    "open",
    "preclose",
    "sellamount",
    "volume",
    "vwap",
}

TransformBuilder = Callable[[Node, str | None], Node]
_TRANSFORMS: dict[str, TransformBuilder] = {}

__all__ = [
    "FactorRecipe",
    "LastShortcut",
    "MetricShortcut",
    "Shortcut",
    "TransformShortcut",
    "last",
    "metric",
    "transform",
]


@dataclass(frozen=True)
class Shortcut:
    """Build-time expansion that turns an input node into a DAG fragment."""

    name: str
    alias: str | None = None

    def __call__(self, source: Node) -> Node:
        raise NotImplementedError


@dataclass(frozen=True)
class MetricShortcut(Shortcut):
    """Named market metric shortcut."""

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric {self.name!r}.")
        validate_public_alias(self.output)

    @property
    def output(self) -> str:
        return self.alias or self.name

    def __call__(self, source: Node) -> Node:
        output = self.output
        if self.name == "volume":
            return Aggregate("1m", "sum", value_col="volume", alias=output)(Col("volume")(source))
        if self.name == "no":
            return Aggregate("1m", "sum", value_col="no", alias=output)(Col("no")(source))
        if self.name == "amount":
            row = (Col("price") * Col("volume")).alias("amount")(source)
            return Aggregate("1m", "sum", value_col="amount", alias=output)(row)
        if self.name == "buyamount":
            row = (Col("price") * Col("volume")).alias("amount")(Where(Side("buy"))(source))
            return Aggregate("1m", "sum", value_col="amount", alias=output)(row)
        if self.name == "sellamount":
            row = (Col("price") * Col("volume")).alias("amount")(Where(Side("sell"))(source))
            return Aggregate("1m", "sum", value_col="amount", alias=output)(row)
        if self.name == "vwap":
            return (metric("amount")(source) / metric("volume")(source)).alias(output)
        if self.name == "close":
            return Aggregate("1m", "last", value_col="close", alias=output)(
                Col("price").alias("close")(Where(Flag("is_last"))(source))
            )
        if self.name == "open":
            return Aggregate("1m", "first", value_col="open", alias=output)(
                Col("price").alias("open")(Where(Flag("is_first"))(source))
            )
        if self.name == "high":
            return Aggregate("1m", "max", value_col="high", alias=output)(Col("price").alias("high")(source))
        if self.name == "low":
            return Aggregate("1m", "min", value_col="low", alias=output)(Col("price").alias("low")(source))
        return Node(
            kind="frame",
            op="metric_reserved",
            params={"name": "preclose", "alias": output},
            inputs={"input": source},
        )


@dataclass(frozen=True)
class LastShortcut(Shortcut):
    """Filter rows at or after one minute."""

    minute: int = 0

    def __post_init__(self) -> None:
        if self.name != "last":
            raise ValueError("LastShortcut name must be 'last'.")
        if self.alias is not None:
            raise ValueError("last() does not support alias.")
        if not isinstance(self.minute, int) or isinstance(self.minute, bool):
            raise TypeError("last() minute must be an integer.")

    def __call__(self, source: Node) -> Node:
        return Where(Threshold("minute", op=">=", value=self.minute))(source)


@dataclass(frozen=True)
class TransformShortcut(Shortcut):
    """Named transform shortcut resolved through a build-time registry."""

    def __post_init__(self) -> None:
        if self.alias is not None:
            validate_public_alias(self.alias)

    def __call__(self, source: Node) -> Node:
        try:
            builder = _TRANSFORMS[self.name]
        except KeyError:
            raise ValueError(f"Transform shortcut {self.name!r} is not registered.") from None
        return builder(source, self.alias)


class FactorRecipe:
    """Base class for user-defined factor-family templates."""

    def build(self) -> Model:
        raise NotImplementedError


def metric(name: str, *, alias: str | None = None) -> MetricShortcut:
    """Return a build-time metric shortcut."""
    return MetricShortcut(name=name, alias=alias)


def last(minute: int) -> LastShortcut:
    """Return a shortcut that keeps rows whose minute is at or after minute."""
    return LastShortcut(name="last", minute=minute)


def transform(name: str, *, alias: str | None = None) -> TransformShortcut:
    """Return a build-time transform shortcut."""
    return TransformShortcut(name=name, alias=alias)
