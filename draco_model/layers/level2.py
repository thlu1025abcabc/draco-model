"""Level-2 market-data construction layers."""
from __future__ import annotations

import datetime

import polars as pl

from draco_model.core import Layer, Node
from draco_model.runtime.execution import EvalContext, FrameInfo, register_executor, register_info


TRADES_WITH_WAIT_COLUMNS = (
    "date",
    "secu_code",
    "minute",
    "price",
    "side",
    "volume",
    "vw_wait_time",
    "is_first",
    "is_last",
    "no",
)


class TradesWithWaitBar(Layer):
    """Aggregate matched trades to minute/price/side bars."""

    op = "trades_with_wait_bar"

    def __call__(self, trades: Node, orders: Node) -> Node:
        return super().__call__({"trades": trades, "orders": orders})


def match_trade_waits(trades: pl.LazyFrame, orders: pl.LazyFrame) -> pl.LazyFrame:
    """Return valid trades with order-to-deal wait time in seconds."""
    active_orders = orders.filter(~pl.col("order_type").is_in([-1, -11])).select(
        "secu_code",
        "order_id",
        "order_time",
    )
    matched = (
        trades.filter(pl.col("side").is_in([0, 1]))
        .select(
            "date",
            "secu_code",
            "deal_time",
            "deal_id",
            (pl.col("price") / 100).cast(pl.Float64).alias("price"),
            pl.col("volume").cast(pl.Float64),
            pl.col("side").cast(pl.Int64),
            pl.min_horizontal("buy_id", "sell_id").alias("order_id"),
        )
        .sort("secu_code", "deal_time", "deal_id")
        .with_row_index("sort_int", offset=1)
        .with_columns(pl.col("sort_int").cast(pl.Int64))
        .join(active_orders, on=["secu_code", "order_id"], how="left")
        .with_columns(
            pl.col("deal_time")
            .cast(pl.String)
            .str.pad_start(9, "0")
            .str.strptime(pl.Time, "%H%M%S%3f")
            .alias("__deal_time"),
            pl.col("order_time")
            .cast(pl.String)
            .str.pad_start(9, "0")
            .str.strptime(pl.Time, "%H%M%S%3f")
            .alias("__order_time"),
        )
    )
    wait = (
        pl.lit(datetime.date(1970, 1, 1)).dt.combine(pl.col("__deal_time"))
        - pl.lit(datetime.date(1970, 1, 1)).dt.combine(pl.col("__order_time"))
    )
    return (
        matched.with_columns(
            pl.when(
                (pl.col("__deal_time") > datetime.time(11, 31))
                & (pl.col("__order_time") <= datetime.time(11, 31))
            )
            .then(wait - datetime.timedelta(hours=1, minutes=30))
            .otherwise(wait)
            .alias("__wait")
        )
        .with_columns(
            pl.when(pl.col("__wait") < datetime.timedelta(0))
            .then(None)
            .otherwise(pl.col("__wait"))
            .alias("__wait")
        )
        .with_columns((pl.col("__wait").dt.total_microseconds() / 1e6).alias("wait_time"))
        .select(
            "date",
            "secu_code",
            "deal_time",
            "price",
            "side",
            "volume",
            "wait_time",
            "sort_int",
        )
    )


def aggregate_trades(events: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate matched trade events to the public minute-bar contract."""
    minute = (
        pl.when(pl.col("deal_time") < 93000000)
        .then(925)
        .when(pl.col("deal_time") >= 145700000)
        .then(1500)
        .when(pl.col("deal_time") < 130000000)
        .then(pl.col("deal_time").clip(upper_bound=112959999) // 100000)
        .otherwise(pl.col("deal_time") // 100000)
    )
    grouped = (
        events.with_columns(minute.cast(pl.Int64).alias("minute"))
        .group_by("date", "secu_code", "minute", "price", "side")
        .agg(
            pl.col("volume").sum(),
            ((pl.col("volume") * pl.col("wait_time")).sum() / pl.col("volume").sum()).alias(
                "vw_wait_time"
            ),
            pl.col("sort_int").min().alias("is_first"),
            pl.col("sort_int").max().alias("is_last"),
            pl.col("volume").count().alias("no"),
        )
    )
    return (
        grouped.with_columns(
            pl.col("is_first") == pl.col("is_first").min().over("date", "secu_code", "minute"),
            pl.col("is_last") == pl.col("is_last").max().over("date", "secu_code", "minute"),
        )
        .select(list(TRADES_WITH_WAIT_COLUMNS))
        .sort("date", "secu_code", "minute", "side", "price")
    )


@register_executor("trades_with_wait_bar")
def _trades_with_wait_bar(node: Node, context: EvalContext) -> pl.LazyFrame:
    trades = context.evaluate(node.inputs["trades"])
    orders = context.evaluate(node.inputs["orders"])
    return aggregate_trades(match_trade_waits(trades, orders))


@register_info("trades_with_wait_bar")
def _trades_with_wait_bar_info(
    node: Node,
    parent_infos: dict[str, FrameInfo],
    context: EvalContext,
) -> FrameInfo:
    return FrameInfo.from_columns(
        TRADES_WITH_WAIT_COLUMNS,
        identity_keys=("date", "secu_code", "minute", "price", "side"),
    )
