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


QUOTES_MIN_BAR_COLUMNS = (
    "date",
    "secu_code",
    "minute",
    "price",
    "side",
    "volume",
    "is_first",
    "is_last",
    "no",
)

# SH (secu_code >= 600000) uses order-type codes 0/10 for active orders and
# -1/-11 for cancels; SZ uses 2/12 (priced) and 1/11/3/13 (price discovered from
# trades), with cancels carried in the trade stream as side -1/-11.
_EVENT_COLUMNS = ("secu_code", "date", "order_time", "price", "volume", "order_id", "side")


def _event_select(*, time_col: str = "order_time", price_div: bool, order_id, side):
    """Project a raw order/trade frame to the shared quote/cancel event schema."""
    price = (pl.col("price") / 100) if price_div else pl.col("price")
    return [
        pl.col("secu_code").cast(pl.Int64),
        pl.col("date"),
        pl.col(time_col).cast(pl.Int64).alias("order_time"),
        price.cast(pl.Float64).alias("price"),
        pl.col("volume").cast(pl.Float64),
        order_id.cast(pl.Int64).alias("order_id"),
        side.cast(pl.Int64).alias("side"),
    ]


def _split_sh(orders: pl.LazyFrame, trades: pl.LazyFrame, date: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Build SH quote and cancel events from the order stream (plus auction trades)."""
    quotes = orders.filter(pl.col("order_type").is_in([0, 10])).with_columns(
        pl.col("order_type").replace([0, 10], [0, 1]).alias("side")
    ).select(_event_select(price_div=True, order_id=pl.col("order_id"), side=pl.col("side")))
    # Closing call auction has been part of SH trades since 2018-08-20.
    if int(date) >= 20180820:
        auction = pl.col("deal_time").is_between(93000000, 145710000)
    else:
        auction = pl.col("deal_time") >= 93000000
    add_quotes = trades.filter(auction).select(
        _event_select(
            time_col="deal_time",
            price_div=True,
            order_id=pl.max_horizontal("buy_id", "sell_id"),
            side=pl.col("side"),
        )
    )
    quotes = pl.concat([quotes, add_quotes])
    cancels = orders.filter(pl.col("order_type").is_in([-1, -11])).with_columns(
        pl.col("order_type").replace([-1, -11], [0, 1]).alias("side")
    ).select(_event_select(price_div=True, order_id=pl.col("order_id"), side=pl.col("side")))
    return quotes, cancels


def _split_sz(orders: pl.LazyFrame, trades: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Build SZ quote and cancel events, discovering market-order prices from trades."""
    priced = orders.filter(pl.col("order_type").is_in([2, 12])).with_columns(
        pl.col("order_type").replace([2, 12], [0, 1]).alias("side")
    ).select(_event_select(price_div=True, order_id=pl.col("order_id"), side=pl.col("side")))
    market = orders.filter(pl.col("order_type").is_in([1, 11, 3, 13])).with_columns(
        pl.col("order_type").replace([1, 11, 3, 13], [0, 1, 0, 1]).alias("side")
    ).select(_event_select(price_div=True, order_id=pl.col("order_id"), side=pl.col("side")))
    lookup = trades.filter(pl.col("side").is_in([0, 1])).select(
        pl.col("secu_code").cast(pl.Int64),
        pl.col("date"),
        pl.col("deal_time").cast(pl.Int64).alias("order_time"),
        (pl.col("price") / 100).cast(pl.Float64).alias("price"),
        pl.col("volume").cast(pl.Float64),
        pl.col("buy_id").cast(pl.Int64),
        pl.col("sell_id").cast(pl.Int64),
        pl.col("deal_id").cast(pl.Int64),
    )
    # Discover each market order's fill price: buy orders take the max matched
    # trade price, sell orders the min; buy price wins when both exist.
    joined_buy = market.join(
        lookup.rename({"buy_id": "order_id"}), on=["secu_code", "order_id"], how="left"
    ).group_by("secu_code", "order_id").agg(pl.col("price_right").max())
    joined_sell = market.join(
        lookup.rename({"sell_id": "order_id"}), on=["secu_code", "order_id"], how="left"
    ).group_by("secu_code", "order_id").agg(pl.col("price_right").min())
    discovered = joined_buy.rename({"price_right": "price"}).join(
        joined_sell, on=["secu_code", "order_id"], how="left"
    ).with_columns(pl.col("price").fill_null(pl.col("price_right"))).select(
        pl.exclude("price_right")
    ).filter(pl.col("price").is_not_null()).join(
        market, on=["secu_code", "order_id"], how="left"
    ).select(
        pl.col("secu_code").cast(pl.Int64),
        pl.col("date"),
        pl.col("order_time").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("order_id").cast(pl.Int64),
        pl.col("side").cast(pl.Int64),
    )
    quotes = pl.concat([priced, discovered])
    # Cancels live in the SZ trade stream (side -1/-11, deal price 0); recover the
    # cancelled price by joining the surviving order id back onto the quotes.
    cancels = trades.filter(pl.col("side").is_in([-1, -11])).select(
        pl.col("secu_code").cast(pl.Int64),
        pl.col("date"),
        pl.col("deal_time").cast(pl.Int64).alias("order_time"),
        pl.col("volume").cast(pl.Float64),
        (pl.col("buy_id") + pl.col("sell_id")).cast(pl.Int64).alias("order_id"),
        pl.col("side").replace([-1, -11], [0, 1]).cast(pl.Int64).alias("side"),
    ).join(
        quotes.select("secu_code", "order_id", "price"), on=["secu_code", "order_id"], how="left"
    ).filter(pl.col("price").is_not_null()).select(list(_EVENT_COLUMNS))
    return quotes, cancels


def split_orders_cancels(
    trades: pl.LazyFrame, orders: pl.LazyFrame, date: str
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split raw level-2 orders/trades into quote and cancel events."""
    sh_orders = orders.filter(pl.col("order_type").is_in([0, 10, -1, -11]))
    sz_orders = orders.filter(~pl.col("order_type").is_in([0, 10, -1, -11]))
    sh_trades = trades.filter(pl.col("secu_code") >= 600000)
    sz_trades = trades.filter(pl.col("secu_code") < 600000)
    sz_quotes, sz_cancels = _split_sz(sz_orders, sz_trades)
    sh_quotes, sh_cancels = _split_sh(sh_orders, sh_trades, date)
    quotes = pl.concat([sz_quotes, sh_quotes]).select(list(_EVENT_COLUMNS))
    cancels = pl.concat([sz_cancels, sh_cancels]).select(list(_EVENT_COLUMNS))
    return quotes, cancels


def _add_sort_int(events: pl.LazyFrame) -> pl.LazyFrame:
    """Assign a deterministic per-row sequence index for first/last detection."""
    return events.sort("secu_code", "order_time", "order_id", "price", "side").with_columns(
        sort_int=pl.lit(1, dtype=pl.Int64)
    ).with_columns(pl.col("sort_int").cum_sum())


def _add_cancel_wait(cancels: pl.LazyFrame, orders: pl.LazyFrame) -> pl.LazyFrame:
    """Attach the cancel-to-order wait time in seconds, excluding the lunch break."""
    active_orders = orders.filter(~pl.col("order_type").is_in([-1, -11])).select(
        pl.col("secu_code").cast(pl.Int64),
        pl.col("order_id").cast(pl.Int64),
        pl.col("order_time").cast(pl.Int64).alias("__order_time"),
    )
    matched = cancels.join(active_orders, on=["secu_code", "order_id"], how="left").with_columns(
        pl.col("order_time").cast(pl.String).str.pad_start(9, "0").str.strptime(pl.Time, "%H%M%S%3f").alias("__cancel"),
        pl.col("__order_time").cast(pl.String).str.pad_start(9, "0").str.strptime(pl.Time, "%H%M%S%3f").alias("__placed"),
    )
    wait = (
        pl.lit(datetime.date(1970, 1, 1)).dt.combine(pl.col("__cancel"))
        - pl.lit(datetime.date(1970, 1, 1)).dt.combine(pl.col("__placed"))
    )
    return (
        matched.with_columns(
            pl.when(
                (pl.col("__cancel") > datetime.time(11, 31))
                & (pl.col("__placed") <= datetime.time(11, 31))
            )
            .then(wait - datetime.timedelta(hours=1, minutes=30))
            .otherwise(wait)
            .alias("__wait")
        )
        .with_columns(
            pl.when(pl.col("__wait") < datetime.timedelta(0)).then(None).otherwise(pl.col("__wait")).alias("__wait")
        )
        .with_columns((pl.col("__wait").dt.total_microseconds() / 1e6).alias("wait_time"))
        .select("secu_code", "date", "order_time", "price", "side", "volume", "wait_time", "sort_int")
    )


def aggregate_bar(events: pl.LazyFrame, *, with_wait: bool) -> pl.LazyFrame:
    """Aggregate quote/cancel events to the minute/price/side bar contract."""
    minute = (
        pl.when(pl.col("order_time") < 93000000)
        .then(925)
        .when(pl.col("order_time") >= 145700000)
        .then(1500)
        .when(pl.col("order_time") < 130000000)
        .then(pl.col("order_time").clip(upper_bound=112959999) // 100000)
        .otherwise(pl.col("order_time") // 100000)
    )
    aggs = [pl.col("volume").sum()]
    if with_wait:
        aggs.append(
            ((pl.col("volume") * pl.col("wait_time")).sum() / pl.col("volume").sum()).alias("vw_wait_time")
        )
    aggs += [
        pl.col("sort_int").min().alias("is_first"),
        pl.col("sort_int").max().alias("is_last"),
        pl.col("volume").count().alias("no"),
    ]
    grouped = events.with_columns(minute.cast(pl.Int64).alias("minute")).group_by(
        "date", "secu_code", "minute", "price", "side"
    ).agg(aggs)
    columns = TRADES_WITH_WAIT_COLUMNS if with_wait else QUOTES_MIN_BAR_COLUMNS
    return (
        grouped.with_columns(
            pl.col("is_first") == pl.col("is_first").min().over("date", "secu_code", "minute"),
            pl.col("is_last") == pl.col("is_last").max().over("date", "secu_code", "minute"),
        )
        .select(list(columns))
        .sort("date", "secu_code", "minute", "side", "price")
    )


class QuotesMinBar(Layer):
    """Aggregate active limit/market orders to minute/price/side quote bars."""

    op = "quotes_min_bar"

    def __call__(self, trades: Node, orders: Node) -> Node:
        return super().__call__({"trades": trades, "orders": orders})


class CancelsMinBar(Layer):
    """Aggregate order cancellations to minute/price/side bars with wait time."""

    op = "cancels_min_bar"

    def __call__(self, trades: Node, orders: Node) -> Node:
        return super().__call__({"trades": trades, "orders": orders})


@register_executor("quotes_min_bar")
def _quotes_min_bar(node: Node, context: EvalContext) -> pl.LazyFrame:
    trades = context.evaluate(node.inputs["trades"])
    orders = context.evaluate(node.inputs["orders"])
    quotes, _ = split_orders_cancels(trades, orders, context.eval_date)
    return aggregate_bar(_add_sort_int(quotes), with_wait=False)


@register_executor("cancels_min_bar")
def _cancels_min_bar(node: Node, context: EvalContext) -> pl.LazyFrame:
    trades = context.evaluate(node.inputs["trades"])
    orders = context.evaluate(node.inputs["orders"])
    _, cancels = split_orders_cancels(trades, orders, context.eval_date)
    cancels = _add_cancel_wait(_add_sort_int(cancels), orders)
    return aggregate_bar(cancels, with_wait=True)


@register_info("quotes_min_bar")
def _quotes_min_bar_info(
    node: Node,
    parent_infos: dict[str, FrameInfo],
    context: EvalContext,
) -> FrameInfo:
    return FrameInfo.from_columns(
        QUOTES_MIN_BAR_COLUMNS,
        identity_keys=("date", "secu_code", "minute", "price", "side"),
    )


@register_info("cancels_min_bar")
def _cancels_min_bar_info(
    node: Node,
    parent_infos: dict[str, FrameInfo],
    context: EvalContext,
) -> FrameInfo:
    return FrameInfo.from_columns(
        TRADES_WITH_WAIT_COLUMNS,
        identity_keys=("date", "secu_code", "minute", "price", "side"),
    )
