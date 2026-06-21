"""Snapshot (order-book) minute-bar construction layer."""
from __future__ import annotations

import polars as pl

from draco_model.core import Layer, Node
from draco_model.runtime.execution import EvalContext, FrameInfo, register_executor, register_info


_LEVELS = range(1, 11)
_AVOI_LEVELS = range(1, 6)

SNAPSHOT_MIN_BAR_COLUMNS = (
    *(f"AskPrice{level}" for level in _LEVELS),
    *(f"BidPrice{level}" for level in _LEVELS),
    *(f"AskVolume{level}" for level in _LEVELS),
    *(f"BidVolume{level}" for level in _LEVELS),
    *(f"aVOI{level}" for level in _AVOI_LEVELS),
    "secu_code",
    "minute",
    "date",
)


def _split_continuous(snapshot: pl.LazyFrame) -> pl.LazyFrame:
    """Keep continuous-session ticks (drops the 11:30-13:00 lunch gap)."""
    return pl.concat(
        [
            snapshot.filter(pl.col("tick_time").is_between(0, 113000000 - 1)),
            snapshot.filter(pl.col("tick_time").is_between(130000000, 145700000 - 1)),
        ]
    )


def _fill_book(snapshot: pl.LazyFrame) -> pl.LazyFrame:
    """Forward-fill empty book levels: top of book from the other side, deeper
    levels from the level above, and missing volumes to zero."""
    snapshot = snapshot.with_columns(
        [pl.col(f"AskPrice{level}").replace([0], [None]) for level in _LEVELS]
        + [pl.col(f"BidPrice{level}").replace([0], [None]) for level in _LEVELS]
    ).with_columns(
        pl.col("AskPrice1").fill_null(pl.col("BidPrice1")),
        pl.col("BidPrice1").fill_null(pl.col("AskPrice1")),
    )
    for level in range(1, 10):
        for side in ("AskPrice", "BidPrice"):
            snapshot = snapshot.with_columns(pl.col(f"{side}{level + 1}").fill_null(pl.col(f"{side}{level}")))
    for side in ("AskVolume", "BidVolume"):
        snapshot = snapshot.with_columns(pl.col(f"{side}{level}").fill_null(0) for level in _LEVELS)
    return snapshot


def _append_lost_bars(bar: pl.LazyFrame) -> pl.LazyFrame:
    """Complete the secu_code x minute grid so every stock has every minute."""
    grid = bar.select("secu_code").unique().sort("secu_code").join(
        bar.select("minute").unique().sort("minute"), how="cross"
    )
    return bar.join(grid, on=["secu_code", "minute"], how="right")


def _aux_price(snapshot: pl.LazyFrame) -> pl.LazyFrame:
    """Per-stock fallback top-of-book price: the last pre-open trade price (kept
    on the raw 1/100 vendor scale). Stocks with no pre-open trade get a null
    fallback, so their leading empty-book bars stay null rather than carrying a
    previous-close value -- this layer intentionally takes no daily_k input."""
    return (
        snapshot.filter(pl.col("tick_time") <= 93000000)
        .sort("secu_code", "tick_time")
        .group_by("secu_code")
        .agg(pl.col("price").replace(0, None).forward_fill().last().alias("aux_price"))
    )


def _add_avoi(snapshot: pl.LazyFrame) -> pl.LazyFrame:
    """Add per-tick order-flow-imbalance terms aVOI1..aVOI5."""
    return snapshot.with_columns(
        (
            (pl.col(f"BidPrice{i}") <= pl.col(f"BidPrice{i}").shift(1).over("secu_code")).cast(pl.Int64)
            * (pl.col(f"BidVolume{i}") * pl.col(f"BidPrice{i}") + 1).log()
            - (pl.col(f"BidPrice{i}") == pl.col(f"BidPrice{i}").shift(1).over("secu_code")).cast(pl.Int64)
            * (pl.col(f"BidVolume{i}") * pl.col(f"BidPrice{i}") + 1).log().shift(1).over("secu_code")
            - (pl.col(f"AskPrice{i}") >= pl.col(f"AskPrice{i}").shift(1).over("secu_code")).cast(pl.Int64)
            * (pl.col(f"AskVolume{i}") * pl.col(f"AskPrice{i}") + 1).log()
            + (pl.col(f"AskPrice{i}") == pl.col(f"AskPrice{i}").shift(1).over("secu_code")).cast(pl.Int64)
            * (pl.col(f"AskVolume{i}") * pl.col(f"AskPrice{i}") + 1).log().shift(1).over("secu_code")
        ).alias(f"aVOI{i}")
        for i in _AVOI_LEVELS
    )


def build_snapshot_bar(snapshot: pl.LazyFrame, date: str) -> pl.LazyFrame:
    """Aggregate raw order-book ticks into the per-minute snapshot bar contract."""
    aux = _aux_price(snapshot)
    snapshot = _fill_book(_split_continuous(snapshot))
    snapshot = _add_avoi(snapshot.filter(pl.col("tick_time") >= 93000000).sort("secu_code", "tick_time"))
    bar = snapshot.group_by("secu_code", (pl.col("tick_time") // 100000).alias("minute")).agg(
        [(pl.col(f"AskPrice{i}").mean() / 100).alias(f"AskPrice{i}") for i in _LEVELS]
        + [(pl.col(f"BidPrice{i}").mean() / 100).alias(f"BidPrice{i}") for i in _LEVELS]
        + [pl.col(f"AskVolume{i}").mean().alias(f"AskVolume{i}") for i in _LEVELS]
        + [pl.col(f"BidVolume{i}").mean().alias(f"BidVolume{i}") for i in _LEVELS]
        + [pl.col(f"aVOI{i}").sum().alias(f"aVOI{i}") for i in _AVOI_LEVELS]
    )
    bar = _append_lost_bars(bar).sort("secu_code", "minute")
    bar = bar.with_columns(
        pl.col("AskPrice1").forward_fill().over("secu_code"),
        pl.col("BidPrice1").forward_fill().over("secu_code"),
    )
    bar = bar.join(aux, on="secu_code", how="left").with_columns(
        pl.col("AskPrice1").fill_null(pl.col("aux_price")),
        pl.col("BidPrice1").fill_null(pl.col("aux_price")),
    ).drop("aux_price")
    bar = _fill_book(bar)
    bar = bar.with_columns(pl.col(f"aVOI{i}").fill_null(0) for i in _AVOI_LEVELS)
    return bar.with_columns(pl.lit(date).alias("date")).select(list(SNAPSHOT_MIN_BAR_COLUMNS))


class SnapshotMinBar(Layer):
    """Aggregate raw order-book ticks to per-minute level-1..10 snapshot bars."""

    op = "snapshot_min_bar"

    def __call__(self, snapshot: Node) -> Node:
        return super().__call__({"snapshot": snapshot})


@register_executor("snapshot_min_bar")
def _snapshot_min_bar(node: Node, context: EvalContext) -> pl.LazyFrame:
    snapshot = context.evaluate(node.inputs["snapshot"])
    return build_snapshot_bar(snapshot, context.eval_date)


@register_info("snapshot_min_bar")
def _snapshot_min_bar_info(
    node: Node,
    parent_infos: dict[str, FrameInfo],
    context: EvalContext,
) -> FrameInfo:
    return FrameInfo.from_columns(
        SNAPSHOT_MIN_BAR_COLUMNS,
        identity_keys=("date", "secu_code", "minute"),
    )
