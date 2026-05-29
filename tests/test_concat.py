from __future__ import annotations

from pathlib import Path

import polars as pl

from draco_model import Engine, Model, Node
from draco_model.runtime.execution import EvalContext, register_executor
from draco_model.layers import Auction, Concat, DailyAgg, Field, Input


@register_executor("concat_left_intraday")
def _concat_left_intraday(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date],
            "secu_code": [1],
            "minute": [930],
            "value": [1.0],
        }
    ).lazy()


@register_executor("concat_right_intraday")
def _concat_right_intraday(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date],
            "secu_code": [1],
            "minute": [931],
            "value": [2.0],
        }
    ).lazy()


@register_executor("concat_daily")
def _concat_daily(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date],
            "secu_code": [1],
            "value": [9.0],
        }
    ).lazy()


def test_concat_broadcast_respects_intraday_minute_domain(sample_root: Path) -> None:
    intraday = Auction("drop")(Field("close")(Input(source="trades_tbar")))
    daily = DailyAgg(value_col="volume", agg="sum")(Field("volume")(Input(source="trades_tbar")))
    merged = Concat()({"close": intraday, "vol": daily})
    model = Model(name="merge_probe", universe="ex2kamt", output=merged)

    frame = Engine(data_root=sample_root).evaluate(model, merged, "20170103").collect()

    minutes = set(frame["minute"].unique().to_list())
    assert 925 not in minutes
    assert 1500 not in minutes


def test_concat_left_joins_daily_after_aligning_all_intraday(sample_root: Path) -> None:
    left = Node(kind="frame", op="concat_left_intraday")
    right = Node(kind="frame", op="concat_right_intraday")
    daily = Node(kind="frame", op="concat_daily")
    merged = Concat()({"left": left, "daily": daily, "right": right})
    model = Model(name="concat_align_probe", universe="ex2kamt", output=merged)

    frame = Engine(data_root=sample_root).evaluate(model, merged, "20170103").collect().sort("minute")

    assert frame["minute"].to_list() == [930, 931]
    assert frame["daily"].to_list() == [9.0, 9.0]
    assert frame["left"].to_list() == [1.0, None]
    assert frame["right"].to_list() == [None, 2.0]
