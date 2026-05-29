from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model, Node
from draco_model.layers import Aggregate, Auction, Field, Fill, Input, Resample, RatioField
from draco_model.runtime.execution import EvalContext, register_executor


@register_executor("ordered_null_test_frame")
def _ordered_null_test_frame(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date, context.eval_date, context.eval_date],
            "secu_code": [1, 1, 1],
            "minute": [930, 931, 932],
            "close": [None, 10.5, 10.3],
        }
    ).lazy()


@register_executor("fill_test_frame")
def _fill_test_frame(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date, context.eval_date, context.eval_date, context.eval_date, context.eval_date],
            "secu_code": [1, 1, 1, 2, 2],
            "minute": [930, 931, 932, 930, 931],
            "value": [None, 10.0, None, None, 20.0],
        }
    ).lazy()


@register_executor("resample_sum_null_test_frame")
def _resample_sum_null_test_frame(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date, context.eval_date, context.eval_date],
            "secu_code": [1, 1, 1],
            "minute": [930, 931, 935],
            "volume": [None, None, 3.0],
        }
    ).lazy()


def test_fill_numeric_literal_replaces_nulls(sample_root: Path) -> None:
    filled = Fill(0)(Node(kind="frame", op="fill_test_frame"))
    model = Model(name="literal_fill_probe", universe="ex2kamt", output=filled)

    frame = Engine(data_root=sample_root).evaluate(model, filled, "20170103").collect()

    assert frame.sort(["secu_code", "minute"])["value"].to_list() == [0.0, 10.0, 0.0, 0.0, 20.0]


def test_fill_ffill_forward_fills_within_stock_date(sample_root: Path) -> None:
    filled = Fill("ffill")(Node(kind="frame", op="fill_test_frame"))
    model = Model(name="ffill_probe", universe="ex2kamt", output=filled)

    frame = Engine(data_root=sample_root).evaluate(model, filled, "20170103").collect()

    assert frame.sort(["secu_code", "minute"])["value"].to_list() == [None, 10.0, 10.0, None, 20.0]


def test_price_resample_simple_fields_use_explicit_aggregation(sample_root: Path) -> None:
    engine = Engine(data_root=sample_root)
    raw = Input(source="trades_tbar")
    open_node = Resample("5m", "first")(Field("open")(raw))
    close_node = Resample("5m", "last")(Field("close")(raw))
    high_node = Resample("5m", "max")(Field("high")(raw))
    low_node = Resample("5m", "min")(Field("low")(raw))
    model = Model(name="price_resample_probe", universe="ex2kamt", output=close_node)

    open_frame = engine.evaluate(model, open_node, "20170103").collect()
    close_frame = engine.evaluate(model, close_node, "20170103").collect()
    high_frame = engine.evaluate(model, high_node, "20170103").collect()
    low_frame = engine.evaluate(model, low_node, "20170103").collect()

    assert open_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["open"].to_list() == [10.1]
    assert close_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["close"].to_list() == [10.3]
    assert high_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["high"].to_list() == [10.5]
    assert low_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["low"].to_list() == [10.1]
    assert close_frame.filter((pl.col("secu_code") == 2) & (pl.col("minute") == 930))["close"].to_list() == []


def test_price_resample_uses_explicit_aggregation(sample_root: Path) -> None:
    close = Resample("5m", "max")(Field("close")(Input(source="trades_tbar")))
    model = Model(name="explicit_resample_probe", universe="ex2kamt", output=close)

    frame = Engine(data_root=sample_root).evaluate(model, close, "20170103").collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["close"].to_list() == [10.5]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["close"].to_list() == [None]


def test_price_resample_first_and_last_skip_nulls_after_ordering(sample_root: Path) -> None:
    raw = Node(kind="frame", op="ordered_null_test_frame")
    first_close = Resample("5m", "first")(raw)
    last_close = Resample("5m", "last")(raw)
    model = Model(name="ordered_null_probe", universe="ex2kamt", output=last_close)

    first_frame = Engine(data_root=sample_root).evaluate(model, first_close, "20170103").collect()
    last_frame = Engine(data_root=sample_root).evaluate(model, last_close, "20170103").collect()

    first_row = first_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    last_row = last_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))

    assert first_row["close"].to_list() == [10.5]
    assert last_row["close"].to_list() == [10.3]


def test_resample_sum_keeps_all_null_group_null(sample_root: Path) -> None:
    volume = Resample("5m", "sum")(Node(kind="frame", op="resample_sum_null_test_frame"))
    model = Model(name="resample_null_sum_probe", universe="ex2kamt", output=volume)

    frame = Engine(data_root=sample_root).evaluate(model, volume, "20170103").collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume"].to_list() == [None]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["volume"].to_list() == [3.0]


def test_price_resample_keeps_auction_bars_separate(sample_root: Path) -> None:
    close = Resample("5m", "last")(Field("close")(Input(source="trades_tbar")))
    model = Model(name="price_auction_probe", universe="ex2kamt", output=close)

    frame = Engine(data_root=sample_root).evaluate(model, close, "20170103").collect()

    assert {925, 930, 1500}.issubset(set(frame.filter(pl.col("secu_code") == 1)["minute"].to_list()))
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 925))["close"].to_list() == [9.85]


def test_auction_merge_aggregates_ratio_payload(sample_root: Path) -> None:
    vwap = Auction("merge", agg="sum")(RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar")))
    model = Model(name="auction_merge_vwap_probe", universe="ex2kamt", output=vwap)

    frame = Engine(data_root=sample_root).evaluate(model, vwap, "20170103").collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["vwap"].to_list() == pytest.approx([250.5 / 25.0])
    assert row["__ratio_vwap_num"].to_list() == pytest.approx([250.5])
    assert row["__ratio_vwap_den"].to_list() == pytest.approx([25.0])


def test_auction_merge_requires_explicit_aggregation(sample_root: Path) -> None:
    close = Auction("merge")(Field("close")(Input(source="trades_tbar")))
    model = Model(name="auction_merge_requires_agg", universe="ex2kamt", output=close)

    with pytest.raises(ValueError, match="requires agg"):
        Engine(data_root=sample_root).evaluate(model, close, "20170103").collect()


def test_resample_aggregates_ratio_payload_before_dividing(sample_root: Path) -> None:
    vwap = Resample("5m", "sum")(RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar")))
    model = Model(name="resample_vwap_probe", universe="ex2kamt", output=vwap)

    frame = Engine(data_root=sample_root).evaluate(model, vwap, "20170103").collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["vwap"].to_list() == pytest.approx([673.0 / 65.0])
    assert row["__ratio_vwap_num"].to_list() == pytest.approx([673.0])
    assert row["__ratio_vwap_den"].to_list() == pytest.approx([65.0])


def test_resample_can_aggregate_ratio_public_field(sample_root: Path) -> None:
    raw_vwap = RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar"))
    resampled = Resample("5m", "mean", apply_to="field")(raw_vwap)
    model = Model(name="resample_vwap_field_probe", universe="ex2kamt", output=resampled)

    frame = Engine(data_root=sample_root).evaluate(model, resampled, "20170103").collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["vwap"].to_list() == pytest.approx([(152.0 / 15.0 + 10.5 + 10.3) / 3.0])
    assert "__ratio_vwap_num" not in frame.columns
    assert "__ratio_vwap_den" not in frame.columns


def test_aggregate_layer_resamples_intraday_ratio_components(sample_root: Path) -> None:
    vwap = Aggregate("5m", "sum", apply_to="components")(
        RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar"))
    )
    model = Model(name="aggregate_vwap_probe", universe="ex2kamt", output=vwap)

    frame = Engine(data_root=sample_root).evaluate(model, vwap, "20170103").collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["vwap"].to_list() == pytest.approx([673.0 / 65.0])
    assert row["__ratio_vwap_num"].to_list() == pytest.approx([673.0])
    assert row["__ratio_vwap_den"].to_list() == pytest.approx([65.0])


def test_close_fill_state_forward_fills_then_uses_daily_preclose(sample_root: Path) -> None:
    extra_source = sample_root / "null_close_tbar" / "20170103.parquet"
    extra_source.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "SecuCode": [2, 2],
            "MinBar": [930, 931],
            "Price": [None, 21.0],
            "Side": [0, 0],
            "Volume": [0.0, 10.0],
            "No": [1, 1],
            "isfirst": [True, True],
            "islast": [True, True],
        }
    ).write_parquet(extra_source)

    close = Fill("state")(Field("close")(Input(source="null_close_tbar")))
    model = Model(name="close_fill_probe", universe="ex2kamt", output=close)

    frame = Engine(data_root=sample_root).evaluate(model, close, "20170103").collect()

    assert frame.sort("minute")["close"].to_list() == [20.0, 21.0]


def test_high_fill_state_uses_matching_close_state_after_transforms(sample_root: Path) -> None:
    high = Fill("state")(
        Resample("5m", "max")(Auction("drop")(Field("high")(Input(source="trades_tbar"))))
    )
    model = Model(name="high_fill_probe", universe="ex2kamt", output=high)

    frame = Engine(data_root=sample_root).evaluate(model, high, "20170103").collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["high"].to_list() == [10.5]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["high"].to_list() == [10.3]


def test_preclose_fill_state_uses_previous_close_state(sample_root: Path) -> None:
    preclose = Fill("state")(Auction("drop")(Field("preclose")(Input(source="trades_tbar"))))
    model = Model(name="preclose_fill_probe", universe="ex2kamt", output=preclose)

    frame = Engine(data_root=sample_root).evaluate(model, preclose, "20170103").collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["preclose"].to_list() == [9.5]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))["preclose"].to_list() == [10.2]


def test_fill_state_requires_supported_field_chain(sample_root: Path) -> None:
    close = Fill("state")(Auction("drop")(Field("close")(Input(source="trades_tbar"))))
    model = Model(name="price_fill_probe", universe="ex2kamt", output=close)

    frame = Engine(data_root=sample_root).evaluate(model, close, "20170103").collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))["close"].to_list() == [10.2]


def test_preclose_field_is_reserved(sample_root: Path) -> None:
    preclose = Field("preclose")(Input(source="trades_tbar"))
    model = Model(name="preclose_probe", universe="ex2kamt", output=preclose)

    with pytest.raises(ValueError, match="preclose field is reserved"):
        Engine(data_root=sample_root).evaluate(model, preclose, "20170103").collect()


def test_ratio_field_fill_state_uses_close_state(sample_root: Path) -> None:
    raw_vwap = RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar"))
    filled_vwap = Fill("state")(raw_vwap)
    close_state = Fill("state")(Field("close")(Input(source="trades_tbar")))
    model = Model(name="vwap_fill_probe", universe="ex2kamt", output=filled_vwap)

    engine = Engine(data_root=sample_root)
    raw_frame = engine.evaluate(model, raw_vwap, "20170103").collect()
    filled_frame = engine.evaluate(model, filled_vwap, "20170103").collect()
    close_frame = engine.evaluate(model, close_state, "20170103").collect()

    row_filter = (pl.col("secu_code") == 1) & (pl.col("minute") == 935)
    assert raw_frame.filter(row_filter)["vwap"].to_list() == [None]
    assert filled_frame.filter(row_filter)["vwap"].to_list() == close_frame.filter(row_filter)["close"].to_list()
    assert "__ratio_vwap_num" not in filled_frame.columns


def test_fill_state_replays_auction_merge_with_close_last(sample_root: Path) -> None:
    source = sample_root / "auction_state_tbar" / "20170103.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "SecuCode": [1, 1, 1],
            "MinBar": [925, 930, 931],
            "Price": [9.0, 10.0, None],
            "Amount": [9.0, 10.0, None],
            "Volume": [1.0, 1.0, None],
            "No": [1, 1, 1],
            "Side": [0, 0, 0],
            "isfirst": [True, True, True],
            "islast": [True, True, True],
        }
    ).write_parquet(source)

    vwap = Fill("state")(
        Auction("merge", agg="sum")(
            RatioField("amount", "volume", alias="vwap")(
                Input(source="auction_state_tbar")
            )
        )
    )
    model = Model(name="auction_state_probe", universe="ex2kamt", output=vwap)

    frame = Engine(data_root=sample_root).evaluate(model, vwap, "20170103").collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["vwap"].to_list() == [9.5]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))["vwap"].to_list() == [10.0]


def test_price_transform_layers_build_current_transform_nodes(sample_root: Path) -> None:
    chained = Resample("5m", "last")(Auction("drop")(Field("close")(Input(source="trades_tbar"))))
    model = Model(name="price_chain_probe", universe="ex2kamt", output=chained)

    frame = Engine(data_root=sample_root).evaluate(model, chained, "20170103").collect()

    assert [node.op for node in model.nodes()] == ["input", "field", "auction", "resample"]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["close"].to_list() == [10.3]
