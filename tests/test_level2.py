from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model, metric
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.layers import CancelsMinBar, QuotesMinBar, Source, TradesWithWaitBar
from draco_model.layers.level2 import (
    QUOTES_MIN_BAR_COLUMNS,
    TRADES_WITH_WAIT_COLUMNS,
    _add_cancel_wait,
    _add_sort_int,
    aggregate_trades,
    match_trade_waits,
    split_orders_cancels,
)


DATE = "20170103"


def _engine(data_root: Path) -> Engine:
    return Engine(data_root=data_root, trading_calendar=TradingCalendar([DATE]))


def test_steptrades_source_normalizes_vendor_columns_and_codes(tmp_path: Path) -> None:
    directory = tmp_path / "steptrades"
    directory.mkdir()
    pl.DataFrame(
        {
            "SecuCode": [22, 600000, 800001],
            "DealTime": [93000000] * 3,
            "BuyID": [1, 2, 3],
            "SellID": [4, 5, 6],
            "DealID": [1, 2, 3],
            "Price": [1000, 2000, 3000],
            "Volume": [10, 20, 30],
            "Side": [0, 1, 0],
        }
    ).write_parquet(directory / f"{DATE}.parquet")
    node = Source("steptrades")
    out = _engine(tmp_path).evaluate(Model("source", "ex2kamt", {"value": node}), node, DATE).collect()
    assert out.columns == ["date", "secu_code", "deal_time", "buy_id", "sell_id", "deal_id", "price", "volume", "side"]
    assert out.sort("secu_code")["secu_code"].to_list() == [1872, 600000]
    assert out.schema["price"] == pl.Int64
    assert out.schema["volume"] == pl.Float64


def test_steporders_source_reports_missing_columns(tmp_path: Path) -> None:
    directory = tmp_path / "steporders"
    directory.mkdir()
    pl.DataFrame({"OrderTime": [93000000]}).write_parquet(directory / f"{DATE}.parquet")
    node = Source("steporders")
    with pytest.raises(ValueError, match="missing fixed schema columns"):
        _engine(tmp_path).evaluate(Model("source", "ex2kamt", {"value": node}), node, DATE).collect()


def test_match_trade_waits_corrects_lunch_break() -> None:
    trades = pl.DataFrame({"date": [DATE], "secu_code": [1], "deal_time": [130030000], "buy_id": [1], "sell_id": [2], "deal_id": [1], "price": [2000], "volume": [10.0], "side": [0]}).lazy()
    orders = pl.DataFrame({"date": [DATE], "secu_code": [1], "order_time": [113000000], "order_id": [1], "order_type": [2]}).lazy()
    out = match_trade_waits(trades, orders).collect()
    assert out["wait_time"].to_list() == [30.0]
    assert out["price"].to_list() == [20.0]


def test_aggregate_trades_builds_minute_price_side_rows() -> None:
    events = pl.DataFrame(
        {
            "date": [DATE] * 4,
            "secu_code": [1] * 4,
            "deal_time": [92500000, 93000500, 93030000, 150000000],
            "price": [9.9, 10.0, 10.0, 10.8],
            "side": [0, 0, 0, 1],
            "volume": [100.0, 10.0, 30.0, 7.0],
            "wait_time": [0.0, 2.0, 4.0, 0.0],
            "sort_int": [1, 2, 3, 4],
        }
    ).lazy()
    out = aggregate_trades(events).collect()
    continuous = out.filter((pl.col("minute") == 930) & (pl.col("price") == 10.0)).row(0, named=True)
    assert continuous["volume"] == 40.0
    assert continuous["no"] == 2
    assert continuous["vw_wait_time"] == 3.5
    assert set(out["minute"].to_list()) == {925, 930, 1500}


def test_trades_with_wait_bar_is_a_named_model_output(tmp_path: Path) -> None:
    trades_dir = tmp_path / "steptrades"
    orders_dir = tmp_path / "steporders"
    trades_dir.mkdir()
    orders_dir.mkdir()
    pl.DataFrame({"SecuCode": [1, 600000], "DealTime": [93025000, 93015000], "BuyID": [12, 1], "SellID": [99, 2], "DealID": [1, 1], "Price": [2010, 1005], "Volume": [50.0, 80.0], "Side": [0, 0]}).write_parquet(trades_dir / f"{DATE}.parquet")
    pl.DataFrame({"SecuCode": [1, 600000], "OrderTime": [93020000, 93010000], "OrderID": [12, 1], "OrderType": [1, 0], "Price": [2010, 1005], "Volume": [50.0, 80.0]}).write_parquet(orders_dir / f"{DATE}.parquet")
    bars = TradesWithWaitBar()(Source("steptrades"), Source("steporders"))
    close = metric("close")(bars)
    engine = _engine(tmp_path)
    model = Model("bars", None, {"trades_wtminbar": bars, "close": close})
    info = engine._infer_info(model, bars, DATE)
    outputs = engine.evaluate_outputs(model, DATE)
    bar_frame = outputs["trades_wtminbar"].collect()
    close_frame = outputs["close"].collect()
    assert list(outputs) == ["trades_wtminbar", "close"]
    assert info.identity_keys == ("date", "secu_code", "minute", "price", "side")
    assert bar_frame.columns == list(TRADES_WITH_WAIT_COLUMNS)
    assert close_frame.columns == ["date", "secu_code", "minute", "close"]
    assert close_frame["close"].sort().to_list() == [10.05, 20.1]

    with pytest.raises(ValueError, match="requires Model.universe"):
        engine.collect(model, [DATE])
    with pytest.raises(ValueError, match="requires Model.universe"):
        engine.collect_many([model], [DATE])


def test_sh_order_stream_builds_quotes_and_cancels() -> None:
    orders = pl.DataFrame(
        {
            "secu_code": [600000, 600000, 600000, 600000],
            "date": [DATE] * 4,
            "order_time": [93100000, 93100000, 93200000, 93200000],
            "order_id": [11, 12, 13, 14],
            "order_type": [0, 10, -1, -11],
            "price": [1000, 1010, 990, 1020],
            "volume": [10.0, 20.0, 5.0, 7.0],
        }
    ).lazy()
    # a pre-open SH trade is never folded into the continuous quote stream
    trades = pl.DataFrame(
        {
            "secu_code": [600000], "date": [DATE], "deal_time": [92500000],
            "buy_id": [1], "sell_id": [2], "deal_id": [1], "price": [1000], "volume": [1.0], "side": [0],
        }
    ).lazy()

    quotes, cancels = split_orders_cancels(trades, orders, DATE)
    q = quotes.collect().sort("order_id")
    c = cancels.collect().sort("order_id")

    assert q["side"].to_list() == [0, 1]  # order_type 0 -> buy, 10 -> sell
    assert q["price"].to_list() == pytest.approx([10.0, 10.1])  # vendor price scaled by 1/100
    assert c["side"].to_list() == [0, 1]  # cancel order_type -1 -> 0, -11 -> 1
    assert c["price"].to_list() == pytest.approx([9.9, 10.2])


def test_sz_market_order_price_discovered_and_cancel_wait() -> None:
    orders = pl.DataFrame(
        {
            "secu_code": [1, 1],
            "date": [DATE, DATE],
            "order_time": [93100000, 93105000],
            "order_id": [21, 22],
            "order_type": [2, 1],  # priced limit buy, market buy (price from trades)
            "price": [1000, 0],
            "volume": [10.0, 30.0],
        }
    ).lazy()
    trades = pl.DataFrame(
        {
            "secu_code": [1, 1],
            "date": [DATE, DATE],
            "deal_time": [93110000, 93200000],
            "buy_id": [22, 22],
            "sell_id": [99, 0],
            "deal_id": [1, 2],
            "price": [1005, 0],
            "volume": [30.0, 5.0],
            "side": [0, -1],  # a real fill of order 22, then a buy-side cancel
        }
    ).lazy()

    quotes, cancels = split_orders_cancels(trades, orders, DATE)
    q = quotes.collect().sort("order_id")
    assert q["order_id"].to_list() == [21, 22]
    assert q["price"].to_list() == pytest.approx([10.0, 10.05])  # priced keeps its price, market takes the fill price

    resolved = _add_cancel_wait(_add_sort_int(cancels), orders).collect()
    assert resolved["side"].to_list() == [0]
    assert resolved["price"].to_list() == pytest.approx([10.05])  # cancel price recovered from the quote
    assert resolved["wait_time"].to_list() == [55.0]  # 09:32:00 - 09:31:05


def test_quotes_and_cancels_min_bar_named_outputs(tmp_path: Path) -> None:
    trades_dir = tmp_path / "steptrades"
    orders_dir = tmp_path / "steporders"
    trades_dir.mkdir()
    orders_dir.mkdir()
    pl.DataFrame(
        {
            "SecuCode": [1, 1],
            "DealTime": [93110000, 93200000],
            "BuyID": [22, 22],
            "SellID": [99, 0],
            "DealID": [1, 2],
            "Price": [1005, 0],
            "Volume": [30.0, 5.0],
            "Side": [0, -1],
        }
    ).write_parquet(trades_dir / f"{DATE}.parquet")
    pl.DataFrame(
        {
            "SecuCode": [1, 1],
            "OrderTime": [93100000, 93105000],
            "OrderID": [21, 22],
            "OrderType": [2, 1],
            "Price": [1000, 0],
            "Volume": [10.0, 30.0],
        }
    ).write_parquet(orders_dir / f"{DATE}.parquet")

    quotes = QuotesMinBar()(Source("steptrades"), Source("steporders"))
    cancels = CancelsMinBar()(Source("steptrades"), Source("steporders"))
    engine = _engine(tmp_path)
    model = Model("bars", None, {"quotes_minbar": quotes, "cancels_minbar": cancels})
    info_q = engine._infer_info(model, quotes, DATE)
    info_c = engine._infer_info(model, cancels, DATE)
    outputs = engine.evaluate_outputs(model, DATE)
    quote_frame = outputs["quotes_minbar"].collect()
    cancel_frame = outputs["cancels_minbar"].collect()

    assert quote_frame.columns == list(QUOTES_MIN_BAR_COLUMNS)
    assert cancel_frame.columns == list(TRADES_WITH_WAIT_COLUMNS)
    assert info_q.identity_keys == ("date", "secu_code", "minute", "price", "side")
    assert info_c.identity_keys == ("date", "secu_code", "minute", "price", "side")
    assert 10.05 in quote_frame["price"].to_list()  # market order price discovered from trade
    assert cancel_frame["vw_wait_time"].to_list() == [55.0]
