from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model, metric
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.layers import Source, TradesWithWaitBar
from draco_model.layers.level2 import TRADES_WITH_WAIT_COLUMNS, aggregate_trades, match_trade_waits


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
    pl.DataFrame({"SecuCode": [1, 600000], "OrderTime": [93020000, 93010000], "OrderID": [12, 1], "OrderType": [1, 0]}).write_parquet(orders_dir / f"{DATE}.parquet")
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
