from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.layers import Source, SnapshotMinBar
from draco_model.layers.snapshot import SNAPSHOT_MIN_BAR_COLUMNS


DATE = "20170103"


def _book(ask1: int, bid1: int, ask_vol: float, bid_vol: float) -> dict:
    """A full 10-level book pinned to the level-1 ask/bid for simplicity."""
    return {
        **{f"AskPrice{level}": ask1 for level in range(1, 11)},
        **{f"BidPrice{level}": bid1 for level in range(1, 11)},
        **{f"AskVolume{level}": ask_vol for level in range(1, 11)},
        **{f"BidVolume{level}": bid_vol for level in range(1, 11)},
    }


def _write_orderbook(tmp_path: Path) -> None:
    ob_dir = tmp_path / "orderbook"
    ob_dir.mkdir()
    rows = [
        {"SecuCode": 1, "TickTime": 92500000, "Price": 995, **_book(1000, 999, 50.0, 50.0)},
        {"SecuCode": 1, "TickTime": 93000000, "Price": 1000, **_book(1000, 999, 100.0, 100.0)},
        {"SecuCode": 1, "TickTime": 93030000, "Price": 1005, **_book(1010, 1009, 200.0, 200.0)},
        {"SecuCode": 1, "TickTime": 93100000, "Price": 1010, **_book(1020, 1019, 300.0, 300.0)},
    ]
    pl.DataFrame(rows).write_parquet(ob_dir / f"{DATE}.parquet")


def test_snapshot_min_bar_outputs_full_grid(tmp_path: Path) -> None:
    _write_orderbook(tmp_path)
    node = SnapshotMinBar()(Source("orderbook"))
    engine = Engine(data_root=tmp_path, trading_calendar=TradingCalendar([DATE]))
    model = Model("snapshot", None, {"snapshot_minbar": node})

    info = engine._infer_info(model, node, DATE)
    out = engine.evaluate_outputs(model, DATE)["snapshot_minbar"].collect()

    assert out.columns == list(SNAPSHOT_MIN_BAR_COLUMNS)
    assert info.identity_keys == ("date", "secu_code", "minute")
    # pre-open tick (92500000) only seeds aux; bars cover the two traded minutes
    assert out["minute"].sort().to_list() == [930, 931]
    assert out.select(pl.exclude("date").is_null().any()).to_numpy().any() == False

    bar_930 = out.filter(pl.col("minute") == 930).row(0, named=True)
    bar_931 = out.filter(pl.col("minute") == 931).row(0, named=True)
    # AskPrice1 is the per-minute mean of the vendor price, scaled by 1/100
    assert bar_930["AskPrice1"] == pytest.approx((1000 + 1010) / 2 / 100)
    assert bar_931["AskPrice1"] == pytest.approx(1020 / 100)
    assert bar_930["AskVolume1"] == pytest.approx((100.0 + 200.0) / 2)
