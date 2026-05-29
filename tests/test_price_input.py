from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model, Node
from draco_model.data.source import SourceCatalog
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.data.universe import UniverseCatalog
from draco_model.layers import Auction, Field, Fill, Input
from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.runtime.execution import EvalContext


def test_input_scans_raw_source_without_intraday_grid(sample_root: Path) -> None:
    raw = Input(source="trades_tbar")
    model = Model(name="raw_probe", universe="ex2kamt", output=raw)

    frame = Engine(data_root=sample_root).evaluate(model, raw, "20170103").collect()

    assert {"date", "secu_code", "minute", "price", "volume", "no"}.issubset(set(frame.columns))
    assert frame.height < 2 * 239


def test_intraday_grid_caches_base_universe_minute_grid(sample_root: Path) -> None:
    minute_calendar = MinuteCalendar()
    grid_cache = {}
    context = EvalContext(
        model=Model(name="grid_probe", universe="ex2kamt", output=Input(source="trades_tbar")),
        eval_date="20170103",
        sources=SourceCatalog(sample_root, minute_calendar),
        universes=UniverseCatalog(sample_root),
        minute_calendar=minute_calendar,
        trading_calendar=TradingCalendar(["20170103", "20170104"]),
        evaluate=lambda node: pl.DataFrame().lazy(),
        infer_schema=lambda node: pytest.fail("intraday_grid should not infer node schemas"),
        grid_cache=grid_cache,
    )

    one_day = context.intraday_grid("ex2kamt", ["20170103"]).collect()
    two_days = context.intraday_grid("ex2kamt", ["20170103", "20170104"]).collect()

    assert list(grid_cache) == [("ex2kamt", "20170103")]
    assert isinstance(grid_cache[("ex2kamt", "20170103")], pl.DataFrame)
    assert grid_cache[("ex2kamt", "20170103")].height == 2 * 239
    assert one_day.height == 2 * 239
    assert two_days.height == 2 * 239 * 2
    assert sorted(two_days["date"].unique().to_list()) == ["20170103", "20170104"]


def test_engine_evaluate_cache_is_scoped_by_universe(sample_root: Path) -> None:
    path = sample_root / "universe" / "oneonly" / "20170103.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"secu_code": [1]}).write_parquet(path)

    close = Field("close")(Input(source="trades_tbar"))
    engine = Engine(data_root=sample_root)

    broad = Model(name="broad", universe="ex2kamt", output=close)
    narrow = Model(name="narrow", universe="oneonly", output=close)

    assert engine.evaluate(broad, close, "20170103").collect().height == engine.evaluate(
        narrow, close, "20170103"
    ).collect().height
    assert (broad.universe, close.id, "20170103") in engine._memory
    assert (narrow.universe, close.id, "20170103") in engine._memory


def test_fill_state_close_recovers_after_null_minute(sample_root: Path) -> None:
    output = Fill("state")(Auction("drop")(Field("close")(Input(source="trades_tbar"))))
    model = Model(name="fill_probe", universe="ex2kamt", output=output)

    frame = (
        Engine(data_root=sample_root)
        .evaluate(model, output, "20170103")
        .collect()
        .sort(["secu_code", "minute"])
    )

    row_931 = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))
    row_935 = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))
    assert row_931.height == 1
    assert row_931["close"][0] == 10.2
    assert row_935.height == 1
    assert row_935["close"][0] == 10.3


def test_engine_requires_trading_calendar_file(tmp_path: Path) -> None:
    output = Node(kind="frame", op="constant_test_frame", params={"value": 1})
    model = Model(name="x", universe="u", output=output)

    with pytest.raises(FileNotFoundError, match="trading_days.parquet"):
        Engine(data_root=tmp_path / "data").collect(model, dates=["20170103"])


def test_trading_calendar_reads_external_first_column(tmp_path: Path) -> None:
    external = tmp_path / "external" / "trading_days.parquet"
    external.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"": ["20170103", "20170104"]}).write_parquet(external)

    calendar = TradingCalendar.from_data_root(tmp_path / "data")

    assert calendar.previous_sessions("20170104", 2) == ["20170103", "20170104"]
