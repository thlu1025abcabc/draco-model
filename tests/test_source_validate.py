from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model
from draco_model.data.source import SourceCatalog, _FIXED_SOURCE_SCHEMAS, _standardize_columns
from draco_model.layers import Source
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS


def test_fixed_source_schema_does_not_scan_files(tmp_path: Path) -> None:
    catalog = SourceCatalog(tmp_path)

    trade_schema = catalog.schema("trades_tbar", ["20170103"])
    cancel_schema = catalog.schema("cancels_tbar", ["20170103"])
    quote_schema = catalog.schema("quotes_tbar", ["20170103"])
    daily_schema = catalog.schema("daily_k", ["20170103"])
    snapshot_schema = catalog.schema("snapshot_tbar", ["20170103"])
    universe_schema = catalog.schema("universe/ex2kamt", ["20170103"])

    assert trade_schema == (
        "secu_code",
        "minute",
        "price",
        "side",
        "volume",
        "vw_wait_time",
        "is_first",
        "is_last",
        "no",
        "date",
    )
    assert cancel_schema == trade_schema
    assert quote_schema == (
        "secu_code",
        "minute",
        "price",
        "side",
        "volume",
        "is_first",
        "is_last",
        "no",
        "date",
    )
    assert daily_schema == (
        "sec_code",
        "date",
        "open",
        "high",
        "low",
        "close",
        "shares",
        "amount",
        "limit_up",
        "limit_down",
        "preclose",
        "isSuspend",
        "isST",
        "adjfactor",
        "total_share",
        "float_share",
        "free_share",
        "list_date",
        "secu_code",
    )
    assert snapshot_schema == (
        *(f"AskPrice{level}" for level in range(1, 11)),
        *(f"BidPrice{level}" for level in range(1, 11)),
        *(f"AskVolume{level}" for level in range(1, 11)),
        *(f"BidVolume{level}" for level in range(1, 11)),
        *(f"aVOI{level}" for level in range(1, 6)),
        "secu_code",
        "minute",
        "date",
    )
    assert universe_schema == (
        "sec_code",
        "preclose",
        "close",
        "adjfactor",
        "secu_code",
        "date",
    )
    assert catalog._scans == {}


def test_fixed_source_schemas_match_standardized_representative_columns() -> None:
    for source, fixed_schema in _FIXED_SOURCE_SCHEMAS.items():
        actual = _standardize_columns(_representative_source_frame(source).lazy(), "20170103").collect_schema().names()
        missing = [column for column in fixed_schema if column not in actual]
        assert missing == [], f"{source} fixed schema missing from standardized columns: {missing}"


def test_fixed_source_missing_column_raises_clear_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="draco_model.data.source")
    _write_trading_days(tmp_path)
    path = tmp_path / "data" / "trades_tbar" / "20170103.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "SecuCode": [1],
            "MinBar": [930],
            "Price": [10.0],
            "Side": [0],
            "Volume": [1.0],
            "isfirst": [True],
            "islast": [True],
            "No": [1],
        }
    ).write_parquet(path)
    raw = Source("trades_tbar")

    with pytest.raises(ValueError, match="missing fixed schema columns.*vw_wait_time"):
        Engine(data_root=tmp_path / "data").evaluate(Model("bad_source", "ex2kamt", raw), raw, "20170103").collect()
    assert "source.fixed_schema_missing source=trades_tbar date=20170103" in caplog.text


def test_fixed_source_plans_have_expected_keys_and_grain(tmp_path: Path) -> None:
    _write_trading_days(tmp_path)
    engine = Engine(data_root=tmp_path / "data")
    engine._ensure_calendar()
    expected = {
        "trades_tbar": (KEY_COLUMNS, "raw"),
        "cancels_tbar": (KEY_COLUMNS, "raw"),
        "quotes_tbar": (KEY_COLUMNS, "raw"),
        "snapshot_tbar": (KEY_COLUMNS, "raw"),
        "daily_k": (DAILY_KEY_COLUMNS, "daily"),
        "universe/ex2kamt": (DAILY_KEY_COLUMNS, "daily"),
    }

    for source, (keys, grain) in expected.items():
        node = Source(source)
        schema = engine._infer_schema(Model(f"schema_{source.replace('/', '_')}", "ex2kamt", node), node, "20170103")
        assert schema.keys == keys
        assert schema.grain == grain


def test_unknown_source_schema_falls_back_to_scan(tmp_path: Path) -> None:
    path = tmp_path / "foo" / "20170103.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"SecuCode": [1], "MinBar": [930], "Price": [10.0]}).write_parquet(path)

    catalog = SourceCatalog(tmp_path)

    assert catalog.schema("foo", ["20170103"]) == ("secu_code", "minute", "price", "date")
    assert set(catalog._scans.keys()) == {("foo", "20170103")}


def test_validate_minutes_rejects_non_int64_minute(tmp_path: Path) -> None:
    catalog = SourceCatalog(tmp_path)
    frame = pl.DataFrame(
        {"date": ["20170103"], "secu_code": [1], "minute": [930.0]}
    ).lazy()

    with pytest.raises(ValueError, match="Float64"):
        catalog._validate_minutes(frame, "foo", "20170103")


def test_validate_minutes_rejects_off_grid_value(tmp_path: Path) -> None:
    path = tmp_path / "foo" / "20170103.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"secu_code": [1], "minute": [9999], "price": [10.0]}).write_parquet(path)

    catalog = SourceCatalog(tmp_path)
    with pytest.raises(ValueError, match="9999"):
        catalog.scan("foo", ["20170103"])


def test_validate_minutes_runs_once_per_source_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for date in ["20170103", "20170104"]:
        path = tmp_path / "foo" / f"{date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"secu_code": [1], "minute": [930], "price": [10.0]}).write_parquet(path)

    catalog = SourceCatalog(tmp_path)
    original = SourceCatalog._validate_minutes
    calls: list[tuple[str, str]] = []

    def spy(self: SourceCatalog, frame: pl.LazyFrame, source: str, date: str) -> None:
        calls.append((source, date))
        original(self, frame, source, date)

    monkeypatch.setattr(SourceCatalog, "_validate_minutes", spy)

    catalog.scan("foo", ["20170103"])
    catalog.scan("foo", ["20170103", "20170104"])

    assert calls == [("foo", "20170103"), ("foo", "20170104")]
    assert set(catalog._scans.keys()) == {("foo", "20170103"), ("foo", "20170104")}


def _write_trading_days(tmp_path: Path) -> None:
    path = tmp_path / "external" / "trading_days.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": ["20170103"]}).write_parquet(path)


def _representative_source_frame(source: str) -> pl.DataFrame:
    if source in {"trades_tbar", "cancels_tbar"}:
        return pl.DataFrame(
            {
                "SecuCode": [1],
                "MinBar": [930],
                "Price": [10.0],
                "Side": [0],
                "Volume": [1.0],
                "vw_wait_time": [0.0],
                "isfirst": [True],
                "islast": [True],
                "No": [1],
            }
        )
    if source == "quotes_tbar":
        return pl.DataFrame(
            {
                "SecuCode": [1],
                "MinBar": [930],
                "Price": [10.0],
                "Side": [0],
                "Volume": [1.0],
                "isfirst": [True],
                "islast": [True],
                "No": [1],
            }
        )
    if source == "daily_k":
        return pl.DataFrame(
            {
                "sec_code": ["000001.SZ"],
                "trading_day": ["2017-01-03"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "shares": [100.0],
                "amount": [1000.0],
                "limit_up": [11.0],
                "limit_down": [9.0],
                "preclose": [9.5],
                "isSuspend": [False],
                "isST": [False],
                "adjfactor": [1.0],
                "total_share": [1000.0],
                "float_share": [900.0],
                "free_share": [800.0],
                "list_date": ["19910403"],
            }
        )
    if source == "snapshot_tbar":
        return pl.DataFrame(
            {
                **{f"AskPrice{level}": [10.0 + level] for level in range(1, 11)},
                **{f"BidPrice{level}": [9.0 - level] for level in range(1, 11)},
                **{f"AskVolume{level}": [100.0 + level] for level in range(1, 11)},
                **{f"BidVolume{level}": [90.0 + level] for level in range(1, 11)},
                **{f"aVOI{level}": [float(level)] for level in range(1, 6)},
                "SecuCode": [1],
                "MinBar": [930],
            }
        )
    if source == "universe/ex2kamt":
        return pl.DataFrame(
            {
                "sec_code": ["000001.SZ"],
                "preclose": [9.5],
                "close": [10.0],
                "adjfactor": [1.0],
            }
        )
    raise ValueError(f"Unsupported representative source {source!r}.")
