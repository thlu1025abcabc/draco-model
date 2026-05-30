from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model.data.source import SourceCatalog


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
