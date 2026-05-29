from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model.data.source import SourceCatalog


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
