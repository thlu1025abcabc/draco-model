from __future__ import annotations

from pathlib import Path

import polars as pl


class UniverseCatalog:
    """Scan universe membership files for evaluation dates."""

    def __init__(self, data_root: str | Path) -> None:
        """Bind the catalog to the data root."""
        self.data_root = Path(data_root)

    def scan(self, universe: str, eval_date: str) -> pl.LazyFrame:
        """Return the security codes in a universe on eval_date."""
        path = self.data_root / "universe" / universe / f"{eval_date}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing universe file: {path}")
        frame = pl.scan_parquet(path)
        columns = frame.collect_schema().names()
        if "secu_code" in columns:
            return frame.select(pl.col("secu_code").cast(pl.Int64)).unique()
        if "SecuCode" in columns:
            return frame.select(pl.col("SecuCode").cast(pl.Int64).alias("secu_code")).unique()
        if "sec_code" in columns:
            return frame.select(pl.col("sec_code").cast(pl.Utf8).str.slice(0, 6).cast(pl.Int64).alias("secu_code")).unique()
        raise ValueError(f"Universe file {path} must contain secu_code, SecuCode, or sec_code.")
