"""Generate the snapshot minute bar for one session for acceptance checks.

Build ``snapshot_minbar`` from the raw ``orderbook`` source under a data root,
print a summary, optionally write the vendor-format parquet, and diff against a
golden parquet. Pass a different ``date`` to verify another session.

This layer takes no ``daily_k`` input: a stock whose book is empty on both sides
from the open (with no pre-open trade price) keeps ``null`` bars, so a diff
against a golden that was back-filled from previous close will flag those rows.

Examples
--------
    python sample/snapshot_minbar.py 20251107
    python sample/snapshot_minbar.py 20251107 --out sample/output/snapshot/20251107.parquet
    python sample/snapshot_minbar.py 20251107 --data-root D:/prod/level2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from draco_model import Engine, Model
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.layers import SnapshotMinBar, Source


# orderbook + the golden snapshot_tbar currently live here.
DEFAULT_DATA_ROOT = Path(r"D:\draco-model\data")

GOLDEN_KEYS = ["SecuCode", "MinBar"]


def build(data_root: Path, date: str) -> pl.DataFrame:
    """Evaluate snapshot_minbar for one date and return the standard frame."""
    node = SnapshotMinBar()(Source("orderbook"))
    model = Model("snapshot_minbar", None, {"snapshot_minbar": node})
    engine = Engine(data_root=data_root, trading_calendar=TradingCalendar([date]))
    return engine.evaluate_outputs(model, date)["snapshot_minbar"].collect()


def vendor(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert the standard columns to the production vendor parquet contract."""
    return frame.drop("date").rename({"secu_code": "SecuCode", "minute": "MinBar"})


def summarize(frame: pl.DataFrame) -> None:
    """Print shape, minute/stock coverage, and any remaining null bars."""
    minute = frame["minute"]
    nulls = sum(frame[column].null_count() for column in frame.columns)
    print(f"rows={frame.height:,}  cols={frame.width}")
    print(f"minute range: {minute.min()}..{minute.max()}  distinct={minute.n_unique()}")
    print(f"secu_code distinct: {frame['secu_code'].n_unique():,}")
    print(f"null cells: {nulls:,}")
    print(frame.head(5))


def diff_golden(produced: pl.DataFrame, golden_path: Path) -> bool:
    """Diff produced bars against a golden parquet; every column must match."""
    gold = pl.read_parquet(golden_path)
    left = vendor(produced).select(gold.columns).sort(GOLDEN_KEYS)
    right = gold.sort(GOLDEN_KEYS)
    if left.height != right.height:
        print(f"  FAIL rows: produced={left.height:,} golden={right.height:,}")
        return False
    bad = []
    for column in gold.columns:
        ne = int((~((left[column].cast(pl.Float64) - right[column].cast(pl.Float64)).abs() <= 1e-4)).sum())
        if ne:
            bad.append(f"{column}={ne:,}")
    if bad:
        print(f"  FAIL value columns differ: {', '.join(bad)}")
        return False
    print(f"  PASS vs golden ({right.height:,} rows)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate snapshot_minbar for one session.")
    parser.add_argument("date", help="Session date, e.g. 20251107.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root holding orderbook/<date>.parquet (and optionally snapshot_tbar/<date>.parquet).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional parquet path to write the vendor-format bars.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Golden vendor parquet to diff against. "
        "Defaults to <data-root>/snapshot_tbar/<date>.parquet when present.",
    )
    args = parser.parse_args()

    date = args.date.replace("-", "")
    bars = build(args.data_root, date)
    summarize(bars)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        vendor(bars).write_parquet(args.out)
        print("wrote", args.out)

    golden = args.golden
    if golden is None:
        candidate = args.data_root / "snapshot_tbar" / f"{date}.parquet"
        golden = candidate if candidate.exists() else None
    if golden is not None:
        return 0 if diff_golden(bars, golden) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
