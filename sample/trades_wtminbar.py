"""Generate the ``trades_wtminbar`` bars for one session for acceptance checks.

Build ``trades_wtminbar`` from ``steptrades`` + ``steporders`` under a data
root, print a summary, optionally write the vendor-format parquet, and diff it
against a golden parquet. Pass a different ``date`` to verify another session.

Examples
--------
    python sample/trades_wtminbar.py 20260618
    python sample/trades_wtminbar.py 20260618 --out sample/output/20260618.parquet
    python sample/trades_wtminbar.py 20260101 --data-root D:/prod/level2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from draco_model import Engine, Model
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.layers import Source, TradesWithWaitBar


# steptrades/steporders + the golden trades_wtminbar currently live here.
DEFAULT_DATA_ROOT = Path(r"D:\draco-bars\golden_sources")

# Production parquet uses vendor column names and drops the date column.
VENDOR_COLUMNS = {
    "secu_code": "SecuCode",
    "minute": "MinBar",
    "price": "Price",
    "side": "Side",
    "volume": "Volume",
    "is_first": "isfirst",
    "is_last": "islast",
    "no": "No",
}

GOLDEN_KEYS = ["SecuCode", "MinBar", "Price", "Side"]


def build_bars(data_root: Path, date: str) -> pl.DataFrame:
    """Evaluate ``trades_wtminbar`` for one date and return the standard frame."""
    node = TradesWithWaitBar()(Source("steptrades"), Source("steporders"))
    model = Model("trades_wtminbar", None, {"trades_wtminbar": node})
    engine = Engine(data_root=data_root, trading_calendar=TradingCalendar([date]))
    return engine.evaluate_outputs(model, date)["trades_wtminbar"].collect()


def vendor_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert standard columns to the production vendor parquet contract."""
    return frame.drop("date").rename(VENDOR_COLUMNS)


def summarize(frame: pl.DataFrame) -> None:
    """Print shape, minute coverage, null counts, and a small preview."""
    minute = frame["minute"]
    nulls = {column: frame[column].null_count() for column in frame.columns if frame[column].null_count()}
    print(f"rows={frame.height:,}  cols={frame.width}")
    print(f"minute range: {minute.min()}..{minute.max()}  distinct={minute.n_unique()}")
    print(f"secu_code distinct: {frame['secu_code'].n_unique():,}")
    print(f"null counts: {nulls or 'none'}")
    print(frame.head(8))


def compare_to_golden(produced: pl.DataFrame, golden_path: Path) -> bool:
    """Diff produced bars against a golden vendor parquet; return True on match."""
    left = vendor_frame(produced).sort(GOLDEN_KEYS)
    golden = pl.read_parquet(golden_path).select(left.columns).sort(GOLDEN_KEYS)
    if left.height != golden.height:
        print(f"FAIL row count: produced={left.height:,} golden={golden.height:,}")
        return False
    try:
        assert_frame_equal(left, golden, check_exact=False, atol=1e-9, rtol=1e-6)
    except AssertionError as error:
        print(f"FAIL value mismatch vs {golden_path}:")
        print(str(error).splitlines()[0])
        return False
    print(f"PASS exact match vs golden ({golden.height:,} rows)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate trades_wtminbar for one session.")
    parser.add_argument("date", help="Session date, e.g. 20260618.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root holding steptrades/<date>.parquet and steporders/<date>.parquet.",
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
        "Defaults to <data-root>/trades_wtminbar/<date>.parquet when present.",
    )
    args = parser.parse_args()

    date = args.date.replace("-", "")
    bars = build_bars(args.data_root, date)
    summarize(bars)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        vendor_frame(bars).write_parquet(args.out)
        print("wrote", args.out)

    golden = args.golden
    if golden is None:
        candidate = args.data_root / "trades_wtminbar" / f"{date}.parquet"
        golden = candidate if candidate.exists() else None
    if golden is not None:
        return 0 if compare_to_golden(bars, golden) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
