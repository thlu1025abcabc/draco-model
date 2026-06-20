"""Generate level-2 minute bars for one session for acceptance checks.

Build any of ``trades_wtminbar`` / ``quotes_minbar`` / ``cancels_minbar`` from
``steptrades`` + ``steporders`` under a data root, print a summary, optionally
write the vendor-format parquet, and diff against a golden parquet. Pass a
different ``date`` to verify another session.

Value columns (Volume/No/Price/Side and vw_wait_time) must match the golden
exactly. ``isfirst``/``islast`` may differ on a few simultaneous-event tie
breaks for ``quotes`` (see README); they are reported but do not fail the diff.

Examples
--------
    python sample/level2_bars.py 20260618
    python sample/level2_bars.py 20260618 --bar quotes
    python sample/level2_bars.py 20260618 --out-dir sample/output
    python sample/level2_bars.py 20260101 --bar cancels --data-root D:/prod/level2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from draco_model import Engine, Model
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.layers import CancelsMinBar, QuotesMinBar, Source, TradesWithWaitBar


# steptrades/steporders + the golden bars currently live here.
DEFAULT_DATA_ROOT = Path(r"D:\draco-bars\golden_sources")

# bar name -> (layer, golden subdirectory under the data root)
BARS = {
    "trades": (TradesWithWaitBar, "trades_wtminbar"),
    "quotes": (QuotesMinBar, "quotes_minbar"),
    "cancels": (CancelsMinBar, "cancels_minbar"),
}

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

# Columns that must match the golden exactly; isfirst/islast are tolerated.
EXACT_COLUMNS = {"SecuCode", "MinBar", "Price", "Side", "Volume", "No", "vw_wait_time"}


def build(bar: str, data_root: Path, date: str) -> pl.DataFrame:
    """Evaluate one bar for a date and return the standard frame."""
    layer, _ = BARS[bar]
    node = layer()(Source("steptrades"), Source("steporders"))
    model = Model(bar, None, {bar: node})
    engine = Engine(data_root=data_root, trading_calendar=TradingCalendar([date]))
    return engine.evaluate_outputs(model, date)[bar].collect()


def vendor(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert standard columns to the production vendor parquet contract."""
    return frame.drop("date").rename(VENDOR_COLUMNS)


def summarize(bar: str, frame: pl.DataFrame) -> None:
    """Print a one-line shape/coverage summary for a bar."""
    minute = frame["minute"]
    print(
        f"[{bar}] rows={frame.height:,}  minute={minute.min()}..{minute.max()}  "
        f"secu={frame['secu_code'].n_unique():,}"
    )


def diff_golden(produced: pl.DataFrame, golden_path: Path) -> bool:
    """Diff a produced bar against its golden parquet; tolerate isfirst/islast ties."""
    gold = pl.read_parquet(golden_path)
    left = vendor(produced).select(gold.columns).sort(GOLDEN_KEYS)
    right = gold.sort(GOLDEN_KEYS)
    if left.height != right.height:
        print(f"  FAIL rows: produced={left.height:,} golden={right.height:,}")
        return False
    exact_bad, tie_bad = [], []
    for col in gold.columns:
        if left[col].dtype == pl.Boolean:
            ne = int((left[col] != right[col]).sum())
        else:
            ne = int((~((left[col].cast(pl.Float64) - right[col].cast(pl.Float64)).abs() <= 1e-6)).sum())
        if not ne:
            continue
        (exact_bad if col in EXACT_COLUMNS else tie_bad).append(f"{col}={ne:,}")
    if exact_bad:
        print(f"  FAIL value columns differ: {', '.join(exact_bad)}")
        return False
    note = f"  (tie-break diffs: {', '.join(tie_bad)})" if tie_bad else ""
    print(f"  PASS vs golden ({right.height:,} rows){note}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate level-2 minute bars for one session.")
    parser.add_argument("date", help="Session date, e.g. 20260618.")
    parser.add_argument("--bar", choices=[*BARS, "all"], default="all", help="Which bar(s) to build.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root holding steptrades/<date>.parquet and steporders/<date>.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional dir to write <golden_name>/<date>.parquet vendor output.",
    )
    args = parser.parse_args()

    date = args.date.replace("-", "")
    bars = list(BARS) if args.bar == "all" else [args.bar]
    ok = True
    for bar in bars:
        frame = build(bar, args.data_root, date)
        summarize(bar, frame)
        _, golden_name = BARS[bar]
        if args.out_dir is not None:
            out = args.out_dir / golden_name / f"{date}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            vendor(frame).write_parquet(out)
            print(f"  wrote {out}")
        golden = args.data_root / golden_name / f"{date}.parquet"
        if golden.exists():
            ok = diff_golden(frame, golden) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
