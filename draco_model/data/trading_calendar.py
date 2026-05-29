from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl


class TradingCalendar:
    """Ordered set of trading sessions used for lookback windows."""

    def __init__(self, dates: Iterable[str]) -> None:
        """Create a calendar from date-like values normalized to YYYYMMDD."""
        sessions = sorted({_normalize_date(date) for date in dates})
        if not sessions:
            raise ValueError("TradingCalendar requires at least one date.")
        self._sessions = tuple(sessions)
        self._index = {date: idx for idx, date in enumerate(self._sessions)}

    @classmethod
    def from_data_root(cls, data_root: str | Path) -> "TradingCalendar":
        """Load trading sessions from external/trading_days.parquet (required)."""
        root = Path(data_root)
        path = root.parent / "external" / "trading_days.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Trading calendar file is required but missing: {path}"
            )
        frame = pl.read_parquet(path)
        column = "date" if "date" in frame.columns else "trading_day"
        if column not in frame.columns:
            column = frame.columns[0]
        return cls(str(value) for value in frame[column].to_list())

    def previous_sessions(self, eval_date: str, lookback_days: int) -> list[str]:
        """Return the inclusive lookback window ending at eval_date."""
        date = _normalize_date(eval_date)
        idx = self._index.get(date)
        if idx is None:
            raise ValueError(f"eval_date {date!r} is not in the trading calendar.")
        start = idx - lookback_days + 1
        if start < 0:
            raise ValueError(f"Not enough trading history for {date!r}, lookback_days={lookback_days}.")
        return list(self._sessions[start : idx + 1])


def _normalize_date(value: object) -> str:
    return str(value).replace("-", "")
