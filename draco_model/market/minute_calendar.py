from __future__ import annotations

import polars as pl


AUCTION_MINUTES = (925, 1500)


def _range_minutes(start: int, end: int) -> list[int]:
    out: list[int] = []
    hour = start // 100
    minute = start % 100
    while hour * 100 + minute <= end:
        out.append(hour * 100 + minute)
        minute += 1
        if minute == 60:
            hour += 1
            minute = 0
    return out


_MINBARS = (925, *_range_minutes(930, 1129), *_range_minutes(1300, 1456), 1500)


class MinuteCalendar:
    """Fixed A-share minute-bar calendar including auction bars."""

    VERSION = "ashare-fixed-v1"

    def __init__(self) -> None:
        self._bucket_maps: dict[int, pl.DataFrame] = {}

    def minbars(self) -> list[int]:
        """Return valid minute bars in HHMM integer format."""
        return list(_MINBARS)

    def bucket_map(self, interval: int) -> pl.LazyFrame:
        """Return a cached minute-to-resample-bucket mapping."""
        if interval < 1:
            raise ValueError("frequency interval must be >= 1.")
        if interval not in self._bucket_maps:
            self._bucket_maps[interval] = self._build_bucket_map(interval)
        return self._bucket_maps[interval].lazy()

    def _build_bucket_map(self, interval: int) -> pl.DataFrame:
        continuous = [minute for minute in _MINBARS if minute not in AUCTION_MINUTES]
        rows = []
        for idx, minute in enumerate(continuous):
            rows.append({"minute": minute, "__bucket_minute": continuous[(idx // interval) * interval]})
        return pl.DataFrame(rows)
