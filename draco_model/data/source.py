from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from draco_model.market.minute_calendar import MinuteCalendar


logger = logging.getLogger(__name__)


class SourceCatalog:
    """Scan dated parquet sources and normalize common market columns."""

    def __init__(self, data_root: str | Path, minute_calendar: MinuteCalendar | None = None) -> None:
        """Bind the catalog to a data root and minute calendar."""
        self.data_root = Path(data_root)
        self.minute_calendar = minute_calendar or MinuteCalendar()
        self._scans: dict[tuple[str, str], pl.LazyFrame] = {}

    def scan(self, source: str, dates: list[str]) -> pl.LazyFrame:
        """Scan one source across dates as a single LazyFrame."""
        logger.debug("source.scan source=%s dates=%s", source, dates)
        frames = [self._scan_date(source, date) for date in dates]
        if not frames:
            raise ValueError("Source scan requires at least one date.")
        return pl.concat(frames, how="diagonal_relaxed")

    def schema(self, source: str, dates: list[str]) -> tuple[str, ...]:
        """Return normalized source columns, using fixed contracts without scanning known sources."""
        if not dates:
            raise ValueError("Source schema requires at least one date.")
        if source in _FIXED_SOURCE_SCHEMAS:
            logger.debug("source.schema.fixed source=%s columns=%d", source, len(_FIXED_SOURCE_SCHEMAS[source]))
            return _FIXED_SOURCE_SCHEMAS[source]
        logger.debug("source.schema.scan source=%s dates=%s", source, dates)
        return tuple(self.scan(source, dates).collect_schema().names())

    def _scan_date(self, source: str, date: str) -> pl.LazyFrame:
        key = (source, date)
        if key in self._scans:
            logger.debug("source.scan_date.cache_hit source=%s date=%s", source, date)
            return self._scans[key]
        path = self.data_root / source / f"{date}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        logger.debug("source.scan_date.start source=%s date=%s path=%s", source, date, path)
        frame = pl.scan_parquet(path)
        frame = _standardize_columns(frame, date)
        self._validate_fixed_schema(frame, source, date)
        self._validate_minutes(frame, source, date)
        self._scans[key] = frame
        logger.debug("source.scan_date.done source=%s date=%s", source, date)
        return frame

    def _validate_fixed_schema(self, frame: pl.LazyFrame, source: str, date: str) -> None:
        fixed = _FIXED_SOURCE_SCHEMAS.get(source)
        if fixed is None:
            return
        actual = tuple(frame.collect_schema().names())
        missing = [column for column in fixed if column not in actual]
        if missing:
            logger.error(
                "source.fixed_schema_missing source=%s date=%s missing=%s actual=%s",
                source,
                date,
                missing,
                list(actual),
            )
            raise ValueError(
                f"Source {source!r} date {date} is missing fixed schema columns: {missing}. "
                f"Actual normalized columns: {list(actual)}."
            )
        logger.debug("source.fixed_schema_ok source=%s date=%s columns=%d", source, date, len(fixed))

    def _validate_minutes(self, frame: pl.LazyFrame, source: str, date: str) -> None:
        schema = frame.collect_schema()
        if "minute" not in schema.names():
            return
        if schema["minute"] != pl.Int64:
            logger.error(
                "source.minute_dtype_invalid source=%s date=%s dtype=%s",
                source,
                date,
                schema["minute"],
            )
            raise ValueError(
                f"Source {source!r} date {date} has minute column with dtype "
                f"{schema['minute']}, expected Int64."
            )
        allowed = set(self.minute_calendar.minbars())
        invalid = (
            frame.filter(~pl.col("minute").is_in(allowed))
            .select("minute")
            .unique()
            .sort("minute")
            .limit(5)
            .collect()
        )
        if invalid.height:
            values = invalid["minute"].to_list()
            logger.error("source.minute_grid_invalid source=%s date=%s values=%s", source, date, values)
            raise ValueError(f"Source {source!r} date {date} has minute bars outside fixed grid: {values}.")


def _standardize_columns(frame: pl.LazyFrame, date: str) -> pl.LazyFrame:
    columns = frame.collect_schema().names()
    renames = {}
    for source, target in {
        "SecuCode": "secu_code",
        "MinBar": "minute",
        "Price": "price",
        "Amount": "amount",
        "Volume": "volume",
        "No": "no",
        "Side": "side",
        "isfirst": "is_first",
        "islast": "is_last",
        "trading_day": "date",
    }.items():
        if source in columns and target not in columns:
            renames[source] = target
    if renames:
        frame = frame.rename(renames)
        columns = [renames.get(column, column) for column in columns]
    if "sec_code" in columns and "secu_code" not in columns:
        frame = frame.with_columns(
            pl.col("sec_code").cast(pl.Utf8).str.slice(0, 6).cast(pl.Int64).alias("secu_code")
        )
        columns.append("secu_code")
    if "date" not in columns:
        frame = frame.with_columns(pl.lit(date).alias("date"))
        columns.append("date")
    else:
        frame = frame.with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("-", "").alias("date"))
    casts = []
    if "secu_code" in columns:
        casts.append(pl.col("secu_code").cast(pl.Utf8).str.slice(0, 6).cast(pl.Int64).alias("secu_code"))
    if "minute" in columns:
        casts.append(pl.col("minute").cast(pl.Int64))
    if "is_first" in columns:
        casts.append(pl.col("is_first").cast(pl.Boolean))
    if "is_last" in columns:
        casts.append(pl.col("is_last").cast(pl.Boolean))
    if casts:
        frame = frame.with_columns(casts)
    return frame


_TRADE_CANCEL_TBAR_SCHEMA = (
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

_QUOTE_TBAR_SCHEMA = (
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

_DAILY_K_SCHEMA = (
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

_SNAPSHOT_TBAR_SCHEMA = (
    *(f"AskPrice{level}" for level in range(1, 11)),
    *(f"BidPrice{level}" for level in range(1, 11)),
    *(f"AskVolume{level}" for level in range(1, 11)),
    *(f"BidVolume{level}" for level in range(1, 11)),
    *(f"aVOI{level}" for level in range(1, 6)),
    "secu_code",
    "minute",
    "date",
)

_UNIVERSE_EX2KAMT_SCHEMA = (
    "sec_code",
    "preclose",
    "close",
    "adjfactor",
    "secu_code",
    "date",
)

_FIXED_SOURCE_SCHEMAS = {
    "trades_tbar": _TRADE_CANCEL_TBAR_SCHEMA,
    "cancels_tbar": _TRADE_CANCEL_TBAR_SCHEMA,
    "quotes_tbar": _QUOTE_TBAR_SCHEMA,
    "daily_k": _DAILY_K_SCHEMA,
    "snapshot_tbar": _SNAPSHOT_TBAR_SCHEMA,
    "universe/ex2kamt": _UNIVERSE_EX2KAMT_SCHEMA,
}
