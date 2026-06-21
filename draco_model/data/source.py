from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS


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

    def identity_keys(self, source: str, dates: list[str]) -> tuple[str, ...]:
        """Return normalized row identity columns for a source."""
        if not dates:
            raise ValueError("Source identity keys require at least one date.")
        fixed = _FIXED_SOURCE_IDENTITY_KEYS.get(source)
        if fixed is not None:
            return fixed
        columns = self.schema(source, dates)
        raise ValueError(
            f"Source {source!r} has no registered identity keys. "
            f"Register the source before using it in a DAG. Normalized columns: {list(columns)}."
        )

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
        frame = _standardize_columns(frame, date, source)
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


def _standardize_columns(frame: pl.LazyFrame, date: str, source: str | None = None) -> pl.LazyFrame:
    columns = frame.collect_schema().names()
    renames = {}
    for vendor, target in {
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
        "DealTime": "deal_time",
        "BuyID": "buy_id",
        "SellID": "sell_id",
        "DealID": "deal_id",
        "OrderTime": "order_time",
        "OrderID": "order_id",
        "OrderType": "order_type",
        "TickTime": "tick_time",
    }.items():
        if vendor in columns and target not in columns:
            renames[vendor] = target
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
    if source in _LEVEL2_SOURCE_CASTS:
        frame = frame.with_columns(
            [
                pl.col(column).cast(dtype)
                for column, dtype in _LEVEL2_SOURCE_CASTS[source].items()
                if column in columns
            ]
        )
        if "secu_code" in columns:
            frame = frame.with_columns(pl.col("secu_code").replace(_TRANSFER_CODES))
            frame = frame.filter(pl.col("secu_code") <= _MAX_SECU_CODE)
    return frame


_STEPTRADES_SCHEMA = (
    "date",
    "secu_code",
    "deal_time",
    "buy_id",
    "sell_id",
    "deal_id",
    "price",
    "volume",
    "side",
)

_STEPORDERS_SCHEMA = (
    "date",
    "secu_code",
    "order_time",
    "order_id",
    "order_type",
    "price",
    "volume",
)

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

_ORDERBOOK_SCHEMA = (
    "secu_code",
    "date",
    "tick_time",
    "price",
    *(f"AskPrice{level}" for level in range(1, 11)),
    *(f"AskVolume{level}" for level in range(1, 11)),
    *(f"BidPrice{level}" for level in range(1, 11)),
    *(f"BidVolume{level}" for level in range(1, 11)),
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
    "steptrades": _STEPTRADES_SCHEMA,
    "steporders": _STEPORDERS_SCHEMA,
    "orderbook": _ORDERBOOK_SCHEMA,
    "trades_tbar": _TRADE_CANCEL_TBAR_SCHEMA,
    "cancels_tbar": _TRADE_CANCEL_TBAR_SCHEMA,
    "quotes_tbar": _QUOTE_TBAR_SCHEMA,
    "daily_k": _DAILY_K_SCHEMA,
    "snapshot_tbar": _SNAPSHOT_TBAR_SCHEMA,
    "universe/ex2kamt": _UNIVERSE_EX2KAMT_SCHEMA,
}

_TICK_TBAR_IDENTITY_KEYS = (*KEY_COLUMNS, "price", "side")

_FIXED_SOURCE_IDENTITY_KEYS = {
    "steptrades": ("date", "secu_code", "deal_id"),
    "steporders": ("date", "secu_code", "order_time", "order_id", "order_type"),
    "orderbook": ("date", "secu_code", "tick_time"),
    "trades_tbar": _TICK_TBAR_IDENTITY_KEYS,
    "cancels_tbar": _TICK_TBAR_IDENTITY_KEYS,
    "quotes_tbar": _TICK_TBAR_IDENTITY_KEYS,
    "daily_k": DAILY_KEY_COLUMNS,
    "snapshot_tbar": KEY_COLUMNS,
    "universe/ex2kamt": DAILY_KEY_COLUMNS,
}

_TRANSFER_CODES = {22: 1872, 600849: 601607, 43: 1914, 601313: 601360}
_MAX_SECU_CODE = 700000

_LEVEL2_SOURCE_CASTS = {
    "steptrades": {
        "deal_time": pl.Int64,
        "buy_id": pl.Int64,
        "sell_id": pl.Int64,
        "deal_id": pl.Int64,
        "price": pl.Int64,
        "volume": pl.Float64,
        "side": pl.Int64,
    },
    "steporders": {
        "order_time": pl.Int64,
        "order_id": pl.Int64,
        "order_type": pl.Int64,
        "price": pl.Int64,
        "volume": pl.Float64,
    },
    "orderbook": {
        "tick_time": pl.Int64,
        "price": pl.Int64,
    },
}
