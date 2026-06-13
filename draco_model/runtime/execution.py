from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, MutableMapping

import polars as pl

from draco_model.core import Model, Node
from draco_model.data.source import SourceCatalog
from draco_model.data.trading_calendar import TradingCalendar
from draco_model.data.universe import UniverseCatalog
from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS


Executor = Callable[[Node, "EvalContext"], pl.LazyFrame]
InfoBuilder = Callable[[Node, dict[str, "FrameInfo"], "EvalContext"], "FrameInfo"]

_EXECUTORS: dict[str, Executor] = {}
_INFO_BUILDERS: dict[str, InfoBuilder] = {}


@dataclass(frozen=True)
class FieldInfo:
    """Metadata for one physical column inside a frame."""

    name: str
    column: str
    operator: str = "identity"
    components: tuple[str, ...] = ()
    source: str | None = None
    lookback_days: int = 1
    component_agg: bool = False
    grain_path: tuple[tuple[str, str], ...] = ()
    is_public: bool = True
    is_payload: bool = False
    identity_order: int | None = None

    @property
    def is_identity(self) -> bool:
        """Return whether this column participates in the frame row identity."""
        return self.identity_order is not None


@dataclass(frozen=True)
class FrameInfo:
    """Frame contract derived from per-column field metadata."""

    fields: dict[str, FieldInfo] = field(default_factory=dict)

    @classmethod
    def from_columns(
        cls,
        columns: tuple[str, ...],
        *,
        identity_keys: tuple[str, ...] = (),
        fields: dict[str, FieldInfo] | None = None,
        source: str | None = None,
        lookback_days: int = 1,
        grain_path: tuple[tuple[str, str], ...] = (),
    ) -> "FrameInfo":
        provided = fields or {}
        identity_order = {column: idx for idx, column in enumerate(identity_keys)}
        out: dict[str, FieldInfo] = {}
        for column in columns:
            info = provided.get(column)
            if info is None:
                payload = column.startswith("__")
                info = FieldInfo(
                    name=column,
                    column=column,
                    source=source,
                    lookback_days=lookback_days,
                    grain_path=grain_path,
                    is_public=not payload,
                    is_payload=payload,
                )
            payload = info.is_payload or column.startswith("__")
            out[column] = replace(
                info,
                column=column,
                identity_order=identity_order.get(column, info.identity_order),
                is_payload=payload,
                is_public=info.is_public and not payload,
            )
        return cls(out)

    @property
    def columns(self) -> tuple[str, ...]:
        """Return physical output columns in frame order."""
        return tuple(self.fields)

    @property
    def identity_keys(self) -> tuple[str, ...]:
        """Return row identity columns derived from field metadata."""
        identity_fields = [
            info
            for info in self.fields.values()
            if info.identity_order is not None
        ]
        return tuple(info.column for info in sorted(identity_fields, key=lambda info: int(info.identity_order)))

    @property
    def keys(self) -> tuple[str, ...]:
        """Compatibility alias for row identity columns."""
        return self.identity_keys

    @property
    def grain(self) -> str:
        """Return a coarse debug label inferred from row identity and field metadata."""
        return infer_grain_label(self)

    def value_columns(self) -> list[str]:
        """Return public value columns, excluding identity and payload columns."""
        return [
            info.column
            for info in self.fields.values()
            if info.is_public and not info.is_identity and not info.is_payload
        ]

    def payload_columns(self) -> list[str]:
        """Return internal payload columns."""
        return [info.column for info in self.fields.values() if info.is_payload]

    def field_for(self, column: str) -> FieldInfo:
        """Return field metadata for a frame column; every frame column has an entry."""
        return self.fields[column]

    def single_value_column(self, *, subject: str = "Operator") -> str:
        """Return the single public value column or raise a clear error."""
        values = self.value_columns()
        if len(values) != 1:
            raise ValueError(f"{subject} requires exactly one public value column, got {values}.")
        return values[0]

    def merged_source_context(self) -> tuple[str | None, int, tuple[tuple[str, str], ...]]:
        """Return the (source, lookback_days, grain_path) a derived field inherits."""
        infos = self._source_context_fields()
        lookback = max((info.lookback_days for info in infos), default=1)
        if not infos or any(info.source is None for info in infos):
            return None, lookback, ()
        sources = {info.source for info in infos}
        grain_paths = {info.grain_path for info in infos}
        if len(sources) != 1 or len(grain_paths) != 1:
            return None, lookback, ()
        return next(iter(sources)), lookback, next(iter(grain_paths))

    def _source_context_fields(self) -> tuple[FieldInfo, ...]:
        values = tuple(
            info
            for info in self.fields.values()
            if info.is_public and not info.is_identity and not info.is_payload
        )
        if values:
            return values
        return tuple(info for info in self.fields.values() if not info.is_payload)


def merge_source_contexts(schemas: list[FrameInfo]) -> tuple[str | None, int, tuple[tuple[str, str], ...]]:
    """Merge per-frame source contexts for a derived field with multiple inputs."""
    contexts = [schema.merged_source_context() for schema in schemas]
    lookback = max((lookback for _, lookback, _ in contexts), default=1)
    if any(source is None for source, _, _ in contexts):
        return None, lookback, ()
    sources = {source for source, _, _ in contexts}
    grain_paths = {grain_path for _, _, grain_path in contexts}
    if len(sources) == 1 and len(grain_paths) == 1:
        return next(iter(sources)), lookback, next(iter(grain_paths))
    return None, lookback, ()


def infer_grain_label(info: FrameInfo) -> str:
    """Infer a coarse frame label for logging and legacy checks."""
    keys = info.identity_keys
    if keys == DAILY_KEY_COLUMNS:
        return "daily"
    if set(KEY_COLUMNS).issubset(set(keys)):
        if len(keys) > len(KEY_COLUMNS):
            return "raw"
        value_fields = [
            field
            for field in info.fields.values()
            if field.is_public and not field.is_identity and not field.is_payload
        ]
        if value_fields and all(not field.grain_path for field in value_fields):
            return "raw"
        return "minute"
    return "unknown"


def can_collect(info: FrameInfo) -> bool:
    """Return whether a frame can be formatted as a daily factor output."""
    return info.identity_keys == DAILY_KEY_COLUMNS and "value" in info.value_columns()


def ordered_union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Return first-seen ordered union of column groups."""
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for column in group:
            if column not in seen:
                seen.add(column)
                out.append(column)
    return tuple(out)


def resolve_identity_join_on(
    left_identity: tuple[str, ...],
    right_identity: tuple[str, ...],
    on: tuple[str, ...] | None,
    *,
    how: str,
    default: str = "intersection",
) -> tuple[str, ...]:
    """Return a validated join key for two identity contracts."""
    if default not in {"intersection", "left"}:
        raise ValueError("Join default must be 'intersection' or 'left'.")
    if on is None:
        if default == "left":
            join_on = left_identity
        else:
            right_set = set(right_identity)
            join_on = tuple(column for column in left_identity if column in right_set)
    else:
        join_on = tuple(on)
    if not join_on:
        raise ValueError(f"Join how={how!r} requires at least one shared identity key or explicit on.")

    missing_left = [column for column in join_on if column not in left_identity]
    missing_right = [column for column in join_on if column not in right_identity]
    if missing_left or missing_right:
        raise ValueError(
            f"Join how={how!r} on columns must be identity columns in both inputs; "
            f"missing_left={missing_left}, missing_right={missing_right}."
        )

    join_on_set = set(join_on)
    right_set = set(right_identity)
    missing_shared = [column for column in left_identity if column in right_set and column not in join_on_set]
    if missing_shared:
        raise ValueError(
            f"Join how={how!r} on columns must include shared identity columns; "
            f"missing_shared={missing_shared}."
        )
    return join_on


def left_join_identity(left: FrameInfo, *rights: FrameInfo, on: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Return output identity for a left join anchored by the left identity."""
    join_on = left.identity_keys if on is None else tuple(on)
    identity = left.identity_keys
    for right in rights:
        resolve_identity_join_on(identity, right.identity_keys, join_on, how="left")
        identity = ordered_union(identity, right.identity_keys)
    return identity


@dataclass(frozen=True)
class TraceStep:
    """Materialized output for one traced frame node."""

    index: int
    resolved_name: str
    node: Node
    frame: pl.DataFrame


def register_executor(op: str) -> Callable[[Executor], Executor]:
    """Register the function that evaluates nodes with a given op name."""

    def decorator(executor: Executor) -> Executor:
        _EXECUTORS[op] = executor
        return executor

    return decorator


def register_info(op: str) -> Callable[[InfoBuilder], InfoBuilder]:
    """Register the frame-info builder for nodes with a given op name."""

    def decorator(builder: InfoBuilder) -> InfoBuilder:
        _INFO_BUILDERS[op] = builder
        return builder

    return decorator


def get_executor(op: str) -> Executor:
    """Return the registered executor for an op or raise a clear error."""
    try:
        return _EXECUTORS[op]
    except KeyError:
        raise ValueError(f"Unsupported node op {op!r}.") from None


def get_info_builder(op: str) -> InfoBuilder | None:
    """Return the registered frame-info builder for an op, if one exists."""
    return _INFO_BUILDERS.get(op)


@dataclass(frozen=True)
class EvalContext:
    """Runtime services shared by node executors during one evaluation."""

    model: Model
    eval_date: str
    sources: SourceCatalog
    universes: UniverseCatalog
    minute_calendar: MinuteCalendar
    trading_calendar: TradingCalendar
    evaluate: Callable[[Node], pl.LazyFrame]
    infer_info: Callable[[Node], FrameInfo]
    grid_cache: MutableMapping[tuple[str, str, tuple[int, ...]], pl.DataFrame]

    def intraday_grid(self, universe: str, dates: list[str], minutes: tuple[int, ...] | None = None) -> pl.LazyFrame:
        """Build or reuse the universe-by-minute grid for intraday inputs."""
        grid_minutes = minutes or tuple(self.minute_calendar.minbars())
        key = (universe, self.eval_date, grid_minutes)
        if key not in self.grid_cache:
            universe_frame = self.universes.scan(universe, self.eval_date).select("secu_code")
            minute_frame = pl.DataFrame({"minute": list(grid_minutes)}).lazy()
            self.grid_cache[key] = universe_frame.join(minute_frame, how="cross").collect()
        frames = [
            self.grid_cache[key]
            .lazy()
            .with_columns(pl.lit(date).alias("date"))
            .select(list(KEY_COLUMNS))
            for date in dates
        ]
        return pl.concat(frames, how="vertical")


def format_factor_output(frame: pl.LazyFrame, factor_name: str, eval_date: str) -> pl.LazyFrame:
    """Normalize a daily frame into the public factor output schema."""
    columns = frame.collect_schema().names()
    if "value" not in columns:
        raise ValueError("Model output must contain a value column.")
    return (
        frame.filter(pl.col("date") == eval_date)
        .select([*DAILY_KEY_COLUMNS, "value"])
        .with_columns(pl.lit(factor_name).alias("factor_name"))
        .select(["date", "secu_code", "factor_name", "value"])
    )
