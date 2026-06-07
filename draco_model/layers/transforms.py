from __future__ import annotations

import logging
from dataclasses import replace

import polars as pl

from draco_model.core import Layer, Node
from draco_model.layers.aggregate import parse_minute_frequency
from draco_model.market.minute_calendar import AUCTION_MINUTES
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import (
    can_grid,
    EvalContext,
    FieldInfo,
    FrameInfo,
    left_join_identity,
    register_executor,
    register_info,
)


logger = logging.getLogger(__name__)


class Grid(Layer):
    """Align a raw or minute-grain frame to an explicit universe-by-minute grid."""

    op = "grid"

    def __init__(
        self,
        frequency: str | None = None,
        *,
        auction: str | None = None,
        name: str | None = None,
    ) -> None:
        if frequency is not None:
            _normalize_grid_frequency(frequency)
        if auction is not None and auction not in {"keep", "drop", "merge"}:
            raise ValueError("Grid auction must be 'keep', 'drop', or 'merge'.")
        super().__init__(name=name, frequency=frequency, auction=auction)


class FillNull(Layer):
    """Fill nulls in a single public field."""

    op = "fill_null"

    def __init__(self, value: int | float | str = "state", *, name: str | None = None) -> None:
        super().__init__(name=name, value=value)

    def __call__(self, frame: Node) -> Node:
        return Node(kind="frame", op=self.op, params=dict(self.params), inputs={"input": frame}, name=self.name)


@register_executor("grid")
def _grid(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    schema = context.infer_info(parent)
    info = _grid_info_from_parent(schema)
    frequency, auction = _grid_policy(node, schema)
    dates = context.trading_calendar.previous_sessions(context.eval_date, _grid_lookback(parent, schema))
    minutes = _grid_minutes(context, frequency, auction)
    logger.debug(
        "grid.start node_id=%s frequency=%s auction=%s dates=%s minutes=%d",
        node.id,
        frequency,
        auction,
        dates,
        len(minutes),
    )
    grid = context.intraday_grid(context.model.universe, dates, minutes)
    return (
        grid.join(context.evaluate(parent), on=list(KEY_COLUMNS), how="left")
        .select(list(info.columns))
        .sort(list(KEY_COLUMNS))
    )


@register_info("grid")
def _grid_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    return _grid_info_from_parent(parent_infos["input"])


def _grid_info_from_parent(parent: FrameInfo) -> FrameInfo:
    if not can_grid(parent):
        raise ValueError("Grid requires a raw or minute frame with date/secu_code/minute keys.")
    grid_info = FrameInfo.from_columns(KEY_COLUMNS, identity_keys=KEY_COLUMNS)
    identity_keys = left_join_identity(grid_info, parent)
    return FrameInfo.from_columns(parent.columns, identity_keys=identity_keys, fields=parent.fields)


@register_executor("fill_null")
def _fill_null(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    schema = context.infer_info(parent)
    output_info = _fill_null_info_from_parent(schema)
    value = node.params.get("value", "state")
    value_col = _single_value_column(schema)
    field_info = _field_for_column(schema, value_col)
    logger.debug(
        "fill_null.start node_id=%s mode=%s value_col=%s grain=%s keys=%s",
        node.id,
        value,
        value_col,
        schema.grain,
        schema.keys,
    )

    if _is_numeric_fill(value):
        logger.debug("fill_null.numeric node_id=%s value_col=%s value=%s", node.id, value_col, value)
        return context.evaluate(parent).with_columns(pl.col(value_col).fill_null(value).alias(value_col)).select(list(output_info.columns))
    if value == "ffill":
        logger.debug("fill_null.ffill node_id=%s value_col=%s", node.id, value_col)
        return (
            context.evaluate(parent)
            .sort(list(schema.keys))
            .with_columns(pl.col(value_col).forward_fill().over(list(DAILY_KEY_COLUMNS)).alias(value_col))
            .select(list(output_info.columns))
        )
    if value != "state":
        raise ValueError("FillNull supports numeric literals, 'ffill', or 'state'.")

    if field_info.operator == "preclose":
        close_state = _close_state_from_node(node, field_info, context)
        logger.debug("fill_null.state_preclose node_id=%s output_column=%s", node.id, field_info.column)
        return _preclose_from_state(close_state, field_info.column)
    logger.debug("fill_null.state node_id=%s value_col=%s", node.id, value_col)
    parent_frame = context.evaluate(parent)
    close_state = _close_state_from_node(node, field_info, context, align_keys=parent_frame.select(list(KEY_COLUMNS)))
    return (
        parent_frame.join(close_state.select([*KEY_COLUMNS, "__close_state"]), on=list(KEY_COLUMNS), how="left")
        .with_columns(pl.col(value_col).fill_null(pl.col("__close_state")).alias(value_col))
        .drop("__close_state")
        .select(list(output_info.columns))
    )


@register_info("fill_null")
def _fill_null_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    return _fill_null_info_from_parent(parent_infos["input"])


def _fill_null_info_from_parent(parent: FrameInfo) -> FrameInfo:
    value_col = _single_value_column(parent)
    info = _field_for_column(parent, value_col)
    fields = dict(parent.fields)
    fields[value_col] = replace(info, name=value_col, column=value_col, components=(), component_agg=False)
    return FrameInfo.from_columns(
        parent.columns,
        identity_keys=parent.keys,
        fields=fields,
    )


def _close_state_from_node(
    node: Node,
    info: FieldInfo,
    context: EvalContext,
    *,
    align_keys: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    if info.source is None:
        raise ValueError("FillNull('state') requires source lineage.")
    close_node = _build_close_state_subtree(info.source, info.lookback_days, info.grain_path)
    logger.debug(
        "close_state.build_from_lineage node_id=%s source=%s lookback_days=%d grain_steps=%d",
        node.id,
        info.source,
        info.lookback_days,
        len(info.grain_path),
    )
    close = context.evaluate(close_node)
    if align_keys is not None:
        state_keys = pl.concat(
            [
                close.select(list(KEY_COLUMNS)),
                align_keys.select(list(KEY_COLUMNS)),
            ]
        ).unique()
        close = state_keys.join(close, on=list(KEY_COLUMNS), how="left")
    dates = context.trading_calendar.previous_sessions(context.eval_date, info.lookback_days)
    logger.debug("close_state.daily_preclose dates=%s", dates)
    daily_preclose = _daily_preclose_frame(context, dates)
    return (
        close.join(daily_preclose, on=list(DAILY_KEY_COLUMNS), how="left")
        .sort(list(KEY_COLUMNS))
        .with_columns(
            pl.col("close")
            .forward_fill()
            .over(list(DAILY_KEY_COLUMNS))
            .fill_null(pl.col("__daily_preclose"))
            .alias("__close_state")
        )
        .select([*KEY_COLUMNS, "__close_state", "__daily_preclose"])
    )


def _preclose_from_state(state: pl.LazyFrame, output_column: str) -> pl.LazyFrame:
    return (
        state.sort(list(KEY_COLUMNS))
        .with_columns(
            pl.col("__close_state")
            .shift(1)
            .over(list(DAILY_KEY_COLUMNS))
            .fill_null(pl.col("__daily_preclose"))
            .alias(output_column)
        )
        .select([*KEY_COLUMNS, output_column])
    )


def _daily_preclose_frame(context: EvalContext, dates: list[str]) -> pl.LazyFrame:
    frame = context.sources.scan("daily_k", dates)
    columns = frame.collect_schema().names()
    missing = [column for column in [*DAILY_KEY_COLUMNS, "preclose"] if column not in columns]
    if missing:
        raise ValueError(f"daily_k preclose source is missing columns: {missing}.")
    return frame.select([*DAILY_KEY_COLUMNS, pl.col("preclose").alias("__daily_preclose")])


def _grid_policy(node: Node, schema: FrameInfo) -> tuple[str, str]:
    frequency = node.params.get("frequency")
    auction = node.params.get("auction")
    inferred_frequency, inferred_auction = _infer_grid_policy_from_schema(schema)
    resolved_auction = str(auction) if auction is not None else inferred_auction
    if resolved_auction not in {"keep", "drop", "merge"}:
        raise ValueError("Grid auction must be 'keep', 'drop', or 'merge'.")
    return (
        _normalize_grid_frequency(str(frequency)) if frequency is not None else inferred_frequency,
        resolved_auction,
    )


def _infer_grid_policy_from_schema(schema: FrameInfo) -> tuple[str, str]:
    policies = {
        policy
        for info in schema.fields.values()
        if (policy := _grid_policy_from_grain_path(info.grain_path)) is not None
    }
    if not policies:
        return "1m", "keep"
    if len(policies) != 1:
        raise ValueError("Grid could not infer one minute frequency; pass frequency=... explicitly.")
    return next(iter(policies))


def _grid_policy_from_grain_path(grain_path: tuple[tuple[str, str], ...]) -> tuple[str, str] | None:
    frequency: str | None = None
    auction_removed = False
    for step_frequency, step_auction in grain_path:
        try:
            frequency = _normalize_grid_frequency(step_frequency)
        except ValueError:
            continue
        if step_auction in {"drop", "merge"}:
            auction_removed = True
    if frequency is None:
        return None
    auction = "drop" if auction_removed else "keep"
    return frequency, auction


def _normalize_grid_frequency(frequency: str) -> str:
    text = frequency.strip().lower()
    parse_minute_frequency(text)
    return text


def _grid_minutes(context: EvalContext, frequency: str, auction: str) -> tuple[int, ...]:
    interval = parse_minute_frequency(frequency)
    if interval == 1:
        minutes = tuple(context.minute_calendar.minbars())
    else:
        buckets = (
            context.minute_calendar.bucket_map(interval)
            .select("__bucket_minute")
            .unique()
            .sort("__bucket_minute")
            .collect()["__bucket_minute"]
            .to_list()
        )
        minutes = (*AUCTION_MINUTES[:1], *(int(minute) for minute in buckets), *AUCTION_MINUTES[1:])
    if auction in {"drop", "merge"}:
        minutes = tuple(minute for minute in minutes if minute not in AUCTION_MINUTES)
    return tuple(minutes)


def _grid_lookback(parent: Node, schema: FrameInfo) -> int:
    if parent.op == "source":
        return int(parent.params.get("lookback_days", 1))
    values = [info.lookback_days for info in schema.fields.values()]
    return max(values) if values else 1


def _build_close_state_subtree(source_name: str, lookback_days: int, grain_path: tuple[tuple[str, str], ...]) -> Node:
    from draco_model.layers.aggregate import Aggregate
    from draco_model.layers.filters import Flag, Where
    from draco_model.layers.operators import Col
    from draco_model.layers.source import Source

    node = Col("price").alias("close")(Where(Flag("is_last"))(Source(source_name, lookback_days=lookback_days)))
    for frequency, auction in grain_path:
        node = Aggregate(
            frequency,
            "last",
            value_col="close",
            alias="close",
            auction=auction,
            apply_to="field",
        )(node)
    return node


def _single_value_column(schema: FrameInfo) -> str:
    values = schema.value_columns()
    if len(values) != 1:
        raise ValueError(f"FillNull requires exactly one public value column, got {values}.")
    return values[0]


def _field_for_column(schema: FrameInfo, column: str) -> FieldInfo:
    if column in schema.fields:
        return schema.fields[column]
    return FieldInfo(column, column)


def _is_numeric_fill(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
