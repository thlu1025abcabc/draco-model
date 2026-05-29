from __future__ import annotations

import polars as pl

from draco_model.core import Layer, Node
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FrameSchema, register_executor, register_schema


class Concat(Layer):
    """Horizontally align multiple frames by their key columns."""

    op = "concat"


@register_executor("concat")
def _concat(node: Node, context: EvalContext) -> pl.LazyFrame:
    intraday_frames = []
    daily_frames = []
    for input_name, parent in node.inputs.items():
        frame = context.evaluate(parent)
        columns = list(context.infer_schema(parent).columns)
        keys = _key_columns(columns)
        values = _value_columns(columns)
        renames = _concat_renames(input_name, values)
        renamed = frame.rename(renames).select([*keys, *renames.values()])
        if keys == list(KEY_COLUMNS):
            intraday_frames.append(renamed)
        elif keys == list(DAILY_KEY_COLUMNS):
            daily_frames.append(renamed)

    if intraday_frames:
        out = pl.concat(intraday_frames, how="align")
        for daily_frame in daily_frames:
            out = out.join(daily_frame, on=list(DAILY_KEY_COLUMNS), how="left")
        return out
    return pl.concat(daily_frames, how="align")


@register_schema("concat")
def _concat_schema(node: Node, parent_schemas: dict[str, FrameSchema], context: EvalContext) -> FrameSchema:
    intraday_values: list[str] = []
    daily_values: list[str] = []
    for input_name, schema in parent_schemas.items():
        columns = list(schema.columns)
        keys = _key_columns(columns)
        values = list(_concat_renames(input_name, _value_columns(columns)).values())
        if keys == list(KEY_COLUMNS):
            intraday_values.extend(values)
        elif keys == list(DAILY_KEY_COLUMNS):
            daily_values.extend(values)
    if intraday_values:
        return FrameSchema((*KEY_COLUMNS, *intraday_values, *daily_values))
    return FrameSchema((*DAILY_KEY_COLUMNS, *daily_values))


def _value_columns(columns: list[str]) -> list[str]:
    keys = set(_key_columns(columns))
    return [column for column in columns if column not in keys and not column.startswith("__")]


def _key_columns(columns: list[str]) -> list[str]:
    if list(columns[:3]) == list(KEY_COLUMNS) or all(column in columns for column in KEY_COLUMNS):
        return list(KEY_COLUMNS)
    if list(columns[:2]) == list(DAILY_KEY_COLUMNS) or all(column in columns for column in DAILY_KEY_COLUMNS):
        return list(DAILY_KEY_COLUMNS)
    raise ValueError(f"Frame does not contain recognized key columns: {columns}.")


def _concat_renames(input_name: str, values: list[str]) -> dict[str, str]:
    if len(values) == 1:
        target = input_name
        if target in KEY_COLUMNS or target in DAILY_KEY_COLUMNS:
            raise ValueError(f"Concat input name {input_name!r} conflicts with key columns.")
        return {values[0]: target}
    return {column: f"{input_name}__{column}" for column in values}
