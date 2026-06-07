from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace

import polars as pl

from draco_model.core import Layer, Node
from draco_model.layers.names import validate_public_alias
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import EvalContext, FieldInfo, FrameInfo, register_executor, register_info


logger = logging.getLogger(__name__)


class Join(Layer):
    """Horizontally align multiple frames by key columns."""

    op = "join"

    def __call__(self, inputs: Node | Mapping[str, Node]) -> Node:
        if isinstance(inputs, Mapping):
            for input_name in inputs:
                validate_public_alias(input_name, subject="Join input name")
        return super().__call__(inputs)


class Project(Layer):
    """Keep only key columns and public fields."""

    op = "project"


@register_executor("join")
def _join(node: Node, context: EvalContext) -> pl.LazyFrame:
    intraday_frames = []
    daily_frames = []
    parent_schemas = {
        input_name: context.infer_info(parent)
        for input_name, parent in node.inputs.items()
    }
    info = _join_info_from_inputs(parent_schemas)
    logger.debug(
        "join.start node_id=%s inputs=%s output_grain=%s keys=%s columns=%d",
        node.id,
        list(node.inputs),
        info.grain,
        info.keys,
        len(info.columns),
    )
    for input_name, parent in node.inputs.items():
        schema = parent_schemas[input_name]
        frame = context.evaluate(parent)
        renames = _renames(input_name, schema)
        selected = frame.rename(renames).select([*schema.keys, *renames.values()])
        if schema.keys == KEY_COLUMNS:
            intraday_frames.append(selected)
        elif schema.keys == DAILY_KEY_COLUMNS:
            daily_frames.append(selected)
        else:
            logger.error("join.unrecognized_keys input=%s keys=%s", input_name, schema.keys)
            raise ValueError(f"Join input {input_name!r} does not have recognized keys.")
    logger.debug(
        "join.frames node_id=%s intraday=%d daily=%d",
        node.id,
        len(intraday_frames),
        len(daily_frames),
    )
    if intraday_frames:
        out = pl.concat(intraday_frames, how="align")
        for daily in daily_frames:
            out = out.join(daily, on=list(DAILY_KEY_COLUMNS), how="left")
        return out.select(list(info.columns))
    return pl.concat(daily_frames, how="align").select(list(info.columns))


@register_info("join")
def _join_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    return _join_info_from_inputs(parent_infos)


def _join_info_from_inputs(parent_schemas: dict[str, FrameInfo]) -> FrameInfo:
    has_intraday = any(schema.keys == KEY_COLUMNS for schema in parent_schemas.values())
    keys = KEY_COLUMNS if has_intraday else DAILY_KEY_COLUMNS
    columns = list(keys)
    fields: dict[str, FieldInfo] = {}
    for input_name, schema in parent_schemas.items():
        renames = _renames(input_name, schema)
        columns.extend(renames.values())
        for source, target in renames.items():
            info = schema.fields.get(source, FieldInfo(source, source))
            component_renames = tuple(renames.get(component, component) for component in info.components)
            fields[target] = replace(
                info,
                name=target if info.is_public else info.name,
                column=target,
                components=component_renames,
                is_payload=target.startswith("__") or info.is_payload,
                is_public=info.is_public and not target.startswith("__"),
                identity_order=None,
            )
    return FrameInfo.from_columns(
        tuple(dict.fromkeys(columns)),
        identity_keys=keys,
        fields=fields,
    )


@register_executor("project")
def _project(node: Node, context: EvalContext) -> pl.LazyFrame:
    parent = node.inputs["input"]
    info = context.infer_info(parent)
    output_info = _project_info_from_parent(info)
    return context.evaluate(parent).select(list(output_info.columns))


@register_info("project")
def _project_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    return _project_info_from_parent(parent_infos["input"])


def _project_info_from_parent(parent: FrameInfo) -> FrameInfo:
    columns = (*parent.keys, *parent.value_columns())
    fields = {
        column: replace(parent.fields[column], components=(), component_agg=False)
        for column in columns
        if column in parent.fields
    }
    return FrameInfo.from_columns(
        columns,
        identity_keys=parent.keys,
        fields=fields,
    )


def _renames(input_name: str, schema: FrameInfo) -> dict[str, str]:
    validate_public_alias(input_name, subject="Join input name")
    values = [column for column in schema.columns if column not in schema.keys]
    public = set(schema.value_columns())
    out: dict[str, str] = {}
    for column in values:
        if column in public:
            target = input_name if len(public) == 1 else f"{input_name}__{column}"
            if target in KEY_COLUMNS or target in DAILY_KEY_COLUMNS:
                raise ValueError(f"Join input name {input_name!r} conflicts with key columns.")
            out[column] = target
        else:
            clean = column.lstrip("_")
            out[column] = f"__{input_name}_{clean}"
    return out
