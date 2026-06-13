from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace

import polars as pl

from draco_model.core import Layer, Node
from draco_model.layers.names import validate_public_alias
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.runtime.execution import (
    EvalContext,
    FieldInfo,
    FrameInfo,
    left_join_identity,
    ordered_union as _ordered_union,
    register_executor,
    register_info,
    resolve_identity_join_on,
)


logger = logging.getLogger(__name__)
IDENTITY_INTERSECTION = "identity_intersection"


class Join(Layer):
    """Horizontally align multiple frames by key columns."""

    op = "join"

    def __init__(
        self,
        *,
        how: str = "full",
        on: str | tuple[str, ...] | list[str] | None = None,
        name: str | None = None,
    ) -> None:
        normalized_how = how.strip().lower()
        if normalized_how not in {"full", "left"}:
            raise ValueError("Join how must be 'full' or 'left'.")
        normalized_on = _normalize_join_on(on)
        super().__init__(
            name=name,
            how=None if normalized_how == "full" else normalized_how,
            on=normalized_on,
        )

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
    parent_schemas = {
        input_name: context.infer_info(parent)
        for input_name, parent in node.inputs.items()
    }
    how = str(node.params.get("how", "full"))
    on = _join_on_param(node)
    info = _join_info_from_inputs(parent_schemas, how=how, on=on)
    logger.debug(
        "join.start node_id=%s how=%s on=%s inputs=%s output_grain=%s keys=%s columns=%d",
        node.id,
        how,
        on,
        list(node.inputs),
        info.grain,
        info.keys,
        len(info.columns),
    )
    if how == "left":
        return _left_join(node, context, parent_schemas, info, on)
    return _full_join(node, context, parent_schemas, info, on)


def _full_join(
    node: Node,
    context: EvalContext,
    parent_schemas: dict[str, FrameInfo],
    info: FrameInfo,
    on: tuple[str, ...] | str | None,
) -> pl.LazyFrame:
    items = list(node.inputs.items())
    base_name, base_node = items[0]
    base_schema = parent_schemas[base_name]
    current_identity = base_schema.keys
    current_columns = _join_input_columns(base_name, base_schema)
    out = _select_join_input(context.evaluate(base_node), base_name, base_schema)
    for input_name, parent in items[1:]:
        schema = parent_schemas[input_name]
        join_on = _join_on_for_step(current_identity, schema.keys, on, how="full")
        right_columns = _join_input_columns(input_name, schema)
        _check_join_column_overlap(current_columns, right_columns, join_on, how="full")
        right = _select_join_input(context.evaluate(parent), input_name, schema)
        out = out.join(right, on=list(join_on), how="full", coalesce=True)
        current_identity = _ordered_union(current_identity, schema.keys)
        current_columns = _ordered_union(current_columns, right_columns)
    return out.select(list(info.columns)).sort(list(info.keys))


def _left_join(
    node: Node,
    context: EvalContext,
    parent_schemas: dict[str, FrameInfo],
    info: FrameInfo,
    on: tuple[str, ...] | str | None,
) -> pl.LazyFrame:
    items = list(node.inputs.items())
    base_name, base_node = items[0]
    base_schema = parent_schemas[base_name]
    current_identity = base_schema.keys
    current_columns = _join_input_columns(base_name, base_schema)
    left_on = base_schema.keys if on is None else on
    out = _select_join_input(context.evaluate(base_node), base_name, base_schema)
    for input_name, parent in items[1:]:
        schema = parent_schemas[input_name]
        join_on = _join_on_for_step(current_identity, schema.keys, left_on, how="left")
        right_columns = _join_input_columns(input_name, schema)
        _check_join_column_overlap(current_columns, right_columns, join_on, how="left")
        right = _select_join_input(context.evaluate(parent), input_name, schema)
        out = out.join(right, on=list(join_on), how="left")
        current_identity = _ordered_union(current_identity, schema.keys)
        current_columns = _ordered_union(current_columns, right_columns)
    return out.select(list(info.columns)).sort(list(info.keys))


@register_info("join")
def _join_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    return _join_info_from_inputs(parent_infos, how=str(node.params.get("how", "full")), on=_join_on_param(node))


def _join_info_from_inputs(
    parent_schemas: dict[str, FrameInfo],
    *,
    how: str = "full",
    on: tuple[str, ...] | str | None = None,
) -> FrameInfo:
    _validate_mixed_daily_join(parent_schemas, how)
    if how == "left":
        return _left_join_info_from_inputs(parent_schemas, on)
    return _full_join_info_from_inputs(parent_schemas, on)


def _full_join_info_from_inputs(parent_schemas: dict[str, FrameInfo], on: tuple[str, ...] | str | None) -> FrameInfo:
    schemas = list(parent_schemas.values())
    identity_keys = schemas[0].keys
    for schema in schemas[1:]:
        _join_on_for_step(identity_keys, schema.keys, on, how="full")
        identity_keys = _ordered_union(identity_keys, schema.keys)
    columns, fields = _join_output_columns_and_fields(parent_schemas, identity_keys)
    return FrameInfo.from_columns(
        tuple(dict.fromkeys(columns)),
        identity_keys=identity_keys,
        fields=fields,
    )


def _left_join_info_from_inputs(parent_schemas: dict[str, FrameInfo], on: tuple[str, ...] | str | None) -> FrameInfo:
    schemas = list(parent_schemas.values())
    left = schemas[0]
    if on == IDENTITY_INTERSECTION:
        identity_keys = left.keys
        for schema in schemas[1:]:
            _join_on_for_step(identity_keys, schema.keys, on, how="left")
            identity_keys = _ordered_union(identity_keys, schema.keys)
    else:
        identity_keys = left_join_identity(left, *schemas[1:], on=on)
    columns, fields = _join_output_columns_and_fields(parent_schemas, identity_keys)
    return FrameInfo.from_columns(
        tuple(dict.fromkeys(columns)),
        identity_keys=identity_keys,
        fields=fields,
    )


def _join_output_columns_and_fields(
    parent_schemas: dict[str, FrameInfo],
    identity_keys: tuple[str, ...],
) -> tuple[list[str], dict[str, FieldInfo]]:
    columns = list(identity_keys)
    seen = set(columns)
    fields: dict[str, FieldInfo] = {}
    for input_name, schema in parent_schemas.items():
        renames = _renames(input_name, schema)
        for source, target in renames.items():
            if target in seen:
                raise ValueError(f"Join output column conflict after renaming: {target!r}.")
            seen.add(target)
            columns.append(target)
            fields[target] = _renamed_field(schema, renames, source, target)
    return columns, fields


def _validate_mixed_daily_join(parent_schemas: dict[str, FrameInfo], how: str) -> None:
    identities = [schema.keys for schema in parent_schemas.values()]
    has_daily = any(identity == DAILY_KEY_COLUMNS for identity in identities)
    all_daily = all(identity == DAILY_KEY_COLUMNS for identity in identities)
    if has_daily and not all_daily and how != "left":
        raise ValueError(
            "Join how='full' cannot mix daily identity frames with non-daily identity frames; "
            "use Join(how='left', on=('date', 'secu_code')) to choose an anchor explicitly."
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


def _select_join_input(frame: pl.LazyFrame, input_name: str, schema: FrameInfo) -> pl.LazyFrame:
    renames = _renames(input_name, schema)
    return frame.rename(renames).select([*schema.keys, *renames.values()])


def _join_input_columns(input_name: str, schema: FrameInfo) -> tuple[str, ...]:
    renames = _renames(input_name, schema)
    return (*schema.keys, *renames.values())


def _check_join_column_overlap(
    left_columns: tuple[str, ...],
    right_columns: tuple[str, ...],
    join_on: tuple[str, ...],
    *,
    how: str,
) -> None:
    join_on_set = set(join_on)
    left_set = set(left_columns)
    overlap = [column for column in right_columns if column in left_set and column not in join_on_set]
    if overlap:
        raise ValueError(
            f"Join how={how!r} would produce overlapping non-join columns after renaming: {overlap}."
        )


def _join_on_for_step(
    left_identity: tuple[str, ...],
    right_identity: tuple[str, ...],
    on: tuple[str, ...] | str | None,
    *,
    how: str,
) -> tuple[str, ...]:
    if on == IDENTITY_INTERSECTION:
        return resolve_identity_join_on(left_identity, right_identity, None, how=how)
    return resolve_identity_join_on(left_identity, right_identity, on, how=how)


def _renamed_field(schema: FrameInfo, renames: dict[str, str], source: str, target: str) -> FieldInfo:
    info = schema.fields[source]
    component_renames = tuple(renames.get(component, component) for component in info.components)
    return replace(
        info,
        name=target if info.is_public else info.name,
        column=target,
        components=component_renames,
        is_payload=target.startswith("__") or info.is_payload,
        is_public=info.is_public and not target.startswith("__"),
        identity_order=None,
    )


def _normalize_join_on(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...] | str | None:
    if value is None:
        return None
    if value == IDENTITY_INTERSECTION:
        return IDENTITY_INTERSECTION
    if isinstance(value, str):
        columns = (value,)
    else:
        columns = tuple(str(column) for column in value)
    if not columns:
        raise ValueError("Join on must contain at least one column.")
    return columns


def _join_on_param(node: Node) -> tuple[str, ...] | str | None:
    return _normalize_join_on(node.params.get("on"))
