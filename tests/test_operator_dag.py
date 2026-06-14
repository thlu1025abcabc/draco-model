from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import draco_model
import draco_model.layers as layer_api
import draco_model.runtime.engine as engine_module
from draco_model import Engine, Layer, Model
from draco_model.core import Node, resolve_node_names
from draco_model.layers import Aggregate, Col, FillNull, Grid, Join, Op, Project, Side, Source, Threshold, Where
from draco_model.layers.aggregate import _auction_merge_targets
from draco_model.layers.operators import _window_op
from draco_model.market.schema import DAILY_KEY_COLUMNS, KEY_COLUMNS
from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.recipes import FactorRecipe, Shortcut, metric, transform
from draco_model.runtime.execution import (
    EvalContext,
    FieldInfo,
    FrameInfo,
    get_info_builder,
    left_join_identity,
    register_executor,
    register_info,
    resolve_identity_join_on,
)


class _Context:
    minute_calendar = MinuteCalendar()


BUILTIN_FRAME_OPS = (
    "_grid_project",
    "_grid_source",
    "aggregate",
    "column",
    "fill_null",
    "join",
    "metric_reserved",
    "op",
    "project",
    "rename",
    "source",
    "where",
)


def _test_join_frame(name: str) -> Node:
    return Node(kind="frame", op="test_join_frame", params={"name": name})


@register_executor("test_join_frame")
def _test_join_frame_executor(node: Node, context: EvalContext) -> pl.LazyFrame:
    name = str(node.params["name"])
    if name == "left_price":
        return pl.DataFrame(
            {
                "date": ["20170103", "20170103", "20170103"],
                "secu_code": [1, 1, 1],
                "minute": [930, 930, 931],
                "price": [10.0, 10.1, 10.2],
                "left_value": [100, 101, 102],
            }
        ).lazy()
    if name == "right_side":
        return pl.DataFrame(
            {
                "date": ["20170103", "20170103", "20170103", "20170103"],
                "secu_code": [1, 1, 1, 1],
                "minute": [930, 930, 931, 932],
                "side": ["buy", "sell", "buy", "sell"],
                "right_value": [1, 2, 3, 4],
            }
        ).lazy()
    if name == "right_lot":
        return pl.DataFrame(
            {
                "date": ["20170103", "20170103"],
                "secu_code": [1, 1],
                "minute": [930, 931],
                "lot": [7, 8],
                "third_value": [70, 80],
            }
        ).lazy()
    if name == "daily_value":
        return pl.DataFrame(
            {
                "date": ["20170103"],
                "secu_code": [1],
                "daily_value": [42.0],
            }
        ).lazy()
    if name == "daily_minute_value":
        return pl.DataFrame(
            {
                "date": ["20170103"],
                "secu_code": [1],
                "minute": [935],
            }
        ).lazy()
    raise ValueError(f"Unsupported test join frame {name!r}.")


@register_info("test_join_frame")
def _test_join_frame_info(node: Node, parent_infos: dict[str, FrameInfo], context: EvalContext) -> FrameInfo:
    name = str(node.params["name"])
    if name == "left_price":
        return FrameInfo.from_columns(
            (*KEY_COLUMNS, "price", "left_value"),
            identity_keys=(*KEY_COLUMNS, "price"),
        )
    if name == "right_side":
        return FrameInfo.from_columns(
            (*KEY_COLUMNS, "side", "right_value"),
            identity_keys=(*KEY_COLUMNS, "side"),
        )
    if name == "right_lot":
        return FrameInfo.from_columns(
            (*KEY_COLUMNS, "lot", "third_value"),
            identity_keys=(*KEY_COLUMNS, "lot"),
        )
    if name == "daily_value":
        return FrameInfo.from_columns(
            ("date", "secu_code", "daily_value"),
            identity_keys=("date", "secu_code"),
        )
    if name == "daily_minute_value":
        return FrameInfo.from_columns(
            ("date", "secu_code", "minute"),
            identity_keys=("date", "secu_code"),
        )
    raise ValueError(f"Unsupported test join frame {name!r}.")


def test_magic_arithmetic_builds_operator_nodes() -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    volume = metric("volume")(raw)
    vwap = (amount / volume).alias("vwap")
    row_amount = (Col("price") * Col("volume")).alias("amount")(raw)

    assert vwap.op == "op"
    assert vwap.params["name"] == "div"
    assert vwap.params["alias"] == "vwap"
    assert row_amount.op == "op"
    assert row_amount.params["mode"] == "row"
    assert row_amount.params["name"] == "mul"


def test_metric_shortcut_is_not_a_runtime_layer() -> None:
    close = metric("close")

    assert isinstance(close, Shortcut)
    assert not isinstance(close, Layer)
    assert not isinstance(close, Node)
    assert not hasattr(layer_api, "Metric")
    assert draco_model.metric is metric
    assert draco_model.transform is transform
    assert draco_model.Shortcut is Shortcut
    assert draco_model.FactorRecipe is FactorRecipe


def test_transform_shortcut_placeholder_requires_registered_transform() -> None:
    raw = Source("trades_tbar")

    with pytest.raises(ValueError, match="Transform shortcut 'rank' is not registered"):
        transform("rank")(raw)


def test_factor_recipe_placeholder_build_raises() -> None:
    with pytest.raises(NotImplementedError):
        FactorRecipe().build()


def test_public_aliases_cannot_use_payload_prefix() -> None:
    raw = Source("trades_tbar")

    with pytest.raises(ValueError, match="must not start with '__'"):
        metric("volume", alias="__volume")(raw)
    with pytest.raises(ValueError, match="must not start with '__'"):
        Aggregate("1d", "last", value_col="volume", alias="__value")
    with pytest.raises(ValueError, match="must not start with '__'"):
        Col("price").alias("__price")
    with pytest.raises(ValueError, match="must not start with '__'"):
        Join()({"__volume": metric("volume")(raw)})
    with pytest.raises(ValueError, match="must not be a key column"):
        metric("volume", alias="minute")(raw)
    with pytest.raises(ValueError, match="must not be a key column"):
        Aggregate("1d", "last", value_col="volume", alias="date")


def test_join_rejects_removed_concat_how() -> None:
    with pytest.raises(ValueError, match="'full' or 'left'"):
        Join(how="concat")


def test_aggregate_requires_alias_when_value_col_is_identity(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    bad = Aggregate("1d", "last", value_col="minute")(raw)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    with pytest.raises(ValueError, match="conflicts with identity columns"):
        engine._infer_info(Model("bad_identity_aggregate", "ex2kamt", bad), bad, "20170103")

    ok = Aggregate("1d", "last", value_col="minute", alias="last_minute")(raw)
    info = engine._infer_info(Model("identity_aggregate_alias", "ex2kamt", ok), ok, "20170103")

    assert info.keys == ("date", "secu_code")
    assert "last_minute" in info.value_columns()


def test_left_join_identity_explicit_on_allows_identity_fanout() -> None:
    left = FrameInfo.from_columns(
        (*KEY_COLUMNS, "price", "left_value"),
        identity_keys=(*KEY_COLUMNS, "price"),
    )
    right = FrameInfo.from_columns(
        (*KEY_COLUMNS, "side", "right_value"),
        identity_keys=(*KEY_COLUMNS, "side"),
    )

    with pytest.raises(ValueError, match="price"):
        left_join_identity(left, right)
    assert left_join_identity(left, right, on=KEY_COLUMNS) == (*KEY_COLUMNS, "price", "side")


def test_resolve_identity_join_on_requires_identity_columns_and_shared_keys() -> None:
    left = (*KEY_COLUMNS, "price")
    right = (*KEY_COLUMNS, "price", "side")

    assert resolve_identity_join_on(left, right, None, how="full") == (*KEY_COLUMNS, "price")
    with pytest.raises(ValueError, match="identity columns"):
        resolve_identity_join_on(left, right, (*KEY_COLUMNS, "close"), how="left")
    with pytest.raises(ValueError, match="shared identity columns"):
        resolve_identity_join_on(left, right, KEY_COLUMNS, how="left")


def test_left_join_identity_accepts_daily_on_for_mixed_grain() -> None:
    minute = FrameInfo.from_columns((*KEY_COLUMNS, "value"), identity_keys=KEY_COLUMNS)
    daily = FrameInfo.from_columns((*DAILY_KEY_COLUMNS, "daily_value"), identity_keys=DAILY_KEY_COLUMNS)

    assert left_join_identity(minute, daily, on=DAILY_KEY_COLUMNS) == KEY_COLUMNS
    with pytest.raises(ValueError, match="missing_right"):
        left_join_identity(minute, daily)


def test_join_rejects_on_that_omits_shared_identity(sample_root: Path) -> None:
    joined = Join(how="left", on=KEY_COLUMNS)(
        {
            "trades": Source("trades_tbar"),
            "cancels": Source("cancels_tbar"),
        }
    )
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    with pytest.raises(ValueError, match="shared identity columns"):
        engine._infer_info(Model("bad_join_on", "ex2kamt", joined), joined, "20170103")


def test_join_left_with_explicit_on_materializes_union_identity(sample_root: Path) -> None:
    left = _test_join_frame("left_price")
    right = _test_join_frame("right_side")
    joined = Join(how="left", on=KEY_COLUMNS)({"left": left, "right": right})
    model = Model("left_join_identity", "ex2kamt", joined)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    info = engine._infer_info(model, joined, "20170103")
    frame = engine.evaluate(model, joined, "20170103").collect()

    assert info.keys == (*KEY_COLUMNS, "price", "side")
    assert tuple(frame.columns) == (*KEY_COLUMNS, "price", "side", "left", "right")
    assert frame.height == 5
    assert frame.select(info.keys).unique().height == 5
    assert frame.filter(pl.col("minute") == 930)["right"].to_list() == [1, 2, 1, 2]


def test_join_left_supports_multiple_inputs(sample_root: Path) -> None:
    left = _test_join_frame("left_price")
    right = _test_join_frame("right_side")
    third = _test_join_frame("right_lot")
    joined = Join(how="left", on=KEY_COLUMNS)({"left": left, "right": right, "third": third})
    model = Model("left_join_three_inputs", "ex2kamt", joined)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    info = engine._infer_info(model, joined, "20170103")
    frame = engine.evaluate(model, joined, "20170103").collect()

    assert info.keys == (*KEY_COLUMNS, "price", "side", "lot")
    assert tuple(frame.columns) == (*KEY_COLUMNS, "price", "side", "lot", "left", "right", "third")
    assert frame.height == 5
    assert frame.select(info.keys).unique().height == 5
    assert frame.filter((pl.col("minute") == 930) & (pl.col("side") == "buy"))["third"].to_list() == [70, 70]


def test_join_full_defaults_to_pairwise_identity_intersection(sample_root: Path) -> None:
    left = _test_join_frame("left_price")
    right = _test_join_frame("right_side")
    joined = Join(how="full")({"left": left, "right": right})
    model = Model("full_join_identity", "ex2kamt", joined)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    info = engine._infer_info(model, joined, "20170103")
    frame = engine.evaluate(model, joined, "20170103").collect()

    assert info.keys == (*KEY_COLUMNS, "price", "side")
    assert tuple(frame.columns) == (*KEY_COLUMNS, "price", "side", "left", "right")
    assert frame.height == 6
    assert frame.select(info.keys).unique().height == 6
    unmatched = frame.filter(pl.col("minute") == 932)
    assert unmatched["price"].to_list() == [None]
    assert unmatched["left"].to_list() == [None]
    assert unmatched["side"].to_list() == ["sell"]
    assert unmatched["right"].to_list() == [4]


def test_join_full_rejects_mixed_daily_and_minute_identities(sample_root: Path) -> None:
    joined = Join()({"minute_feature": _test_join_frame("left_price"), "daily_feature": _test_join_frame("daily_value")})
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    with pytest.raises(ValueError, match="cannot mix daily identity"):
        engine._infer_info(Model("bad_mixed_full_join", "ex2kamt", joined), joined, "20170103")


def test_join_left_allows_explicit_mixed_daily_and_minute_join(sample_root: Path) -> None:
    joined = Join(how="left", on=DAILY_KEY_COLUMNS)(
        {"minute_feature": _test_join_frame("left_price"), "daily_feature": _test_join_frame("daily_value")}
    )
    model = Model("mixed_left_join", "ex2kamt", joined)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    info = engine._infer_info(model, joined, "20170103")
    frame = engine.evaluate(model, joined, "20170103").collect()

    assert info.keys == (*KEY_COLUMNS, "price")
    assert tuple(frame.columns) == (*KEY_COLUMNS, "price", "minute_feature", "daily_feature")
    assert frame.height == 3
    assert frame["daily_feature"].to_list() == [42.0, 42.0, 42.0]


def test_metric_amount_uses_price_times_volume(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    model = Model("amount_probe", "ex2kamt", amount)

    frame = Engine(data_root=sample_root).evaluate(model, amount, "20170103").collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert [node.op for node in model.nodes()] == ["source", "op", "aggregate", "project"]
    assert row["amount"].to_list() == pytest.approx([152.0])


def test_buyamount_expands_to_side_filter_and_product(tmp_path: Path) -> None:
    _write_market_fixture(tmp_path)
    raw = Source("trades_tbar")
    buyamount = metric("buyamount")(raw)
    sellamount = metric("sellamount")(raw)
    amount = metric("amount")(raw)
    engine = Engine(data_root=tmp_path / "data")

    buy = engine.evaluate(Model("buyamount", "ex2kamt", buyamount), buyamount, "20170103").collect()
    sell = engine.evaluate(Model("sellamount", "ex2kamt", sellamount), sellamount, "20170103").collect()
    total = engine.evaluate(Model("amount", "ex2kamt", amount), amount, "20170103").collect()

    assert "where" in [node.op for node in Model("buyamount", "ex2kamt", buyamount).nodes()]
    assert buy["buyamount"].to_list() == pytest.approx([100.0])
    assert sell["sellamount"].to_list() == pytest.approx([200.0])
    assert total["amount"].to_list() == pytest.approx([300.0])


def test_threshold_filter_keeps_matching_rows(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    filtered = Where(Threshold("volume", op=">", value=10))(raw)

    frame = Engine(data_root=sample_root).evaluate(
        Model("large_volume_rows", "ex2kamt", filtered),
        filtered,
        "20170103",
    ).collect()

    rows = frame.select(["minute", "volume"]).sort("minute").to_dict(as_series=False)
    assert rows == {"minute": [932, 933], "volume": [30.0, 20.0]}


def test_where_stores_condition_in_params_not_subtree() -> None:
    raw = Source("trades_tbar")
    other = Source("quotes_tbar")
    raw_where = Where(Side("buy"))(raw)
    other_where = Where(Side("buy"))(other)

    assert raw_where.inputs["frame"] is raw
    assert set(raw_where.inputs) == {"frame"}
    assert set(other_where.inputs) == {"frame"}
    assert raw_where.params["condition"] == {"op": "side", "params": {"side": "buy"}}
    assert raw_where.params == other_where.params


def test_filtered_raw_column_keeps_field_source_context(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    filtered_price = Col("price").alias("filtered_price")(Where(Threshold("minute", op="==", value=931))(raw))
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    schema = engine._infer_info(Model("filtered_price_source_context", "ex2kamt", filtered_price), filtered_price, "20170103")
    info = schema.fields["filtered_price"]

    assert info.source == "trades_tbar"
    assert info.lookback_days == 1
    assert info.grain_path == ()


def test_vwap_component_and_field_aggregation(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    by_components = Aggregate("5m", "sum", apply_to="components")(vwap)
    by_field = Aggregate("5m", "mean", apply_to="field")(vwap)
    engine = Engine(data_root=sample_root)

    component_frame = engine.evaluate(Model("vwap_components", "ex2kamt", by_components), by_components, "20170103").collect()
    field_frame = engine.evaluate(Model("vwap_field", "ex2kamt", by_field), by_field, "20170103").collect()

    component_row = component_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    field_row = field_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert component_row["vwap"].to_list() == pytest.approx([673.0 / 65.0])
    assert field_row["vwap"].to_list() == pytest.approx([(152.0 / 15.0 + 10.5 + 10.3) / 3.0])
    assert any(column.startswith("__op_vwap") for column in component_frame.columns)
    assert not any(column.startswith("__op_vwap") for column in field_frame.columns)


def test_auction_merge_maps_before_aggregation(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    merged_1m = Aggregate("1m", "sum", value_col="volume", auction="merge")(Col("volume")(raw))
    merged_5m = Aggregate("5m", "sum", value_col="volume", auction="merge")(Col("volume")(raw))
    merged_open = Aggregate("5m", "first", value_col="open", auction="merge")(metric("open")(raw))
    merged_close = Aggregate("5m", "last", value_col="close", auction="merge")(metric("close")(raw))
    engine = Engine(data_root=sample_root)

    frame_1m = engine.evaluate(Model("auction_merge_1m", "ex2kamt", merged_1m), merged_1m, "20170103").collect()
    frame_5m = engine.evaluate(Model("auction_merge_5m", "ex2kamt", merged_5m), merged_5m, "20170103").collect()
    open_frame = engine.evaluate(Model("auction_merge_open", "ex2kamt", merged_open), merged_open, "20170103").collect()
    close_frame = engine.evaluate(Model("auction_merge_close", "ex2kamt", merged_close), merged_close, "20170103").collect()

    assert frame_1m.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume"].to_list() == [25.0]
    assert frame_1m.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 1456))["volume"].to_list() == [10.0]
    assert frame_1m.filter(pl.col("minute").is_in([925, 1500])).height == 0
    assert frame_5m.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume"].to_list() == [75.0]
    assert frame_5m.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 1455))["volume"].to_list() == [10.0]
    assert open_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["open"].to_list() == [9.85]
    assert close_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 1455))["close"].to_list() == [10.85]


def test_auction_merge_targets_follow_output_frequency() -> None:
    context = _Context()

    assert _auction_merge_targets(context, 1) == (930, 1456)
    assert _auction_merge_targets(context, 5) == (930, 1455)
    assert _auction_merge_targets(context, 15) == (930, 1445)


# Grid-like null bars should not contribute to daily mean denominators.
def test_daily_aggregate_applies_auction_policy(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    volume = metric("volume")(raw)
    keep = Aggregate("1d", "mean", value_col="volume", alias="value", auction="keep")(volume)
    drop = Aggregate("1d", "mean", value_col="volume", alias="value", auction="drop")(volume)
    merge = Aggregate("1d", "mean", value_col="volume", alias="value", auction="merge")(volume)
    engine = Engine(data_root=sample_root)

    keep_value = engine.evaluate(Model("daily_keep", "ex2kamt", keep), keep, "20170103").collect()["value"].to_list()
    drop_value = engine.evaluate(Model("daily_drop", "ex2kamt", drop), drop, "20170103").collect()["value"].to_list()
    merge_value = engine.evaluate(Model("daily_merge", "ex2kamt", merge), merge, "20170103").collect()["value"].to_list()

    assert keep_value == pytest.approx([85.0 / 5.0])
    assert drop_value == pytest.approx([65.0 / 3.0])
    assert merge_value == pytest.approx([72.5 / 4.0])


def test_scalar_arithmetic_on_metric(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    scaled = (metric("volume")(raw) * 100).alias("volume_x100")
    frame = Engine(data_root=sample_root).evaluate(
        Model("scaled_volume", "ex2kamt", scaled),
        scaled,
        "20170103",
    ).collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume_x100"].to_list() == [1500.0]


def test_scalar_arithmetic_component_aggregation_keeps_payload(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    scaled = (metric("volume")(raw) * 100).alias("volume_x100")
    aggregated = Aggregate("5m", "sum", apply_to="components")(scaled)

    frame = Engine(data_root=sample_root).evaluate(
        Model("scaled_volume_5m", "ex2kamt", aggregated),
        aggregated,
        "20170103",
    ).collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume_x100"].to_list() == [6500.0]
    assert "__op_volume_x100_0" in frame.columns


def test_rolling_operator_uses_generic_op(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    volume = metric("volume")(raw)
    corr = Op("rolling_corr", amount, volume, window=2, alias="corr_2")

    frame = Engine(data_root=sample_root).evaluate(Model("corr", "ex2kamt", corr), corr, "20170103").collect()

    assert corr.op == "op"
    assert corr.params["name"] == "rolling_corr"
    assert "corr_2" in frame.columns
    assert {"__op_corr_2_0", "__op_corr_2_1"}.issubset(frame.columns)


def test_rolling_operator_requires_window() -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    volume = metric("volume")(raw)

    with pytest.raises(ValueError, match="requires a positive integer window"):
        Op("rolling_corr", amount, volume)
    with pytest.raises(ValueError, match="requires a positive integer window"):
        Op("rolling_corr", amount, volume, window=0)


def test_rolling_cross_day_option_controls_minute_grouping(tmp_path: Path) -> None:
    _write_two_day_rolling_fixture(tmp_path)
    raw = Source("trades_tbar", lookback_days=2)
    amount = metric("amount")(raw)
    volume = metric("volume")(raw)
    reset = Op("rolling_corr", amount, volume, window=2, alias="corr")
    cross_day = Op("rolling_corr", amount, volume, window=2, alias="corr", cross_day=True)
    engine = Engine(data_root=tmp_path / "data")

    reset_frame = engine.evaluate(Model("reset_corr", "ex2kamt", reset), reset, "20170104").collect()
    cross_frame = engine.evaluate(Model("cross_corr", "ex2kamt", cross_day), cross_day, "20170104").collect()

    reset_value = reset_frame.filter((pl.col("date") == "20170104") & (pl.col("minute") == 930))["corr"].to_list()
    cross_value = cross_frame.filter((pl.col("date") == "20170104") & (pl.col("minute") == 930))["corr"].to_list()
    assert reset_value == [None]
    assert cross_value == pytest.approx([1.0])


def test_nested_operator_preserves_parent_payload(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    shifted = (vwap + 1).alias("vwap_plus_one")

    frame = Engine(data_root=sample_root).evaluate(
        Model("shifted_vwap", "ex2kamt", shifted),
        shifted,
        "20170103",
    ).collect()

    assert {"__op_vwap_plus_one_0", "__operand0_op_vwap_0", "__operand0_op_vwap_1"}.issubset(frame.columns)


def test_project_drops_operator_payload(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    projected = Project()(vwap)

    frame = Engine(data_root=sample_root).evaluate(Model("projected", "ex2kamt", projected), projected, "20170103").collect()

    assert "vwap" in frame.columns
    assert not any(column.startswith("__op_vwap") for column in frame.columns)


def test_join_preserves_payload_with_prefix(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    joined = Join()({"vwap": vwap, "close": metric("close")(raw)})

    frame = Engine(data_root=sample_root).evaluate(Model("joined", "ex2kamt", joined), joined, "20170103").collect()

    assert {"vwap", "close"}.issubset(frame.columns)
    assert any(column.startswith("__vwap_op_vwap") for column in frame.columns)


def test_fillnull_state_for_vwap_and_preclose(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    filled_vwap = FillNull("state")(vwap)
    preclose = FillNull("state")(metric("preclose")(raw))
    engine = Engine(data_root=sample_root)

    vwap_frame = engine.evaluate(Model("filled_vwap", "ex2kamt", filled_vwap), filled_vwap, "20170103").collect()
    preclose_frame = engine.evaluate(Model("preclose", "ex2kamt", preclose), preclose, "20170103").collect()

    assert vwap_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["vwap"].to_list() == [10.3]
    assert preclose_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["preclose"].to_list() == [9.85]
    assert not any(column.startswith("__op_vwap") for column in vwap_frame.columns)


def test_fillnull_drops_payload_and_blocks_component_aggregation(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    filled = FillNull("state")(vwap)
    aggregated = Aggregate("5m", "sum", apply_to="components")(filled)

    with pytest.raises(ValueError, match="not supported after FillNull"):
        Engine(data_root=sample_root).evaluate(
            Model("filled_component_agg", "ex2kamt", aggregated),
            aggregated,
            "20170103",
        ).collect()


def test_fillnull_state_uses_grain_path_after_operator(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    null_price = Col("price").alias("probe")(Where(Threshold("minute", op="==", value=931))(raw))
    five_minute = Aggregate("5m", "last", value_col="probe", alias="probe")(null_price)
    shifted = (five_minute + 0).alias("probe_shifted")
    filled = FillNull("state")(shifted)

    frame = Engine(data_root=sample_root).evaluate(
        Model("filled_after_operator", "ex2kamt", filled),
        filled,
        "20170103",
    ).collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["probe_shifted"].to_list() == [10.3]


def test_fillnull_state_rejects_multi_source_field_info(tmp_path: Path) -> None:
    _write_multi_source_fixture(tmp_path)
    trade_volume = metric("volume")(Source("trades_tbar"))
    quote_volume = metric("volume")(Source("quotes_tbar"))
    filled = FillNull("state")((trade_volume - quote_volume).alias("net_volume"))

    with pytest.raises(ValueError, match="requires FieldInfo.source"):
        Engine(data_root=tmp_path / "data").evaluate(
            Model("multi_source_fill", "ex2kamt", filled),
            filled,
            "20170103",
        ).collect()


def test_grid_aligns_minute_frame_to_universe_calendar(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    gridded = Grid()(metric("volume")(raw))

    frame = Engine(data_root=sample_root).evaluate(Model("grid_volume", "ex2kamt", gridded), gridded, "20170103").collect()

    assert frame.height == 2 * len(MinuteCalendar().minbars())
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume"].to_list() == [15.0]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 934))["volume"].to_list() == [None]
    assert frame.filter((pl.col("secu_code") == 2) & (pl.col("minute") == 930))["volume"].to_list() == [None]


def test_grid_uses_explicit_resampled_frequency(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    volume_5m = Aggregate("5m", "sum", value_col="volume")(metric("volume")(raw))
    gridded = Grid("5m")(volume_5m)

    frame = Engine(data_root=sample_root).evaluate(Model("grid_volume_5m", "ex2kamt", gridded), gridded, "20170103").collect()
    minutes = frame.filter(pl.col("secu_code") == 1)["minute"].unique().sort().to_list()

    assert 934 not in minutes
    assert 935 in minutes
    assert 925 in minutes
    assert 1500 in minutes


def test_grid_infers_frequency_and_removed_auction_from_grain_path(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    volume_5m = Aggregate("5m", "sum", value_col="volume", auction="drop")(metric("volume")(raw))
    volume_15m = Aggregate("15m", "sum", value_col="volume", auction="keep")(volume_5m)
    gridded = Grid()((volume_15m + 0).alias("volume_shifted"))

    frame = Engine(data_root=sample_root).evaluate(
        Model("grid_inferred_15m_drop", "ex2kamt", gridded),
        gridded,
        "20170103",
    ).collect()
    minutes = frame.filter(pl.col("secu_code") == 1)["minute"].unique().sort().to_list()

    assert 945 in minutes
    assert 935 not in minutes
    assert 925 not in minutes
    assert 1500 not in minutes


def test_grid_aligns_raw_source_before_metric(sample_root: Path) -> None:
    gridded = Grid()(Source("trades_tbar"))
    engine = Engine(data_root=sample_root)

    raw_frame = engine.evaluate(Model("grid_raw", "ex2kamt", gridded), gridded, "20170103").collect()

    assert raw_frame.height == 2 * len(MinuteCalendar().minbars()) + 1
    assert raw_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume"].sort().to_list() == [5.0, 10.0]
    assert raw_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 934))["price"].to_list() == [None]
    assert raw_frame.filter((pl.col("secu_code") == 2) & (pl.col("minute") == 930))["price"].to_list() == [None]

    volume = metric("volume")(gridded)
    volume_frame = engine.evaluate(Model("grid_raw_volume", "ex2kamt", volume), volume, "20170103").collect()

    assert volume_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume"].to_list() == [15.0]
    assert volume_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 934))["volume"].to_list() == [None]
    assert volume_frame.filter((pl.col("secu_code") == 2) & (pl.col("minute") == 930))["volume"].to_list() == [None]


def test_grid_broadcasts_daily_frame_to_minutes(sample_root: Path) -> None:
    daily = _test_join_frame("daily_value")
    gridded = Grid()(daily)
    model = Model("grid_daily", "ex2kamt", gridded)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    info = engine._infer_info(model, gridded, "20170103")
    frame = engine.evaluate(model, gridded, "20170103").collect()

    assert info.keys == KEY_COLUMNS
    assert tuple(frame.columns) == (*KEY_COLUMNS, "daily_value")
    assert frame.height == 2 * len(MinuteCalendar().minbars())
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["daily_value"].to_list() == [42.0]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 934))["daily_value"].to_list() == [42.0]
    assert frame.filter((pl.col("secu_code") == 2) & (pl.col("minute") == 930))["daily_value"].to_list() == [None]


def test_grid_rejects_value_column_that_conflicts_with_output_identity(sample_root: Path) -> None:
    daily_minute = _test_join_frame("daily_minute_value")
    gridded = Grid()(daily_minute)
    model = Model("grid_daily_minute_conflict", "ex2kamt", gridded)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    with pytest.raises(ValueError, match="conflict with value columns"):
        engine._infer_info(model, gridded, "20170103")


def test_fillnull_ffill_and_numeric_modes(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    close = Grid()(metric("close")(raw))
    ffilled = FillNull("ffill")(close)
    zero_filled = FillNull(0)(close)
    engine = Engine(data_root=sample_root)

    ffilled_frame = engine.evaluate(Model("ffill_close", "ex2kamt", ffilled), ffilled, "20170103").collect()
    zero_frame = engine.evaluate(Model("zero_close", "ex2kamt", zero_filled), zero_filled, "20170103").collect()

    assert ffilled_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))["close"].to_list() == [10.2]
    assert ffilled_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["close"].to_list() == [10.3]
    assert zero_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))["close"].to_list() == [0.0]
    assert zero_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["close"].to_list() == [0.0]


def test_aggregate_apply_to_field_drops_payload(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    daily = Aggregate("1d", "mean", value_col="vwap", alias="value")(vwap)
    projected = Project()(daily)
    engine = Engine(data_root=sample_root)

    daily_frame = engine.evaluate(Model("daily_vwap", "ex2kamt", daily), daily, "20170103").collect()
    projected_frame = engine.evaluate(Model("projected_daily_vwap", "ex2kamt", projected), projected, "20170103").collect()

    assert not any(column.startswith("__op_vwap") for column in daily_frame.columns)
    assert not any(column.startswith("__op_vwap") for column in projected_frame.columns)


def test_preclose_without_state_raises(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    preclose = metric("preclose")(raw)

    with pytest.raises(ValueError, match="preclose metric is reserved"):
        Engine(data_root=sample_root).evaluate(Model("bad_preclose", "ex2kamt", preclose), preclose, "20170103").collect()


def test_preclose_source_context_comes_from_parent_field_info(sample_root: Path) -> None:
    raw = Source("trades_tbar", lookback_days=2)
    derived = Col("price").alias("px")(raw)
    preclose = metric("preclose", alias="prev_close")(derived)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    schema = engine._infer_info(Model("preclose_source_context", "ex2kamt", preclose), preclose, "20170104")
    info = schema.fields["prev_close"]

    assert info.source == "trades_tbar"
    assert info.lookback_days == 2


def test_preclose_alias_uses_operator_metadata(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    preclose = FillNull("state")(metric("preclose", alias="prev_close")(raw))

    frame = Engine(data_root=sample_root).evaluate(
        Model("aliased_preclose", "ex2kamt", preclose),
        preclose,
        "20170103",
    ).collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["prev_close"].to_list() == [9.85]


def test_close_alias_preclose_is_not_reserved_preclose(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    close_named_preclose = FillNull("state")(metric("close", alias="preclose")(raw))

    frame = Engine(data_root=sample_root).evaluate(
        Model("close_named_preclose", "ex2kamt", close_named_preclose),
        close_named_preclose,
        "20170103",
    ).collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["preclose"].to_list() == [10.2]


def test_collect_daily_output(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    output = Aggregate("1d", "last", value_col="close", alias="value")(metric("close")(raw))
    result = Engine(data_root=sample_root).collect(Model("close_last", "ex2kamt", output), dates=["20170103"])

    assert result.to_dict(as_series=False) == {
        "date": ["20170103"],
        "secu_code": [1],
        "factor_name": ["close_last"],
        "value": [10.85],
    }


def test_collect_accepts_custom_single_output_column(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    output = Aggregate("1d", "sum", value_col="amount", alias="amount")(metric("amount")(raw))
    result = Engine(data_root=sample_root).collect(
        Model("daily_amount", "ex2kamt", output),
        dates=["20170103"],
        output_columns=["amount"],
    )

    assert result.to_dict(as_series=False) == {
        "date": ["20170103"],
        "secu_code": [1],
        "factor_name": ["daily_amount"],
        "value": [880.0],
    }


def test_collect_unpivots_multiple_output_columns(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = Aggregate("1d", "sum", value_col="amount", alias="amount")(metric("amount")(raw))
    volume = Aggregate("1d", "sum", value_col="volume", alias="volume")(metric("volume")(raw))
    output = Join()({"amount": amount, "volume": volume})

    result = Engine(data_root=sample_root).collect(
        Model("trade_totals", "ex2kamt", output),
        dates=["20170103"],
        output_columns=["amount", "volume"],
    )

    assert result.to_dict(as_series=False) == {
        "date": ["20170103", "20170103"],
        "secu_code": [1, 1],
        "factor_name": ["trade_totals__amount", "trade_totals__volume"],
        "value": [880.0, 85.0],
    }


def test_collect_rejects_unknown_output_column(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    output = Aggregate("1d", "sum", value_col="amount", alias="amount")(metric("amount")(raw))

    with pytest.raises(ValueError, match="requested"):
        Engine(data_root=sample_root).collect(
            Model("daily_amount", "ex2kamt", output),
            dates=["20170103"],
            output_columns=["missing"],
        )


def test_collect_rejects_scalar_output_columns(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    output = Aggregate("1d", "sum", value_col="amount", alias="amount")(metric("amount")(raw))

    with pytest.raises(TypeError, match="list or tuple"):
        Engine(data_root=sample_root).collect(
            Model("daily_amount", "ex2kamt", output),
            dates=["20170103"],
            output_columns="amount",  # type: ignore[arg-type]
        )


def test_collect_emits_run_logging(sample_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="draco_model.runtime.engine")
    raw = Source("trades_tbar")
    output = Aggregate("1d", "last", value_col="close", alias="value")(metric("close")(raw))

    Engine(data_root=sample_root).collect(Model("close_last", "ex2kamt", output), dates=["20170103"])

    messages = [record.getMessage() for record in caplog.records]
    assert any("collect.start model=close_last universe=ex2kamt" in message for message in messages)
    assert any("collect.done model=close_last universe=ex2kamt" in message for message in messages)


def test_collect_concatenates_multiple_dates(tmp_path: Path) -> None:
    _write_two_day_collect_fixture(tmp_path)
    raw = Source("trades_tbar")
    output = Aggregate("1d", "last", value_col="close", alias="value")(metric("close")(raw))

    result = Engine(data_root=tmp_path / "data").collect(
        Model("close_last", "ex2kamt", output),
        dates=["20170103", "20170104"],
    )

    assert result.sort("date").to_dict(as_series=False) == {
        "date": ["20170103", "20170104"],
        "secu_code": [1, 1],
        "factor_name": ["close_last", "close_last"],
        "value": [10.0, 20.0],
    }


def test_collect_rejects_minute_output_even_with_value_column(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    output = metric("volume", alias="value")(raw)

    with pytest.raises(ValueError, match="requires a daily output"):
        Engine(data_root=sample_root).collect(Model("minute_value", "ex2kamt", output), dates=["20170103"])


def test_trace_and_mermaid_show_expanded_operator_dag(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    model = Model("trace_vwap", "ex2kamt", (metric("amount")(raw) / metric("volume")(raw)).alias("vwap"))

    steps = Engine(data_root=sample_root).trace(model, "20170103")
    mermaid = model.explain_mermaid()

    assert [step.node.op for step in steps] == [
        "source",
        "op",
        "aggregate",
        "project",
        "column",
        "aggregate",
        "project",
        "op",
    ]
    assert "op" in mermaid
    assert "ratio_field" not in mermaid


def test_frame_infos_match_materialized_columns(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    filled = FillNull("state")(vwap)
    corr = Op("rolling_corr", metric("amount")(raw), metric("volume")(raw), window=2, alias="corr")
    daily = Aggregate("1d", "mean", value_col="vwap", alias="daily_vwap")(vwap)
    minute_features = Join(how="left")({"filled": filled, "corr": corr})
    output = Project()(Join(how="left", on=DAILY_KEY_COLUMNS)({"minute_features": minute_features, "daily": daily}))
    model = Model("plan_columns", "ex2kamt", output)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    for node in model.nodes():
        if node.kind != "frame":
            continue
        schema = engine._infer_info(model, node, "20170103")
        frame = engine.evaluate(model, node, "20170103").collect()
        assert tuple(frame.columns) == schema.columns


def test_info_inference_memoizes_shared_nodes(sample_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = Source("trades_tbar")
    vwap = (metric("amount")(raw) / metric("volume")(raw)).alias("vwap")
    model = Model("memoized_schema", "ex2kamt", vwap)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()
    calls: dict[str, int] = {}
    original_get_info_builder = engine_module.get_info_builder

    def counting_get_info_builder(op: str):
        calls[op] = calls.get(op, 0) + 1
        return original_get_info_builder(op)

    monkeypatch.setattr(engine_module, "get_info_builder", counting_get_info_builder)

    schema = engine._infer_info(model, vwap, "20170103")
    calls_after_first_infer = dict(calls)
    cached_schema = engine._infer_info(model, vwap, "20170103")

    assert cached_schema is schema
    assert calls == calls_after_first_infer
    assert calls["source"] == 1


def test_builtin_frame_layers_register_info_builders() -> None:
    for op in BUILTIN_FRAME_OPS:
        assert get_info_builder(op) is not None


def test_package_import_bootstraps_builtin_layer_registry() -> None:
    code = f"""
import draco_model
from draco_model.runtime.execution import get_executor, get_info_builder

for op in {BUILTIN_FRAME_OPS!r}:
    get_executor(op)
    assert get_info_builder(op) is not None, op
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_join_evaluates_distinct_sources(tmp_path: Path) -> None:
    _write_multi_source_fixture(tmp_path)
    trade_volume = metric("volume")(Source("trades_tbar"))
    quote_volume = metric("volume")(Source("quotes_tbar"))
    joined = Join()({"trade_volume": trade_volume, "quote_volume": quote_volume})

    frame = Engine(data_root=tmp_path / "data").evaluate(
        Model("multi_source", "ex2kamt", joined),
        joined,
        "20170103",
    ).collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["trade_volume"].to_list() == [10.0]
    assert row["quote_volume"].to_list() == [7.0]


def _write_market_fixture(tmp_path: Path) -> None:
    external = tmp_path / "external"
    data = tmp_path / "data"
    external.mkdir()
    pl.DataFrame({"date": ["20170103"]}).write_parquet(external / "trading_days.parquet")
    (data / "universe" / "ex2kamt").mkdir(parents=True)
    pl.DataFrame({"secu_code": [1]}).write_parquet(data / "universe" / "ex2kamt" / "20170103.parquet")
    (data / "trades_tbar").mkdir()
    pl.DataFrame(
        {
            "SecuCode": [1, 1],
            "MinBar": [930, 930],
            "Price": [10.0, 20.0],
            "Side": [0, 1],
            "Volume": [10.0, 10.0],
            "vw_wait_time": [0.0, 0.0],
            "isfirst": [True, True],
            "islast": [True, True],
            "No": [1, 1],
        }
    ).write_parquet(data / "trades_tbar" / "20170103.parquet")
    (data / "daily_k").mkdir()
    pl.DataFrame(_daily_k_fixture_data(["000001.SZ"], ["2017-01-03"], [9.5])).write_parquet(
        data / "daily_k" / "20170103.parquet"
    )


def _write_two_day_rolling_fixture(tmp_path: Path) -> None:
    external = tmp_path / "external"
    data = tmp_path / "data"
    external.mkdir()
    pl.DataFrame({"date": ["20170103", "20170104"]}).write_parquet(external / "trading_days.parquet")
    (data / "universe" / "ex2kamt").mkdir(parents=True)
    pl.DataFrame({"secu_code": [1]}).write_parquet(data / "universe" / "ex2kamt" / "20170104.parquet")
    (data / "trades_tbar").mkdir()
    for date, volumes in {"20170103": [1.0, 2.0], "20170104": [3.0, 4.0]}.items():
        pl.DataFrame(
            {
                "SecuCode": [1, 1],
                "MinBar": [930, 931],
                "Price": [10.0, 10.0],
                "Side": [0, 0],
                "Volume": volumes,
                "vw_wait_time": [0.0, 0.0],
                "isfirst": [True, True],
                "islast": [True, True],
                "No": [1, 1],
            }
        ).write_parquet(data / "trades_tbar" / f"{date}.parquet")


def _write_two_day_collect_fixture(tmp_path: Path) -> None:
    external = tmp_path / "external"
    data = tmp_path / "data"
    external.mkdir()
    pl.DataFrame({"date": ["20170103", "20170104"]}).write_parquet(external / "trading_days.parquet")
    (data / "trades_tbar").mkdir(parents=True)
    for date, price in {"20170103": 10.0, "20170104": 20.0}.items():
        pl.DataFrame(
            {
                "SecuCode": [1],
                "MinBar": [930],
                "Price": [price],
                "Side": [0],
                "Volume": [1.0],
                "vw_wait_time": [0.0],
                "isfirst": [True],
                "islast": [True],
                "No": [1],
            }
        ).write_parquet(data / "trades_tbar" / f"{date}.parquet")


def _write_multi_source_fixture(tmp_path: Path) -> None:
    external = tmp_path / "external"
    data = tmp_path / "data"
    external.mkdir()
    pl.DataFrame({"date": ["20170103"]}).write_parquet(external / "trading_days.parquet")
    (data / "trades_tbar").mkdir(parents=True)
    pl.DataFrame(
        {
            "SecuCode": [1],
            "MinBar": [930],
            "Price": [10.0],
            "Side": [0],
            "Volume": [10.0],
            "vw_wait_time": [0.0],
            "isfirst": [True],
            "islast": [True],
            "No": [1],
        }
    ).write_parquet(data / "trades_tbar" / "20170103.parquet")
    (data / "quotes_tbar").mkdir()
    pl.DataFrame(
        {
            "SecuCode": [1],
            "MinBar": [930],
            "Price": [10.0],
            "Side": [0],
            "Volume": [7.0],
            "isfirst": [True],
            "islast": [True],
            "No": [1],
        }
    ).write_parquet(data / "quotes_tbar" / "20170103.parquet")


def _daily_k_fixture_data(sec_codes: list[str], trading_days: list[str], precloses: list[float]) -> dict:
    n = len(sec_codes)
    return {
        "sec_code": sec_codes,
        "trading_day": trading_days,
        "open": [10.0] * n,
        "high": [11.0] * n,
        "low": [9.0] * n,
        "close": [10.5] * n,
        "shares": [100.0] * n,
        "amount": [1000.0] * n,
        "limit_up": [11.0] * n,
        "limit_down": [9.0] * n,
        "preclose": precloses,
        "isSuspend": [False] * n,
        "isST": [False] * n,
        "adjfactor": [1.0] * n,
        "total_share": [1000.0] * n,
        "float_share": [900.0] * n,
        "free_share": [800.0] * n,
        "list_date": ["19910403"] * n,
    }


def test_merged_source_context_uses_max_lookback() -> None:
    info = FrameInfo.from_columns(
        ("date", "secu_code", "minute", "a", "b"),
        identity_keys=KEY_COLUMNS,
        fields={
            "a": FieldInfo("a", "a", source="trades_tbar", lookback_days=5),
            "b": FieldInfo("b", "b", source="trades_tbar", lookback_days=3),
        },
    )

    assert info.merged_source_context() == ("trades_tbar", 5, ())


def test_mixed_lookback_operands_inherit_max_lookback(sample_root: Path) -> None:
    slow = metric("volume")(Source("trades_tbar", lookback_days=2))
    fast = metric("no")(Source("trades_tbar"))
    combined = (slow + fast).alias("combined")
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    schema = engine._infer_info(Model("mixed_lookback", "ex2kamt", combined), combined, "20170104")

    assert schema.fields["combined"].lookback_days == 2


def test_window_op_rejects_non_frame_operands() -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)

    with pytest.raises(ValueError, match="exactly two frame"):
        Op("rolling_corr", Col("price"), Col("volume"), window=2)
    with pytest.raises(ValueError, match="exactly two frame"):
        Op("rolling_beta", amount, 2.0, window=2)


def test_rolling_corr_guards_float_negative_variance() -> None:
    base = 1.0e8
    frame = pl.DataFrame(
        {
            "date": ["20170103"] * 4,
            "secu_code": [1] * 4,
            "minute": [930, 931, 932, 933],
            "y": [base, base + 3e-4, base + 1e-4, base + 2e-4],
            "x": [base + 2e-4, base + 1e-4, base + 3e-4, base],
        }
    ).lazy()

    out = _window_op(frame, KEY_COLUMNS, "rolling_corr", [pl.col("y"), pl.col("x")], "corr", 3, False, []).collect()

    corr = out["corr"]
    assert not corr.fill_null(0.0).is_nan().any()
    assert corr.drop_nulls().abs().max() <= 1.0


def test_aggregate_minute_frequency_requires_minute_input(sample_root: Path) -> None:
    daily = Aggregate("1d", "sum", value_col="volume", alias="volume")(metric("volume")(Source("trades_tbar")))
    five_minute = Aggregate("5m", "sum", value_col="volume", alias="volume")(daily)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    with pytest.raises(ValueError, match="minute-grain input"):
        engine._infer_info(Model("daily_to_minute", "ex2kamt", five_minute), five_minute, "20170103")


def test_fillnull_state_requires_minute_input(sample_root: Path) -> None:
    daily = Aggregate("1d", "last", value_col="close", alias="close")(metric("close")(Source("trades_tbar")))
    filled = FillNull("state")(daily)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    with pytest.raises(ValueError, match="minute-grain input"):
        engine._infer_info(Model("daily_state_fill", "ex2kamt", filled), filled, "20170103")


def test_conflicting_names_on_identical_structure_raise() -> None:
    combined = Source("trades_tbar", name="left_raw") + Source("trades_tbar", name="right_raw")

    with pytest.raises(ValueError, match="conflicting names"):
        Model("conflicting_names", "ex2kamt", combined).nodes()


def test_duplicate_structure_keeps_explicit_name() -> None:
    combined = Source("trades_tbar") + Source("trades_tbar", name="raw_trades")

    names = resolve_node_names(Model("duplicate_structure", "ex2kamt", combined).nodes())

    assert "raw_trades" in names.values()


def test_infer_info_requires_registered_info_builder(sample_root: Path) -> None:
    node = Node(kind="frame", op="executor_only_test_frame", params={})
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    with pytest.raises(ValueError, match="frame-info builder"):
        engine._infer_info(Model("executor_only", "ex2kamt", node), node, "20170103")
