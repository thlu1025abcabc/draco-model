from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model
from draco_model.layers import Aggregate, Col, FillNull, Join, Metric, Op, Project, Side, Source, Threshold, Where
from draco_model.layers.aggregate import _auction_merge_targets
from draco_model.market.minute_calendar import MinuteCalendar
from draco_model.runtime.execution import get_plan_builder


class _Context:
    minute_calendar = MinuteCalendar()


def test_magic_arithmetic_builds_operator_nodes() -> None:
    raw = Source("trades_tbar")
    amount = Metric("amount", raw)
    volume = Metric("volume", raw)
    vwap = (amount / volume).alias("vwap")
    row_amount = (Col("price") * Col("volume")).alias("amount")(raw)

    assert vwap.op == "op"
    assert vwap.params["name"] == "div"
    assert vwap.params["alias"] == "vwap"
    assert row_amount.op == "op"
    assert row_amount.params["mode"] == "row"
    assert row_amount.params["name"] == "mul"


def test_public_aliases_cannot_use_payload_prefix() -> None:
    raw = Source("trades_tbar")

    with pytest.raises(ValueError, match="must not start with '__'"):
        Metric("volume", raw, alias="__volume")
    with pytest.raises(ValueError, match="must not start with '__'"):
        Aggregate("1d", "last", value_col="volume", alias="__value")
    with pytest.raises(ValueError, match="must not start with '__'"):
        Col("price").alias("__price")
    with pytest.raises(ValueError, match="must not start with '__'"):
        Join()({"__volume": Metric("volume", raw)})
    with pytest.raises(ValueError, match="must not be a key column"):
        Metric("volume", raw, alias="minute")
    with pytest.raises(ValueError, match="must not be a key column"):
        Aggregate("1d", "last", value_col="volume", alias="date")


def test_metric_amount_uses_price_times_volume(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = Metric("amount", raw)
    model = Model("amount_probe", "ex2kamt", amount)

    frame = Engine(data_root=sample_root).evaluate(model, amount, "20170103").collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert [node.op for node in model.nodes()] == ["source", "op", "aggregate"]
    assert row["amount"].to_list() == pytest.approx([152.0])


def test_buyamount_expands_to_side_filter_and_product(tmp_path: Path) -> None:
    _write_market_fixture(tmp_path)
    raw = Source("trades_tbar")
    buyamount = Metric("buyamount", raw)
    sellamount = Metric("sellamount", raw)
    amount = Metric("amount", raw)
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


def test_vwap_component_and_field_aggregation(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (Metric("amount", raw) / Metric("volume", raw)).alias("vwap")
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
    assert any(column.startswith("__op_vwap") for column in field_frame.columns)


def test_auction_merge_maps_before_aggregation(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    merged_1m = Aggregate("1m", "sum", value_col="volume", auction="merge")(Col("volume")(raw))
    merged_5m = Aggregate("5m", "sum", value_col="volume", auction="merge")(Col("volume")(raw))
    merged_open = Aggregate("5m", "first", value_col="open", auction="merge")(Metric("open", raw))
    merged_close = Aggregate("5m", "last", value_col="close", auction="merge")(Metric("close", raw))
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


def test_daily_aggregate_applies_auction_policy(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    volume = Metric("volume", raw)
    keep = Aggregate("1d", "mean", value_col="volume", alias="value", auction="keep")(volume)
    drop = Aggregate("1d", "mean", value_col="volume", alias="value", auction="drop")(volume)
    merge = Aggregate("1d", "mean", value_col="volume", alias="value", auction="merge")(volume)
    engine = Engine(data_root=sample_root)

    keep_value = engine.evaluate(Model("daily_keep", "ex2kamt", keep), keep, "20170103").collect()["value"].to_list()
    drop_value = engine.evaluate(Model("daily_drop", "ex2kamt", drop), drop, "20170103").collect()["value"].to_list()
    merge_value = engine.evaluate(Model("daily_merge", "ex2kamt", merge), merge, "20170103").collect()["value"].to_list()

    assert keep_value == pytest.approx([85.0 / 7.0])
    assert drop_value == pytest.approx([65.0 / 5.0])
    assert merge_value == pytest.approx([72.5 / 6.0])


def test_scalar_arithmetic_on_metric(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    scaled = (Metric("volume", raw) * 100).alias("volume_x100")
    frame = Engine(data_root=sample_root).evaluate(
        Model("scaled_volume", "ex2kamt", scaled),
        scaled,
        "20170103",
    ).collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume_x100"].to_list() == [1500.0]


def test_scalar_arithmetic_component_aggregation_keeps_payload(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    scaled = (Metric("volume", raw) * 100).alias("volume_x100")
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
    amount = Metric("amount", raw)
    volume = Metric("volume", raw)
    corr = Op("rolling_corr", amount, volume, window=2, alias="corr_2")

    frame = Engine(data_root=sample_root).evaluate(Model("corr", "ex2kamt", corr), corr, "20170103").collect()

    assert corr.op == "op"
    assert corr.params["name"] == "rolling_corr"
    assert "corr_2" in frame.columns
    assert {"__op_corr_2_0", "__op_corr_2_1"}.issubset(frame.columns)


def test_rolling_operator_requires_window() -> None:
    raw = Source("trades_tbar")
    amount = Metric("amount", raw)
    volume = Metric("volume", raw)

    with pytest.raises(ValueError, match="requires a positive integer window"):
        Op("rolling_corr", amount, volume)
    with pytest.raises(ValueError, match="requires a positive integer window"):
        Op("rolling_corr", amount, volume, window=0)


def test_rolling_cross_day_option_controls_minute_grouping(tmp_path: Path) -> None:
    _write_two_day_rolling_fixture(tmp_path)
    raw = Source("trades_tbar", lookback_days=2)
    amount = Metric("amount", raw)
    volume = Metric("volume", raw)
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
    vwap = (Metric("amount", raw) / Metric("volume", raw)).alias("vwap")
    shifted = (vwap + 1).alias("vwap_plus_one")

    frame = Engine(data_root=sample_root).evaluate(
        Model("shifted_vwap", "ex2kamt", shifted),
        shifted,
        "20170103",
    ).collect()

    assert {"__op_vwap_plus_one_0", "__operand0_op_vwap_0", "__operand0_op_vwap_1"}.issubset(frame.columns)


def test_project_drops_operator_payload(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (Metric("amount", raw) / Metric("volume", raw)).alias("vwap")
    projected = Project()(vwap)

    frame = Engine(data_root=sample_root).evaluate(Model("projected", "ex2kamt", projected), projected, "20170103").collect()

    assert "vwap" in frame.columns
    assert not any(column.startswith("__op_vwap") for column in frame.columns)


def test_join_preserves_payload_with_prefix(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (Metric("amount", raw) / Metric("volume", raw)).alias("vwap")
    joined = Join()({"vwap": vwap, "close": Metric("close", raw)})

    frame = Engine(data_root=sample_root).evaluate(Model("joined", "ex2kamt", joined), joined, "20170103").collect()

    assert {"vwap", "close"}.issubset(frame.columns)
    assert any(column.startswith("__vwap_op_vwap") for column in frame.columns)


def test_fillnull_state_for_vwap_and_preclose(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (Metric("amount", raw) / Metric("volume", raw)).alias("vwap")
    filled_vwap = FillNull("state")(vwap)
    preclose = FillNull("state")(Metric("preclose", raw))
    engine = Engine(data_root=sample_root)

    vwap_frame = engine.evaluate(Model("filled_vwap", "ex2kamt", filled_vwap), filled_vwap, "20170103").collect()
    preclose_frame = engine.evaluate(Model("preclose", "ex2kamt", preclose), preclose, "20170103").collect()

    assert vwap_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["vwap"].to_list() == [10.3]
    assert preclose_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["preclose"].to_list() == [9.85]
    assert any(column.startswith("__op_vwap") for column in vwap_frame.columns)


def test_fillnull_ffill_and_numeric_modes(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    close = Metric("close", raw)
    ffilled = FillNull("ffill")(close)
    zero_filled = FillNull(0)(close)
    engine = Engine(data_root=sample_root)

    ffilled_frame = engine.evaluate(Model("ffill_close", "ex2kamt", ffilled), ffilled, "20170103").collect()
    zero_frame = engine.evaluate(Model("zero_close", "ex2kamt", zero_filled), zero_filled, "20170103").collect()

    assert ffilled_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))["close"].to_list() == [10.2]
    assert ffilled_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["close"].to_list() == [10.3]
    assert zero_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))["close"].to_list() == [0.0]
    assert zero_frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["close"].to_list() == [0.0]


def test_daily_aggregate_preserves_payload_until_project(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (Metric("amount", raw) / Metric("volume", raw)).alias("vwap")
    daily = Aggregate("1d", "mean", value_col="vwap", alias="value")(vwap)
    projected = Project()(daily)
    engine = Engine(data_root=sample_root)

    daily_frame = engine.evaluate(Model("daily_vwap", "ex2kamt", daily), daily, "20170103").collect()
    projected_frame = engine.evaluate(Model("projected_daily_vwap", "ex2kamt", projected), projected, "20170103").collect()

    assert any(column.startswith("__op_vwap") for column in daily_frame.columns)
    assert not any(column.startswith("__op_vwap") for column in projected_frame.columns)


def test_preclose_without_state_raises(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    preclose = Metric("preclose", raw)

    with pytest.raises(ValueError, match="preclose metric is reserved"):
        Engine(data_root=sample_root).evaluate(Model("bad_preclose", "ex2kamt", preclose), preclose, "20170103").collect()


def test_preclose_alias_uses_operator_metadata(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    preclose = FillNull("state")(Metric("preclose", raw, alias="prev_close"))

    frame = Engine(data_root=sample_root).evaluate(
        Model("aliased_preclose", "ex2kamt", preclose),
        preclose,
        "20170103",
    ).collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["prev_close"].to_list() == [9.85]


def test_close_alias_preclose_is_not_reserved_preclose(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    close_named_preclose = FillNull("state")(Metric("close", raw, alias="preclose"))

    frame = Engine(data_root=sample_root).evaluate(
        Model("close_named_preclose", "ex2kamt", close_named_preclose),
        close_named_preclose,
        "20170103",
    ).collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["preclose"].to_list() == [10.2]


def test_collect_daily_output(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    output = Aggregate("1d", "last", value_col="close", alias="value")(Metric("close", raw))
    result = Engine(data_root=sample_root).collect(Model("close_last", "ex2kamt", output), dates=["20170103"])

    assert result.to_dict(as_series=False) == {
        "date": ["20170103"],
        "secu_code": [1],
        "factor_name": ["close_last"],
        "value": [10.85],
    }


def test_collect_concatenates_multiple_dates(tmp_path: Path) -> None:
    _write_two_day_collect_fixture(tmp_path)
    raw = Source("trades_tbar")
    output = Aggregate("1d", "last", value_col="close", alias="value")(Metric("close", raw))

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
    output = Metric("volume", raw, alias="value")

    with pytest.raises(ValueError, match="requires a daily output"):
        Engine(data_root=sample_root).collect(Model("minute_value", "ex2kamt", output), dates=["20170103"])


def test_trace_and_mermaid_show_expanded_operator_dag(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    model = Model("trace_vwap", "ex2kamt", (Metric("amount", raw) / Metric("volume", raw)).alias("vwap"))

    steps = Engine(data_root=sample_root).trace(model, "20170103")
    mermaid = model.explain_mermaid()

    assert [step.node.op for step in steps] == ["source", "op", "aggregate", "column", "aggregate", "op"]
    assert "op" in mermaid
    assert "ratio_field" not in mermaid


def test_frame_plans_match_materialized_columns(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    vwap = (Metric("amount", raw) / Metric("volume", raw)).alias("vwap")
    filled = FillNull("state")(vwap)
    corr = Op("rolling_corr", Metric("amount", raw), Metric("volume", raw), window=2, alias="corr")
    daily = Aggregate("1d", "mean", value_col="vwap", alias="daily_vwap")(vwap)
    output = Project()(Join()({"filled": filled, "corr": corr, "daily": daily}))
    model = Model("plan_columns", "ex2kamt", output)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    for node in model.nodes():
        if node.kind != "frame":
            continue
        schema = engine._infer_schema(model, node, "20170103")
        frame = engine.evaluate(model, node, "20170103").collect()
        assert tuple(frame.columns) == schema.columns


def test_builtin_frame_layers_register_plans() -> None:
    for op in ["aggregate", "column", "fill_null", "join", "metric_reserved", "op", "project", "rename", "source", "where"]:
        assert get_plan_builder(op) is not None


def test_join_evaluates_distinct_sources(tmp_path: Path) -> None:
    _write_multi_source_fixture(tmp_path)
    trade_volume = Metric("volume", Source("trades_tbar"))
    quote_volume = Metric("volume", Source("quotes_tbar"))
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
    pl.DataFrame({"sec_code": ["000001.SZ"], "trading_day": ["2017-01-03"], "preclose": [9.5]}).write_parquet(
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
