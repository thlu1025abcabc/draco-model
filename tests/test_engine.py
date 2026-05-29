from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model, Node
from draco_model.layers import Aggregate, DailyAgg, Input, RatioField
from draco_model.runtime.execution import EvalContext, register_executor


@register_executor("unordered_intraday_test_frame")
def _unordered_intraday_test_frame(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date, context.eval_date, context.eval_date],
            "secu_code": [1, 1, 1],
            "minute": [1500, 930, 1456],
            "close": [100.0, 10.0, 50.0],
        }
    ).lazy()


@register_executor("daily_sum_null_test_frame")
def _daily_sum_null_test_frame(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date, context.eval_date, context.eval_date],
            "secu_code": [1, 1, 2],
            "minute": [930, 931, 930],
            "volume": [None, None, 4.0],
        }
    ).lazy()


def test_engine_dispatches_registered_executor(engine_data_root: Path) -> None:
    output = Node(kind="frame", op="constant_test_frame", params={"value": 7})
    model = Model(name="constant_factor", universe="unused", output=output)

    result = Engine(data_root=engine_data_root).collect(model, dates=["20170103"])

    assert result.to_dict(as_series=False) == {
        "date": ["20170103"],
        "secu_code": [1],
        "factor_name": ["constant_factor"],
        "value": [7.0],
    }


def test_collect_requires_at_least_one_date(tmp_path: Path) -> None:
    output = Node(kind="frame", op="constant_test_frame", params={"value": 7})
    model = Model(name="constant_factor", universe="unused", output=output)

    with pytest.raises(ValueError, match="at least one date"):
        Engine(data_root=tmp_path / "data").collect(model, dates=[])


def test_daily_agg_first_and_last_use_minute_order(engine_data_root: Path) -> None:
    raw = Node(kind="frame", op="unordered_intraday_test_frame")
    first_output = DailyAgg(value_col="close", agg="first")(raw)
    last_output = DailyAgg(value_col="close", agg="last")(raw)
    engine = Engine(data_root=engine_data_root)

    first_result = engine.collect(Model(name="first_probe", universe="unused", output=first_output), dates=["20170103"])
    last_result = engine.collect(Model(name="last_probe", universe="unused", output=last_output), dates=["20170103"])

    assert first_result["value"].to_list() == [10.0]
    assert last_result["value"].to_list() == [100.0]


def test_daily_agg_sum_keeps_all_null_group_null(engine_data_root: Path) -> None:
    raw = Node(kind="frame", op="daily_sum_null_test_frame")
    output = DailyAgg(value_col="volume", agg="sum")(raw)
    result = Engine(data_root=engine_data_root).collect(
        Model(name="daily_null_sum_probe", universe="unused", output=output),
        dates=["20170103"],
    ).sort("secu_code")

    assert result["value"].to_list() == [None, 4.0]


def test_aggregate_layer_can_do_daily_aggregation(engine_data_root: Path) -> None:
    raw = Node(kind="frame", op="unordered_intraday_test_frame")
    output = Aggregate("daily", "last", value_col="close", alias="value")(raw)

    result = Engine(data_root=engine_data_root).collect(
        Model(name="daily_aggregate_probe", universe="unused", output=output),
        dates=["20170103"],
    )

    assert result["value"].to_list() == [100.0]


def test_daily_agg_can_apply_to_ratio_components(sample_root: Path) -> None:
    raw_vwap = RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar"))
    by_field = DailyAgg(value_col="vwap", agg="mean", apply_to="field")(raw_vwap)
    by_components = DailyAgg(value_col="vwap", agg="sum", apply_to="components")(raw_vwap)
    engine = Engine(data_root=sample_root)

    field_frame = engine.evaluate(Model("daily_vwap_field", "ex2kamt", by_field), by_field, "20170103").collect()
    component_frame = engine.evaluate(
        Model("daily_vwap_components", "ex2kamt", by_components),
        by_components,
        "20170103",
    ).collect()

    field_value = field_frame.filter(pl.col("secu_code") == 1)["value"].to_list()
    component_value = component_frame.filter(pl.col("secu_code") == 1)["value"].to_list()
    assert field_value == pytest.approx([(9.85 + 152.0 / 15.0 + 10.5 + 10.3 + 10.85) / 5.0])
    assert component_value == pytest.approx([880.0 / 85.0])
    assert field_value != pytest.approx(component_value)
