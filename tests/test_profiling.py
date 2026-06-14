from __future__ import annotations

from pathlib import Path

import pytest

from draco_model import Engine, Model, profile_plan
from draco_model.layers import Aggregate, Join, Source
from draco_model.recipes import metric


def test_profile_plan_marks_shared_cache_candidates() -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    vwap = metric("vwap")(raw)

    profile = profile_plan(
        [
            Model("amount", "ex2kamt", amount),
            Model("vwap", "ex2kamt", vwap),
        ]
    )

    amount_aggregates = [
        node
        for node in profile.nodes
        if node.op == "aggregate" and node.params.get("alias") == "amount"
    ]
    assert len(amount_aggregates) == 1
    assert amount_aggregates[0].ref_count == 2
    assert amount_aggregates[0].model_count == 2
    assert amount_aggregates[0].cache_candidate
    assert profile.summary()["cache_candidate_count"] >= 1
    assert profile.to_frame().height == len(profile.nodes)


def test_profile_plan_does_not_materialize_sources_as_candidates() -> None:
    raw = Source("trades_tbar")
    profile = profile_plan(
        [
            Model("amount", "ex2kamt", metric("amount")(raw)),
            Model("volume", "ex2kamt", metric("volume")(raw)),
        ]
    )

    source_nodes = [node for node in profile.nodes if node.op == "source"]
    assert len(source_nodes) == 1
    assert source_nodes[0].ref_count == 2
    assert not source_nodes[0].cache_candidate


def test_engine_profiler_records_collect_events(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    output = Aggregate("1d", "sum", value_col="amount", alias="value")(amount)
    model = Model("daily_amount", "ex2kamt", output)
    engine = Engine(data_root=sample_root)

    with engine.profiler() as profiler:
        result = engine.collect(model, ["20170103"])

    assert result.columns == ["date", "secu_code", "factor_name", "value"]
    events = profiler.events()
    event_names = [event.event for event in events]
    assert "collect.start" in event_names
    assert "collect.materialize.end" in event_names
    assert any(
        event.event == "eval.cache_miss" and event.fields.get("op") == "source"
        for event in events
    )
    assert profiler.summary()["counts"]["collect.start"] == 1
    assert profiler.to_frame().height == len(events)


def test_collect_many_returns_long_factor_output(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = Aggregate("1d", "sum", value_col="amount", alias="value")(metric("amount")(raw))
    volume = Aggregate("1d", "sum", value_col="volume", alias="value")(metric("volume")(raw))
    engine = Engine(data_root=sample_root)

    result = engine.collect_many(
        [
            Model("daily_amount", "ex2kamt", amount),
            Model("daily_volume", "ex2kamt", volume),
        ],
        ["20170103"],
    ).sort("factor_name")

    assert result.to_dict(as_series=False) == {
        "date": ["20170103", "20170103"],
        "secu_code": [1, 1],
        "factor_name": ["daily_amount", "daily_volume"],
        "value": [880.0, 85.0],
    }


def test_collect_many_reuses_shared_candidate(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    volume = metric("volume")(raw)
    vwap = (amount / volume).alias("vwap")
    amount_daily = Aggregate("1d", "sum", value_col="amount", alias="value")(amount)
    vwap_daily = Aggregate("1d", "mean", value_col="vwap", alias="value")(vwap)
    engine = Engine(data_root=sample_root)

    with engine.profiler() as profiler:
        engine.collect_many(
            [
                Model("daily_amount", "ex2kamt", amount_daily),
                Model("daily_vwap", "ex2kamt", vwap_daily),
            ],
            ["20170103"],
        )

    events = profiler.events()
    assert any(
        event.event == "batch_cache.materialize" and event.fields.get("node_id") == amount.id
        for event in events
    )
    assert any(
        event.event == "batch_cache.hit" and event.fields.get("node_id") == amount.id
        for event in events
    )


def test_collect_many_respects_min_cache_ref_count(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    volume = metric("volume")(raw)
    vwap = (amount / volume).alias("vwap")
    amount_daily = Aggregate("1d", "sum", value_col="amount", alias="value")(amount)
    vwap_daily = Aggregate("1d", "mean", value_col="vwap", alias="value")(vwap)
    engine = Engine(data_root=sample_root)

    with engine.profiler() as profiler:
        engine.collect_many(
            [
                Model("daily_amount", "ex2kamt", amount_daily),
                Model("daily_vwap", "ex2kamt", vwap_daily),
            ],
            ["20170103"],
            min_cache_ref_count=3,
        )

    assert not any(
        event.event.startswith("batch_cache.") and event.fields.get("node_id") == amount.id
        for event in profiler.events()
    )


def test_collect_many_unpivots_multiple_output_columns(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = Aggregate("1d", "sum", value_col="amount", alias="amount")(metric("amount")(raw))
    volume = Aggregate("1d", "sum", value_col="volume", alias="volume")(metric("volume")(raw))
    output = Join()({"amount": amount, "volume": volume})
    engine = Engine(data_root=sample_root)

    result = engine.collect_many(
        [Model("trade_totals", "ex2kamt", output)],
        ["20170103"],
        output_columns=["amount", "volume"],
    )

    assert result.to_dict(as_series=False) == {
        "date": ["20170103", "20170103"],
        "secu_code": [1, 1],
        "factor_name": ["trade_totals__amount", "trade_totals__volume"],
        "value": [880.0, 85.0],
    }


def test_collect_many_rejects_duplicate_model_names(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = Aggregate("1d", "sum", value_col="amount", alias="value")(metric("amount")(raw))
    volume = Aggregate("1d", "sum", value_col="volume", alias="value")(metric("volume")(raw))

    with pytest.raises(ValueError, match="unique model names"):
        Engine(data_root=sample_root).collect_many(
            [
                Model("duplicate", "ex2kamt", amount),
                Model("duplicate", "ex2kamt", volume),
            ],
            ["20170103"],
        )
