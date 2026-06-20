from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model as PublicModel, profile_plan
from draco_model.core import Node
from draco_model.layers import Aggregate, Source
from draco_model.recipes import metric
from draco_model.runtime.profiling import Profiler


def Model(name: str, universe: str, output: Node | dict[str, Node]) -> PublicModel:
    """Test helper for single-output model setup under the dict-only public API."""
    if isinstance(output, Node):
        output = {"value": output}
    return PublicModel(name, universe, output)


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


def test_profile_plan_counts_shared_nodes_across_named_outputs() -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    sum_daily = Aggregate("1d", "sum", value_col="amount", alias="sum")(amount)
    mean_daily = Aggregate("1d", "mean", value_col="amount", alias="mean")(amount)

    profile = profile_plan(
        [
            Model(
                "amount_stats",
                "ex2kamt",
                {"sum": sum_daily, "mean": mean_daily},
            )
        ]
    )

    amount_nodes = [
        node
        for node in profile.nodes
        if node.op == "aggregate" and node.params.get("alias") == "amount"
    ]
    assert len(amount_nodes) == 1
    assert amount_nodes[0].ref_count == 2
    assert amount_nodes[0].model_count == 1
    assert amount_nodes[0].cache_candidate


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
    ).sort(["factor_name", "date", "secu_code"])

    assert result.to_dict(as_series=False) == {
        "date": ["20170103", "20170103", "20170103", "20170103"],
        "secu_code": [1, 2, 1, 2],
        "factor_name": ["daily_amount", "daily_amount", "daily_volume", "daily_volume"],
        "value": [880.0, None, 85.0, None],
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


def test_collect_many_concats_named_outputs(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = Aggregate("1d", "sum", value_col="amount", alias="amount")(metric("amount")(raw))
    volume = Aggregate("1d", "sum", value_col="volume", alias="volume")(metric("volume")(raw))
    engine = Engine(data_root=sample_root)

    result = engine.collect_many(
        [Model("trade_totals", "ex2kamt", {"amount": amount, "volume": volume})],
        ["20170103"],
    )

    assert result.to_dict(as_series=False) == {
        "date": ["20170103", "20170103", "20170103", "20170103"],
        "secu_code": [1, 2, 1, 2],
        "factor_name": [
            "trade_totals__amount",
            "trade_totals__amount",
            "trade_totals__volume",
            "trade_totals__volume",
        ],
        "value": [880.0, None, 85.0, None],
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


def test_empty_profiles_carry_typed_schema() -> None:
    plan_frame = profile_plan([]).to_frame()
    assert plan_frame.height == 0
    assert plan_frame.schema["node_id"] == pl.Utf8
    assert plan_frame.schema["ref_count"] == pl.Int64
    assert plan_frame.schema["models"] == pl.List(pl.Utf8)
    assert plan_frame.schema["cache_candidate"] == pl.Boolean

    event_frame = Profiler().to_frame()
    assert event_frame.height == 0
    assert event_frame.schema["event"] == pl.Utf8
    assert event_frame.schema["at_ms"] == pl.Float64
    assert event_frame.schema["node_id"] == pl.Utf8


def test_collect_many_matches_single_collect(sample_root: Path) -> None:
    raw = Source("trades_tbar")
    amount = metric("amount")(raw)
    volume = metric("volume")(raw)
    m1 = Model("ratio", "ex2kamt", Aggregate("1d", "last", value_col="ratio", alias="value")((amount / volume).alias("ratio")))
    m2 = Model("amount", "ex2kamt", Aggregate("1d", "sum", value_col="amount", alias="value")(amount))
    dates = ["20170103"]

    single = pl.concat(
        [
            Engine(data_root=sample_root).collect(m1, dates),
            Engine(data_root=sample_root).collect(m2, dates),
        ]
    ).sort(["factor_name", "date", "secu_code"])
    batch = Engine(data_root=sample_root).collect_many([m1, m2], dates).sort(["factor_name", "date", "secu_code"])

    assert single.equals(batch)
