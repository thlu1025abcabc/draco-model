from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model, profile_plan
from draco_model.layers import Aggregate, Source
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
