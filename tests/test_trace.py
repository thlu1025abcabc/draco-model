from __future__ import annotations

from pathlib import Path

import polars as pl

from draco_model import Engine, Model, Node, TraceStep
from draco_model.layers import Auction, Field, Filter, Input, Resample, Threshold


def test_trace_materializes_single_frame_node(engine_data_root: Path) -> None:
    output = Node(kind="frame", op="constant_test_frame", params={"value": 7})
    model = Model(name="constant_factor", universe="unused", output=output)

    steps = Engine(data_root=engine_data_root).trace(model, date="20170103")

    assert len(steps) == 1
    assert isinstance(steps[0], TraceStep)
    assert steps[0].index == 0
    assert steps[0].resolved_name == "constant_test_frame_0"
    assert steps[0].node is output
    assert steps[0].node.op == "constant_test_frame"
    assert steps[0].frame.to_dict(as_series=False) == {
        "date": ["20170103"],
        "secu_code": [1],
        "value": [7.0],
    }


def test_trace_materializes_price_chain_in_graph_order(sample_root: Path) -> None:
    output = Resample("5m", "last", name="close_5m")(
        Auction("drop", name="drop_auction")(
            Field("close", name="close_1m")(
                Input(source="trades_tbar", name="raw_trades")
            )
        )
    )
    model = Model(name="price_chain_probe", universe="ex2kamt", output=output)
    engine = Engine(data_root=sample_root)

    steps = engine.trace(model, date="20170103")
    evaluated = engine.evaluate(model, output, "20170103").collect().sort(["secu_code", "minute"])

    assert [step.index for step in steps] == [0, 1, 2, 3]
    assert [step.node.op for step in steps] == ["input", "field", "auction", "resample"]
    assert [step.resolved_name for step in steps] == ["raw_trades", "close_1m", "drop_auction", "close_5m"]
    assert steps[-1].frame.sort(["secu_code", "minute"]).to_dict(as_series=False) == evaluated.to_dict(as_series=False)


def test_trace_skips_condition_nodes_and_keeps_filter_output_clean(engine_data_root: Path) -> None:
    frame = Node(kind="frame", op="constant_test_frame", params={"value": 7})
    filtered = Filter(Threshold("value", op=">", value=5))(frame)
    model = Model(name="filtered_factor", universe="unused", output=filtered)

    steps = Engine(data_root=engine_data_root).trace(model, date="20170103")

    assert [step.node.op for step in steps] == ["constant_test_frame", "filter"]
    assert all(step.node.kind == "frame" for step in steps)
    assert steps[-1].frame.columns == ["date", "secu_code", "value"]
    assert steps[-1].frame.filter(pl.col("value") == 7.0).height == 1
