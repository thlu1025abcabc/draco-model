from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model, Node
from draco_model.layers import Filter, Threshold


def test_filter_does_not_leak_internal_condition_column(engine_data_root: Path) -> None:
    frame = Node(kind="frame", op="constant_test_frame", params={"value": 7})
    filtered = Filter(Threshold("value", op=">", value=5))(frame)
    model = Model(name="filtered_factor", universe="unused", output=filtered)

    result = Engine(data_root=engine_data_root).evaluate(model, filtered, "20170103").collect()

    assert result.columns == ["date", "secu_code", "value"]
    assert result["value"].to_list() == [7.0]
