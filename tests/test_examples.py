from __future__ import annotations

from examples.close_last import model


def test_close_last_example_builds_current_dag() -> None:
    assert model.name == "close_last"
    assert [node.op for node in model.nodes()] == ["input", "field", "auction", "daily_agg"]
