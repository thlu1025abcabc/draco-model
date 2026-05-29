from __future__ import annotations

from draco_model import Model, Node
from draco_model.layers import Auction, Field, Fill, Input, Resample


def test_layers_build_static_dag_and_reuse_input_node() -> None:
    raw = Input(source="trades_tbar")
    close = Field("close")(raw)
    auctioned = Auction("drop")(close)
    resampled = Resample("5m", "last")(auctioned)
    model = Model(name="close_probe", universe="ex2kamt", output=resampled)

    nodes = model.nodes()
    assert [node.op for node in nodes] == ["input", "field", "auction", "resample"]
    assert len({node.id for node in nodes}) == 4
    assert nodes[1].inputs["input"] is raw


def test_mermaid_includes_graph_layers() -> None:
    output = Fill("state")(Field("close")(Input(source="trades_tbar")))
    model = Model(name="close_state", universe="ex2kamt", output=output)

    mermaid = model.explain_mermaid()
    assert "input" in mermaid
    assert "field" in mermaid
    assert "fill" in mermaid


def test_mermaid_uses_explicit_node_names() -> None:
    output = Resample("5m", "last", name="close_5m")(
        Field("close", name="close_1m")(
            Input(source="trades_tbar", name="raw_trades")
        )
    )
    model = Model(name="named_graph", universe="ex2kamt", output=output)

    mermaid = model.explain_mermaid()

    assert "raw_trades" in mermaid
    assert "close_1m" in mermaid
    assert "close_5m" in mermaid


def test_mermaid_uses_stable_default_node_names() -> None:
    output = Fill("state")(Field("close")(Input(source="trades_tbar")))
    model = Model(name="auto_named_graph", universe="ex2kamt", output=output)

    mermaid = model.explain_mermaid()

    assert "input_0" in mermaid
    assert "field_0" in mermaid
    assert "fill_0" in mermaid


def test_transform_layers_build_plain_nodes() -> None:
    close = Field("close")(Input(source="trades_tbar", lookback_days=3))

    auctioned = Auction("drop")(close)
    resampled = Resample("5m", "last")(close)
    filled = Fill("state")(close)

    assert auctioned.op == "auction"
    assert auctioned.params == {"mode": "drop"}
    assert auctioned.inputs["input"] is close
    assert resampled.op == "resample"
    assert resampled.params["frequency"] == "5m"
    assert resampled.params["agg"] == "last"
    assert filled.op == "fill"
    assert filled.params["value"] == "state"
    assert set(filled.inputs) == {"input", "close_state"}
    assert filled.inputs["input"] is close
    assert filled.inputs["close_state"].id == close.id


def test_fill_state_builds_close_state_subtree_for_non_close_field() -> None:
    high = Resample("5m", "max")(Field("high")(Input(source="trades_tbar")))
    filled = Fill("state")(high)
    close_state = filled.inputs["close_state"]

    assert close_state.op == "resample"
    assert close_state.params == {"frequency": "5m", "agg": "last"}
    assert close_state.inputs["input"].op == "field"
    assert close_state.inputs["input"].params == {"name": "close"}


def test_nested_transform_layers_build_expected_graph_order() -> None:
    layered = Fill("state")(
        Resample("5m", "last")(Auction("drop")(Field("close")(Input(source="trades_tbar"))))
    )

    assert [node.op for node in Model(name="layered", universe="ex2kamt", output=layered).nodes()] == [
        "input",
        "field",
        "auction",
        "resample",
        "fill",
    ]


def test_structural_node_ids_are_stable_for_same_graph() -> None:
    left = Resample("5m", "last")(Field("close")(Input(source="trades_tbar")))
    right = Resample("5m", "last")(Field("close")(Input(source="trades_tbar")))

    assert left.id == right.id
    assert [node.id for node in Model("left", "ex2kamt", left).nodes()] == [
        node.id for node in Model("right", "ex2kamt", right).nodes()
    ]


def test_structural_node_ids_change_when_params_change() -> None:
    last = Resample("5m", "last")(Field("close")(Input(source="trades_tbar")))
    first = Resample("5m", "first")(Field("close")(Input(source="trades_tbar")))

    assert last.id != first.id


def test_structural_node_ids_sort_named_inputs() -> None:
    close = Field("close")(Input(source="trades_tbar"))
    volume = Field("volume")(Input(source="trades_tbar"))

    first = Node(kind="frame", op="pair", inputs={"close": close, "volume": volume})
    second = Node(kind="frame", op="pair", inputs={"volume": volume, "close": close})
    swapped = Node(kind="frame", op="pair", inputs={"close": volume, "volume": close})

    assert first.id == second.id
    assert first.id != swapped.id


def test_explicit_node_id_is_preserved() -> None:
    node = Node(kind="frame", op="manual", id="manual_id")

    assert node.id == "manual_id"


def test_node_name_does_not_change_structural_id() -> None:
    named = Field("close", name="named_close")(Input(source="trades_tbar", name="named_input"))
    unnamed = Field("close")(Input(source="trades_tbar"))

    assert named.id == unnamed.id
