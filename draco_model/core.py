from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Node:
    """Immutable DAG node used by layers, conditions, and model outputs."""

    kind: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, "Node"] = field(default_factory=dict)
    id: str = ""
    name: str | None = None

    def __post_init__(self) -> None:
        """Assign a deterministic id from node structure when none is provided."""
        if not self.id:
            object.__setattr__(self, "id", _structural_id(self.kind, self.op, self.params, self.inputs))

    def __hash__(self) -> int:
        return hash(self.id)

    def alias(self, name: str) -> "Node":
        """Return this frame node with a new public output alias where supported."""
        from draco_model.layers.operators import alias_node

        return alias_node(self, name)

    def __add__(self, other: Any) -> "Node":
        from draco_model.layers.operators import Op

        return Op("add", self, other)

    def __radd__(self, other: Any) -> "Node":
        from draco_model.layers.operators import Op

        return Op("add", other, self)

    def __sub__(self, other: Any) -> "Node":
        from draco_model.layers.operators import Op

        return Op("sub", self, other)

    def __rsub__(self, other: Any) -> "Node":
        from draco_model.layers.operators import Op

        return Op("sub", other, self)

    def __mul__(self, other: Any) -> "Node":
        from draco_model.layers.operators import Op

        return Op("mul", self, other)

    def __rmul__(self, other: Any) -> "Node":
        from draco_model.layers.operators import Op

        return Op("mul", other, self)

    def __truediv__(self, other: Any) -> "Node":
        from draco_model.layers.operators import Op

        return Op("div", self, other)

    def __rtruediv__(self, other: Any) -> "Node":
        from draco_model.layers.operators import Op

        return Op("div", other, self)


class Layer:
    """Base class for graph layers that turn input nodes into output nodes."""

    op: str = "layer"
    output_kind: str = "frame"

    def __init__(self, *, name: str | None = None, **params: Any) -> None:
        """Store non-null layer parameters on the graph node."""
        self.name = name
        self.params = {key: value for key, value in params.items() if value is not None}

    def __call__(self, inputs: Node | Mapping[str, Node]) -> Node:
        """Build a node connected to one input node or a named input mapping."""
        return Node(
            kind=self.output_kind,
            op=self.op,
            params=dict(self.params),
            inputs=_normalize_inputs(inputs),
            name=self.name,
        )


@dataclass(frozen=True)
class Condition:
    """Boolean expression descriptor used by filtering layers."""

    op: str
    params: dict[str, Any]

    def to_node(self, frame: Node) -> Node:
        """Attach this condition to the frame it will be evaluated against."""
        return Node(kind="condition", op=self.op, params=dict(self.params), inputs={"frame": frame})


class Model:
    """Named factor graph rooted at a frame node."""

    def __init__(self, name: str, universe: str, output: Node) -> None:
        """Create a model for one universe and one frame output node."""
        if output.kind != "frame":
            raise ValueError("Model output must be a frame node.")
        self.name = name
        self.universe = universe
        self.output = output

    def nodes(self) -> list[Node]:
        """Return graph nodes in dependency-first topological order."""
        return _topological_nodes(self.output)

    def explain_mermaid(self) -> str:
        """Render the model DAG as a Mermaid flowchart."""
        nodes = self.nodes()
        names = resolve_node_names(nodes)
        aliases = {node.id: f"n{idx}" for idx, node in enumerate(nodes)}
        lines = ["flowchart TD"]
        for node in nodes:
            lines.append(f"  {aliases[node.id]}[\"{_escape_mermaid(_node_label(node, names[node.id]))}\"]")
        for node in nodes:
            for input_name, parent in node.inputs.items():
                lines.append(
                    f"  {aliases[parent.id]} -->|\"{_escape_mermaid(input_name)}\"| {aliases[node.id]}"
                )
        return "\n".join(lines)


def _normalize_inputs(inputs: Node | Mapping[str, Node]) -> dict[str, Node]:
    if isinstance(inputs, Node):
        return {"input": inputs}
    normalized = dict(inputs)
    if not normalized:
        raise ValueError("Layer inputs cannot be empty.")
    for key, value in normalized.items():
        if not isinstance(value, Node):
            raise TypeError(f"Layer input {key!r} must be a Node.")
    return normalized


def _topological_nodes(output: Node) -> list[Node]:
    out: list[Node] = []
    visited: set[str] = set()

    def visit(node: Node) -> None:
        if node.id in visited:
            return
        visited.add(node.id)
        for parent in node.inputs.values():
            visit(parent)
        out.append(node)

    visit(output)
    return out


def resolve_node_names(nodes: list[Node]) -> dict[str, str]:
    """Return stable display names for nodes, using explicit names when present."""
    out: dict[str, str] = {}
    seen: set[str] = set()
    counters: dict[str, int] = {}
    for node in nodes:
        display = node.name or _default_node_name(node, counters)
        if display in seen:
            raise ValueError(f"Duplicate node name {display!r}.")
        seen.add(display)
        out[node.id] = display
    return out


def _node_label(node: Node, display_name: str) -> str:
    params = json.dumps(node.params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    head = f"{display_name}\\n{node.kind}: {node.op}"
    return f"{head}\\n{params}" if params != "{}" else head


def _escape_mermaid(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', "'").replace("\n", "<br/>")


def _structural_id(kind: str, op: str, params: dict[str, Any], inputs: dict[str, Node]) -> str:
    payload = {
        "kind": kind,
        "op": op,
        "params": _normalize_id_value(params),
        "inputs": [(name, child.id) for name, child in sorted(inputs.items())],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2b(encoded, digest_size=16).hexdigest()
    return f"n_{digest}"


def _normalize_id_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_id_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_id_value(item) for key, item in sorted(value.items())}
    raise TypeError(f"Node params contain non-serializable value for structural id: {value!r}.")


def _default_node_name(node: Node, counters: dict[str, int]) -> str:
    base = _node_name_base(node)
    index = counters.get(base, 0)
    counters[base] = index + 1
    return f"{base}_{index}"


def _node_name_base(node: Node) -> str:
    return _name_part(node.op)


def _name_part(value: Any) -> str:
    text = str(value).strip().lower()
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    cleaned = "_".join(part for part in "".join(out).split("_") if part)
    return cleaned or "node"
