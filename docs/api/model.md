# Model, Node, and Graph Metadata

## `Model`

```python
Model(
    name: str,
    universe: str | None,
    output: dict[str, Node],
)
```

Named factor graph rooted at one or more frame outputs.

| Parameter | Description |
|---|---|
| `name` | Factor/model name. Used as `factor_name` for a single `{"value": node}` output and as the prefix for other named outputs. |
| `universe` | Universe source name, such as `"ex2kamt"`, or `None` for outputs that must not be universe-aligned. |
| `output` | Dictionary mapping output names to root frame nodes. Every output node must have `kind == "frame"`. |

`universe=None` is intended for dataset-like outputs such as market bars. Such models support `evaluate()`, `evaluate_outputs()`, and `trace()`. They cannot use `collect()`, `collect_many()`, or `Grid()`, because those operations require an explicit stock universe.

### Methods

```python
Model.nodes() -> list[Node]
```

Return graph nodes in dependency-first topological order.

```python
Model.explain_mermaid() -> str
```

Return a Mermaid flowchart of the model DAG.

## `Node`

```python
Node(
    kind: str,
    op: str,
    params: dict[str, Any] = {},
    inputs: dict[str, Node] = {},
    id: str = "",
    name: str | None = None,
)
```

Immutable DAG node used by layers and model outputs.

`id` is structural and deterministic. `name` is display metadata used by trace and Mermaid rendering; it does not participate in the structural id.

### Methods

```python
Node.alias(name: str) -> Node
```

Return a node exposing its single public field under a new alias where supported.

`Node` also supports arithmetic operators: `+`, `-`, `*`, and `/`.

## `TraceStep`

Returned by `Engine.trace()`.

| Field | Description |
|---|---|
| `index` | Step number in trace order. |
| `resolved_name` | Explicit or generated display name. |
| `node` | The frame node that was evaluated. |
| `frame` | Materialized Polars dataframe for that node. |
