# Debugging

The DAG is inspectable at the node level. This is useful because metric recipes expand into ordinary graph nodes.

## List Nodes

```python
for node in model.nodes():
    print(node.id, node.op, node.params)
```

Nodes are returned in dependency-first topological order.

## Trace Materialized Frames

```python
engine = Engine(data_root="data")

for step in engine.trace(model, "20170103"):
    print(step.index, step.resolved_name, step.node.op, step.frame.shape)
```

`trace()` materializes each frame node and returns `TraceStep` objects.

## Render Mermaid

```python
print(model.explain_mermaid())
```

The returned string is a Mermaid flowchart. It uses resolved display names, node kinds, operations, and node parameters.

## Common Errors

| Error | Meaning |
|---|---|
| `Engine.collect requires a daily output` | The model output is not daily grain. Use `Aggregate("1d", ..., alias="value")` or call `evaluate()`. |
| `requires a public 'value' column` | The daily output does not expose a public `value` field. |
| `must not start with '__'` | Public aliases cannot use the internal payload prefix. |
| `missing fixed schema columns` | A known source parquet file does not satisfy its fixed source contract. |

## Logging

Draco uses the Python standard library `logging` package. Library code emits logs but does not configure handlers, formatters, or global log levels.

Configure logging in your application:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
```

Use `INFO` to observe high-level runs:

- `collect.start`
- `collect.done`
- `trace.start`
- `trace.done`

Use `DEBUG` for node-level and layer-level details:

- source scans and schema contract checks
- engine cache hits and misses
- aggregate frequency, auction policy, and output layout
- fill-null mode and close-state construction
- join input grain and output layout

Data contract failures, such as missing fixed source columns or invalid minute bars, are logged at `ERROR` before raising.
