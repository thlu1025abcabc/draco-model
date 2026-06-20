# Quickstart

This page shows the shortest path from a raw market source to a daily factor output.

## Build a Daily Factor

```python
from draco_model import Engine, Model
from draco_model.recipes import metric
from draco_model.layers import Aggregate, Source

raw = Source("trades_tbar")
close = metric("close")(raw)
output = Aggregate("1d", "last", value_col="close", alias="value")(close)

model = Model(name="close_last", universe="ex2kamt", output={"value": output})
df = Engine(data_root="data").collect(model, dates=["20170103"])
```

`metric("close")(raw)` expands into a real DAG. It is not a black-box field executor. You can inspect that DAG with `model.nodes()` or `model.explain_mermaid()`.

## Evaluate Intermediate Nodes

Use `Engine.evaluate()` when you want the output of any frame node, including minute-grain frames.

```python
engine = Engine(data_root="data")
minute_close = engine.evaluate(model, close, "20170103").collect()
```

Use `Engine.collect()` only for final factor output. Each model output must be daily grain with `(date, secu_code)` keys and exactly one public value column. `collect()` left joins each output to the model universe for each date, renames the output column to `value`, adds `factor_name`, and returns one long dataframe. Use dictionaries such as `{"amount": amount_node, "volume": volume_node}` when one model should emit multiple factors.

## Trace a Model

```python
for step in engine.trace(model, "20170103"):
    print(step.index, step.resolved_name, step.node.op, step.frame.shape)
```

Trace materializes each frame node in dependency order and is intended for debugging, validation, and visual inspection.
