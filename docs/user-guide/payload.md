# Payload Semantics

Payload columns are internal columns whose names start with `__`. They carry operator components and debug metadata across the DAG.

## Default Behavior

Unless a layer explicitly drops payload, payload columns are preserved. `Aggregate(apply_to="field")` and `FillNull()` are explicit drop points: they project the result to key columns and public fields.

`Project()` is the explicit public-field projection layer:

```python
from draco_model.layers import Project

public_only = Project()(frame)
```

## Operator Components

Arithmetic and rolling operators can produce internal component columns. For a vwap-like field:

```python
vwap = (Metric("amount", raw) / Metric("volume", raw)).alias("vwap")
```

The public field is `vwap`, while internal payload stores the operands needed for component aggregation.

## FillNull

`FillNull()` drops old payload columns. After fill, the public column is the authoritative field value and no longer supports `Aggregate(apply_to="components")`; use `apply_to="field"` for later aggregation.

For `FillNull("state")`, close-state is built from the field's `FieldInfo.source` and `grain_path`, then aligned to the frame keys being filled. This keeps state fill aligned with prior resampling and explicit grid rows, and rejects multi-source fields whose close-state source would be ambiguous.

## Aggregate

`Aggregate(apply_to="field")` aggregates the public output and drops payload at the target grain.

`Aggregate(apply_to="components")` aggregates operator components and recomputes the public output. Use this when the operator has meaningful component aggregation semantics.

## Naming Rules

Public aliases and `Join()` input names must not start with `__` and must not use key column names: `date`, `secu_code`, or `minute`.
