# Payload Semantics

Payload columns are internal columns whose names start with `__`. They carry operator components and lineage/debug state across the DAG.

## Default Behavior

Unless a layer explicitly drops payload, payload columns are preserved.

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

`FillNull()` preserves payload columns, but the filled public field is no longer marked as fully recomputable from its old components. After fill, the public column is the authoritative field value.

For `FillNull("state")`, close-state is built from the field's source lineage and aggregate path, then aligned to the frame keys being filled. This keeps state fill aligned with prior resampling and explicit grid rows, and rejects multi-source fields whose close-state source would be ambiguous.

## Aggregate

`Aggregate(apply_to="field")` aggregates the public output and retains payload at the target grain for lineage/debugging. Retained payload does not promise that the aggregated public field can be recomputed from it.

`Aggregate(apply_to="components")` aggregates operator components and recomputes the public output. Use this when the operator has meaningful component aggregation semantics.

## Naming Rules

Public aliases and `Join()` input names must not start with `__` and must not use key column names: `date`, `secu_code`, or `minute`.
