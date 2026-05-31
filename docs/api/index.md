# API Reference

This section documents public APIs exported from `draco_model` and `draco_model.layers`.

## Core Runtime

- [Engine](engine.md)
- [Model, Node, and Graph Metadata](model.md)

## Layers

- [Source](layers/source.md)
- [Metric](layers/metrics.md)
- [Operators](layers/operators.md)
- [Filters](layers/filters.md)
- [Aggregate](layers/aggregate.md)
- [FillNull](layers/transforms.md)
- [Join and Project](layers/combine.md)

## Conventions

All layer constructors return or produce `Node` objects. A `Node` describes a lazy frame computation, not an eager dataframe.

Most layers follow this shape:

```python
output_node = Layer(...)(input_node)
```

`Source(...)`, `Metric(...)`, `Op(...)`, and `Col(...)` are convenience APIs that create or expand graph nodes directly.
