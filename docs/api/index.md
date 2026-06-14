# API Reference

This section documents public APIs exported from `draco_model`, `draco_model.layers`, and `draco_model.recipes`.

## Core Runtime

- [Engine](engine.md)
- [Model, Node, and Graph Metadata](model.md)
- [Recipes and Shortcuts](recipes.md)

## Layers

- [Source](layers/source.md)
- [Operators](layers/operators.md)
- [Filters](layers/filters.md)
- [Aggregate](layers/aggregate.md)
- [Grid and FillNull](layers/transforms.md)
- [Join and Project](layers/combine.md)

## Conventions

Layer constructors return or produce `Node` objects. A `Node` describes a lazy frame computation, not an eager dataframe.

Most layers follow this shape:

```python
output_node = Layer(...)(input_node)
```

`Source(...)`, `Op(...)`, and `Col(...)` are convenience APIs that create or expand graph nodes directly.

`metric(...)` and `transform(...)` are build-time shortcuts, not runtime layers. They only become graph nodes after being applied to a `Node`.
