# Draco Model Documentation

`draco-model` is a Polars-backed factor DAG library. Public fields are expanded into explicit graph nodes such as `Source`, `Where`, `Op`, `Aggregate`, `Join`, `Project`, and `FillNull`.

The documentation is split into a narrative user guide and a Sphinx-style API reference.

## User Guide

- [Quickstart](user-guide/quickstart.md)
- [Data Sources](user-guide/data-sources.md)
- [Aggregation](user-guide/aggregation.md)
- [Rolling Operators](user-guide/rolling.md)
- [Payload Semantics](user-guide/payload.md)
- [Debugging](user-guide/debugging.md)

## API Reference

- [API Overview](api/index.md)
- [Engine](api/engine.md)
- [Model, Node, and Graph Metadata](api/model.md)
- [Recipes and Shortcuts](api/recipes.md)
- [Source](api/layers/source.md)
- [Operators](api/layers/operators.md)
- [Filters](api/layers/filters.md)
- [Aggregate](api/layers/aggregate.md)
- [FillNull](api/layers/transforms.md)
- [Join and Project](api/layers/combine.md)

## Design Contracts

- `Engine.collect()` accepts only daily output with `(date, secu_code)` keys and a public `value` column.
- Rolling `window` is a row/bar count at the input grain, not a calendar-day count.
- `Source(..., lookback_days=...)` is explicit; rolling operators do not automatically expand source lookback.
- Internal payload is retained unless a layer explicitly drops it with `Project()`.
- Fixed source schemas are treated as checked data contracts.
