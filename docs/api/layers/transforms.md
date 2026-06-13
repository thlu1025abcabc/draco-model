# Transform Layers

## Grid

```python
Grid(frequency: str | None = None, *, auction: str | None = None, name: str | None = None)
```

Align a frame to an explicit universe-by-minute grid.

`Grid()` is an API wrapper around a system grid source and `Join(how="left")`. It builds `Model.universe × minute calendar` for the evaluation dates implied by field lookback metadata or source lookback, then left-joins the input frame onto that grid. Missing public and payload values remain null.

When `frequency` is omitted, `Grid()` infers the target minute frequency from field `grain_path`; raw sources fall back to the full 1m calendar. If any minute step in that path used `auction="drop"` or `auction="merge"`, the inferred grid excludes auction bars because later resampling cannot recreate removed auction keys. Frames with mixed inferred grid policies raise an error; pass `frequency=...` or `auction=...` explicitly to override inference.

Minute or raw frames join on `(date, secu_code, minute)`. Daily frames join on `(date, secu_code)`, broadcasting daily values to every minute in the grid. It is explicit: `Source(...)` still scans the source as-is unless wrapped by `Grid()`.

Grid determines join keys from frame identity keys, not from ordinary columns. If a value column would collide with the output grid identity, for example a daily value column named `minute`, `Grid()` raises; alias that value before gridding.

`Grid()` is not sticky. It guarantees the row set of the frame it returns, but later layers are still allowed to change that row set. For example, `Where(...)` can filter grid-created rows and `Aggregate(...)` only groups rows that survive upstream filters. For flag-based metrics such as `Metric("close")` and `Metric("open")`, gridding the raw source first does not keep missing minutes: grid-created rows have null `is_last` / `is_first`, and those null flags are treated as false by the metric filter.

To produce a complete minute panel where missing close/open bars remain as null values, grid the metric output:

```python
close_panel = Grid()(Metric("close", Source("trades_tbar")))
```

Use `Metric("close", Grid()(Source("trades_tbar")))` only when you intentionally want `close` to be computed from real `is_last=True` rows and are fine with missing minutes disappearing before the final output.

## FillNull

```python
FillNull(value: int | float | str = "state", *, name: str | None = None)
```

Fill nulls in a single public field.

## Parameters

| Name | Type | Description |
|---|---|---|
| `value` | `int | float | str` | Numeric fill value, `"ffill"`, or `"state"`. |
| `name` | `str | None` | Optional display name. |

## Modes

| Mode | Behavior |
|---|---|
| Numeric literal | Replace nulls with that value. |
| `"ffill"` | Forward-fill over `(date, secu_code)`. |
| `"state"` | Fill from close-state logic. |

## State Fill

`FillNull("state")` builds or evaluates a close-state frame. For ordinary fields, nulls are filled with forward-filled close, falling back to `daily_k.preclose`.

The close-state frame is built from the field's recorded `FieldInfo.source` and `grain_path`, then aligned to the frame keys being filled. A field that has been resampled before fill uses close at the same target grain, and explicit grid rows can receive forward-filled state values. If a field combines multiple sources and no single `FieldInfo.source` is available, state fill raises a clear error instead of choosing a source implicitly.

For reserved `Metric("preclose")`, state fill produces previous close semantics:

```python
preclose = FillNull("state")(Metric("preclose", raw))
```

## Notes

The input frame must contain exactly one public value column.

`FillNull()` drops old payload columns. A filled public field is no longer eligible for `Aggregate(apply_to="components")`; use `apply_to="field"` for later aggregation.
