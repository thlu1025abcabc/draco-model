# Transform Layers

## Grid

```python
Grid(frequency: str | None = None, *, auction: str | None = None, name: str | None = None)
```

Align a raw or minute-grain frame to an explicit universe-by-minute grid.

`Grid()` left-joins the input frame onto `Model.universe × minute calendar` for the evaluation dates implied by field lookback metadata or source lookback. Missing public and payload values remain null.

When `frequency` is omitted, `Grid()` infers the target minute frequency from field `grain_path`; raw sources fall back to the full 1m calendar. If any minute step in that path used `auction="drop"` or `auction="merge"`, the inferred grid excludes auction bars because later resampling cannot recreate removed auction keys. Frames with mixed inferred grid policies raise an error; pass `frequency=...` or `auction=...` explicitly to override inference.

`Grid()` requires a frame with `(date, secu_code, minute)` keys. It is explicit: `Source(...)` still scans the source as-is unless wrapped by `Grid()`.

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

The close-state frame is built from the field's recorded source lineage and aggregate path, then aligned to the frame keys being filled. A field that has been resampled before fill uses close at the same target grain, and explicit grid rows can receive forward-filled state values. If a field combines multiple sources and no single source lineage is available, state fill raises a clear error instead of choosing a source implicitly.

For reserved `Metric("preclose")`, state fill produces previous close semantics:

```python
preclose = FillNull("state")(Metric("preclose", raw))
```

## Notes

The input frame must contain exactly one public value column.

`FillNull()` preserves payload columns, but the filled public field is no longer marked as recomputable from old components.
