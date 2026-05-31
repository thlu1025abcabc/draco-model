# FillNull

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

For reserved `Metric("preclose")`, state fill produces previous close semantics:

```python
preclose = FillNull("state")(Metric("preclose", raw))
```

## Notes

The input frame must contain exactly one public value column.

`FillNull()` preserves payload columns, but the filled public field is no longer marked as recomputable from old components.
