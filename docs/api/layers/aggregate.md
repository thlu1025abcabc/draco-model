# Aggregate

```python
Aggregate(
    frequency: str,
    agg: str,
    *,
    apply_to: str = "field",
    value_col: str | None = None,
    alias: str | None = None,
    auction: str = "keep",
    name: str | None = None,
)
```

Aggregate raw, minute, or daily fields to a target frequency.

## Parameters

| Name | Type | Description |
|---|---|---|
| `frequency` | `str` | `"1m"`, minute frequency such as `"5m"`, `"1d"`, or `"daily"`. |
| `agg` | `str` | Aggregation method: `sum`, `mean`, `max`, `min`, `std`, `median`, `first`, or `last`. |
| `apply_to` | `str` | `"field"` or `"components"`. |
| `value_col` | `str | None` | Public value column to aggregate. If omitted, aggregate public value columns. |
| `alias` | `str | None` | Optional output alias. Requires exactly one selected value column. |
| `auction` | `str` | `"keep"`, `"drop"`, or `"merge"`. |
| `name` | `str | None` | Optional display name. |

## Returns

Layer object. Call it with a frame node to produce a `Node`.

```python
out = Aggregate("1d", "last", value_col="close", alias="value")(close)
```

## Notes

- `sum` is null-safe: all-null groups remain null.
- `auction="merge"` maps auction minutes to the first and last non-auction target bars.
- Daily aggregation applies auction handling before daily grouping.
- `apply_to="field"` aggregates public fields directly, then applies public projection so payload columns are dropped.
- `apply_to="components"` is appropriate for ratio-like fields whose numerator and denominator should be aggregated before recomputing the ratio. It is not supported after `FillNull()`; filled fields should use `apply_to="field"`.
- If `value_col` is an identity column, pass a non-key `alias`; aggregate outputs cannot reuse identity column names as public values.

## Raises

| Error | Condition |
|---|---|
| `ValueError` | `apply_to` is not `"field"` or `"components"`. |
| `ValueError` | `auction` is not `"keep"`, `"drop"`, or `"merge"`. |
| `ValueError` | `alias` is used while aggregating multiple value columns. |
| `ValueError` | The output column would conflict with an identity column name. |
