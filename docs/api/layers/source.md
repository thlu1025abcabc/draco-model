# Source

```python
Source(source: str, *, lookback_days: int = 1, name: str | None = None) -> Node
```

Create a raw source frame node.

## Parameters

| Name | Type | Description |
|---|---|---|
| `source` | `str` | Source name under `data_root`, such as `"trades_tbar"` or `"daily_k"`. |
| `lookback_days` | `int` | Number of trading sessions to scan, including the evaluation date. Must be at least 1. |
| `name` | `str | None` | Optional display name for trace and Mermaid output. |

## Returns

`Node`

## Notes

- `Source` does not add an intraday grid.
- Known sources use fixed schemas for stable planning.
- Extra normalized columns are dropped according to the fixed source contract.
- Missing fixed-contract columns raise `ValueError`.

## Examples

```python
raw = Source("trades_tbar")
raw_5d = Source("trades_tbar", lookback_days=5)
daily = Source("daily_k")
```
