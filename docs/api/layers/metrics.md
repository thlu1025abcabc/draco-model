# Metric

```python
Metric(name: str, source: Node, *, alias: str | None = None) -> Node
```

Expand a named market metric into its semantic DAG recipe.

## Parameters

| Name | Type | Description |
|---|---|---|
| `name` | `str` | Metric name. |
| `source` | `Node` | Input source or frame node. |
| `alias` | `str | None` | Optional public output alias. |

## Supported Metrics

| Metric | Expansion |
|---|---|
| `volume` | `Col("volume") -> Aggregate("1m", "sum")` |
| `no` | `Col("no") -> Aggregate("1m", "sum")` |
| `amount` | `Col("price") * Col("volume") -> Aggregate("1m", "sum")` |
| `buyamount` | `Where(Side("buy")) -> price * volume -> Aggregate("1m", "sum")` |
| `sellamount` | `Where(Side("sell")) -> price * volume -> Aggregate("1m", "sum")` |
| `vwap` | `Metric("amount") / Metric("volume")` |
| `open` | price rows filtered by `is_first`, then first aggregation |
| `close` | price rows filtered by `is_last`, then last aggregation |
| `high` | max price aggregation |
| `low` | min price aggregation |
| `preclose` | reserved; use with `FillNull("state")` |

## Notes

`amount` always uses `price * volume`, even if the raw source contains an `amount` column.

`preclose` cannot be evaluated directly:

```python
preclose = FillNull("state")(Metric("preclose", raw))
```

## Examples

```python
raw = Source("trades_tbar")

volume = Metric("volume", raw)
vwap = Metric("vwap", raw, alias="minute_vwap")
close = Metric("close", raw)
```
