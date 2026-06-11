# Aggregation

`Aggregate` handles raw-to-minute aggregation, minute resampling, daily aggregation, and auction handling.

```python
from draco_model.layers import Aggregate, Metric, Source

raw = Source("trades_tbar")
volume = Metric("volume", raw)

volume_5m = Aggregate("5m", "sum", value_col="volume")(volume)
daily_value = Aggregate("1d", "sum", value_col="volume", alias="value")(volume)
```

## Frequencies

| Frequency | Output keys | Meaning |
|---|---|---|
| `"1m"` | `(date, secu_code, minute)` | Aggregate raw rows into minute bars. |
| `"5m"`, `"15m"` | `(date, secu_code, minute)` | Bucket continuous minutes by the minute calendar. |
| `"1d"`, `"daily"` | `(date, secu_code)` | Aggregate to daily grain. |

## Aggregation Methods

Supported methods are `sum`, `mean`, `max`, `min`, `std`, `median`, `first`, and `last`.

`sum` is null-safe: a group that is entirely null remains null instead of becoming `0`.

## Auction Policy

| `auction` | Behavior |
|---|---|
| `"keep"` | Keep auction bars as normal rows. |
| `"drop"` | Drop auction minutes. |
| `"merge"` | Merge auction minutes into the first and last non-auction target bars for the output frequency. |

For example, at 1m frequency, opening auction `925` merges into `930`, and closing auction `1500` merges into `1456`. At 5m frequency, closing auction `1500` merges into `1455`.

Daily aggregation applies the auction policy before daily grouping.

## `apply_to`

`apply_to="field"` aggregates the public field directly and drops payload by applying the public projection after aggregation.

`apply_to="components"` aggregates operator components separately, then recomputes the public output. This requires aggregatable components and is not available after `FillNull()`. It is important for ratio-like fields such as vwap:

```python
amount = Metric("amount", raw)
volume = Metric("volume", raw)
vwap = (amount / volume).alias("vwap")

vwap_5m = Aggregate("5m", "sum", apply_to="components")(vwap)
```
