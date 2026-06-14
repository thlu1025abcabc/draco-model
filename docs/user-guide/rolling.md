# Rolling Operators

Rolling operators are created with `Op(...)`.

```python
from draco_model.recipes import metric
from draco_model.layers import Op, Source

raw = Source("trades_tbar", lookback_days=5)
amount = metric("amount")(raw)
volume = metric("volume")(raw)

corr = Op("rolling_corr", amount, volume, window=5, alias="corr_5")
cross_day = Op("rolling_corr", amount, volume, window=5, alias="corr_5_cross", cross_day=True)
```

## Window Unit

`window` is a row/bar count at the input grain. It is not a natural-day or trading-day count.

- Minute input: `window=5` means the latest five minute rows within the rolling group.
- Daily input: `window=5` means the latest five daily rows. If there is exactly one row per trading day, this behaves like five trading days.

## Grouping

For minute-grain inputs:

- `cross_day=False` is the default. Rolling groups by `(date, secu_code)` and resets each day.
- `cross_day=True` groups by `secu_code` and can use rows from previous trading days.

For daily-grain inputs, rolling groups by `secu_code`.

## Operands and Degenerate Windows

Rolling operators require exactly two frame (Node) operands. Passing `Col(...)` or scalar
operands raises at construction time.

When a window has zero variance (or float error makes the computed variance non-positive),
the output is null. `rolling_corr` results are clipped to `[-1, 1]`.

## Lookback

`Source(..., lookback_days=...)` is explicit. Rolling operators do not automatically increase source lookback based on `window`.

If you request a cross-day rolling window but scan only one day, the operator has only one day of rows available.
