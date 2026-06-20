# Recipes and Shortcuts

```python
metric(name: str, *, alias: str | None = None) -> MetricShortcut
last(minute: int) -> LastShortcut
transform(name: str, *, alias: str | None = None) -> TransformShortcut
```

Recipes are build-time helpers. They are not runtime `Layer` objects, do not create a single graph node, and do not register executors or info builders. A shortcut becomes part of the DAG only after it is applied to a `Node`.

```python
from draco_model.layers import Source
from draco_model.recipes import last, metric

raw = Source("trades_tbar")
close = metric("close")(raw)
late_raw = last(1400)(raw)
```

## Metric Shortcuts

| Metric | Expansion |
|---|---|
| `volume` | `Col("volume") -> Aggregate("1m", "sum")` |
| `no` | `Col("no") -> Aggregate("1m", "sum")` |
| `amount` | `Col("price") * Col("volume") -> Aggregate("1m", "sum")` |
| `buyamount` | `Where(Side("buy")) -> price * volume -> Aggregate("1m", "sum")` |
| `sellamount` | `Where(Side("sell")) -> price * volume -> Aggregate("1m", "sum")` |
| `vwap` | `metric("amount")(source) / metric("volume")(source)` |
| `open` | price rows filtered by `is_first`, then first aggregation |
| `close` | price rows filtered by `is_last`, then last aggregation |
| `high` | max price aggregation |
| `low` | min price aggregation |
| `preclose` | reserved; use with `FillNull("state")` |

`amount` always uses `price * volume`, even if the raw source contains an `amount` column.

`open` and `close` are flag-based shortcuts. They first filter rows by `is_first` or `is_last`, then aggregate the surviving prices. Null flags are treated as false. If the input frame was created by `Grid()(Source(...))`, missing source minutes have null `price`, `is_first`, and `is_last`; `metric("close")(gridded_source)` therefore filters those rows out instead of returning a null close row.

For a complete minute panel with null close/open values on missing bars, grid after the metric:

```python
close_panel = Grid()(metric("close")(raw))
```

`preclose` cannot be evaluated directly:

```python
preclose = FillNull("state")(metric("preclose")(raw))
```

## Filter Shortcuts

| Shortcut | Expansion |
|---|---|
| `last(minute)` | `Where(Threshold("minute", op=">=", value=minute))` |

`last(minute)` keeps rows whose `minute` is at or after the threshold for every stock and date.

## Transform Shortcuts

`transform(name)` is a placeholder for future build-time transform shortcuts. No built-in transforms are registered yet, so applying an unknown transform raises a clear `ValueError`.

## FactorRecipe

```python
class LiqMoments(FactorRecipe):
    def build(self) -> Model:
        raw = Source(self.source)
        liq = metric(self.liq_name)(raw)
        moment = transform(self.op_name)(liq)
        ranked = transform("rank")(moment)
        return Model(self.name, self.universe, {"value": ranked})
```

`FactorRecipe` is a user-extensible factor-family base class. It is intentionally not a fixed pipeline container. Parameter spaces, mutation, batch generation, and cartesian products are left for later design.
