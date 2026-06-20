# Level-2 Bars

```python
TradesWithWaitBar(*, name: str | None = None)
```

Construct minute/price/side trade bars by matching `steptrades` with `steporders`.

## Input

Call the layer with two nodes:

```python
bars = TradesWithWaitBar()(trades, orders)
```

- `trades` must provide the fixed `steptrades` columns.
- `orders` must provide the fixed `steporders` columns.

Inactive order types `-1` and `-11` are excluded. Wait time is measured from the matched order to the trade, with the lunch break removed. Negative waits become null.

## Output

The output identity is `(date, secu_code, minute, price, side)` and the columns are:

```text
date, secu_code, minute, price, side, volume,
vw_wait_time, is_first, is_last, no
```

The result is a normal frame node. It can be consumed by existing recipes and layers or exposed directly as a named model output.

## Example

```python
from draco_model import Engine, Model
from draco_model.layers import Source, TradesWithWaitBar

trades = Source("steptrades")
orders = Source("steporders")
bars = TradesWithWaitBar()(trades, orders)
model = Model("trades_wtminbar", None, {"trades_wtminbar": bars})

outputs = Engine(data_root="data").evaluate_outputs(model, "20260618")
trades_wtminbar = outputs["trades_wtminbar"]
```

`evaluate_outputs()` preserves output names and grain and returns `LazyFrame` values. Storage and transport remain the caller's responsibility.

The model name and output name are independent identifiers. `universe=None` declares that this dataset output is not aligned to a stock universe.
