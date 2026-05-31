# Operators

Operators create arithmetic, row-level expression, or rolling nodes.

## `Col`

```python
Col(name: str)
```

Reference a raw column inside the frame passed to the expression.

```python
row_amount = (Col("price") * Col("volume")).alias("amount")(raw)
```

## `Op`

```python
Op(name: str, *operands: Any, alias: str | None = None, **params: Any) -> Node | OpExpr
```

Create a generic operator.

## Supported Operators

| Operator | Notes |
|---|---|
| `add` | Arithmetic addition. |
| `sub` | Arithmetic subtraction. |
| `mul` | Arithmetic multiplication. |
| `div` | Division; division by zero yields null. |
| `rolling_corr` | Rolling correlation. Requires `window`. |
| `rolling_beta` | Rolling beta. Requires `window`. |
| `rolling_alpha` | Rolling alpha. Requires `window`. |

## Rolling Parameters

| Name | Type | Description |
|---|---|---|
| `window` | `int` | Positive row/bar window. Required for rolling operators. |
| `cross_day` | `bool` | For minute inputs, group by `secu_code` instead of `(date, secu_code)`. Defaults to `False`. |

## Examples

```python
amount = Metric("amount", raw)
volume = Metric("volume", raw)

vwap = (amount / volume).alias("vwap")
corr = Op("rolling_corr", amount, volume, window=5, alias="corr_5")
```

## Raises

| Error | Condition |
|---|---|
| `ValueError` | Unsupported operator name. |
| `ValueError` | Rolling operator missing a positive integer `window`. |
| `ValueError` | Frame-level `Op` mixes `Node` operands with `Col` expressions. |
