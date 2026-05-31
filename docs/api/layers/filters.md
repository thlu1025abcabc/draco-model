# Filters

Filters use a `Condition` and `Where` to keep matching rows.

## `Where`

```python
Where(condition: Condition, *, name: str | None = None)
```

Apply a condition to a frame.

```python
filtered = Where(Side("buy"))(raw)
```

## Conditions

### `Side`

```python
Side(side: str)
```

Semantic side filter. `side` must be `"buy"` or `"sell"`.

Execution maps these to the normalized `side` codes used by the current data source.

### `Flag`

```python
Flag(column: str)
```

Boolean column condition used by metric recipes, such as `is_first` and `is_last`.

### `Threshold`

```python
Threshold(column: str, *, op: str = ">", value: Any)
```

Compare one column with a literal threshold. Supported operators are `>`, `>=`, `<`, `<=`, `==`, `=`, `!=`, and `<>`.

### `TopQuantile`

```python
TopQuantile(column: str, *, q: float, over: list[str] | tuple[str, ...])
```

Keep rows whose column value is at or above the group quantile.

## Examples

```python
buy_rows = Where(Side("buy"))(raw)
large_rows = Where(Threshold("volume", op=">", value=1000))(raw)
top_volume = Where(TopQuantile("volume", q=0.8, over=["date", "secu_code"]))(features)
```
