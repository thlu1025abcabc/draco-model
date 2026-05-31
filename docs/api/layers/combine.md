# Join and Project

## `Join`

```python
Join()(inputs: Mapping[str, Node]) -> Node
```

Horizontally align multiple frames by key columns.

```python
features = Join()({
    "close": Metric("close", raw),
    "volume": Metric("volume", raw),
})
```

## Join Semantics

- Minute inputs use `(date, secu_code, minute)` keys.
- Daily inputs use `(date, secu_code)` keys.
- If any input is minute grain, the output is minute grain.
- Daily inputs can be joined onto minute outputs by `(date, secu_code)`.
- Payload columns are retained with input-name prefixes.

Join input names are public names. They must not start with `__` and must not be key column names.

## `Project`

```python
Project()(input: Node) -> Node
```

Keep only key columns and public fields. This is the explicit way to drop internal payload.

```python
public_features = Project()(features)
```

## Examples

```python
joined = Join()({
    "vwap": Metric("vwap", raw),
    "close": Metric("close", raw),
})

public = Project()(joined)
```
