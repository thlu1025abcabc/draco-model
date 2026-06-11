# Join and Project

## `Join`

```python
Join(how="full", on=None)(inputs: Mapping[str, Node]) -> Node
```

Horizontally align multiple frames by key columns.

```python
features = Join()({
    "close": Metric("close", raw),
    "volume": Metric("volume", raw),
})
```

## Join Semantics

`how="full"` is the default and performs ordered SQL full joins:

- `on=None` uses the identity-key intersection for each join step.
- Explicit `on=(...)` joins on those columns.
- `on="identity_intersection"` joins each step on the current left identity and right identity intersection.
- Output identity keys are the ordered union of all input identity keys.
- Output rows are sorted by the output identity keys.
- Mixed daily and minute/raw identity inputs are rejected for full joins. Use `Join(how="left", on=("date", "secu_code"))` when one side is daily and the other side is minute/raw, so the anchoring policy is explicit.

`how="left"` performs ordered left joins:

- `on=None` uses the first input's identity keys as join keys.
- Explicit `on=(...)` joins on those columns.
- `on="identity_intersection"` joins each step on the current left identity and right identity intersection.
- Every right input must contain the join columns.
- Output identity keys are the ordered union of all input identity keys.
- Output rows are sorted by the output identity keys.

For every pairwise join step, join columns must be identity columns in both inputs and must include all identity columns shared by those two inputs. For example, two raw sources with identity `(date, secu_code, minute, price, side)` cannot be joined only on `(date, secu_code, minute)`, because `price` and `side` would be silently mismatched. After input-name prefixing, overlapping non-join physical columns raise instead of relying on Polars suffixes.

When `how="left"` is used with mixed grain inputs, pass `on` explicitly if the left identity contains columns the daily input does not have. For example, minute-left plus daily-right should usually use `on=("date", "secu_code")`.

If a workflow combines several minute features plus daily features, join the minute features first, then left-join the daily features on `(date, secu_code)`. A single multi-input `Join(how="left", on=("date", "secu_code"))` would also apply that daily key to the minute-minute join steps, which is rejected because those steps share `minute`.

Payload columns are retained with input-name prefixes.

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
