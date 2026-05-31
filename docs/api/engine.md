# Engine

```python
Engine(
    data_root: str | Path = "data",
    *,
    minute_calendar: MinuteCalendar | None = None,
    trading_calendar: TradingCalendar | None = None,
)
```

Evaluate model DAGs against local parquet data.

## Parameters

| Name | Type | Description |
|---|---|---|
| `data_root` | `str | Path` | Root directory containing source parquet data. |
| `minute_calendar` | `MinuteCalendar | None` | Optional intraday minute calendar. |
| `trading_calendar` | `TradingCalendar | None` | Optional trading calendar. If omitted, it is loaded from `external/trading_days.parquet`. |

## Methods

### `collect`

```python
Engine.collect(model: Model, dates: list[str] | tuple[str, ...]) -> pl.DataFrame
```

Evaluate `model.output` for one or more dates and return the public daily factor output.

Returns a dataframe with columns:

```text
date, secu_code, factor_name, value
```

The model output must be daily grain with `(date, secu_code)` keys and must expose a public `value` column.

### `evaluate`

```python
Engine.evaluate(model: Model, node: Node, eval_date: str) -> pl.LazyFrame
```

Evaluate any node in a model for one date. Use this for intermediate nodes and minute-grain outputs.

### `trace`

```python
Engine.trace(model: Model, date: str) -> list[TraceStep]
```

Materialize each frame node in dependency order. Use this for debugging or validation.

## Examples

```python
engine = Engine(data_root="data")

daily = engine.collect(model, dates=["20170103", "20170104"])
minute = engine.evaluate(model, close_node, "20170103").collect()
steps = engine.trace(model, "20170103")
```

## Raises

| Error | Condition |
|---|---|
| `ValueError` | `collect()` is called with no dates. |
| `ValueError` | `collect()` output is not daily grain. |
| `ValueError` | `collect()` output has no public `value` column. |
