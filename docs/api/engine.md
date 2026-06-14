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
Engine.collect(
    model: Model,
    dates: list[str] | tuple[str, ...],
    *,
    output_columns: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame
```

Evaluate `model.output` for one or more dates and return the public daily factor output.

Returns a dataframe with columns:

```text
date, secu_code, factor_name, value
```

The model output must be daily grain with `(date, secu_code)` keys and must expose the requested public output columns. `output_columns=None` defaults to `("value",)`.

If one output column is requested, `factor_name` is the model name. If multiple output columns are requested, the final daily wide frame is unpivoted and factor names use `model_name__column_name`.

### `collect_many`

```python
Engine.collect_many(
    models: list[Model] | tuple[Model, ...],
    dates: list[str] | tuple[str, ...],
    *,
    output_columns: list[str] | tuple[str, ...] | None = None,
    min_cache_ref_count: int = 2,
    exclude_cache_ops: Iterable[str] = ("source",),
) -> pl.DataFrame
```

Evaluate multiple models and return one long factor dataframe with the same schema as `collect()`.

`collect_many()` builds a logical union profile per universe to find shared structural nodes, then executes model/date outputs through the normal demand-driven path. Batch cache candidates are materialized on first use and reused within that `collect_many()` call. The cache is in-memory and discarded after the call returns.

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

### `profile_plan`

```python
Engine.profile_plan(models: list[Model] | tuple[Model, ...]) -> PlanProfile
```

Return a static shared-node profile for a group of models. This is equivalent to calling `profile_plan(models)`.

### `profiler`

```python
with engine.profiler() as profiler:
    engine.collect(model, ["20170103"])
```

Collect runtime profiling events for Engine calls inside the context. The profiler records event counts, elapsed times, node ids, ops, and cache hit/miss events without changing execution semantics.

## Examples

```python
engine = Engine(data_root="data")

daily = engine.collect(model, dates=["20170103", "20170104"])
multi = engine.collect(model, dates=["20170103"], output_columns=["value1", "value2"])
many = engine.collect_many([model_a, model_b], dates=["20170103"])
minute = engine.evaluate(model, close_node, "20170103").collect()
steps = engine.trace(model, "20170103")

with engine.profiler() as profiler:
    engine.collect(model, dates=["20170103"])
events = profiler.to_frame()
```

## Raises

| Error | Condition |
|---|---|
| `ValueError` | `collect()` is called with no dates. |
| `ValueError` | `collect()` output is not daily grain. |
| `ValueError` | `collect()` output does not expose the requested public output columns. |
| `ValueError` | `collect_many()` is called with no models or duplicate model names. |
