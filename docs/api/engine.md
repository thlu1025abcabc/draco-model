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
) -> pl.DataFrame
```

Evaluate the model outputs for one or more dates and return the public daily factor output.

Returns a dataframe with columns:

```text
date, secu_code, factor_name, value
```

Each model output must be daily grain with `(date, secu_code)` keys and exactly one public value column. `collect()` left joins each output to the model universe for each date, renames that public value column to `value`, adds `factor_name`, then concatenates all outputs into one long dataframe. This guarantees every factor has the same `(date, secu_code)` rows for the requested universe.

`collect()` requires `model.universe` to be a non-null universe name. Use `evaluate_outputs()` for models with `universe=None`.

For a single `{"value": node}` output, `factor_name` is the model name. For named outputs such as `{"amount": amount_node}`, factor names use `model_name__output_name`.

### `collect_many`

```python
Engine.collect_many(
    models: list[Model] | tuple[Model, ...],
    dates: list[str] | tuple[str, ...],
    *,
    min_cache_ref_count: int = 2,
    exclude_cache_ops: Iterable[str] = ("source",),
) -> pl.DataFrame
```

Evaluate multiple models and return one long factor dataframe with the same schema as `collect()`.

Every model passed to `collect_many()` must have a non-null universe.

`collect_many()` builds a logical union profile per universe to find shared structural nodes, then executes model/date outputs through the normal demand-driven path. Batch cache candidates are materialized on first use and reused within that `collect_many()` call. The cache is in-memory and discarded after the call returns.

### `evaluate`

```python
Engine.evaluate(model: Model, node: Node, eval_date: str) -> pl.LazyFrame
```

Evaluate any node in a model for one date. Use this for intermediate nodes and minute-grain outputs.

### `evaluate_outputs`

```python
Engine.evaluate_outputs(model: Model, eval_date: str) -> dict[str, pl.LazyFrame]
```

Evaluate every named `Model.output` for one date and return a mapping with the same output names. Unlike `collect()`, this method does not require daily grain, join to the universe, rename value columns, or materialize the result.

Use it when a model represents datasets such as minute bars, including models with `universe=None`. The caller owns later materialization or transport concerns.

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
multi = engine.collect(
    Model("trade_totals", "ex2kamt", {"amount": amount_node, "volume": volume_node}),
    dates=["20170103"],
)
many = engine.collect_many([model_a, model_b], dates=["20170103"])
minute = engine.evaluate(model, close_node, "20170103").collect()
outputs = engine.evaluate_outputs(model, "20170103")
steps = engine.trace(model, "20170103")

with engine.profiler() as profiler:
    engine.collect(model, dates=["20170103"])
events = profiler.to_frame()
```

## Raises

| Error | Condition |
|---|---|
| `ValueError` | `collect()` is called with no dates. |
| `ValueError` | A `collect()` output is not daily grain. |
| `ValueError` | A `collect()` output does not expose exactly one public value column. |
| `ValueError` | `collect_many()` is called with no models or duplicate model names. |
