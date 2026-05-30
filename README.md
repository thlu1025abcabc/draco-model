# Draco Model

`draco-model` 是一个基于 Polars LazyFrame 的因子 DAG 原型。当前版本已经切到新的 General Operator DAG：字段不再是黑盒 `Field` executor，而是展开成可 trace 的 `Source -> Where -> Op -> Aggregate -> Join/Project` 图。

```python
from draco_model import Engine, Model
from draco_model.layers import Aggregate, Metric, Source

raw = Source("trades_tbar")
close = Metric("close", raw)
output = Aggregate("1d", "last", value_col="close", alias="value")(close)

model = Model(name="close_last", universe="ex2kamt", output=output)
df = Engine(data_root="data").collect(model, dates=["20170103"])
```

## 核心 API

- `Source("trades_tbar")` 扫描一个 raw source，不自动补 intraday grid。
- `Metric("name", raw)` 是字段 recipe shorthand；它会展开成真实 DAG。
- `Col("price")` 是 raw column reference，只能应用到 frame 上，例如 `(Col("price") * Col("volume")).alias("amount")(raw)`。
- `Op("name", ...)` 是统一 operator 入口，支持 `add/sub/mul/div/rolling_corr/rolling_beta/rolling_alpha`。
- `Node` 和 `Col` 支持 magic arithmetic：`+ - * /` 都会生成 `op` 节点。
- `Where(Side("buy"))` / `Where(Side("sell"))` 是语义 side filter；执行时映射到当前数据中的 `side == 0/1`。
- `Aggregate(frequency, agg, ...)` 统一处理 raw -> 1m、分钟 resample、daily agg 和 auction 逻辑。
- `Join()` 横向合并多个 frame，保留 public fields 和 internal payload。
- `Project()` 显式丢弃 internal payload，只保留 key columns + public fields。
- `FillNull(value)` 支持固定数值、`"ffill"` 和 `"state"`；`"state"` 使用 close_state 填 public field。

除非显式运行 `Project()`，其他 layer 都会保留 internal payload。聚合类 layer 会把 payload 聚合到目标粒度；`FillNull()` 会保留 payload 列，但填充后的 public field 不再把旧 payload 标记为可重算 components。
Public alias 和 `Join()` input name 不能以 `__` 开头，也不能使用 `date`、`secu_code`、`minute` 这些 key column 名。

旧 public API 已删除：`Field`、`RatioField`、`Auction`、`Resample`、`DailyAgg`、`Concat` 不再导出。

## Metric Recipes

内置 `Metric` 语义：

- `Metric("volume", raw)`：`Col("volume") -> Aggregate("1m", "sum")`。
- `Metric("no", raw)`：`Col("no") -> Aggregate("1m", "sum")`。`no` 表示 records 数量，不表示价格顺序。
- `Metric("amount", raw)`：`Col("price") * Col("volume") -> Aggregate("1m", "sum")`。即使 raw source 有 `amount` 列，也固定使用 `price * volume`。
- `Metric("buyamount", raw)`：`Where(Side("buy")) -> price * volume -> Aggregate("1m", "sum")`。
- `Metric("sellamount", raw)`：`Where(Side("sell")) -> price * volume -> Aggregate("1m", "sum")`。
- `Metric("vwap", raw)`：`Metric("amount", raw) / Metric("volume", raw)`。
- `Metric("open/close")`：分别用 `is_first` / `is_last` 过滤 price 后聚合。
- `Metric("high/low")`：对 price 做 max/min。
- `Metric("preclose")`：直接 evaluate 会报错；必须通过 `FillNull("state")(Metric("preclose", raw))` 使用。

示例：

```python
from draco_model.layers import Aggregate, Col, FillNull, Metric, Source

raw = Source("trades_tbar")

amount = Metric("amount", raw)
volume = Metric("volume", raw)
vwap = (amount / volume).alias("vwap")

vwap_5m = Aggregate("5m", "sum", apply_to="components")(vwap)
mean_minute_vwap = Aggregate("5m", "mean", apply_to="field")(vwap)

row_amount = (Col("price") * Col("volume")).alias("amount")(raw)
preclose = FillNull("state")(Metric("preclose", raw))
```

## Aggregate 语义

`Aggregate(frequency, agg, value_col=None, alias=None, apply_to="field", auction="keep")`：

- `frequency="1m"`：按 `(date, secu_code, minute)` 聚合 raw bucket。
- `frequency="5m"`：按 minute calendar 重采样，auction bars 默认保留。
- `frequency="1d"` / `"daily"`：按 `(date, secu_code)` 聚合。
- `auction="drop"` 删除 auction minutes。
- `auction="merge"` 先把 auction minutes 合入当前目标频率中除 auction 外的第一根/最后一根 bar，再执行聚合；例如 1m 为 `925 -> 930`、`1500 -> 1456`，5m 为 `925 -> 930`、`1500 -> 1455`。Daily aggregate 也会先应用 `auction` 策略，再按日聚合。
- `apply_to="field"` 直接聚合 public output，同时保留 payload。
- `apply_to="components"` 对 operator components 分别聚合，再按原 operator 重算 public output。

`sum` 使用 null-safe sum：如果一个 group 全是 null，结果保持 null，不会被 Polars 默认 sum 写成 0。

## 数据目录

本地 parquet 数据放在 `data/` 和 `external/` 下：

```text
data/
  trades_tbar/
    20170103.parquet
  quotes_tbar/
    20170103.parquet
  cancels_tbar/
    20170103.parquet
  snapshot_tbar/
    20170103.parquet
  daily_k/
    20170103.parquet
  universe/
    ex2kamt/
      20170103.parquet
external/
  trading_days.parquet
```

source 语义：

- `trades_tbar` 是分钟级聚合后的逐笔成交数据。每行表示一个 stock、一个 minute、一个 price、一个 side，并聚合该 bucket 的 `Volume` 和 `No`。
- `quotes_tbar` 是分钟级聚合后的逐笔委托数据，形状同样是 stock/minute/price/side。
- `cancels_tbar` 是分钟级聚合后的逐笔撤单数据，形状同样是 stock/minute/price/side。
- `snapshot_tbar` 是分钟级 snapshot 数据，包含 1-10 档 bid/ask price 和 volume 的平均值。
- `daily_k` 是日频 OHLC 数据，包含 `open`、`high`、`low`、`close`、`preclose`、`volume`、`amount` 等字段。
- `universe/ex2kamt` 定义股票池，并包含一些日频参考字段。
- `external/trading_days.parquet` 是交易日历来源。

source scan 会标准化常见 vendor column name，例如 `SecuCode -> secu_code`、`MinBar -> minute`、`Price -> price`、`Amount -> amount`、`Volume -> volume`、`No -> no`、`Side -> side`、`isfirst -> is_first`、`islast -> is_last`、`trading_day -> date`。

## Trace 和 Mermaid

`Engine.trace(model, date)` 会按拓扑顺序 materialize 每个 frame node；`Model.explain_mermaid()` 会输出同一张 DAG。因为 metric 会展开，`amount`、`buyamount`、`vwap` 这类字段的内部 `where/op/aggregate` 都能在 trace 和图里看到。

## 运行示例

```powershell
python -m examples.close_last
python -m examples.top_volume_close_mean
python -m examples.preclose_fill_state_demo
```

## TODO

- Source 层需要主动提供 normalized schema columns，例如 `SourceCatalog.schema(source, dates)`，减少上层 schema inference 对 `collect_schema()` 的依赖。
- 后续 batch planner 需要合并多个 model / metric 中可复用的 source scan、operator branch 和 join。
- 后续 optimizer 可以把同 source 的 `Metric("amount")`、`Metric("volume")`、`Metric("vwap")` fuse 成更少的 physical plan。
- 继续思考 `close_state` 是否也进入 payload / operator metadata 体系。
- 设计 `FillNull()` 后再次 `Aggregate()` 的 payload 语义：当前 `FillNull()` 会保留 payload，但不会把旧 components 标记为可重算；后续需要决定 filled public field 与 payload/components 的一致性策略。
- 设计 `apply_to="field"` 后保留 payload 的二次聚合语义：当前 payload 会随 field aggregation 保留，但不保证可重算 public field；后续需要明确它是 lineage/debug，还是可参与后续计算的状态。
