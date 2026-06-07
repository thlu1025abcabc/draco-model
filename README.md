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
- `Grid(frequency=None, auction=None)` 显式把 raw/minute frame 对齐到 universe × minute grid；未传 `frequency` 时从 `grain_path` 推断，raw source 默认 1m，可直接作用于 `Source(...)`。
- `Join()` 横向合并多个 frame，保留 public fields 和 internal payload。
- `Project()` 显式丢弃 internal payload，只保留 key columns + public fields。
- `FillNull(value)` 支持固定数值、`"ffill"` 和 `"state"`；`"state"` 使用 field 的 source lineage 和 aggregate path 构造同粒度 close_state，并对齐待填 frame 的 key 后填 public field。

`Engine.collect()` 只接受日频 factor 输出：输出必须是 `(date, secu_code)` grain，并包含 public `value` column。分钟级结果请先显式 `Aggregate("1d", ..., alias="value")`，或直接用 `Engine.evaluate()` / `Engine.trace()` 查看。

除非显式运行 `Project()`，其他 layer 都会保留 internal payload。聚合类 layer 会把 payload 聚合到目标粒度；`FillNull()` 会保留 payload 列，但填充后的 public field 不再把旧 payload 标记为可重算 components。
Payload 的当前特性：

- `FillNull()` 后再次 `Aggregate()` 时，filled public field 以填充后的列为准；保留下来的 payload 仍可用于 lineage/debug，但旧 components 不再表示可以完整重算 filled public field。
- `Aggregate(apply_to="field")` 会直接聚合 public field，并把 payload 一起保留到目标粒度；保留下来的 payload 不承诺可以完整重算聚合后的 public field，主要表达来源和调试信息。

`FillNull("state")` 要求字段有唯一 source lineage；多 source 表达式会显式报错，避免隐式选择某个 source 的 close_state。

Public alias 和 `Join()` input name 不能以 `__` 开头，也不能使用 `date`、`secu_code`、`minute` 这些 key column 名。

旧 public API 已删除：`Field`、`RatioField`、`Auction`、`Resample`、`DailyAgg`、`Concat` 不再导出。

## 文档

更完整的 user guide 和 API reference 放在 [docs/index.md](docs/index.md)。文档按 sklearn / Polars 风格拆成 narrative guide 与 API reference 两层：

- user guide 解释 source schema、aggregation、rolling、payload、debugging 等语义。
- API reference 覆盖 `Engine`、`Model`、`Source`、`Metric`、`Op`、filters、`Aggregate`、`Grid`、`FillNull`、`Join` 和 `Project`。

文档站使用 MkDocs Material：

```powershell
python -m pip install -e .[docs]
python -m mkdocs serve
```

本地预览地址是 `http://127.0.0.1:8000`。构建静态站点：

```powershell
python -m mkdocs build
```

## FrameInfo 约定

运行期元数据收敛为 `FieldInfo` 和 `FrameInfo`。每个 frame layer 都需要注册 executor 和 info builder；执行所需的 layout/spec 由 helper 从 `FrameInfo` 派生，不再维护独立 `FramePlan` / `FrameSchema` / `FrameLineage`：

- `FieldInfo` 记录单列的物理列名、public/payload/identity 角色、source lineage、operator components、`grain_path` 和 `lookback_days`。
- `FrameInfo` 是一组 `FieldInfo`；`columns`、`identity_keys`、`grain`、public value columns 和 payload columns 都从字段信息推断。
- schema inference 统一由 info builder 派生，不再由 executor 结果反推；executor 的最终输出列应按 `FrameInfo.columns` `select`，避免元数据和真实输出漂移。
- `SourceCatalog` 负责注册 source 的固定 schema 和 row identity keys；`Aggregate` 等 `group_by` layer 会把输出 identity 改成 group keys。
- `Grid` 是显式 left join 语义：左侧 grid identity 为 `(date, secu_code, minute)`，右侧必须包含这些列，输出 identity 是所有输入 identity keys 的有序并集；raw source 不会自动补 grid。

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
from draco_model.layers import Aggregate, Col, FillNull, Grid, Metric, Source

raw = Source("trades_tbar")

amount = Metric("amount", raw)
volume = Metric("volume", raw)
vwap = (amount / volume).alias("vwap")

vwap_5m = Aggregate("5m", "sum", apply_to="components")(vwap)
mean_minute_vwap = Aggregate("5m", "mean", apply_to="field")(vwap)
gridded_raw = Grid()(raw)
volume_grid = Grid()(volume)
volume_5m_auto_grid = Grid()(Aggregate("5m", "sum", value_col="volume")(volume))
volume_5m_grid = Grid("5m")(Aggregate("5m", "sum", value_col="volume")(volume))

row_amount = (Col("price") * Col("volume")).alias("amount")(raw)
preclose = FillNull("state")(Metric("preclose", raw))
```

## Rolling 语义

`rolling_corr` / `rolling_beta` / `rolling_alpha` 通过 `Op(...)` 使用，例如：

```python
raw = Source("trades_tbar", lookback_days=5)
amount = Metric("amount", raw)
volume = Metric("volume", raw)

intraday_corr = Op("rolling_corr", amount, volume, window=5, alias="corr_5")
cross_day_corr = Op("rolling_corr", amount, volume, window=5, alias="corr_5_cross", cross_day=True)
```

- `cross_day=False` 是默认行为。分钟级 rolling 按 `(date, secu_code)` 分组，等价于日内重置。
- `cross_day=True` 时，分钟级 rolling 只按 `secu_code` 分组，可以跨交易日使用上一日窗口。
- `Source(..., lookback_days=...)` 由调用方显式指定；rolling operator 不会根据 `window` 自动扩大 lookback。

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

`SourceCatalog.schema(source, dates)` 会优先使用固定 source schema，避免上层 schema inference 依赖 parquet scan。当前固定 schema：

- `trades_tbar` / `cancels_tbar`：`secu_code`、`minute`、`price`、`side`、`volume`、`vw_wait_time`、`is_first`、`is_last`、`no`、`date`。
- `quotes_tbar`：`secu_code`、`minute`、`price`、`side`、`volume`、`is_first`、`is_last`、`no`、`date`。
- `daily_k`：`sec_code`、`date`、`open`、`high`、`low`、`close`、`shares`、`amount`、`limit_up`、`limit_down`、`preclose`、`isSuspend`、`isST`、`adjfactor`、`total_share`、`float_share`、`free_share`、`list_date`、`secu_code`。
- `snapshot_tbar`：`AskPrice1`-`AskPrice10`、`BidPrice1`-`BidPrice10`、`AskVolume1`-`AskVolume10`、`BidVolume1`-`BidVolume10`、`aVOI1`-`aVOI5`、`secu_code`、`minute`、`date`。
- `universe/ex2kamt`：`sec_code`、`preclose`、`close`、`adjfactor`、`secu_code`、`date`。

## Trace 和 Mermaid

`Engine.trace(model, date)` 会按拓扑顺序 materialize 每个 frame node；`Model.explain_mermaid()` 会输出同一张 DAG。因为 metric 会展开，`amount`、`buyamount`、`vwap` 这类字段的内部 `where/op/aggregate` 都能在 trace 和图里看到。

## 运行示例

```powershell
python -m examples.close_last
python -m examples.top_volume_close_mean
python -m examples.preclose_fill_state_demo
```

## TODO

- 后续 batch planner 需要合并多个 model / metric 中可复用的 source scan、operator branch 和 join。
- 后续 optimizer 可以把同 source 的 `Metric("amount")`、`Metric("volume")`、`Metric("vwap")` fuse 成更少的 physical plan。
- 评估是否需要 smart join：根据输入 grain、key 覆盖、source 复用和 public/payload 列需求，减少不必要的宽表 join、重复 scan 或中间 payload 搬运。
- 继续思考 `close_state` 是否也进入 payload / operator metadata 体系。
- 再次整体评估 payload 是否应该默认保留：需要决定 payload 是继续作为跨 layer 的物理列传播，还是改为只在需要 trace/debug/组件重算时保留，或者在部分 layer 后默认 drop。
- 重新设计 `Aggregate(apply_to="field")` 的 payload 处理：当前 payload 会被同一个 `agg` 聚合后保留，但这可能产生误导性的 lineage/debug 信息，后续需要决定是 drop、仅保留 metadata，还是为 payload 定义独立聚合策略。
