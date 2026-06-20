# Draco Model

`draco-model` 是一个基于 Polars LazyFrame 的因子 DAG 原型。当前版本已经切到新的 General Operator DAG：字段不再是黑盒 `Field` executor，而是展开成可 trace 的 `Source -> Where -> Op -> Aggregate -> Join/Project` 图。

```python
from draco_model import Engine, Model
from draco_model.recipes import metric
from draco_model.layers import Aggregate, Source

raw = Source("trades_tbar")
close = metric("close")(raw)
output = Aggregate("1d", "last", value_col="close", alias="value")(close)

model = Model(name="close_last", universe="ex2kamt", output={"value": output})
df = Engine(data_root="data").collect(model, dates=["20170103"])
```

## 核心 API

- `Source("trades_tbar")` 扫描一个 raw source，不自动补 intraday grid。
- `metric("name")(raw)` 是字段 recipe shorthand；它会展开成真实 DAG。
- `last(minute)(raw)` 是时间过滤 shortcut，保留行内 `minute` 大于等于给定阈值的行。
- `Col("price")` 是 raw column reference，只能应用到 frame 上，例如 `(Col("price") * Col("volume")).alias("amount")(raw)`。
- `Op("name", ...)` 是统一 operator 入口，支持 `add/sub/mul/div/rolling_corr/rolling_beta/rolling_alpha`。
- `Node` 和 `Col` 支持 magic arithmetic：`+ - * /` 都会生成 `op` 节点。
- `Where(Side("buy"))` / `Where(Side("sell"))` 是语义 side filter；执行时映射到当前数据中的 `side == 0/1`。
- `Aggregate(frequency, agg, ...)` 统一处理 raw -> 1m、分钟 resample、daily agg 和 auction 逻辑。
- `Grid(frequency=None, auction=None)` 显式把 frame 对齐到 universe × minute grid；minute/raw frame 按 `(date, secu_code, minute)` 对齐，daily frame 按 `(date, secu_code)` broadcast 到分钟 grid。
- `Join(how="full")` 用 SQL full join 横向对齐多个 frame；`Join(how="left", on=...)` 按显式 key 做 left join；显式 `on` 必须覆盖两个输入共享的 identity keys，避免漏掉 `price/side` 这类 key 后产生笛卡尔扇出。`how="full"` 不允许混合 daily identity frame 与 minute/raw frame；混合粒度请显式使用 `Join(how="left", on=("date", "secu_code"))`。两者输出 identity 都是所有输入 identity keys 的有序并集。
- `Project()` 显式丢弃 internal payload，只保留 key columns + public fields。
- `FillNull(value)` 支持固定数值、`"ffill"` 和 `"state"`；`"state"` 使用 field 的 `FieldInfo.source` 和 `grain_path` 构造同粒度 close_state，并对齐待填 frame 的 key 后填 public field。`FillNull()` 后会丢弃 old payload。
- `Model(..., output={"name": node})` 显式声明 factor 输出；`output` 只支持 dictionary mapping。
- `profile_plan(models)` 静态分析一组 model 的共享 structural nodes，并标记适合 batch materialization 的 cache candidate。
- `Engine.collect_many(models, dates, ...)` 批量收集多个 model，并用 batch-scoped materialized cache 复用共享 DAG 节点；输出仍是标准长表。

使用建议：`Source(...)` 不会自动补 grid，但如果 raw/minute source 后续要和 daily feature 或其他粒度横向组合，优先显式套一层 `Grid()`。这样下游节点都落在统一的 `(date, secu_code, minute)` grid 上，能减少 mixed-grain `Join()` 的歧义。

`Engine.collect()` 只接受日频 factor 输出：每个 output 必须是 `(date, secu_code)` grain，并且只有一个 public value column。`collect()` 会按 model universe 对每个 output 做 left join，把该 value column 统一改名为 `value`，增加 `factor_name`，最后 concat 成标准长表。多个因子请用 `Model(..., output={"a": node_a, "b": node_b})` 表达，不再用 `output_columns` 展开宽表。分钟级结果请先显式 `Aggregate("1d", ...)`，或直接用 `Engine.evaluate()` / `Engine.trace()` 查看。

除非显式运行 `Project()`，其他 layer 默认保留 internal payload；`Aggregate(apply_to="field")` 和 `FillNull()` 是例外，它们会在计算后自动投影到 key columns + public fields。
Payload 的当前特性：

- `FillNull()` 后旧 payload 会被丢弃；这表示 filled public field 不再支持 `Aggregate(apply_to="components")`，后续如需聚合请使用 `apply_to="field"`。
- `Aggregate(apply_to="field")` 会直接聚合 public field，并在聚合后自动丢弃 payload。

`FillNull("state")` 要求字段有唯一 `FieldInfo.source`；多 source 表达式会显式报错，避免隐式选择某个 source 的 close_state。

Public alias 和 `Join()` input name 不能以 `__` 开头，也不能使用 `date`、`secu_code`、`minute` 这些 key column 名。

旧 public API 已删除：`Field`、`RatioField`、`Auction`、`Resample`、`DailyAgg`、`Concat` 不再导出。

## Profiling

Profiling 分为不跑数据的静态 DAG 分析和可选的运行时事件记录。它们都不改变 `collect()` 的执行语义。

```python
from draco_model import Engine, Model, profile_plan

plan = profile_plan([amount_model, vwap_model])
print(plan.cache_candidates())

engine = Engine(data_root="data")
with engine.profiler() as profiler:
    engine.collect(amount_model, dates=["20170103"])

print(profiler.summary())
events = profiler.to_frame()
```

`profile_plan()` 用 structural node id 统计一组 model 中重复出现的子图，适合给后续 batch planner / cache 策略写稳定测试。`Engine.profiler()` 只记录 `collect`、`evaluate`、`infer_info`、`eval` 和最终 materialize 的事件、耗时与 cache hit/miss，不会像 `trace()` 那样 materialize 每个中间节点。

`Engine.collect_many()` 先按 universe 构建 logical union profile，再按 model/date 正常 demand-driven 执行；当一个节点满足 `min_cache_ref_count` 且 op 不在 `exclude_cache_ops` 中时，第一次使用会 materialize 到 batch cache，后续 model 直接复用。默认 `exclude_cache_ops=("source",)`，避免把 raw source eagerly materialize。

## 文档

更完整的 user guide 和 API reference 放在 [docs/index.md](docs/index.md)。文档按 sklearn / Polars 风格拆成 narrative guide 与 API reference 两层：

- user guide 解释 source schema、aggregation、rolling、payload、debugging 等语义。
- API reference 覆盖 `Engine`、`Model`、`Source`、`metric`、`Op`、filters、`Aggregate`、`Grid`、`FillNull`、`Join` 和 `Project`。

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

运行期元数据收敛为 `FieldInfo` 和 `FrameInfo`。每个 frame layer 都需要注册 executor 和 info builder；执行所需的 layout/spec 由 helper 从 `FrameInfo` 派生，不再维护独立的 plan/schema/source-tracking class：

- `FieldInfo` 记录单列的物理列名、public/payload/identity 角色、`source`、operator components、`grain_path` 和 `lookback_days`。
- `FrameInfo` 是一组 `FieldInfo`；`columns`、`identity_keys`、`grain`、public value columns 和 payload columns 都从字段信息推断。
- schema inference 统一由 info builder 派生，不再由 executor 结果反推；executor 的最终输出列应按 `FrameInfo.columns` `select`，避免元数据和真实输出漂移。
- `SourceCatalog` 负责注册 source 的固定 schema 和 row identity keys；`Aggregate` 等 `group_by` layer 会把输出 identity 改成 group keys。
- `Join(how="full")` 在 `on=None` 时逐步使用左右 identity keys 的交集；`Join(how="left")` 在 `on=None` 时以最左输入的 identity keys 为 join key。显式 `on` 时按 `on` join，但 `on` 必须是双方 identity key 且覆盖双方共享 identity；输出 identity 都是所有输入 identity keys 的有序并集。`how="full"` 只允许同粒度或全 daily 输入，不允许 daily/minute 混合产生 `minute=null` padding row；混合粒度请使用 `how="left"` 并显式选择 join key。`Grid` 是 system grid source 加 `Join(how="left")` 的 API 应用；raw source 不会自动补 grid，但对 source 显式 `Grid()` 后，下游 feature 会共享统一 minute identity，后续 join 通常更简单。

## Recipe Shortcuts

内置 `metric` 语义：

- `metric("volume")(raw)`：`Col("volume") -> Aggregate("1m", "sum")`。
- `metric("no")(raw)`：`Col("no") -> Aggregate("1m", "sum")`。`no` 表示 records 数量，不表示价格顺序。
- `metric("amount")(raw)`：`Col("price") * Col("volume") -> Aggregate("1m", "sum")`。即使 raw source 有 `amount` 列，也固定使用 `price * volume`。
- `metric("buyamount")(raw)`：`Where(Side("buy")) -> price * volume -> Aggregate("1m", "sum")`。
- `metric("sellamount")(raw)`：`Where(Side("sell")) -> price * volume -> Aggregate("1m", "sum")`。
- `metric("vwap")(raw)`：`metric("amount")(raw) / metric("volume")(raw)`。
- `metric("open/close")`：分别用 `is_first` / `is_last` 过滤 price 后聚合。
- `metric("high/low")`：对 price 做 max/min。
- `metric("preclose")`：直接 evaluate 会报错；必须通过 `FillNull("state")(metric("preclose")(raw))` 使用。
- `last(1400)(raw)`：等价于 `Where(Threshold("minute", op=">=", value=1400))(raw)`，对每个股票/日期保留指定 minute 及之后的行。

示例：

```python
from draco_model.recipes import metric
from draco_model.layers import Aggregate, Col, FillNull, Grid, Join, Source

raw = Source("trades_tbar")
gridded_raw = Grid()(raw)

amount = metric("amount")(raw)
volume = metric("volume")(raw)
vwap = (amount / volume).alias("vwap")

vwap_5m = Aggregate("5m", "sum", apply_to="components")(vwap)
mean_minute_vwap = Aggregate("5m", "mean", apply_to="field")(vwap)
grid_volume = metric("volume")(gridded_raw)
volume_grid = Grid()(volume)
volume_5m_auto_grid = Grid()(Aggregate("5m", "sum", value_col="volume")(volume))
volume_5m_grid = Grid("5m")(Aggregate("5m", "sum", value_col="volume")(volume))
close_grid = Grid()(metric("close")(raw))
daily_vwap = Aggregate("1d", "mean", value_col="vwap", alias="daily_vwap")(vwap)
features = Join(how="left", on=("date", "secu_code"))({
    "minute_volume": grid_volume,
    "daily": daily_vwap,
})

row_amount = (Col("price") * Col("volume")).alias("amount")(raw)
preclose = FillNull("state")(metric("preclose")(raw))
```

`Grid()` 只保证当前 frame 的 row set，不会 sticky 到所有下游 layer。`Where(...)`、`metric(...)(...)`、`Aggregate(...)` 仍然可以改变 row set。尤其是 `metric("close")` / `metric("open")` 会先按 `is_last` / `is_first` 过滤，grid 补出来的缺失分钟这些 flag 是 null，会被当成 false，所以 `metric("close")(Grid()(raw))` 会把缺失分钟过滤掉。若目标是完整分钟面板，并希望缺失 bar 的 close/open 保持为 null，请使用 `Grid()(metric("close")(raw))`。

## Rolling 语义

`rolling_corr` / `rolling_beta` / `rolling_alpha` 通过 `Op(...)` 使用，例如：

```python
raw = Source("trades_tbar", lookback_days=5)
amount = metric("amount")(raw)
volume = metric("volume")(raw)

intraday_corr = Op("rolling_corr", amount, volume, window=5, alias="corr_5")
cross_day_corr = Op("rolling_corr", amount, volume, window=5, alias="corr_5_cross", cross_day=True)
```

- `cross_day=False` 是默认行为。分钟级 rolling 按 `(date, secu_code)` 分组，等价于日内重置。
- `cross_day=True` 时，分钟级 rolling 只按 `secu_code` 分组，可以跨交易日使用上一日窗口。
- `Source(..., lookback_days=...)` 由调用方显式指定；rolling operator 不会根据 `window` 自动扩大 lookback。
- rolling operator 只接受恰好两个 frame 节点操作数；传 `Col(...)` 或标量会在构造期报错。
- 窗口内方差为零（或浮点误差导致非正）时输出 null；`rolling_corr` 结果裁剪到 `[-1, 1]`。

## Aggregate 语义

`Aggregate(frequency, agg, value_col=None, alias=None, apply_to="field", auction="keep")`：

- `frequency="1m"`：按 `(date, secu_code, minute)` 聚合 raw bucket。
- `frequency="5m"`：按 minute calendar 重采样，auction bars 默认保留。
- `frequency="1d"` / `"daily"`：按 `(date, secu_code)` 聚合。
- `auction="drop"` 删除 auction minutes。
- `auction="merge"` 先把 auction minutes 合入当前目标频率中除 auction 外的第一根/最后一根 bar，再执行聚合；例如 1m 为 `925 -> 930`、`1500 -> 1456`，5m 为 `925 -> 930`、`1500 -> 1455`。Daily aggregate 也会先应用 `auction` 策略，再按日聚合。
- `apply_to="field"` 直接聚合 public output，然后自动丢弃 payload。
- `apply_to="components"` 对 operator components 分别聚合，再按原 operator 重算 public output。
- 如果 `value_col` 本身是 identity key，例如 raw source 的 `minute` / `price` / `side`，必须提供非 key 的 `alias`，避免输出 public value 与 identity column 同名。

注意：daily aggregate 的 `auction="merge"` 会在合并 auction bar 和最终 daily 聚合时使用同一个 `agg`。例如 `Aggregate("1d", "mean", auction="merge")` 会先用 `mean` 合并 `925 -> 930` / `1500 -> 1456` 后的同一分钟，再对全天做 `mean`。如果目标语义是“auction volume 先用 `sum` 合入非 auction bar，然后再做 daily `mean`”，需要显式拆成两层：

```python
volume_1m = Aggregate("1m", "sum", value_col="volume", auction="merge")(metric("volume")(raw))
daily_mean = Aggregate("1d", "mean", value_col="volume", alias="value")(volume_1m)
```

`sum` 使用 null-safe sum：如果一个 group 全是 null，结果保持 null，不会被 Polars 默认 sum 写成 0。

## 数据目录

本地 parquet 数据放在 `data/` 和 `external/` 下：

```text
data/
  steptrades/
    20170103.parquet
  steporders/
    20170103.parquet
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

- `steptrades` 是 vendor 逐笔成交源，保留成交时间、买卖委托 ID、成交 ID、原始整数价格、成交量和方向。
- `steporders` 是 vendor 逐笔委托源，保留委托时间、委托 ID 和委托类型。
- `trades_tbar` 是分钟级聚合后的逐笔成交数据。每行表示一个 stock、一个 minute、一个 price、一个 side，并聚合该 bucket 的 `Volume` 和 `No`。
- `quotes_tbar` 是分钟级聚合后的逐笔委托数据，形状同样是 stock/minute/price/side。
- `cancels_tbar` 是分钟级聚合后的逐笔撤单数据，形状同样是 stock/minute/price/side。
- `snapshot_tbar` 是分钟级 snapshot 数据，包含 1-10 档 bid/ask price 和 volume 的平均值。
- `daily_k` 是日频 OHLC 数据，包含 `open`、`high`、`low`、`close`、`preclose`、`volume`、`amount` 等字段。
- `universe/ex2kamt` 定义股票池，并包含一些日频参考字段。
- `external/trading_days.parquet` 是交易日历来源。

source scan 会标准化常见 vendor column name，例如 `SecuCode -> secu_code`、`MinBar -> minute`、`Price -> price`、`Amount -> amount`、`Volume -> volume`、`No -> no`、`Side -> side`、`isfirst -> is_first`、`islast -> is_last`、`trading_day -> date`。

`SourceCatalog.schema(source, dates)` 会优先使用固定 source schema，避免上层 schema inference 依赖 parquet scan。当前固定 schema：

- `steptrades`：`date`、`secu_code`、`deal_time`、`buy_id`、`sell_id`、`deal_id`、`price`、`volume`、`side`。
- `steporders`：`date`、`secu_code`、`order_time`、`order_id`、`order_type`、`price`、`volume`。
- `trades_tbar` / `cancels_tbar`：`secu_code`、`minute`、`price`、`side`、`volume`、`vw_wait_time`、`is_first`、`is_last`、`no`、`date`。
- `quotes_tbar`：`secu_code`、`minute`、`price`、`side`、`volume`、`is_first`、`is_last`、`no`、`date`。
- `daily_k`：`sec_code`、`date`、`open`、`high`、`low`、`close`、`shares`、`amount`、`limit_up`、`limit_down`、`preclose`、`isSuspend`、`isST`、`adjfactor`、`total_share`、`float_share`、`free_share`、`list_date`、`secu_code`。
- `snapshot_tbar`：`AskPrice1`-`AskPrice10`、`BidPrice1`-`BidPrice10`、`AskVolume1`-`AskVolume10`、`BidVolume1`-`BidVolume10`、`aVOI1`-`aVOI5`、`secu_code`、`minute`、`date`。
- `universe/ex2kamt`：`sec_code`、`preclose`、`close`、`adjfactor`、`secu_code`、`date`。

Level-2 原始数据可以构造成标准 bar，并直接作为 named model output：

```python
from draco_model import Engine, Model
from draco_model.layers import Source, TradesWithWaitBar

trades = Source("steptrades")
orders = Source("steporders")
trades_wtminbar = TradesWithWaitBar()(trades, orders)
model = Model("trades_wtminbar", None, {"trades_wtminbar": trades_wtminbar})

outputs = Engine(data_root="data").evaluate_outputs(model, "20260618")
bars = outputs["trades_wtminbar"]
```

`evaluate_outputs()` 保留 model output 名称和原始粒度，返回 `LazyFrame`，不做日频 factor 格式化，也不负责文件或云端传输。后续编排层可以独立决定怎样物化和传递结果。

这里 `"trades_wtminbar"` 的第一个位置是 `Model.name`，字典里的 `"trades_wtminbar"` 是 output name；两者可以不同。`universe=None` 明确表示该 model 不做股票池对齐，因此不能用于 `collect()`、`collect_many()` 或 `Grid()`。

`QuotesMinBar` 和 `CancelsMinBar` 用同样的方式从 `steptrades` + `steporders` 构造委托报价和撤单的 minute/price/side bar：

```python
from draco_model.layers import CancelsMinBar, QuotesMinBar, Source

trades = Source("steptrades")
orders = Source("steporders")
quotes_minbar = QuotesMinBar()(trades, orders)
cancels_minbar = CancelsMinBar()(trades, orders)
```

- `QuotesMinBar` 输出 `(date, secu_code, minute, price, side, volume, is_first, is_last, no)`，没有等待时间。
- `CancelsMinBar` 输出与 `trades_wtminbar` 相同的列（含 `vw_wait_time`），等待时间是撤单时刻减去原始委托时刻、扣除午休。
- 两者都按交易所拆分：沪市（`secu_code >= 600000`）走委托流（报价 `order_type` 0/10、撤单 -1/-11，外加收盘集合竞价从成交补价）；深市走委托流 `order_type` 2/12（自带价）和 1/11/3/13（市价/本方最优，价格从成交反查），撤单来自成交流 `side` -1/-11、价格回查对应报价。
- `is_first`/`is_last` 标记本分钟最早/最晚事件落在哪个 price 档。同一委托被多笔同毫秒成交命中不同价位时会产生排序并列，构造层用 `(secu_code, order_time, order_id, price, side)` 做确定性 tie-break，因此输出可复现（同一输入每次结果一致）。

## Trace 和 Mermaid

`Engine.trace(model, date)` 会按拓扑顺序 materialize 每个 frame node；`Model.explain_mermaid()` 会输出同一张 DAG。因为 metric 会展开，`amount`、`buyamount`、`vwap` 这类字段的内部 `where/op/aggregate` 都能在 trace 和图里看到。

## 运行示例

```powershell
python -m examples.close_last
python -m examples.top_volume_close_mean
python -m examples.preclose_fill_state_demo
```

`sample/level2_bars.py` 按日期构造 level-2 bar 并对 golden 验收，支持用不同日期复跑：

```powershell
python sample/level2_bars.py 20260618              # trades + quotes + cancels，自动对比 golden
python sample/level2_bars.py 20260618 --bar quotes
python sample/level2_bars.py 20260101 --data-root D:/prod/level2 --out-dir sample/output
```

## TODO

1. Fusion graph
2. Smart join
