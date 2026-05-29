# Draco Model

`draco-model` 是一个基于 Polars 的因子计算图原型。用户用 callable layer 构建静态 DAG，然后交给 `Engine` 执行。

```python
from draco_model import Engine, Model
from draco_model.layers import Auction, DailyAgg, Field, Input

close = Auction("drop")(
    Field("close")(
        Input(source="trades_tbar")
    )
)
output = DailyAgg(value_col="close", agg="last")(close)

model = Model(name="close_last", universe="ex2kamt", output=output)
df = Engine(data_root="data").collect(model, dates=["20170103"])
```

## 核心概念

- `Input(source=...)` 扫描一个 raw source，覆盖请求的交易日。它不构造 value field，也不会自动补 intraday grid。
- `Field("name", alias=None)` 从 raw source frame 中构造一个 value column。用 `alias` 可以避免同名字段冲突，例如 `Field("volume", alias="trade_volume")`。`alias` 不能以 `__` 开头，这个前缀保留给内部 payload。
- `RatioField("amount", "volume", alias="vwap")` 构造 ratio 类字段。它保留内部 numerator/denominator payload，后续聚合层可以选择聚合 payload 后再相除，或直接聚合合成后的 public field。`alias` 同样不能以 `__` 开头。
- `Auction("drop")` 删除 auction minutes。`Auction("merge", agg="sum")` 把 `925 -> 930`、`1456 -> 1500` 后聚合，`agg` 支持 `first`、`last`、`sum`。默认 `apply_to="components"`，ratio 字段会先聚合 numerator/denominator，再重新相除；也可以用 `apply_to="field"` 直接聚合 public field。
- `Resample("5m", "last")` 用显式 aggregation 做分钟重采样，例如 `first`、`last`、`max`、`min`、`sum`、`mean`。`sum` 对全 null group 保持 null，不会把缺失误写成 0。默认 `apply_to="components"`，ratio 字段会先聚合 numerator/denominator，再重新相除；`apply_to="field"` 则直接聚合合成后的 public field，并丢弃内部 payload。
- `Aggregate(frequency, agg, ...)` 是统一的聚合入口，`frequency` 可以是分钟频率如 `"5m"`，也可以是 `"daily"` / `"1d"`。它支持 `value_col`、`alias` 和 `apply_to="field" | "components"`；默认 `apply_to="field"`，适合把已经形成的 public field 聚合成新字段。
- `Concat()` 横向合并 frame。混合 intraday 和 daily frame 时，会先用 `pl.concat(..., how="align")` 对齐所有 intraday frame，再把 daily frame 按 `(date, secu_code)` left join 上去。
- `Fill(value)` 填充单 value column 的 null。`Fill(0)` / `Fill(1.5)` 用固定数值填充；`Fill("ffill")` 在每个 `(date, secu_code)` 内 forward fill；`Fill("state")` 用 close_state 填充 price field 的 null。`close` 会先在每个 `(secu_code, date)` 内 forward fill，再用 `daily_k.preclose` 填剩余 null；`open/high/low` 和 ratio 字段用同链路的 close_state 填 null；如果链路包含 `Auction("merge")` 或 `Resample`，close_state 中的 close 聚合固定使用 `last`；`preclose` 使用上一根 close_state。`Fill("state")` 会把 close_state 作为显式 DAG input 构造出来，所以 trace 和 Mermaid 可以看到这条依赖。
- `DailyAgg(value_col=..., agg=...)` 把 intraday frame 聚合成日频 factor value。默认 `apply_to="field"`；如果输入是 ratio 字段，并且希望先聚合 numerator/denominator 再相除，可以显式写 `apply_to="components"`。`sum` 对全 null group 保持 null。
- `Node` / `Layer` 支持可选 `name`。不传时系统会按拓扑顺序生成简洁稳定的 resolved name，例如 `input_0`、`field_0`、`resample_0`；`name` 只用于 trace 和 Mermaid，不改变计算语义，也不参与结构性 `Node.id`。
- `Model.explain_mermaid()` 输出 Mermaid DAG，使用和 trace 一致的 resolved name，方便检查模型结构。
- `Engine.trace(model, date)` 按图顺序 materialize 每个 frame node，返回的 `TraceStep.resolved_name` 可直接和 Mermaid 图对应。

## Field Builders

内置 field builder 按 field name 注册：

- `close`：在 `date`、`secu_code`、`minute` 粒度下，取 `is_last=True` 的 `price`。
- `open`：在 `date`、`secu_code`、`minute` 粒度下，取 `is_first=True` 的 `price`。
- `high`：每个 minute key 的 `price` 最大值。
- `low`：每个 minute key 的 `price` 最小值。
- `volume`：每个 minute key 的 `volume` 加总；如果该 group 全是 null，结果保持 null。
- `no`：每个 minute key 的 `no` 加总；如果该 group 全是 null，结果保持 null。`no` 表示 records 数量，不表示价格顺序。
- `amount`：如果 raw source 有 `amount`，按 minute key 加总；否则使用 `sum(price * volume)` 推导；如果该 group 全是 null，结果保持 null。
- `preclose`：不能直接作为普通 `Field` evaluate；单独写 `Field("preclose")(Input(...))` 会 raise `ValueError`。必须通过 `Fill("state")(Field("preclose")(...))` 使用，由上一根 close_state 推导，第一根用 `daily_k.preclose`。

每个 `Field` builder 应返回 key columns 加一个 public value column。内部 helper column 不应暴露出去。

`vwap` 这类 ratio 字段不再写成 `Field("vwap")`，而是：

```python
from draco_model.layers import Auction, Input, RatioField, Resample

raw = Input(source="trades_tbar")
vwap = Resample("5m", "sum")(
    Auction("merge", agg="sum")(
        RatioField("amount", "volume", alias="vwap")(raw)
    )
)
```

ratio 字段聚合时有两种语义：

```python
from draco_model.layers import Aggregate, DailyAgg, Input, RatioField, Resample

raw_vwap = RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar"))

# 先 sum amount 和 volume，再相除，适合 VWAP 这类 ratio 指标。
vwap_5m = Resample("5m", "sum", apply_to="components")(raw_vwap)
daily_vwap = DailyAgg(value_col="vwap", agg="sum", apply_to="components")(raw_vwap)

# 先算每分钟 vwap，再对 public field 做 mean。
mean_minute_vwap_5m = Resample("5m", "mean", apply_to="field")(raw_vwap)
daily_mean_minute_vwap = Aggregate("daily", "mean", value_col="vwap", alias="value")(raw_vwap)
```

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

source scan 会标准化常见 vendor column name，例如 `SecuCode -> secu_code`、`MinBar -> minute`、`Price -> price`、`Amount -> amount`、`Volume -> volume`、`No -> no`、`isfirst -> is_first`、`islast -> is_last`、`trading_day -> date`。

## 组合字段

当两个 source 产出同名字段时，可以用 `alias` 或 named `Concat` input 避免列名冲突：

```python
from draco_model.layers import Concat, Field, Input

trade_volume = Field("volume", alias="trade_volume")(Input(source="trades_tbar"))
cancel_volume = Field("volume", alias="cancel_volume")(Input(source="cancels_tbar"))

features = Concat()({
    "trade_volume": trade_volume,
    "cancel_volume": cancel_volume,
})
```

`alias` 必须是 public column name，不能以 `__` 开头；`__` 前缀由系统内部 payload 使用，例如 ratio 字段的 numerator/denominator。

## 运行示例

在项目根目录执行：

```powershell
python -m examples.close_last
```

示例会打印 factor result 和 Mermaid DAG。实际运行需要按上面的目录准备本地 parquet 数据。

## TODO

- v1 的 field builder 只按 field name 注册，不按 `(source, field)` 注册。如果未来同名 field 在不同 source 上需要不同语义，再在 v2 里设计 source-aware field registry。
- Source 层需要主动提供 normalized schema columns，例如 `SourceCatalog.schema(source, dates)`，让上层 schema inference 不必通过扫描得到的 LazyFrame 反复 `collect_schema()`。
- 思考 `close_state` 是否也应该进入 payload 体系：目前它是 `Fill("state")` 内部临时状态，后续可以考虑把它作为一种可传递 payload，让 transform / fill 的状态依赖更统一。
- 评估扩展 `Node.kind`：当前主要有 `frame` 和 `condition`，未来可考虑加入 `expression`（列级表达式）、`scalar`（横截面/全局统计值）和 `meta`（calendar、universe、schema、lineage 等 planner 元信息）节点；这是长期架构方向，当前不急做。
- 增加多 model 的 batch planner：一次接收多个 `Model`，合并可复用的 source scan / field / transform 中间节点，按共享 DAG 执行，最后再拆回每个 model 的结果。

