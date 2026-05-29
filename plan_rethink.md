# Draco Model 待重新评估的优化项

收录尚未决定方向、需要进一步思考/测量后再决定是否做的项。每项要做之前先确认：是否还成立、ROI 多大、是否被其他设计变化覆盖。

---

## #5 — `collect_schema()` 在 hot path 反复调用

### 现状

`layers/transforms.py` 里很多 helper 各自调 `collect_schema()`：

| 函数 | 调用 |
|---|---|
| `_value_columns` | `frame.collect_schema().names()` |
| `_value_and_payload_columns` | 内部调 `_ratio_payloads` 又拿一次 schema |
| `_single_value_column` | 调 `_value_columns` 再拿一次 |
| `_fill_executor` | 自己拿 + 调上面几个 helper 各自再拿 |
| `_auction_frame` | `_value_columns(frame)` + `_ratio_payloads(..., frame.collect_schema().names())` |
| `_resample_frame` | 同上 |
| `_aggregate_frame` | `frame.collect_schema().names()` |

一次 `Fill('state')` 链路上的求值能调 5–8 次 `collect_schema()`。

### 为什么是「待重新评估」

**Polars 的 `collect_schema()` 不执行查询**，但会遍历 plan tree 做 schema resolution。在 plan 浅时几乎零成本，plan 深时累积有开销。

需要先**测量**：
- 跑一个典型 model（比如 `examples/close_last`），用 cProfile 或 `time.perf_counter()` 包住每次 `collect_schema()` 调用
- 看实际累积 ms 是不是真的可观

如果实测每次 model run 的 schema resolve 总和 <10ms，**不值得做**——增加 helper 签名复杂度换不到性能。

### 假设决定做，改法

入口拿一次 schema names，往下传：

```python
@register_executor("fill")
def _fill_executor(node, context):
    parent_frame = context.evaluate(node.inputs["input"])
    columns = parent_frame.collect_schema().names()   # 只问一次
    # _value_columns(columns)、_ratio_payloads(values, columns) ...
```

helper 签名从 `_value_columns(frame)` 改成 `_value_columns(columns: list[str])`。`field.py` 里 `FieldBuilder = Callable[[pl.LazyFrame, list[str]], pl.LazyFrame]` 已经是这个模式。

### 影响范围（如果做）

- `draco_model/layers/transforms.py` — `_value_columns`、`_value_and_payload_columns`、`_single_value_column`、`_fill_executor`、`_auction_frame`、`_resample_frame`、`_aggregate_frame`
- `draco_model/layers/combine.py` — `_value_columns`、`_key_columns`、`_concat`
- `draco_model/layers/aggregate.py` — 已经只调一次，OK

### 决策清单

- [ ] 做 profiling，量化 `collect_schema()` 实际成本
- [ ] 如果 <10ms/run → 跳过这一项
- [ ] 如果 >50ms/run → 落地

---

## #6 — `_minute_bucket_map` 每次 Resample 重建（已落地）

### 现状

```python
# layers/transforms.py:222-227
def _minute_bucket_map(minutes: list[int], interval: int) -> pl.LazyFrame:
    continuous = [minute for minute in minutes if minute not in AUCTION_MINUTES]
    rows = []
    for idx, minute in enumerate(continuous):
        rows.append({"minute": minute, "__bucket_minute": continuous[(idx // interval) * interval]})
    return pl.DataFrame(rows).lazy()
```

每次 `_resample_frame` 调用都重建。输入只依赖 `(minutes, interval)`，A 股 minute 列表固定（`MinuteCalendar.VERSION = "ashare-fixed-v1"`），interval 取值有限。

### 为什么是「待重新评估」

构造 ~240 行 list of dicts + `pl.DataFrame` 单次成本很低（亚毫秒级）。**只有 Resample 在 hot path 才值得 cache**：

- 单 model 单 date：可能只用 1-2 次 Resample，cache 收益 ≈ 0
- 100 model × 几十 date：累积有 cache 收益，但每次开销本来就小

跟 #5 一样需要先量化。

### 假设决定做，两种方案

| 方案 | 代码 | 优劣 |
|---|---|---|
| **A. `@lru_cache`** | 函数签名改 `tuple[int, ...]` + `@lru_cache` | 1 行；调用点要 `tuple(minbars())`；全局 cache 不易测试 |
| **B. 挂到 `MinuteCalendar`** | `calendar.bucket_map(interval) -> pl.LazyFrame`，内部 dict cache | 语义对（bucket 属于 calendar）；调用点更干净；calendar 已有 `VERSION` 配合 |

讨论倾向 B，理由：
- `MinuteCalendar` 已有 `VERSION = "ashare-fixed-v1"`，明显在为多 calendar 铺路
- 全局 cache 状态对测试不友好
- 调用点从 `_minute_bucket_map(context.minute_calendar.minbars(), interval)` 变成 `context.minute_calendar.bucket_map(interval)`，更 OO

### 影响范围（如果做）

- `draco_model/market/minute_calendar.py` — 加 `bucket_map` 方法 + 缓存
- `draco_model/layers/transforms.py:149, 222-227` — 改调用点，删本地 helper

### 落地结论

- 已选择方案 B：`MinuteCalendar.bucket_map(interval)`。
- cache 存 `pl.DataFrame`，对外返回 `.lazy()`，避免缓存未 materialize 的 lazy plan。
- `Resample` 直接调用 `context.minute_calendar.bucket_map(interval)`。
- `transforms.py` 中的本地 `_minute_bucket_map` helper 已删除。

---

## 共同主题

#5 和 #6 都属于**没有实测前 ROI 不明朗**的微优化。建议：

1. 先把 #8 / #14 / #4 这些有明确语义/可读性价值的项做掉
2. 等 cloud 部署上来跑出真实 workload 后，用 profiler 看 hot spot
3. 如果 #5 / #6 真的进 top 10，再回头处理

否则容易陷入「凭直觉做微优化但不知道有没有用」的陷阱。
