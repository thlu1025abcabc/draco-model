# Draco Model 架构改进计划

本文件只记录尚未完成、需要继续讨论、或已经明确暂缓的事项。已落地内容不再作为 todo 维护。

---

## 1. Batch planner / optimizer

### 目标

减少重复 scan、重复 join、重复 operator branch，尤其是同 source 的常见 metric 组合。

典型例子：

- `Metric("amount", raw)`
- `Metric("volume", raw)`
- `Metric("vwap", raw)`

这些都来自同一个 `Source("trades_tbar")`，理论上可以 fuse 成更少的 physical plan。

### 可能方向

- batch planner 合并多个 model / metric 中可复用的 source scan。
- optimizer fuse 同 source 的 `amount`、`volume`、`vwap`。
- 对相同子图继续依赖 structural id 去重，但在 physical execution 层做更强共享。

### 风险

- 这是性能优化，不应改变现有 DAG 语义。
- 需要先有 profiling 或至少一个明确 benchmark，避免过早复杂化。

---

## 2. Smart Join 暂缓

### 当前决定

Smart Join 暂时不做。它需要额外维护 row-domain metadata，例如：

- `sorted_by`
- `unique_by_keys`
- `grid_domain`

同时还要严格保证每个 layer 之后的顺序和 key 唯一性。当前成本偏高，不拉进主线。

### 未来若重启

只有当多个输入都能证明满足以下条件时，`Join()` 才可以跳过 key join，改成横向拼接：

- 相同 key domain。
- 相同 frequency / auction grid。
- 已按 canonical keys 排序。
- 每个 identity key 只有一行。
- public columns / payload prefix 规则与普通 `Join()` 完全一致。

需要注意：

- `Grid()(Source(...))` 不能直接视为 unique，因为 raw source 可能在同一 grid key 下有多行。
- `Where(...)` 后即使 filter 保序，也会改变 row domain，不能直接保留 smart-join eligibility。
- fallback join 后不要依赖 Polars 默认输出顺序，应显式 sort。

---

## 建议执行顺序

1. 有 benchmark 后再推进 batch planner / optimizer。
2. Smart Join 继续暂缓。
