# Draco Model 架构改进计划

本文件只记录尚未完成或已暂缓的架构事项。已经落地的内容不再保留在这里，包括：

- registry bootstrap
- schema inference memoization
- Condition 节点瘦身
- `FillNull("state")` 基于 `FieldInfo.grain_path` 的 lineage 修正
- `Grid` layer、grid policy 推断以及 auction removal 传播修正
- `FramePlan` 持有 `FrameSchema`，不再复制 schema 字段

---

## 1. 修正 daily aggregate 的 auction merge 语义

### 背景

`tests/test_operator_dag.py` 中 `test_daily_aggregate_applies_auction_policy` 已标记一个 known small bug。当前测试先记录现状，但还没有重新审视 daily aggregate 下 `auction="merge"` 的准确期望。

需要确认的问题：

- minute 级 `auction="merge"` 已按目标频率把 auction rows 合入非 auction 第一根/最后一根 bar。
- daily aggregate 在按日聚合前也会应用 auction policy。
- 对 `mean`、`sum`、`first`、`last` 等不同 agg，daily 层的 merge/drop/keep 是否都应该有明确且一致的期望。

### 建议方案

1. 先重新列出一个小 fixture 中 keep/drop/merge 的中间 minute rows。
2. 分别计算 daily `sum`、`mean`、`first`、`last` 的预期。
3. 更新 `test_daily_aggregate_applies_auction_policy`，让测试表达真实语义，而不是只保护当前实现。
4. 若实现与新期望不一致，再改 `aggregate.py`。

### 验证

- `test_daily_aggregate_applies_auction_policy` 覆盖 keep/drop/merge。
- 全量测试通过。

---

## 2. 重新设计 `Aggregate(apply_to="field")` 的 payload 处理

### 问题

当前 `Aggregate(apply_to="field")` 会直接聚合 public field，同时把 payload 也按同一个 `agg` 聚合后保留下来。这有时会误导 debug/lineage 语义：

- public field 是聚合后的结果。
- payload 虽然被保留，但不一定还能完整重算这个聚合后的 public field。
- 对 ratio、filled field、state-filled field，payload 的含义尤其容易变模糊。

### 待讨论选项

- `drop`：field aggregation 后默认丢弃 payload，只保留 public field。
- `metadata only`：不保留物理 payload 列，只保留 source / grain / operator metadata。
- `separate strategy`：payload 有独立聚合策略，不默认复用 public field 的 `agg`。
- `keep current but document`：继续保留物理列，但明确它只用于 lineage/debug，不承诺可重算。

### 验证

- 需要覆盖 `vwap`、`FillNull("state")` 后再 aggregate、daily aggregate 等场景。
- README 和 docs 中 payload 语义需要同步。

---

## 3. Payload / metadata 的整体架构

### 需要继续讨论

- payload 是否应该默认跨所有 layer 保留，还是只在 trace/debug/component recompute 需要时保留。
- `close_state` 是否应该进入 payload 或 operator metadata 体系。
- `FieldInfo.name`、`FieldInfo.column`、`source`、`grain_path`、`components` 之间是否需要重新划边界。
- fill 后再次 aggregate 的功能是否需要显式语义，而不是依赖当前 payload 保留行为。

### 暂定原则

- 在没有 `Project()` 的情况下，当前仍保留 payload。
- 保留的 payload 可以用于 debug/lineage，但不能默认理解为一定可以重算当前 public field。
- 后续如果要丢 payload，优先通过 plan/schema 统一决策，而不是在各 executor 里零散 drop。

---

## 4. Batch planner / optimizer

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
- 对相同子图继续依赖 structural id 去重，但在 physical execution 层做更强的共享。

### 风险

- 这是性能优化，不应改变现有 DAG 的语义。
- 需要先有 profiling 或至少一个明确 benchmark，避免过早复杂化。

---

## 5. Smart Join 暂缓

### 当前决定

Smart Join 先不做。原因是它需要额外维护 row-domain metadata，例如：

- `sorted_by`
- `unique_by_keys`
- `grid_domain`

同时还要严格保证每个 layer 之后的顺序和 key 唯一性。这个成本现在偏高，暂时不值得拉进主线。

### 未来若重启

只有当两个输入都能证明满足以下条件时，`Join()` 才可以跳过 key join，改成横向拼接：

- 相同 key domain。
- 相同 frequency / auction grid。
- 已按 canonical keys 排序。
- 每个 key 只有一行。
- public columns / payload prefix 规则与普通 `Join()` 完全一致。

需要注意：

- `Grid()(Source(...))` 不能直接视为 unique，因为 raw source 可能同 key 多行。
- `Where(...)` 后即使 filter 保序，也会改变 row domain，不能直接保留 smart-join eligibility。
- fallback join 后不要依赖 Polars 默认输出顺序，应显式 sort。

---

## 建议执行顺序

1. 修正 daily aggregate 的 auction merge 测试与实现。
2. 重新设计 `Aggregate(apply_to="field")` 的 payload 语义。
3. 继续讨论 payload / metadata 整体架构。
4. 有 benchmark 后再推进 batch planner / optimizer。
5. Smart Join 继续暂缓。
