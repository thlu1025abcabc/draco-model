# Operator DAG 重构 — 评审与整改计划

针对 General Operator DAG 重构的代码评审结论与待办。按优先级排列，每条给出问题、定位、整改方向与验收标准。

> 状态图例：✅ 已完成 ｜ 🟦 部分完成 ｜ ❌ 未开始
> 最近一次核对：`operators.py` / `core.py` / `runtime/execution.py` / `expressions.py` 与上轮评审时一致（未改）；本轮改动集中在 `aggregate.py`(auction)、`data/source.py`+`layers/source.py`(schema)、`names.py`(key column 防护)、README、测试。

## 总体结论

方向正确、质量不错。核心抽象（`Node(kind/op/params/inputs)` + executor/schema 双注册表 + 结构化 id 去重 + memo）干净、可 trace、可画图。主要风险是 schema/执行逻辑双写的隐式契约，以及滚动算子的 lookback 陷阱——这两类**本轮尚未触及**。

---

## ✅ 本轮已完成（plan 之外的改进）

这些不在原 plan 编号里，但确实是这轮做掉的、有价值的工作：

1. **Auction merge 改为按输出频率取目标 bar**：`aggregate.py:282` `_auction_merge_targets`，1m→1456、5m→1455、15m→1445；修掉了 5m/15m 收盘集合竞价落点错误（这个 bug 我上轮没发现）。测试 `test_auction_merge_targets_follow_output_frequency`。
2. **Daily aggregate 现在也先应用 auction 策略**再按日聚合（`aggregate.py:64-71` + `_merge_auction_frame`）。测试 `test_daily_aggregate_applies_auction_policy`。
3. **`SourceCatalog.schema(source, dates)` 固定 schema**：`data/source.py`(+85)，`layers/source.py:33` 改用它，schema 推断不再依赖 parquet `collect_schema()`。解决了旧 README TODO。`test_source_validate.py`(+87) 扩展。
4. **Public alias / Join input name 现在还禁止 key column 名**（`date/secu_code/minute`）：`names.py` + README 第 36 行 + 测试 `test_public_aliases_cannot_use_payload_prefix` 新增断言。
5. **README 增补 payload 语义说明**（即此前确认为有意行为的「payload 透传」「FillNull state 通用化」两条），并更新 auction 文档。

---

## 🔴 高优先级

### ❌ H1. schema 推断与执行逻辑双写，靠隐式列名契约同步
- 定位：算子 `operators.py` `_op_schema` ↔ `_frame_op_executor`；聚合 `aggregate.py:221` `_aggregate_schema_parts` ↔ `aggregate.py:159` `_aggregate_values`。
- 现状：未动。本轮新增的 `_merge_auction_frame` 又多了一处调用 `_aggregate_values`，双写面只增不减。
- 整改：抽一个布局函数返回 `(columns, fields, component_names)`，schema 与 executor 共用，作单一事实源。
- 验收：算子/聚合列名与 fields 只在一处定义；新增「schema 推断列 == 执行产出列」一致性测试。

### ❌ H2. 滚动/窗口算子的 window 与 lookback_days 脱钩（静默全 null）
- 定位：`operators.py` `Op` 构造（不校验 window）、`_window_op`（执行期才读 `window`）、`_common_lookback`。
- 现状：未动。
- 整改：`Op` 构造期对 WINDOW_OPS 强制校验 `window`；让 window 算子把 `lookback_days` 至少抬到 `window`。
- 验收：缺 window 构造期即报错；新增「window=N 时实际扫描 ≥N 天、结果非全 null」测试。

### ❌ H3. 分钟粒度滚动按天重置，未文档化
- 定位：`operators.py` `_window_op` minute 分支 `over(date, secu_code)`。
- 现状：未动；本轮 README 改了不少但未加此说明。
- 整改：README 明确「分钟滚动 = 日内、按天重置」；如需跨日另设语义。
- 验收：README 增补说明。

---

## 🟡 中优先级

### ❌ M1. `Source` 用 `__new__` 返回 `Node`，风格不一致
- 定位：`source.py:10-21`。
- 现状：本轮改了 `source.py`，但只是把 schema 推断换成 `context.sources.schema(...)`；`Source.__new__` 仍在。
- 整改：改为 `def Source(...) -> Node`。
- 验收：示例与测试不变即通过。

### ❌ M2. `.alias()` 在多字段 frame 上的失败推迟到执行期
- 定位：`operators.py` `alias_node` → `_rename` → `_single_value_column`。
- 现状：未动。
- 整改：构造期即检测多字段并报错。
- 验收：对多字段 frame `.alias()` 构造期报错的测试。

### ❌ M3. `collect` 假设日频「value」输出，分钟粒度会静默重复行
- 定位：`execution.py:127` `format_factor_output`。
- 现状：未动。
- 整改：加 grain 守卫（非日频输出报错或显式聚合）。
- 验收：分钟粒度直接 `collect` 报清晰错误的测试。

---

## 🟢 低优先级 / 收尾

- ❌ L1. 结构化 id 不含 `name`（`core.py`）：仅 name 不同的节点会折叠丢显示名。未动。
- ❌ L2. `Condition` 是 frozen dataclass 却持有 dict（`core.py`）：一旦被 hash 即 `TypeError`。未动。
- ❌ L3. `sum_or_null` 依赖 `expr.count()`=非空计数（`expressions.py`）：随 polars 版本变化，建议钉版本 + 加测试。未动。
- ✅ L4. 示例 dead code：已删除 `examples/top_volume_close_mean.py` 未用的 `trace_`。
- ✅ L5. 空 `layers/inputs/` 包已删除。

---

## 测试缺口（补充）— 🟦 部分

本轮新增了 auction / source schema / key-column 相关测试，并补了部分原列缺口：
- ✅ `Threshold` 过滤器已有真实过滤测试；`FillNull` 的 `"ffill"` 和数值填充已有测试。
- ❌ `test_examples.py` 已删 → 三个 example 无冒烟测试，建议加回最小冒烟。
- ✅ 多日期 `collect`、多 source、`Op` 缺 `window` 报错路径已覆盖。

---

## 建议执行顺序

1. H2（window/lookback 校验，最易静默出错）
2. H1（抽 layout 单一事实源，最大长期收益）
3. 补 `Threshold` 与 example 冒烟测试
4. 其余按优先级推进
