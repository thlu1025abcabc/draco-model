# Operator DAG 重构 — 评审与整改计划

针对 General Operator DAG 重构的代码评审结论与待办。按优先级排列，每条给出问题、定位、整改方向与验收标准。

> 状态图例：✅ 已完成 ｜ 🟦 部分完成 ｜ ❌ 未开始
> 本文件只保留**未结清**的待办。已落地或按设计不做的条目已移除（见下行）。

> 已结清并移除：H1（FramePlan 单一事实源，schema/executor 共享布局）、H2-校验（`Op` 构造期强制 window 正整数）、H3-功能（`cross_day` 参数，默认按天重置可跨日）、H4（fixed source schema contract 校验与测试）、M1（`Source` 改工厂函数）、M3（`collect` grain 守卫 `_validate_collect_schema`）、M4（删除 `register_schema` 孤儿注册表）、L3（`sum_or_null` 行为测试）、L4/L5、L6（修正 aggregate plan 返回类型注解）、auction merge 落点 + daily auction 策略、`SourceCatalog.schema` 固定 schema、key-column 别名防护、README payload 语义。
> 按设计不做（默认行为）：H2-lookback（滚动 `lookback_days` 不自动抬到 window）、H3（分钟滚动按天重置 / `cross_day`，无需 README）、M2（`.alias()` 多字段在推断期报错即可）、L2（`Condition` 不会被 hash）。

## 总体结论

方向正确、质量不错。核心抽象（`Node(kind/op/params/inputs)` + executor/plan 双注册表 + 结构化 id 去重 + memo）干净、可 trace、可画图。本轮把 schema/执行双写收敛进了 `FramePlan` 单一事实源（H1 已结清），并把 fixed source schema 收敛成可验证的数据 contract（H4 已结清）。

---

## 🟢 低优先级 / 收尾

- ❌ L1. 结构化 id 不含 `name`（`core.py`）：仅 name 不同的节点会折叠丢显示名。未动。

---

## 测试缺口

- ❌ `test_examples.py` 已删 → 三个 example 无冒烟测试，建议加回最小冒烟。

---

## 建议执行顺序

1. 补 example 冒烟测试
2. 其余按优先级推进
