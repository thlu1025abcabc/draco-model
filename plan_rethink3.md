# Hybrid Frame Architecture

## Summary

采用 **hybrid 模式**重构 frame 表达：所有 `kind="frame"` 都视为 logical frame，普通字段是 compound field 的退化形式；复杂指标通过 `FieldSpec` 描述 components / operation / public output，执行层可以继续用 fused physical plan 保持性能。

核心目标：

- 不新增 `kind="compound_frame"`；`Node.kind` 仍以 `frame` / `condition` 为主。
- 升级 `FrameSchema`，从单纯 `columns` 变成 logical fields + physical columns。
- `RatioField`、未来的 `SubField`、`RollingCorrField`、跨 source compound field 都走同一个 field-spec 模型。
- `Auction` / `Resample` 对 compound field 的 components 做 aggregation，再重算 public output。
- `.project()` 显式把 logical/component frame 投影成只含 public columns 的普通物理 frame，但仍然保持 lazy，不 `.collect()`。

## Key Design

### Frame Schema

新增 logical schema 概念：

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str
    components: tuple[str, ...]
    operation: str
    public: bool = True

@dataclass(frozen=True)
class FrameSchema:
    keys: tuple[str, ...]
    columns: tuple[str, ...]
    fields: dict[str, FieldSpec]
```

普通 field 也是 `FieldSpec`：

```python
Field("close") -> FieldSpec(
    name="close",
    components=("close",),
    operation="identity",
)
```

ratio field：

```python
RatioField("amount", "volume", alias="vwap") -> FieldSpec(
    name="vwap",
    components=("__ratio_vwap_num", "__ratio_vwap_den"),
    operation="ratio",
)
```

### Public API

保留现有 API：

```python
Field("close")(Input(source="trades_tbar"))
RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar"))
```

新增显式 projection：

```python
vwap = RatioField("amount", "volume", alias="vwap")(raw)
vwap_frame = vwap.project()
```

`.project()` 含义：

- 不是 `.collect()`。
- 只是插入 `Node(kind="frame", op="project")`。
- 输出只保留 key columns + public logical fields。
- 丢弃 internal component columns。

## Implementation Changes

### Schema / Runtime

- `FrameSchema` 保留 `columns` 以兼容当前 executor，但新增 `fields`。
- `context.infer_schema(node)` 返回完整 logical schema。
- `Model.output` 仍要求 `kind == "frame"`。
- `Engine.evaluate()` 仍返回 `pl.LazyFrame`，不改变 executor 返回协议。

### Field Builders

- `Field` 生成 identity field spec。
- `RatioField` 继续可用 fused payload physical plan：
  - physical columns: `vwap`, `__ratio_vwap_num`, `__ratio_vwap_den`
  - logical field: `vwap = ratio(num, den)`
- 后续可增加：
  - `SubField(left, right, alias=...)`
  - `AddField(fields, alias=...)`
  - `RollingCorrField(left, right, window=..., alias=...)`
- v1 先只重构 `RatioField`，不要一次性实现所有 compound field 类型。

### Transform Semantics

- `Auction("merge")` / `Resample(...)` 根据 `FrameSchema.fields` 判断 field 类型。
- identity field：聚合 public value column。
- ratio field：用同一个 aggregation 聚合 numerator / denominator，再重算 ratio。
- 继续使用 fused group_by，不退化成两次 group_by + join。
- `Fill("state")` / `Fill(0)` 默认作用在 public output column 上，保持现有语义。
- `Fill("ffill")` v1 保持当前 ratio 行为：ffill components 后重算 ratio。

### Project

新增 `Node.project()`：

```python
def project(self) -> Node:
    return Node(kind="frame", op="project", inputs={"input": self})
```

`project` executor：

- 根据 schema 找 public fields。
- select key columns + public field columns。
- 不输出 internal component columns。

`project` schema：

- fields 保留 public field。
- columns 只保留 keys + public columns。
- component 信息可丢弃，因为 project 后成为普通 physical frame。

## Layer Behavior

- `Concat` 支持 logical frame，但只合并 public fields；如果输入仍带 internal components，v1 要求用户先 `.project()`，错误信息明确。
- `DailyAgg` 作用在 public field output 上，建议输入是 projected frame。
- `Filter` 只允许使用 public columns；如果引用 internal component column，raise clear error。
- `Trace` 对 project 前后的 frame 都可 materialize；project 前 trace 可以看到 internal component columns，project 后只看到 public columns。

## Test Plan

- `RatioField` schema 包含 public `vwap` 和 internal components。
- `RatioField(...).project()` 输出只包含 key columns + `vwap`。
- `Resample("5m", "sum")(RatioField(...))` 仍使用 fused aggregation，结果与当前 payload 实现一致。
- `Auction("merge", agg="sum")(RatioField(...))` 仍先聚合 num/den 再相除。
- `Fill(0)(RatioField(...))` 语义保持：填 public output，不变成填 num/den。
- `Fill("ffill")(RatioField(...))` 语义保持：components ffill 后重算 ratio。
- `Concat` 对未 project 的 ratio frame 报 clear error；对 `.project()` 后输入正常合并。
- 全量运行：
  ```powershell
  C:\Users\th\anaconda3\python.exe -m pytest -q
  C:\Users\th\anaconda3\python.exe -m compileall -q draco_model examples tests
  ```

## Assumptions

- v1 不新增 `kind="compound_frame"`。
- v1 不改变 `Engine.evaluate()` 返回 `pl.LazyFrame` 的协议。
- v1 只把现有 `RatioField` 迁移到 hybrid schema；减法、加法、rolling corr、跨 source compound field 先作为后续扩展。
- `.project()` 是显式 API，不自动插入。
- 性能优先：compound field transform 必须允许 fused executor，不采用“两次 group_by + join”作为默认实现。
