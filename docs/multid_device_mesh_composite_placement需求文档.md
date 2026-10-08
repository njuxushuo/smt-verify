# Multi-Dimensional Device Mesh & Composite Placement 需求文档

## 1. 背景与目标

当前 `smt-verify` 的分布式关系模型实际上是 **1D device mesh + 单 placement**：

```text
world_size = N

Relation =
    Replicate
or  Shard(dim)
or  Partial(sum)
```

当前 `ShardRelation` 直接使用：

```text
single_dim / world_size
rank * local_extent
```

来定义 shard shape 与 global index mapping，因此隐含了：

```text
flat rank == shard coordinate
全部 world_size ranks 都参与同一次 shard
一个 tensor 只有一个 placement
```

这对于当前 1D MVP 是正确的，但不足以覆盖真实的多维并行：

```text
DP × TP
TP × CP
DP × TP × CP
以及其他 N-D DeviceMesh
```

真实目标应升级为：

```text
N-D Device Mesh

mesh axis 0 -> placement 0
mesh axis 1 -> placement 1
...
mesh axis n -> placement n
```

例如：

```text
mesh_shape = [2, 4]

placements = [
    Replicate,
    Shard(dim=1)
]
```

表示：

```text
mesh axis 0, size 2:
    Replicate

mesh axis 1, size 4:
    Shard(tensor_dim=1)
```

又例如：

```text
mesh_shape = [2, 4]

placements = [
    Shard(dim=0),
    Shard(dim=2)
]
```

表示同一个 logical tensor 同时沿两个不同 tensor dimensions 被二维 mesh 切分。

本阶段目标：

> **将当前 1D world-size relation abstraction 升级为 N-D Device Mesh + per-mesh-axis composite placements，同时保持现有 local operator semantics 基本不变。**

本阶段完成后，后续 Reduction 和 Collective 必须建立在新的 mesh abstraction 上。

---

## 2. 核心设计原则

### 2.1 Mesh 与 Placement 必须分离

Mesh 描述：

> **设备如何组织。**

Placement 描述：

> **logical tensor 在每个 mesh axis 上如何分布。**

例如：

```text
world_size = 8
mesh_shape = [2, 4]
```

采用 row-major rank mapping：

```text
rank 0 -> (0, 0)
rank 1 -> (0, 1)
rank 2 -> (0, 2)
rank 3 -> (0, 3)
rank 4 -> (1, 0)
rank 5 -> (1, 1)
rank 6 -> (1, 2)
rank 7 -> (1, 3)
```

tensor：

```text
X [8, 16]
```

relation：

```text
placements = [
    Replicate,
    Shard(1)
]
```

则 local shape：

```text
[8, 4]
```

而：

```text
placements = [
    Shard(0),
    Shard(1)
]
```

则 local shape：

```text
[4, 4]
```

### 2.2 Operator semantics 与 distributed relation semantics 继续分离

当前已经实现的：

```text
Add
Mul
MatMul
Transpose
Reshape
Squeeze
Unsqueeze
Expand
```

都是 rank-local operator semantics。

这些 operator 不应该知道：

```text
mesh shape
mesh coordinate
placement
```

因此：

> **不要给现有 OperatorSemantics 接口增加 mesh / placement 参数。**

保持：

```text
validate_concrete_shapes(...)
shape_constraints(...)
symbolic_execute(...)
```

只处理 local tensor semantics。

本阶段主要修改：

```text
Stage Verification IR
Mesh helpers
RelationSpec
Relation concrete semantics
Relation reduced-shape semantics
Relation value semantics
Relation encoder
Lemma semantic identity / serialization
Tests / JSON fixtures
```

---

## 3. 当前代码中需要被替换的 1D 假设

### 3.1 StageSpec

当前只有 scalar：

```text
world_size
```

没有 mesh topology。

### 3.2 RelationSpec

当前只能表示：

```text
Replicate
or Shard(dim)
or Partial(sum)
```

不能表达：

```text
[Replicate, Shard(dim)]
[Shard(dim0), Shard(dim1)]
[Shard(dim), Partial(sum)]
```

### 3.3 Shard relation

当前 shard shape 使用：

```text
single_shape[dim] / world_size
```

当前 value mapping 使用：

```text
global_index[dim]
=
rank * local_extent + local_index[dim]
```

必须升级为：

```text
mesh-axis size
mesh coordinate
```

而不能继续直接使用 flat rank / whole world_size。

### 3.4 Relation encoder

当前：

```text
distributed_tensors[rank]
```

按 flat rank 解析 local tensor，这一点可以继续保留。

但 relation semantics 必须额外拿到：

```text
mesh
rank <-> mesh coordinate
```

---

## 4. Stage Verification IR 扩展

### 4.1 DeviceMeshSpec

新增：

```python
@dataclass(frozen=True)
class DeviceMeshSpec:
    shape: tuple[int, ...]
```

第一版只需要：

```text
shape
```

暂不增加：

```text
axis names
process groups
physical topology
node/device metadata
```

要求：

```text
rank(mesh) >= 1
每个 mesh extent 都是正整数
prod(mesh.shape) == world_size
```

### 4.2 PlacementSpec

新增：

```python
@dataclass(frozen=True)
class PlacementSpec:
    type: str
    dim: int | None = None
    reduce_op: str | None = None
```

第一版 placement types：

```text
replicate
shard
partial
```

规则：

```text
Replicate:
    type="replicate"
    dim=None
    reduce_op=None

Shard:
    type="shard"
    dim=<tensor_dim>
    reduce_op=None

Partial(sum):
    type="partial"
    dim=None
    reduce_op="sum"
```

### 4.3 RelationSpec

Canonical internal IR 改为：

```python
@dataclass
class RelationSpec:
    single_tensor: str
    distributed_tensors: tuple[str, ...]
    placements: tuple[PlacementSpec, ...]
```

要求：

```text
len(placements) == len(stage.mesh.shape)
```

不要增加：

```text
multi_shard
replicate_shard
shard_partial
```

等组合 relation type。

所有组合都由：

```text
tuple[PlacementSpec, ...]
```

自然表达。

### 4.4 StageSpec

建议改为：

```python
@dataclass
class StageSpec:
    name: str
    world_size: int
    mesh: DeviceMeshSpec
    single: ProgramSpec
    distributed: dict[int, ProgramSpec]
    input_relations: tuple[RelationSpec, ...]
    output_relation: RelationSpec
```

本阶段暂时保留：

```text
world_size
```

以减少已有 verifier / program orchestration 的改动。

但必须保证：

```text
world_size == prod(mesh.shape)
```

因此 `world_size` 是 mesh total size 的一致性字段，而不是 relation semantics 中的 shard factor。

---

## 5. JSON 格式

### 5.1 新 Multi-D Stage

```json
{
  "name": "transpose_2d_mesh",
  "world_size": 4,
  "mesh": {
    "shape": [2, 2]
  }
}
```

### 5.2 新 Relation 格式

```json
{
  "single_tensor": "X",
  "distributed_tensors": ["X0", "X1", "X2", "X3"],
  "placements": [
    {"type": "replicate"},
    {"type": "shard", "dim": 1}
  ]
}
```

或者：

```json
{
  "single_tensor": "X",
  "distributed_tensors": ["X0", "X1", "X2", "X3"],
  "placements": [
    {"type": "shard", "dim": 0},
    {"type": "shard", "dim": 1}
  ]
}
```

或者：

```json
{
  "single_tensor": "X",
  "distributed_tensors": ["X0", "X1", "X2", "X3"],
  "placements": [
    {"type": "shard", "dim": 0},
    {"type": "partial", "reduce_op": "sum"}
  ]
}
```

---

## 6. 旧 1D JSON backward compatibility

已有 1D fixture 不要求全部手工改写。

Loader 必须支持 legacy JSON。

旧 Stage 没有 `mesh` 时：

```text
mesh.shape = [world_size]
```

旧 relation：

```json
{"type":"shard","dim":1}
```

canonicalize 为：

```text
placements = [Shard(1)]
```

旧：

```json
{"type":"replicate"}
```

canonicalize 为：

```text
placements = [Replicate]
```

旧：

```json
{"type":"partial","reduce_op":"sum"}
```

canonicalize 为：

```text
placements = [Partial(sum)]
```

要求：

> **Loader 输出的 canonical `RelationSpec` 始终使用 `placements`，不要在内部同时维护两套 relation representation。**

即：

```text
legacy JSON
      ↓
loader normalization
      ↓
new canonical IR
```

若显式 `mesh.shape` 为 Multi-D，则 relation 必须使用 `placements`。不要猜测 legacy `type="shard"` 属于哪个 mesh axis。

如果同时出现 `placements` 与 legacy `type/dim/reduce_op`，直接拒绝。

---

## 7. Mesh rank mapping

实现统一 helper，例如：

```python
def mesh_size(mesh: DeviceMeshSpec) -> int: ...
def rank_to_coordinate(rank: int, mesh: DeviceMeshSpec) -> tuple[int, ...]: ...
def coordinate_to_rank(coordinate: tuple[int, ...], mesh: DeviceMeshSpec) -> int: ...
```

采用 row-major mapping。

例如：

```text
mesh [2,4]

0 -> (0,0)
1 -> (0,1)
2 -> (0,2)
3 -> (0,3)
4 -> (1,0)
5 -> (1,1)
6 -> (1,2)
7 -> (1,3)
```

必须保证：

```text
coordinate_to_rank(rank_to_coordinate(r)) == r
```

对所有合法 rank 成立。

建议提供能枚举“固定其他 coordinate，仅变化指定 mesh axes”的 helper，供 Partial 和未来 Collective 复用。

不要在多个 relation 实现中重复写 rank arithmetic。

---

## 8. 第一版 Composite Placement 支持范围

必须支持旧 1D：

```text
[Replicate]
[Shard(d)]
[Partial(sum)]
```

Multi-D 必须支持：

```text
[Replicate, Replicate]
[Replicate, Shard(d)]
[Shard(d), Replicate]
[Shard(d1), Shard(d2)]
[Shard(d), Partial(sum)]
[Partial(sum), Shard(d)]
[Replicate, Partial(sum)]
[Partial(sum), Replicate]
```

基础实现应面向：

```text
arbitrary mesh rank >= 1
```

而不是写死 2D。

---

## 9. 第一版限制：同一 tensor dimension 不被多个 mesh axes 重复 Shard

当前 milestone 暂时要求：

```text
同一个 tensor dimension 最多出现一次 Shard(dim)
```

允许：

```text
mesh [2,4]
[Shard(0), Shard(1)]
```

暂不允许：

```text
mesh [2,4]
[Shard(1), Shard(1)]
```

后者涉及 hierarchical/strided shard ordering，需要额外定义 global index mapping。

第一版在 concrete validation 中明确拒绝。

---

## 10. Composite relation concrete shape semantics

定义：

```text
single_shape = [D0, D1, ..., Dn]
mesh_shape   = [M0, M1, ..., Mk]
placements  = [P0, P1, ..., Pk]
```

对每个 tensor dimension `d`，定义 shard factor：

\[
F_d =
\prod_{\{a \mid P_a = Shard(d)\}} M_a
\]

在当前 milestone 下：

```text
Fd = 1
or
Fd = 对应 mesh axis size
```

要求：

\[
single\_shape[d] mod F_d = 0
\]

local shape：

\[
local\_shape[d] = single\_shape[d] / F_d
\]

`Replicate` 和 `Partial(sum)` 都不改变 local shape。

所有 ranks 的 local tensor shape 在当前 regular-shard scope 下必须相同。

示例：

```text
single [8,16]
mesh [2,4]
placements [Replicate, Shard(1)]
local [8,4]
```

```text
single [8,16]
mesh [2,4]
placements [Shard(0), Shard(1)]
local [4,4]
```

```text
single [8,16]
mesh [2,4]
placements [Shard(0), Partial(sum)]
local [4,16]
```

---

## 11. Composite relation reduced-shape semantics

Multi-D 后，relation reduced-shape constraints 直接由 placements 推导。

对每个 tensor dimension `d`：

\[
F_d =
\prod_{\{a \mid P_a = Shard(d)\}} mesh\_shape[a]
\]

要求 reduced：

\[
single'_d mod F_d = 0
\]

以及每个 rank：

\[
local'_{r,d} \cdot F_d = single'_d
\]

当 `Fd = 1` 时自然变成：

\[
local'_{r,d}=single'_d
\]

因此 Replicate 和 Partial 都保持 same shape。

不要继续使用整个 `world_size` 作为每个 shard dimension 的 factor。

---

## 12. Composite relation value semantics

### 12.1 Shard index mapping

对于 mesh coordinate：

```text
c = (c0, c1, ..., ck)
```

如果：

```text
placement[a] = Shard(d)
```

则：

\[
global\_index[d]
=
c_a \cdot local\_extent[d]
+
local\_index[d]
\]

没有 shard 的 tensor dimension：

\[
global\_index[d]=local\_index[d]
\]

### 12.2 Replicate semantics

在 composite placement 中，Replicate 表示：

> 改变该 mesh axis coordinate 时，不产生额外 tensor partition。

例如：

```text
mesh [2,4]
[Replicate, Shard(1)]
```

`(0,0)` 与 `(1,0)` 对应相同 global shard，`(0,1)` 与 `(1,1)` 对应相同 global shard。

当没有 Partial 时，所有 local elements 都必须等于对应 single global element。

### 12.3 Partial(sum) semantics

设 Partial axes 集合：

```text
P = {mesh axes whose placement is Partial(sum)}
```

对固定的：

```text
Shard coordinates
Replicate coordinates
local index
```

single global value必须等于：

> 在所有 Partial-axis coordinates 的 Cartesian product 上对对应 local values 求和。

概念上：

\[
single[g]
=
\sum_{c_P}
local[c_{	ext{fixed}},c_P][l]
\]

其中 `g` 由 Shard placements + local index 得到。

### 12.4 Shard + Partial 示例

```text
mesh = [2,2]
placements = [Shard(0), Partial(sum)]
```

要求：

```text
single shard 0 = local(0,0) + local(0,1)
single shard 1 = local(1,0) + local(1,1)
```

不能错误地把全部 4 ranks 一次性相加。

### 12.5 Replicate + Partial 示例

```text
mesh = [2,2]
placements = [Replicate, Partial(sum)]
```

要求：

```text
single = local(0,0) + local(0,1)
single = local(1,0) + local(1,1)
```

即每个 replicate group 都独立形成同一个 logical tensor。

### 12.6 多个 Partial axes

实现应自然支持多个 Partial axes，即在多个 Partial axes 的 Cartesian product 上求和。不要只写 2D special case。

---

## 13. Relation semantics 的代码组织

当前将：

```text
ReplicateRelation
ShardRelation
PartialRelation
```

作为 mutually-exclusive whole-tensor relation。

Multi-D 后建议改成：

```text
CompositeRelationSemantics
+
PlacementSpec helpers
```

或等价设计。

关键要求：

> 不要简单地分别编码每个 placement，再把独立 relation constraints `AND` 起来。

因为 `Shard + Partial`、`Replicate + Partial` 需要按 mesh group 组合语义。

可以保留 placement-specific validation/helper，但最终 shape/value encoding 必须由 composite relation composer 统一生成。

---

## 14. distributed ranks 继续使用 flat rank

当前：

```python
distributed: dict[int, ProgramSpec]
```

以及：

```text
distributed_tensors[rank]
```

继续保留。

本阶段不要改成多维 nested dictionary。

通过：

```text
flat rank <-> mesh coordinate
```

helper 转换即可。

---

## 15. Operator semantics 不得 mesh-aware

本阶段明确不修改：

```text
AddOperator
MulOperator
MatMulOperator
TransposeOperator
ReshapeOperator
SqueezeOperator
UnsqueezeOperator
ExpandOperator
```

的数学语义。

尤其不要增加：

```python
symbolic_execute(..., mesh, placements)
shape_constraints(..., mesh, placements)
```

当前 operator interface 继续只处理：

```text
local shapes
local symbolic values
attrs
```

Multi-D correctness 应通过：

```text
composite input relation
+
existing local operator semantics
+
composite output relation
+
SMT
```

完成。

---

## 16. Shape reducer 需要的修改

当前 `_relation_constraints(...)` 通过 `get_relation(relation.type)` dispatch。

Multi-D 后应改为 composite relation semantics，并传入：

```text
stage.mesh
relation.placements
flat-rank local shapes
original shapes
```

Operator constraint dispatch 不应改变。

---

## 17. Relation encoder 需要的修改

`_resolve_relation_tensors()` 按 flat rank resolve local tensor，这部分可以保留。

`_encode_one_relation()` 必须将 `stage.mesh` 传入 composite relation value encoder。

Value encoder 内部通过：

```text
rank_to_coordinate()
coordinate_to_rank()
```

构造 shard offsets、partial groups 和 replicate groups。

---

## 18. Validator 需要的修改

新增 Mesh validation：

```text
mesh.shape 非空
每维正整数
prod(mesh.shape) == world_size
```

Relation validation：

```text
len(distributed_tensors) == world_size
len(placements) == mesh rank
```

Placement schema：

```text
replicate:
    dim=None
    reduce_op=None

shard:
    dim 是合法 tensor axis
    reduce_op=None

partial:
    dim=None
    reduce_op=="sum"
```

重复 Shard 相同 tensor dim 本 milestone 拒绝。

Concrete local shape 按 composite shard factors 计算 expected shape，并逐 rank 检查。

---

## 19. Lemma materialization 必须同步升级

lemma semantic identity 必须包含：

```text
mesh shape
composite placements
```

否则不同 mesh/placement 可能错误碰撞。

因此：

```text
CertifiedLemma
semantic_identity_payload()
lemma_to_dict()
```

都必须加入 mesh。

### 19.1 同时修复现有 Op attrs identity bug

当前 `_op_to_dict()` 没有序列化 `attrs`。

本阶段一起修复：

```text
semantic identity 必须区分不同 attrs
lemma persistence 必须保存 attrs
```

如果希望尽量保持旧空 attrs operator 的 identity，可以：

```text
attrs == {} -> payload 中省略 attrs
attrs 非空 -> 写入 attrs
```

但无论如何：

> 两个 attrs 不同的 operator 必须得到不同 semantic identity。

---

## 20. Verifier proof formula 不变

当前核心 proof obligation：

\[
R_{in}\land
eg R_{out}
\]

不需要因为 Multi-D mesh 修改。

变化的是 `Rin / Rout` 的编码，而不是 proof formula。

---

## 21. Single-op Stage policy

当前实际实验继续以 single-op Stage 为主，但本阶段：

> **不要新增 exactly-one-op validator。**

Multi-D mesh refactor 与 Stage splitting 是独立问题。

---

## 22. Unit tests

必须覆盖：

### Mesh helpers

```text
mesh [2,4]
rank 0 -> (0,0)
rank 3 -> (0,3)
rank 4 -> (1,0)
rank 7 -> (1,3)
```

以及 round-trip 和 invalid bounds。

### Loader compatibility

```text
旧 world_size-only Stage
旧 replicate/shard/partial relation
新 Multi-D mesh + placements
```

拒绝：

```text
multi-D mesh + legacy relation shorthand
placements + legacy fields 同时存在
placement count != mesh rank
```

### Concrete relation shapes

```text
[Replicate, Shard]
[Shard, Shard]
[Shard, Partial]
duplicate same tensor shard dim rejected
```

### Reduced shape

检查：

```text
single'_d % shard_factor == 0
local'_d * shard_factor == single'_d
```

### Value semantics

至少覆盖：

```text
[Replicate, Shard]
[Shard, Shard]
[Shard, Partial]
[Replicate, Partial]
```

正确 mapping 与错误 mapping 都要测。

---

## 23. End-to-end Multi-D JSON

建议新增：

```text
input/mesh/
```

### Case A：Replicate × Shard + Transpose

```text
world_size = 4
mesh = [2,2]

single:
X [4,6]
transpose(0,1)
Y [6,4]

input placements:
[Replicate, Shard(1)]

local X:
[4,3]

output placements:
[Replicate, Shard(0)]

local Y:
[3,4]
```

预期：

```text
PROVED
```

### Case B：Shard × Shard + Transpose

```text
world_size = 4
mesh = [2,2]

single:
X [4,6]
transpose(0,1)
Y [6,4]

input placements:
[Shard(0), Shard(1)]

local X:
[2,3]

output placements:
[Shard(1), Shard(0)]

local Y:
[3,2]
```

预期：

```text
PROVED
```

### Case C：Wrong multi-shard candidate

基于 Case B 构造一个 **shape-valid but value-wrong** candidate，确保进入 SMT：

```text
DISPROVED
```

### Case D：Shard × Partial + Transpose

```text
world_size = 4
mesh = [2,2]

single:
X [4,6]
transpose(0,1)
Y [6,4]

input placements:
[Shard(0), Partial(sum)]

local X:
[2,6]

output placements:
[Shard(1), Partial(sum)]

local Y:
[6,2]
```

预期：

```text
PROVED
```

这个 case 必须验证 Partial 只沿对应 mesh axis 分组求和。

### Case E：Legacy 1D regression

现有 1D Replicate/Shard/Partial/MatMul/Broadcast/View fixtures 全部保持原结果。

---

## 24. Regression Requirements

完整测试必须确保：

```text
Add/Mul broadcasting
MatMul 2D
batched MatMul
MatMul batch broadcast
MatMul shard contraction -> Partial
wrong relation -> DISPROVED

Transpose
Reshape
Squeeze
Unsqueeze
Expand

Stage 5 PROVED/DISPROVED/UNKNOWN
Stage 6 lemma materialization
legacy JSON fixtures
```

全部不因 mesh refactor 回归。

---

## 25. 代码结构建议

不强制具体文件名，但建议：

```text
src/
├── stage/
│   ├── model.py
│   ├── loader.py
│   ├── validator.py
│   └── mesh.py
├── semantics/
│   ├── relations.py
│   ├── placements.py
│   └── operators/
├── shape/
│   └── constraints.py
├── verification/
│   └── relation_encoder.py
└── lemma/
    └── model.py
```

原则：

> **源码按长期语义职责组织，不按 migration 步骤组织。**

---

## 26. 明确非目标

本阶段不要实现：

```text
AllReduce
AllGather
ReduceScatter
AllToAll
Collective operator semantics

physical network topology
axis names / DP / TP / CP labels
uneven shard
interleaved/strided shard
同一 tensor dimension 被多个 mesh axes 重复 shard
automatic placement inference
candidate placement generation
stage splitting
whole-graph alignment
exactly-one-op enforcement
storage / stride / alias semantics
```

---

## 27. 实现顺序建议

```text
1. DeviceMeshSpec + PlacementSpec + canonical RelationSpec
2. loader legacy normalization + new Multi-D JSON
3. mesh rank-coordinate helpers
4. validator
5. relation reduced-shape semantics
6. relation value semantics
7. relation_encoder adaptation
8. lemma mesh/placement identity + op.attrs identity fix
9. Multi-D unit tests
10. Multi-D e2e fixtures
11. full legacy regression
```

---

## 28. 最终运行要求

必须运行：

```bash
conda run -n smt pytest -q
```

并单独实际执行所有新增 `input/mesh/` fixtures。

---

## 29. 最终汇报要求

实现完成后必须报告：

```text
1. commit SHA
2. changed files

3. DeviceMeshSpec 最终定义
4. PlacementSpec 最终定义
5. RelationSpec 最终定义

6. legacy 1D JSON canonicalization
7. new Multi-D JSON 示例

8. rank -> mesh coordinate 算法
9. coordinate -> rank 算法

10. composite concrete shape 公式
11. composite reduced-shape 公式
12. Shard global-index mapping
13. Replicate semantics
14. Partial group-sum semantics

15. 是否支持 arbitrary mesh rank
16. same-tensor-dim repeated Shard 为什么暂时拒绝

17. OperatorSemantics interface 是否保持不变
18. Add/Mul/MatMul/View 是否无 mesh-specific logic

19. relation_encoder 如何获得 mesh coordinate

20. lemma identity 如何包含 mesh + placements
21. op.attrs identity bug 如何修复

22. 新增 unit tests 数量
23. 新增 Multi-D e2e JSON 列表
24. 每个 JSON actual result

25. legacy 1D regression results
26. full pytest output

27. verifier proof formula 是否保持不变
28. Stage 6 materialization 是否正常
```

---

## 30. 验收标准

```text
✓ canonical IR 能表达 N-D mesh
✓ 每个 mesh axis 有独立 placement
✓ Replicate + Shard 正确
✓ Shard + Shard 正确
✓ Shard + Partial 正确
✓ rank/coordinate mapping 正确
✓ relation concrete shape semantics 正确
✓ relation reduced-shape semantics 正确
✓ relation value semantics 正确
✓ old 1D JSON backward compatible
✓ old 1D verification results unchanged
✓ existing local operators 不被 mesh 污染
✓ verifier proof formula 不变
✓ lemma identity 包含 mesh + placements
✓ operator attrs 进入 semantic identity
✓ Multi-D positive cases PROVED
✓ shape-valid wrong candidate DISPROVED
✓ full pytest pass
```

完成本阶段后，再继续：

```text
Reduction semantics
→ Collective semantics
```

其中 Collective 从一开始就必须以：

```text
mesh axis
```

为通信 group 的基本单位。
