# View / Layout Operators 需求文档

## 1. 背景与目标

当前 `smt-verify` 已经具备：

```text
Add / Mul
→ arbitrary-rank element-wise broadcasting

MatMul
→ rank >= 2 batched matmul
→ batch trailing broadcasting
→ original-aware reduced-shape constraints
→ arbitrary-rank symbolic execution
```

本阶段新增第三类核心语义模式：

> **View / Layout / Index-remapping semantics**

实现顺序：

```text
Transpose
→ Reshape
→ Squeeze
→ Unsqueeze
→ Expand
```

统一采用三层接口：

```text
validate_concrete_shapes()
shape_constraints()
symbolic_execute()
```

核心抽象：

\[
Y[o] = X[\phi(o)]
\]

其中不同算子的区别主要在于 `φ`：

```text
transpose
→ axis permutation

reshape
→ flat-offset preserving remapping

squeeze
→ 在 input index 中插入 singleton coordinate 0

unsqueeze
→ 从 output index 删除新增 singleton coordinate

expand
→ expanded singleton axis 映射回 input index 0
```

---

## 2. 当前只建模 logical tensor semantics

本阶段明确不建模：

```text
storage layout
stride
physical contiguity
memory aliasing
copy vs view distinction
```

因此：

> `reshape` 统一表示保持逻辑线性元素顺序的 shape reinterpretation。

当前 IR 不单独引入 `view` operator。

未来 frontend 若遇到：

```text
PyTorch view
PyTorch reshape
其他框架 reshape
```

只要在 logical tensor 层保持相同 element order，都可 lower 成：

```text
reshape
```

---

## 3. Operator semantics 与 Relation semantics 分离

View operator 只定义：

```text
local tensor shape/value/index semantics
```

不要硬编码：

```text
Shard(2) -> Shard(3)
Replicate -> Replicate
```

例如：

```text
X ~ Shard(dim=2)

Y = transpose(X, 2, 3)

candidate:
Y ~ Shard(dim=3)
```

应由：

```text
operator semantics
+
existing relation semantics
+
SMT
```

自动证明。

错误 candidate：

```text
Y ~ Shard(dim=2)
```

应得到：

```text
DISPROVED
```

---

## 4. Stage Verification IR 扩展

当前：

```python
@dataclass
class OpSpec:
    type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
```

需要扩展为：

```python
@dataclass
class OpSpec:
    type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, object]
```

要求：

```text
attrs 默认 {}
旧 JSON 不写 attrs 时继续合法
已有 Add/Mul/MatMul fixtures 无需手工补 attrs
```

JSON 示例：

### Transpose

```json
{
  "type": "transpose",
  "inputs": ["X"],
  "outputs": ["Y"],
  "attrs": {
    "dim0": 1,
    "dim1": -1
  }
}
```

### Reshape

```json
{
  "type": "reshape",
  "inputs": ["X"],
  "outputs": ["Y"],
  "attrs": {
    "shape": [2, 12]
  }
}
```

### Squeeze

显式 dim：

```json
{
  "type": "squeeze",
  "inputs": ["X"],
  "outputs": ["Y"],
  "attrs": {
    "dim": -1
  }
}
```

删除所有 singleton dims：

```json
{
  "type": "squeeze",
  "inputs": ["X"],
  "outputs": ["Y"],
  "attrs": {}
}
```

### Unsqueeze

```json
{
  "type": "unsqueeze",
  "inputs": ["X"],
  "outputs": ["Y"],
  "attrs": {
    "dim": -1
  }
}
```

### Expand

```json
{
  "type": "expand",
  "inputs": ["X"],
  "outputs": ["Y"],
  "attrs": {
    "shape": [2, 3, 4]
  }
}
```

---

## 5. attrs 与 TensorSpec.shape

对于：

```text
reshape
expand
```

`attrs.shape` 表示 operator 参数。

`TensorSpec.shape` 表示图中的 tensor metadata。

两者必须一致。

例如：

```text
attrs.shape = [2,12]
Y.shape     = [2,12]
```

否则 static validation 失败。

这是 verification IR 的 consistency check，不视为冗余。

---

## 6. Operator 代码结构调整

推荐调整为：

```text
src/semantics/
├── broadcast.py
├── operators.py
└── operators/
    ├── __init__.py
    ├── elementwise.py
    ├── matmul.py
    └── view.py
```

### `operators.py`

只保留：

```text
OperatorSemantics interface
common exceptions
DECLARED_OPERATOR_TYPES
OPERATOR_REGISTRY
get_operator()
```

### `operators/elementwise.py`

移动：

```text
AddOperator
MulOperator
```

行为不得改变。

### `operators/matmul.py`

移动：

```text
MatMulOperator
MatMulShapeError
infer_matmul_output_shape
```

行为不得改变。

### `operators/view.py`

实现：

```text
TransposeOperator
ReshapeOperator
SqueezeOperator
UnsqueezeOperator
ExpandOperator
```

以及相关 helper。

本次结构调整必须为 behavior-preserving refactor。

---

## 7. Operator interface 调整

因为 View operators 需要 attrs，三个接口需要接收 attrs：

```python
def validate_concrete_shapes(
    self,
    input_shapes,
    output_shapes,
    attrs,
    context,
) -> None:
    ...

def shape_constraints(
    self,
    input_shapes,
    output_shapes,
    original_input_shapes,
    original_output_shapes,
    attrs,
    context,
) -> list[z3.BoolRef]:
    ...

def symbolic_execute(
    self,
    input_tensors,
    output_shapes,
    attrs,
    context,
) -> tuple[SymbolicTensor, ...]:
    ...
```

已有：

```text
Add
Mul
MatMul
```

接收：

```text
attrs = {}
```

并保持原行为。

---

## 8. 统一 Negative Axis

本阶段从一开始统一支持 negative axis。

建议 helper：

```python
def canonicalize_axis(
    axis: int,
    rank: int,
    *,
    allow_end: bool = False,
) -> int:
    ...
```

### 普通 axis

用于：

```text
transpose
squeeze
```

合法范围：

```text
[-rank, rank - 1]
```

例如 rank=4：

```text
-1 -> 3
-2 -> 2
-4 -> 0
```

越界报错。

### Unsqueeze axis

对于 input rank=`r`：

```text
[-r-1, r]
```

合法。

例如 rank=3：

```text
-4 -> 0
-1 -> 3
3  -> 3
```

---

# 9. TransposeOperator

## attrs

必须包含：

```text
dim0
dim1
```

二者必须为整数。

支持 negative axis。

## validate_concrete_shapes()

检查：

```text
exactly one input
exactly one output
input/output rank 相同
dim0/dim1 合法
```

canonicalize 后：

```python
expected = list(input_shape)
expected[dim0], expected[dim1] = expected[dim1], expected[dim0]
```

要求：

```text
output_shape == expected
```

允许：

```text
dim0 == dim1
```

此时为 identity。

## shape_constraints()

reduced output 必须采用同一 permutation：

```python
expected = permute(input_shape)
constraints = [
    output[i] == expected[i]
    for i in range(rank)
]
```

不添加额外 reduction-basis 下界。

## symbolic_execute()

对 output index 交换相同两轴，得到 input index：

```python
input_index = list(output_index)
input_index[dim0], input_index[dim1] = (
    input_index[dim1],
    input_index[dim0],
)
```

然后：

```text
Y[o] = X[input_index]
```

不得写 rank=2/3/4 特判。

---

# 10. ReshapeOperator

## 10.1 语义范围

当前只支持：

> logical flat-order-preserving reshape

不建模：

```text
stride
storage contiguity
copy
aliasing
view legality
```

第一版要求 target shape fully resolved。

暂不支持：

```text
-1 inference
0-as-copy-dimension
```

---

## 10.2 attrs

必须包含：

```text
shape: list[int]
```

所有维度：

```text
> 0
```

且：

```text
attrs.shape == declared output TensorSpec.shape
```

---

## 10.3 validate_concrete_shapes()

检查：

```text
exactly one input
exactly one output
attrs.shape 合法
attrs.shape == output shape
numel(input) == numel(output)
```

---

## 10.4 reshape group helper

建议实现：

```python
infer_reshape_groups(
    original_input_shape,
    original_output_shape,
)
```

它的含义是：

> 在 standard contiguous row-major logical reshape 下，寻找 input/output 两侧可以按相同 element count 对齐的 contiguous dimension groups。

例如：

```text
input  [2,3,4]
output [6,4]
```

可分为：

```text
input [2,3] <-> output [6]
input [4]   <-> output [4]
```

因为：

```text
2*3 == 6
4   == 4
```

例如：

```text
input  [2,3,4]
output [2,12]
```

可分为：

```text
input [2]   <-> output [2]
input [3,4] <-> output [12]
```

该 helper 是：

```text
reduction semantics helper
```

不是 value semantics helper。

---

## 10.5 shape_constraints()

不要只约束：

\[
\prod input'_i = \prod output'_j
\]

还要保持 original contiguous grouping structure。

若 original group：

```text
input axes [a:b]
output axes [c:d]
```

则 reduced 中要求：

\[
\prod_{i=a}^{b-1} input'_i
=
\prod_{j=c}^{d-1} output'_j
\]

并保留整体：

\[
numel(input') = numel(output')
\]

一致性。

这属于 original-structure-preserving shape reduction。

---

## 10.6 symbolic_execute()

Value semantics 使用 flat offset preservation。

对于 output index：

\[
(o_0,\dots,o_n)
\]

计算 row-major flat offset：

\[
offset
=
\sum_i
o_i
\prod_{j>i} outputShape_j
\]

再按 input shape unravel：

```text
flat offset
→ input multi-index
```

最终：

\[
Y[o] = X[unravel(offset,inputShape)]
\]

建议 helper：

```python
ravel_index(index, shape) -> int
unravel_index(offset, shape) -> tuple[int, ...]
```

---

# 11. SqueezeOperator

本阶段最终必须支持：

```text
squeeze(dim)
squeeze()
```

可以先分两步调试，但必须在同一个 milestone 内完成。

## squeeze(dim)

attrs：

```json
{"dim": -1}
```

检查：

```text
axis 合法
input_shape[axis] == 1
output shape == 删除该 axis
```

支持 negative axis。

### shape_constraints()

要求：

\[
input'[axis] = 1
\]

其余 axis 一一对应。

### symbolic_execute()

output index 在被 squeeze 的 axis 插入：

```text
0
```

得到 input index。

---

## squeeze()

attrs：

```json
{}
```

删除 original input 中全部：

```text
extent == 1
```

的 axes。

例如：

```text
[1,2,1,3] -> [2,3]
```

reduced input 中这些 original singleton axes 也必须继续：

```text
== 1
```

symbolic execution 中相应位置插入 0。

---

# 12. UnsqueezeOperator

attrs：

```text
dim
```

必须支持 negative axis。

对于 input rank=`r`：

```text
dim ∈ [-r-1, r]
```

canonicalize 后：

```text
output rank = input rank + 1
output shape = input shape 在 dim 位置插入 1
```

## shape_constraints()

要求：

\[
output'[dim] = 1
\]

其余 axis 对应 input。

## symbolic_execute()

从 output index 删除新增 singleton coordinate：

```python
input_index = output_index[:dim] + output_index[dim+1:]
```

访问 input。

---

# 13. ExpandOperator

Expand 建模为 unary broadcasting。

例如：

```text
input  [2,1,4]
output [2,3,4]
```

语义：

\[
Y[i,j,k] = X[i,0,k]
\]

允许 trailing-aligned expansion。

## attrs

必须包含：

```text
shape
```

第一版要求 fully resolved positive dims。

不支持 PyTorch `-1` shorthand。

## validate_concrete_shapes()

要求：

```text
target rank >= input rank
attrs.shape == declared output shape
```

trailing-align input/output。

每个 aligned axis 必须满足：

```text
input_dim == output_dim
or
input_dim == 1
```

missing leading input dims 等价于隐式 singleton。

禁止：

```text
2 -> 3
```

这种非-singleton expansion。

## shape_constraints()

保持 original expansion pattern：

如果 original：

```text
input_dim == output_dim
```

则：

\[
input' = output'
\]

如果 original：

```text
input_dim = 1
output_dim > 1
```

则：

\[
input' = 1
\]

output reduced dimension保持自己的 reduced variable。

missing leading axes 由 output 决定。

建议实现：

```python
expand_shape_constraints(...)
expand_index(...)
```

不要强行复用 binary broadcast API。

## symbolic_execute()

对 output index：

```text
若对应 input extent == 1:
    input index = 0
否则:
    input index = output index
```

missing leading output axes 不进入 input index。

---

# 14. attrs Static Validation

Operator-specific attrs schema 必须尽早验证。

例如：

```text
transpose
→ 必须 dim0 + dim1

reshape
→ 必须 shape

squeeze
→ 允许 {} 或 {"dim": int}

unsqueeze
→ 必须 dim

expand
→ 必须 shape
```

不要把：

```text
missing attr
wrong type
unknown attr
```

拖到 symbolic execution。

attrs validation 逻辑应尽量放在 operator family 内，不要在 Stage validator 中写巨大 switch。

---

# 15. Unit Tests

至少覆盖：

## Axis helper

```text
rank=4:
-1 -> 3
-2 -> 2
-4 -> 0

-5 -> error
4  -> error
```

unsqueeze rank=3：

```text
-4 -> 0
-1 -> 3
3  -> 3

-5 -> error
4  -> error
```

## Transpose

```text
positive axes
negative axes
same-axis identity
invalid axis
wrong output shape
4D case
symbolic index mapping
```

## Reshape

```text
[2,3,4] -> [6,4]
[2,3,4] -> [2,12]
[2,3,4] -> [24]
invalid numel
attrs/output mismatch
reshape-group inference
reduced-group preservation
symbolic flat-index preservation
```

## Squeeze

```text
positive dim
negative dim
squeeze()
multiple singleton axes
non-singleton explicit dim rejected
wrong output
reduced singleton preservation
symbolic index mapping
```

## Unsqueeze

```text
dim=0
dim=-1
dim=-rank-1
invalid axis
wrong output
reduced inserted axis == 1
symbolic mapping
```

## Expand

```text
same-rank singleton expand
leading-rank expand
no-op same shape
invalid non-singleton expansion
attrs/output mismatch
reduced singleton preservation
symbolic index mapping
```

---

# 16. End-to-End JSON

建议目录：

```text
input/view/
```

至少新增：

## A. Transpose + shard-axis change

Single：

```text
X [2,4,6]
Y [2,6,4]

transpose dim0=1 dim1=2
```

world_size=2。

Input relation：

```text
X -> Shard(dim=1)
```

Local：

```text
Xr [2,2,6]
Yr [2,6,2]
```

Candidate：

```text
Y -> Shard(dim=2)
```

预期：

```text
PROVED
```

## B. Wrong transpose relation

同上，但 candidate：

```text
Y -> Shard(dim=1)
```

预期：

```text
DISPROVED
```

## C. Negative-axis transpose

```text
X [2,3,4,6]
transpose dim0=-1 dim1=-2
Y [2,3,6,4]
```

预期：

```text
PROVED
```

## D. Reshape

```text
X [4,2,3]
reshape -> [4,6]
```

Input：

```text
X -> Shard(dim=0)
```

Output：

```text
Y -> Shard(dim=0)
```

world_size=2。

Local：

```text
Xr [2,2,3]
Yr [2,6]
```

预期：

```text
PROVED
```

## E. Squeeze

```text
X [4,1,6]
squeeze dim=1
Y [4,6]
```

沿 dim=0 shard。

预期：

```text
PROVED
```

## F. Unsqueeze

```text
X [4,6]
unsqueeze dim=1
Y [4,1,6]
```

沿 dim=0 shard。

预期：

```text
PROVED
```

## G. Expand

```text
X [4,1,6]
expand [4,3,6]
Y [4,3,6]
```

沿 dim=0 shard。

预期：

```text
PROVED
```

---

# 17. Stage Scope：当前仍保持单 Stage 单 Operator

本阶段继续遵循当前项目边界：

> **一个 Stage 只包含一个核心 operator。**

也就是说，本阶段新增的 View / Layout operator 仍分别通过独立 Stage 验证：

```text
Transpose Stage
Reshape Stage
Squeeze Stage
Unsqueeze Stage
Expand Stage
```

当前不要为了验证 operator compositionality 而引入：

```text
transpose -> matmul
reshape -> matmul
transpose -> reshape
```

等 multi-op Stage。

原因是：

```text
whole program
    ↓
stage splitting / graph alignment
    ↓
single-op local stages
    ↓
current Stage verifier
```

其中：

```text
stage splitting
multi-op graph decomposition
cross-stage continuity
```

属于更上游的 scheduler / stage partitioning / whole-graph verification 任务，不属于本阶段 operator semantics 的实现范围。

本阶段只要求：

```text
每个 Stage 中有且仅有一个核心 operator
```

并分别验证该 operator 对：

```text
shape
symbolic value
single/distributed relation
```

的影响。

未来即使真实程序中出现：

```text
transpose
→ matmul
```

也应先由上游切分为：

```text
Stage A:
transpose

Stage B:
matmul
```

并通过 stage boundary relation 连接，而不是让当前 Stage verifier 直接支持 multi-op execution。

---

# 18. Regression Requirements

完整回归必须覆盖：

```text
Add broadcasting
Mul broadcasting
MatMul 2D
batched MatMul
batch MatMul broadcasting
Shard-K -> Partial
lemma materialization
wrong relation DISPROVED
```

结构重构不得改变现有结果。

---

# 19. 明确非目标

本阶段不要实现：

```text
physical view / stride semantics
storage aliasing
copy-vs-view distinction

reshape -1 inference
reshape 0 shorthand
expand -1 shorthand

slice
select
concat
split

sum / mean / max
all_reduce
all_gather
reduce_scatter

LayoutRelation
candidate generation
whole-graph alignment
multi-op Stage execution
stage splitting / graph decomposition
```

---

# 20. 最终运行要求

必须实际运行：

```bash
conda run -n smt pytest -q
```

并逐个实际执行：

```text
transpose positive
transpose wrong relation
negative transpose
reshape
squeeze
unsqueeze
expand
```

---

# 21. 最终汇报要求

最终汇报必须包含：

```text
1. commit SHA
2. 新增/修改文件

3. OpSpec.attrs 最终定义
4. 旧 JSON 是否向后兼容
5. operator interface 如何接收 attrs

6. operators.py / operators/ 目录最终结构

7. negative axis 支持规则
8. Transpose semantics
9. Reshape logical semantics
10. infer_reshape_groups 规则
11. Squeeze dim + omitted dim
12. Unsqueeze negative dim
13. Expand trailing alignment

14. unit tests 数量
15. end-to-end JSON 列表
16. 每个 JSON 实跑结果
18. 完整 pytest 结果

19. Add/Mul 是否行为保持
20. MatMul 是否行为保持
21. Relation semantics 是否未修改
22. Verifier formula 是否未修改
23. Lemma format 是否未修改
```

---

# 22. 验收标准

本阶段完成后，系统应具备三类核心 operator semantic pattern：

```text
Element-wise / Broadcast
MatMul / Contraction
View / Index Remapping
```

并完整支持：

```text
Transpose
Reshape
Squeeze
Unsqueeze
Expand
```

至少能够自动验证：

```text
Shard(dim=i)
    ↓ transpose(i,j)
Shard(dim=j)
```

得到：

```text
PROVED
```

而错误 shard axis 得到：

```text
DISPROVED
```

Operator 中不得硬编码 relation transition。
