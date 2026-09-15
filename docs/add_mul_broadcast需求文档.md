# Add/Mul Arbitrary-Rank Broadcasting 需求文档

## 1. 背景

当前 `smt-verify` 中的 `AddOperator` 和 `MulOperator` 已经可以处理任意 rank 的 tensor，但要求：

\[
shape(A)=shape(B)=shape(C)
\]

因此当前可以处理：

```text
[2,3,4] + [2,3,4]
[2,3,4,5] * [2,3,4,5]
```

但尚不支持标准 NumPy/PyTorch 风格的 trailing-dimension broadcasting，例如：

```text
[2,3,4] + [4]
[2,1,4] + [1,3,4]
[2,3,4,5] * [1,3,1,5]
```

本阶段目标是：

> 将 Add/Mul 从 “arbitrary-rank + same-shape only” 扩展为 “arbitrary-rank + standard trailing-dimension broadcasting”，并抽出以后其它 operator 可复用的基础 broadcast primitives。

本阶段只处理 Add/Mul，不扩展其它 operator。

---

## 2. 本阶段目标

实现：

```text
Add:
arbitrary-rank + broadcasting

Mul:
arbitrary-rank + broadcasting
```

同时抽出公共 broadcast helper，使未来以下 operator 可以按需复用：

```text
batched matmul
where
sub
div
其它支持 broadcasting 的 operator
```

注意：

> 共享 broadcast 实现，不代表所有 element-wise operator 自动允许 broadcast。

是否使用 broadcast，仍应由具体 operator semantics 显式决定。

---

## 3. 非目标

本阶段禁止顺手实现：

```text
MatMul batched semantics
MatMul batch broadcasting
Transpose
Sum
Reshape
AllReduce
AllGather
ReduceScatter
Sub
Div
Where
```

也不要修改：

```text
Stage IR schema
Relation semantics
Verifier proof formula
Lemma format
Shape reduction 总体框架
```

本阶段只扩展：

```text
AddOperator
MulOperator
shared broadcast helpers
```

---

## 4. Broadcast 正式语义

采用标准 trailing-dimension broadcasting。

两个 shape 从最后一维开始向前对齐。

对于某一组 aligned dimensions：

\[
a_i,\ b_i
\]

合法条件为：

\[
a_i=b_i
\]

或：

\[
a_i=1
\]

或：

\[
b_i=1
\]

如果某个输入缺少 leading dimension，则该维度视为隐式 `1`。

例如：

```text
A = [2,3,4]
B = [4]
```

逻辑对齐为：

```text
A = [2,3,4]
B = [1,1,4]
```

输出：

```text
[2,3,4]
```

再例如：

```text
A = [2,1,4]
B = [1,3,4]
```

输出：

```text
[2,3,4]
```

非法 example：

```text
A = [2,3,4]
B = [5]
```

因为最后一维：

```text
4 != 5
4 != 1
5 != 1
```

---

## 5. 建议新增公共模块

建议新增：

```text
src/semantics/broadcast.py
```

不要把 broadcast 逻辑分别复制到 Add 和 Mul 中。

至少提供以下三个核心能力。

### 5.1 Concrete broadcast shape inference

建议接口：

```python
def infer_broadcast_shape(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    context: str | None = None,
) -> tuple[int, ...]:
    ...
```

职责：

```text
original concrete shapes
        ↓
检查 broadcast compatibility
        ↓
返回 expected output shape
```

例如：

```python
infer_broadcast_shape((2, 3, 4), (4,))
# (2, 3, 4)

infer_broadcast_shape((2, 1, 4), (1, 3, 4))
# (2, 3, 4)
```

非法 shape 应抛出明确异常。

---

### 5.2 Reduced-shape broadcast constraints

建议接口：

```python
def broadcast_shape_constraints(
    left_shape,
    right_shape,
    output_shape,
    original_left_shape,
    original_right_shape,
    original_output_shape,
    context,
) -> list[z3.BoolRef]:
    ...
```

该函数服务于 Stage 2 shape reduction。

核心原则：

> 必须保持 original broadcast pattern，而不是只要求 reduced shapes 最终仍然能 broadcast。

不能只写：

\[
a'_i=b'_i \lor a'_i=1 \lor b'_i=1
\]

因为这样 solver 可能改变原始 broadcast 结构。

应当根据 original shape 判断该维度属于哪一种模式，并对 reduced shape施加对应约束。

#### Case A：原始两边相等

如果：

\[
a_i=b_i
\]

则要求：

\[
a'_i=b'_i=o'_i
\]

#### Case B：原始左边为 1，右边不是 1

如果：

\[
a_i=1,\quad b_i>1
\]

则要求：

\[
a'_i=1
\]

以及：

\[
o'_i=b'_i
\]

#### Case C：原始右边为 1，左边不是 1

要求：

\[
b'_i=1
\]

以及：

\[
o'_i=a'_i
\]

#### Case D：某一侧原始 rank 较短，缺少该 leading dim

缺失侧没有对应 symbolic dimension。

输出 reduced dimension 直接等于存在侧 reduced dimension。

例如 original：

```text
A = [32,128,4096]
B = [4096]
```

逻辑对齐：

```text
A = [32,128,4096]
B = [ 1,  1,4096]
```

reduced 后要保持：

```text
A'[-1] == B'[-1]
Out'[-1] == A'[-1]
Out'[-2] == A'[-2]
Out'[-3] == A'[-3]
```

再例如：

```text
A = [32,1,4096]
B = [1,128,4096]
```

reduced 后必须保持：

```text
A'[1] == 1
B'[0] == 1
A'[2] == B'[2]
Out'[0] == A'[0]
Out'[1] == B'[1]
Out'[2] == A'[2] == B'[2]
```

---

### 5.3 Symbolic broadcast index projection

建议接口：

```python
def broadcast_index(
    output_index: tuple[int, ...],
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[int, ...]:
    ...
```

职责：

> 将一个 output index 映射成当前 input 应读取的位置。

例如：

```text
Input:
B [4]

Output:
C [2,3,4]
```

对于：

```text
C[1,2,3]
```

应映射：

```text
B[3]
```

又例如：

```text
A [2,1,4]
C [2,3,4]
```

对于：

```text
C[1,2,3]
```

应映射：

```text
A[1,0,3]
```

规则：

```text
从 trailing dimensions 对齐。

如果 input dimension == 1:
    input index = 0
否则:
    input index = 对应 output index

input 不存在的 leading dimensions 不进入 input index。
```

---

## 6. AddOperator 修改要求

### 6.1 validate_concrete_shapes()

当前行为：

```text
left == right == output
```

需要修改为：

```text
expected = infer_broadcast_shape(left, right)
require output == expected
```

仍然要求：

```text
exactly two inputs
exactly one output
```

不限制 rank。

该函数职责仍然只是：

> original program legality。

---

### 6.2 shape_constraints()

删除当前 same-shape constraint 逻辑：

```text
left == right
left == output
```

改为调用：

```python
broadcast_shape_constraints(...)
```

必须使用：

```text
original_input_shapes
original_output_shapes
```

来保持 original broadcast structure。

该函数职责是：

> reduced-shape semantics。

---

### 6.3 symbolic_execute()

计算：

\[
C[i]=A[\phi_A(i)] + B[\phi_B(i)]
\]

其中：

```text
φA = broadcast_index(...)
φB = broadcast_index(...)
```

不要为 rank=1/2/3/4 编写分支。

必须支持 arbitrary rank。

---

## 7. MulOperator 修改要求

Mul 与 Add 使用完全相同的 broadcast shape/index semantics。

区别仅为：

\[
C[i]=A[\phi_A(i)] \times B[\phi_B(i)]
\]

Add 和 Mul 不允许分别复制一份 broadcast shape/index 实现。

---

## 8. 不要过度抽象 Operator 层

本阶段不要引入类似：

```python
class ElementWiseBroadcastOperator(...)
```

并让所有 element-wise operator 自动拥有 broadcast。

推荐：

```text
shared broadcast primitives
        ↑
        │
Add 显式调用
Mul 显式调用
```

未来其它 operator 是否使用 broadcast，由其自身 semantics 决定。

---

## 9. 三个 Operator 接口职责必须保持清晰

### Stage 1

```text
validate_concrete_shapes()
```

负责：

> original program legality。

例如：

```text
[2,3,4] + [4] -> [2,3,4]
```

合法。

---

### Stage 2

```text
shape_constraints()
```

负责：

> reduced-shape semantics，并保持 original broadcast structure。

不是重新检查 original program。

---

### Stage 3

```text
symbolic_execute()
```

负责：

> value semantics，即每个 reduced output element 从哪些 reduced input elements 读取值。

可以保留必要 execution invariant checks。

---

## 10. 必须新增 Unit Tests

至少覆盖以下 8 类测试。

### Test 1：same-shape regression

```text
[2,3,4] + [2,3,4]
→ [2,3,4]
```

确保旧行为不回归。

### Test 2：rank mismatch broadcast

```text
[2,3,4] + [4]
→ [2,3,4]
```

### Test 3：singleton broadcast

```text
[2,1,4] + [1,3,4]
→ [2,3,4]
```

### Test 4：4D Mul broadcast

```text
[2,3,4,5] * [1,3,1,5]
→ [2,3,4,5]
```

### Test 5：invalid broadcast

```text
[2,3,4] + [5]
```

必须在 Stage validation / concrete validation 阶段失败。

### Test 6：wrong output shape

```text
A = [2,3,4]
B = [4]
declared C = [2,3,1]
```

必须失败。

### Test 7：reduction pattern preservation

Original：

```text
A = [32,1,4096]
B = [1,128,4096]
```

必须验证 reduced constraints 强制：

```text
A'[1] == 1
B'[0] == 1
A'[2] == B'[2]
```

不能只检查最终“可 broadcast”。

### Test 8：symbolic index mapping

```text
A = [2,1,4]
B = [1,3,4]
C = [2,3,4]
```

明确检查：

```text
C[1,2,3]
```

对于 Add 使用：

```text
A[1,0,3]
B[0,2,3]
```

即：

\[
C[1,2,3] = A[1,0,3] + B[0,2,3]
\]

---

## 11. 必须新增 End-to-End JSON Examples

除 unit tests 外，再新增至少 4 个真实 Stage JSON，用于完整运行：

```text
load
→ validate
→ reduce
→ symbolic execute
→ relation encode
→ SMT verify
```

建议放在：

```text
input/broadcast/element_wise/
```

如不想调整目录，也可以沿用当前 input 命名风格。

---

### Example A：Add rank-mismatch broadcast

建议：

```text
single:
A [4,3,4]
B [4]
C [4,3,4]

world_size = 2
```

relations：

```text
A -> Shard(dim=0)
B -> Replicate
C -> Shard(dim=0)
```

distributed：

```text
rank0:
A0 [2,3,4]
B0 [4]
C0 [2,3,4]

rank1:
A1 [2,3,4]
B1 [4]
C1 [2,3,4]
```

operator：

```text
C = Add(A,B)
Cr = Add(Ar,Br)
```

预期：

```text
PROVED
```

该 case 体现：

\[
Shard + Replicate \rightarrow Shard
\]

并且 Add 内部发生 broadcast。

---

### Example B：Add singleton broadcast

single：

```text
A [4,1,4]
B [1,3,4]
C [4,3,4]
```

relations：

```text
A -> Shard(dim=0)
B -> Replicate
C -> Shard(dim=0)
```

local：

```text
A_r [2,1,4]
B_r [1,3,4]
C_r [2,3,4]
```

预期：

```text
PROVED
```

---

### Example C：Mul 4D broadcast

single：

```text
A [4,3,2,5]
B [1,3,1,5]
C [4,3,2,5]
```

relations：

```text
A -> Shard(dim=0)
B -> Replicate
C -> Shard(dim=0)
```

local A/C 第一维除以 world_size。

预期：

```text
PROVED
```

---

### Example D：Invalid broadcast

```text
A [2,3,4]
B [5]
C [2,3,4]
```

预期：

```text
load_stage / validate_stage failure
```

不能进入 SMT verification。

---

## 12. 推荐额外新增 Wrong Relation Case

推荐再增加一个合法 broadcast computation，但错误 output relation 的 case。

例如：

```text
A -> Shard(dim=0)
B -> Replicate
```

但 candidate output relation 错误写成：

```text
C -> Replicate
```

预期：

```text
SAT
DISPROVED
```

该 case 用于确认：

> broadcast 支持没有破坏 verifier 对错误 distributed relation 的识别能力。

---

## 13. Regression 要求

必须确保当前已有所有测试继续通过。

尤其：

```text
2D Add
2D Mul
2D MatMul
matmul_shard_to_partial
matmul_shard_to_partial_large
matmul_shard_wrong_replicate
lemma materialization
lemma store
```

均不得回归。

MatMul 本阶段不得修改语义。

---

## 14. 实现完成后的验证要求

必须实际运行：

```bash
conda run -n smt pytest -q
```

并逐个实际运行新增 positive/negative broadcast examples。

最终汇报必须包括：

```text
1. 新增/修改文件
2. broadcast helper API 与职责
3. AddOperator 修改内容
4. MulOperator 修改内容
5. 新增 unit tests 数量及覆盖内容
6. 新增 JSON examples
7. 每个 JSON example 的实际运行结果
8. 完整 pytest 结果
9. 是否修改 MatMul semantics：必须回答 No
10. 是否修改 relation semantics：必须回答 No
```

---

## 15. 验收标准

本阶段验收通过，至少需要满足：

1. Add/Mul 支持 arbitrary-rank standard trailing broadcasting。
2. Broadcast shape inference、reduction constraints、symbolic index mapping 三层职责清晰。
3. Shape reduction 保留 original broadcast pattern。
4. Add/Mul 复用公共 broadcast primitives，不复制实现。
5. 不修改 MatMul、relation、verifier、lemma 核心语义。
6. 至少 8 类 unit tests 完整覆盖。
7. 至少 4 个 end-to-end JSON examples 实际运行。
8. 所有原有 regression tests 继续通过。
9. Wrong relation case 如实现，应得到 DISPROVED。
10. Codex 最终提供 commit SHA 与真实测试输出，供后续人工核验。
