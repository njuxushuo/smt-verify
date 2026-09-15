# MatMul Arbitrary-Rank / Batched Broadcasting 需求文档

## 1. 背景

当前 `smt-verify` 中的 `MatMulOperator` 仅支持二维矩阵乘法：

\[
A:[M,K],\quad B:[K,N],\quad C:[M,N]
\]

这对于基础 MVP 足够，但无法覆盖 Transformer 中常见的高维 batched MatMul，例如：

```text
[B,S,H] @ [H,D] -> [B,S,D]
[B,H,S,D] @ [B,H,D,S] -> [B,H,S,S]
[B,H,M,K] @ [1,H,K,N] -> [B,H,M,N]
```

当前仓库已经具备：

```text
src/semantics/broadcast.py
```

其中提供：

```python
infer_broadcast_shape(...)
broadcast_shape_constraints(...)
broadcast_index(...)
```

这些 helper 已经支持标准 trailing-dimension broadcasting。

本阶段的目标不是重新设计 broadcast，而是：

> 将 MatMul 从 2D-only 扩展为 rank >= 2 的 batched MatMul，并仅在 leading batch dimensions 上复用现有 broadcast primitives。

---

## 2. 本阶段目标

将当前：

```text
MatMul:
2D x 2D only
```

扩展为：

```text
MatMul:
rank >= 2
batched matmul
batch-dimension broadcasting
```

正式语义为：

\[
A:[B_A...,M,K]
\]

\[
B:[B_B...,K,N]
\]

要求：

\[
broadcastable(B_A,B_B)
\]

且：

\[
K_A=K_B
\]

输出：

\[
C:[broadcast(B_A,B_B),M,N]
\]

其中：

```text
A[:-2] = left batch shape
B[:-2] = right batch shape
A[-2:] = [M,K]
B[-2:] = [K,N]
```

只有 batch dimensions 使用 broadcast。

最后两维仍由 MatMul 自身的矩阵乘法语义决定。

---

## 3. 非目标

本阶段不要实现：

```text
1D @ 1D
1D @ 2D
2D @ 1D
PyTorch matmul 的 vector 特殊升维/降维规则
Transpose
Sum
Reshape
Linear
AllReduce
AllGather
ReduceScatter
```

不要修改：

```text
Stage IR schema
Relation semantics
Verifier proof formula
Lemma format
Add/Mul broadcast semantics
```

本阶段只扩展：

```text
MatMulOperator
必要的 MatMul-specific helper
相关 tests / examples / docs
```

除非现有 `broadcast.py` 明显无法复用，否则不要重构 broadcast 模块。

---

## 4. MatMul 正式语义

### 4.1 Rank 要求

第一版仅支持：

\[
rank(A)\ge2
\]

\[
rank(B)\ge2
\]

不支持 rank-1 vector MatMul。

例如：

```text
[8] @ [8]
[8] @ [8,16]
[16,8] @ [8]
```

本阶段都应拒绝。

---

## 5. 支持的主要情况

### Case 1：2D × 2D

```text
[M,K] @ [K,N] -> [M,N]
```

必须保持现有行为。

### Case 2：3D × 2D

```text
[B,M,K] @ [K,N] -> [B,M,N]
```

例如：

```text
[2,3,4] @ [4,5] -> [2,3,5]
```

右边没有 batch dimensions，相当于在所有 batch 上共享。

### Case 3：2D × 3D

```text
[M,K] @ [B,K,N] -> [B,M,N]
```

例如：

```text
[3,4] @ [2,4,5] -> [2,3,5]
```

### Case 4：3D × 3D，相同 batch

```text
[B,M,K] @ [B,K,N] -> [B,M,N]
```

### Case 5：4D × 4D，相同 batch

例如 Transformer attention：

```text
[B,H,S,D] @ [B,H,D,T] -> [B,H,S,T]
```

### Case 6：batch singleton broadcast

```text
[B,H,M,K] @ [1,H,K,N] -> [B,H,M,N]
```

### Case 7：batch rank mismatch

```text
[B,H,M,K] @ [H,K,N] -> [B,H,M,N]
```

右对齐 batch dimensions：

```text
[B,H]
[  H]
```

等价于：

```text
[B,H]
[1,H]
```

### Case 8：多 batch 维同时 broadcast

```text
[2,1,8,M,K] @ [1,4,8,K,N]
-> [2,4,8,M,N]
```

---

## 6. 非法情况

### 6.1 Batch dimensions 不兼容

例如：

```text
[2,3,M,K]
[4,3,K,N]
```

batch：

```text
[2,3]
[4,3]
```

第一维：

```text
2 vs 4
```

不兼容，应在 concrete validation 阶段失败。

### 6.2 Contraction dimension 不匹配

例如：

```text
[2,3,M,16]
[1,3,15,N]
```

即使 batch 可 broadcast，也必须失败：

\[
16\ne15
\]

### 6.3 Output shape 错误

例如：

```text
A = [2,3,8,16]
B = [1,3,16,32]
```

正确输出：

```text
[2,3,8,32]
```

如果声明成其它 shape，必须失败。

---

## 7. Broadcast 复用原则

当前已有：

```python
infer_broadcast_shape(...)
broadcast_shape_constraints(...)
broadcast_index(...)
```

MatMul 不应对完整 tensor shape 调用它们。

错误：

```python
infer_broadcast_shape(left_shape, right_shape)
```

正确：

```python
left_batch = left_shape[:-2]
right_batch = right_shape[:-2]

batch_output = infer_broadcast_shape(
    left_batch,
    right_batch,
)
```

MatMul 的 shape 由：

```text
broadcast(batch dimensions)
+
matrix dimensions
```

组合得到。

不要把最后两维交给 generic broadcast helper。

---

## 8. MatMulOperator 修改要求

### 8.1 `validate_concrete_shapes()`

职责：

> 检查 original MatMul program legality。

必须检查：

```text
exactly two inputs
exactly one output
rank(left) >= 2
rank(right) >= 2
rank(output) >= 2
```

定义：

```python
left_batch = left[:-2]
right_batch = right[:-2]

M = left[-2]
K_left = left[-1]
K_right = right[-2]
N = right[-1]
```

要求：

\[
K_{left}=K_{right}
\]

使用：

```python
batch_output = infer_broadcast_shape(
    left_batch,
    right_batch,
)
```

推导 expected output：

```python
expected_output = batch_output + (M, N)
```

最后要求：

```text
declared output == expected_output
```

---

### 8.2 `shape_constraints()`

职责：

> 定义 reduced MatMul 的合法 shape，并保持 original batch broadcast structure。

需要拆成两部分。

#### A. Batch dimensions

对：

```text
left[:-2]
right[:-2]
output[:-2]
```

调用：

```python
broadcast_shape_constraints(...)
```

original shapes 同样传：

```text
original_left[:-2]
original_right[:-2]
original_output[:-2]
```

要求保持：

```text
original equal
→ reduced equal

original singleton
→ reduced singleton

missing leading batch dim
→ 保持 missing-leading 对齐关系
```

#### B. Matrix dimensions

必须添加：

\[
left'[-1] = right'[-2]
\]

\[
output'[-2] = left'[-2]
\]

\[
output'[-1] = right'[-1]
\]

并继续保留当前 reduction soundness 约束：

\[
K' \ge \min(2,K_{original})
\]

这里的 `K` 指 contraction dimension：

```text
left[-1]
right[-2]
```

不要对 M/N 强制最小值 2。

不要删除 Stage 2 全局已有的：

\[
1\le d'\le d
\]

约束。

---

### 8.3 `symbolic_execute()`

职责：

> 构造 reduced batched MatMul 的 value semantics。

对于：

\[
A:[batch_A...,M,K]
\]

\[
B:[batch_B...,K,N]
\]

\[
C:[batch_C...,M,N]
\]

其中：

\[
batch_C=broadcast(batch_A,batch_B)
\]

需要对每个：

```text
output_batch_index
row i
column j
```

计算：

\[
C[b,i,j]
=
\sum_k
A[\phi_A(b),i,k]
\cdot
B[\phi_B(b),k,j]
\]

其中：

```text
φA = broadcast_index(batch_output_index, left_batch_shape, output_batch_shape)
φB = broadcast_index(batch_output_index, right_batch_shape, output_batch_shape)
```

最终 input index：

```python
left_index = left_batch_index + (row, k)
right_index = right_batch_index + (k, column)
```

不要针对 rank=2/3/4 写分支。

必须统一支持任意 rank >= 2。

---

## 9. 推荐增加 MatMul-specific helper

可以选择增加一个小 helper，例如：

```python
def infer_matmul_output_shape(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    context: str | None = None,
) -> tuple[int, ...]:
    ...
```

职责：

```text
rank check
batch broadcast
K match
output shape inference
```

如果这样可以避免：

```text
validate_concrete_shapes
symbolic_execute
tests
```

重复 shape inference，推荐抽取。

但不要创建过度复杂的 MatMul framework。

---

## 10. Relation Semantics 不需要修改

本阶段的 broadcast / batched MatMul 仍然只负责 operator local semantics。

例如未来可以验证：

```text
A -> Shard(dim=...)
B -> Replicate
C -> Shard(dim=...)
```

或者：

```text
A -> Shard
B -> Shard
C -> Partial
```

是否成立，由已有 relation semantics + SMT verification 决定。

不要在 MatMulOperator 中硬编码：

```text
Shard + Shard -> Partial
```

之类的 relation transition 规则。

MatMul 只定义：

> 每个 rank 上 local MatMul 怎么算。

Relation 是否成立由 verifier 证明。

---

## 11. 必须新增 Unit Tests

至少覆盖以下 10 类测试。

### Test 1：2D regression

```text
[3,4] @ [4,5] -> [3,5]
```

确保旧行为不回归。

### Test 2：3D × 2D

```text
[2,3,4] @ [4,5]
-> [2,3,5]
```

### Test 3：2D × 3D

```text
[3,4] @ [2,4,5]
-> [2,3,5]
```

### Test 4：3D × 3D

```text
[2,3,4] @ [2,4,5]
-> [2,3,5]
```

### Test 5：4D attention-like case

```text
[2,3,4,5] @ [2,3,5,6]
-> [2,3,4,6]
```

### Test 6：batch singleton broadcast

```text
[2,3,4,5] @ [1,3,5,6]
-> [2,3,4,6]
```

### Test 7：batch rank mismatch

```text
[2,3,4,5] @ [3,5,6]
-> [2,3,4,6]
```

### Test 8：invalid batch broadcast

```text
[2,3,4,5] @ [4,3,5,6]
```

必须 validation failure。

### Test 9：invalid contraction K

```text
[2,3,4,5] @ [1,3,6,7]
```

必须 validation failure。

### Test 10：wrong output shape

```text
A = [2,3,4,5]
B = [1,3,5,6]
declared C = [2,3,4,5]
```

必须 validation failure。

---

## 12. 必须新增 Shape Reduction Tests

至少覆盖：

### Reduction Test A：batch singleton preservation

Original：

```text
A = [8,4,16,32]
B = [1,4,32,64]
C = [8,4,16,64]
```

必须证明 reduced 约束保持：

```text
B'[0] == 1
A'[1] == B'[1]
C'[0] == A'[0]
C'[1] == A'[1]
C'[-2] == A'[-2]
C'[-1] == B'[-1]
A'[-1] == B'[-2]
```

### Reduction Test B：K lower bound

如果 original：

```text
K = 32
```

必须证明：

\[
K'\ge2
\]

如果 original：

```text
K = 1
```

则允许：

\[
K'=1
\]

---

## 13. 必须新增 Symbolic Execution Tests

至少明确检查一个 batched + broadcast case。

例如：

```text
A = [2,1,2,3]
B = [1,4,3,2]
C = [2,4,2,2]
```

选择：

```text
C[1,3,0,1]
```

应满足：

\[
C[1,3,0,1]
=
\sum_{k=0}^{2}
A[1,0,0,k]
\cdot
B[0,3,k,1]
\]

使用 Z3 equivalence 检查。

---

## 14. 必须新增 End-to-End JSON Examples

至少新增 4 个。

建议目录：

```text
input/broadcast/matmul_batched/
```

### Example A：3D × 2D，Replicate weight

Single：

```text
A [4,3,4]
B [4,5]
C [4,3,5]
```

world_size = 2

relations：

```text
A -> Shard(dim=0)
B -> Replicate
C -> Shard(dim=0)
```

Local：

```text
A_r [2,3,4]
B_r [4,5]
C_r [2,3,5]
```

预期：

```text
PROVED
```

---

### Example B：4D batched MatMul

Single：

```text
A [4,2,3,4]
B [4,2,4,5]
C [4,2,3,5]
```

relations：

```text
A -> Shard(dim=0)
B -> Shard(dim=0)
C -> Shard(dim=0)
```

Local：

```text
A_r [2,2,3,4]
B_r [2,2,4,5]
C_r [2,2,3,5]
```

预期：

```text
PROVED
```

---

### Example C：batch broadcast MatMul

Single：

```text
A [4,2,3,4]
B [1,2,4,5]
C [4,2,3,5]
```

relations：

```text
A -> Shard(dim=0)
B -> Replicate
C -> Shard(dim=0)
```

Local：

```text
A_r [2,2,3,4]
B_r [1,2,4,5]
C_r [2,2,3,5]
```

预期：

```text
PROVED
```

---

### Example D：invalid batch broadcast

例如：

```text
A [2,3,4,5]
B [4,3,5,6]
```

预期：

```text
load_stage / validate_stage failure
```

---

## 15. 推荐增加一个 Partial 输出 Case

非常推荐复用原来 2D 的经典：

```text
Shard_1(A)
Shard_0(B)
→ MatMul
→ Partial(sum)
```

并扩成 batched 版本。

例如：

```text
A [2,3,4,8]
B [2,3,8,5]
C [2,3,4,5]
```

world_size = 2。

沿 contraction K 做 shard：

```text
A -> Shard(dim=3)
B -> Shard(dim=2)
C -> Partial(sum)
```

Local：

```text
A_r [2,3,4,4]
B_r [2,3,4,5]
C_r [2,3,4,5]
```

每个 rank 计算自己的 partial MatMul。

预期：

```text
PROVED
```

这个 case 很重要，因为它验证：

> batched MatMul 扩展没有破坏原有 Shard-K → Partial(sum) 语义。

---

## 16. Wrong Relation Case

推荐新增一个：

```text
合法 batched MatMul
但错误 candidate output relation
```

例如本应：

```text
Shard
```

却声明：

```text
Replicate
```

或者本应：

```text
Partial(sum)
```

却声明：

```text
Replicate
```

预期：

```text
SAT
DISPROVED
```

---

## 17. Regression 要求

必须确保当前全部测试继续通过。

尤其：

```text
2D MatMul
Add/Mul broadcasting
matmul_shard_to_partial
large MatMul
wrong replicate case
lemma materialization
```

不能回归。

---

## 18. 最终运行要求

完成后必须实际运行：

```bash
conda run -n smt pytest -q
```

并逐个运行新增 MatMul examples。

最终汇报必须包含：

```text
1. commit SHA
2. 新增/修改文件
3. MatMul supported rank / semantics
4. 是否修改 broadcast.py
5. validate_concrete_shapes 修改
6. shape_constraints 修改
7. symbolic_execute 修改
8. 新增 unit tests 数量
9. 新增 end-to-end JSON
10. 每个 JSON 的实际结果
11. 完整 pytest 结果
12. 2D MatMul regression 是否通过
13. Add/Mul broadcast 是否未修改
14. relation semantics 是否未修改
15. verifier formula 是否未修改
```

---

## 19. 验收标准

本阶段完成后，应满足：

```text
MatMul:
rank >= 2
batched matmul
batch trailing broadcasting
original-aware reduced batch pattern preservation
contraction K lower-bound preservation
arbitrary-rank symbolic execution
```

且保持：

```text
Operator semantics
!=
Relation semantics
```

MatMul 只负责 local value/shape semantics。

single/distributed tensor relation 仍由已有 relation layer + SMT verifier 处理。
