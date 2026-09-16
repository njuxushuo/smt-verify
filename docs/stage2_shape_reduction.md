# Stage 2：Symbolic Shape 与 Shape Reduction

阶段二在已校验的 `StageSpec` 上构建 symbolic shape，并在约束满足的前提下缩小为最小 concrete shape。它只处理 tensor shape；不创建 tensor 元素符号值，也不证明单机与分布式程序的值等价。

## Symbolic shape

每个 tensor 的每个维度都会被替换为独立的 Z3 `Int`。例如单机 `A` 的两个维度名为 `single__A__d0` 与 `single__A__d1`；rank 0 上 `A0` 的维度名为 `rank0__A0__d0` 等。所有维度均有 `d >= 1` 约束。原始 concrete 数值不固定 symbolic dimension，只保留 tensor rank 和输入结构。

## 四类约束

1. **Dimension validity**：所有 symbolic dimension 均至少为 1。
2. **Relation shape**：输入 relation 与输出 relation 都生成 shape 约束。
3. **Operator shape**：对 single side 和每个显式 rank 的顺序 op 生成约束。
4. **Shard divisibility**：Shard 的全局维度必须满足 `global_dim % world_size == 0`。

## Relation shape semantics

Relation-specific symbolic shape semantics are implemented in [`src/semantics/relations.py`](../src/semantics/relations.py) and are reached through the relation registry. [`src/shape/constraints.py`](../src/shape/constraints.py) only resolves Stage shapes and aggregates the constraints.

- `replicate`：每个 rank local shape 逐维等于 single shape。
- `shard(dim=d)`：非 shard 维相等；第 `d` 维满足 `world_size * local_dim == global_dim`，并要求全局维度整除 world size。
- `partial(sum)`：本阶段只要求 local shape 逐维等于 single shape；跨 rank 的 value sum 属于后续阶段。

## Operator shape semantics

当前只支持：

- `matmul`：支持 rank >= 2 的 batched MatMul。仅 `A[:-2]` 与 `B[:-2]` 进行 trailing broadcasting，并根据 original shape 保留 batch broadcast pattern；矩阵维度约束为 `A[-1] == B[-2]`、`C[-2] == A[-2]`、`C[-1] == B[-1]`。
- `add`、`mul`：支持 arbitrary-rank standard trailing-dimension broadcasting。reduced shape 约束根据 original shape 逐维保留 equal、left-singleton、right-singleton 或 missing-leading 模式，避免 reduction 改变原始 broadcast pattern。
- `transpose`：reduced output 对 input 执行相同 axis permutation，不增加额外维度下界。
- `reshape`：按 original row-major contiguous dimension groups 保留每组 element count，而不是只约束整体 numel。
- `squeeze`、`unsqueeze`：保留删除或插入的 singleton axis 及其余 axis 映射；omitted-dim squeeze 还保留 original singleton/non-singleton 区分。
- `expand`：按 original equal/singleton/missing-leading pattern 保留 unary trailing expansion。

`sum`、`all_reduce`、`all_gather` 与 `reduce_scatter` 可以由阶段一读取，但阶段二会抛出 `UnsupportedShapeSemanticsError`。

## Operator semantics organization

Stage-level constraint builder does not encode concrete operator rules. Operator interfaces and the registry live in [`src/semantics/operators/__init__.py`](../src/semantics/operators/__init__.py), while element-wise, MatMul, and View implementations are split into family modules in the same package. `get_operator()` resolves every implemented operator and attrs are passed directly to its shape constraints.

## Optimize objective

使用 Z3 `Optimize`，目标固定为所有 tensor symbolic dimensions 的总和：

```text
min Σ_T Σ_i D[T, i]
```

求解成功后，model 中的维度会被 concretize 为 `ReducedShapeResult`，不会改动原始 `StageSpec`。

## 与 TrainVerify 的关系

TrainVerify 的 `rxshape.py` 启发了本实现的 symbolic dimensions、validity constraints、`Optimize` 最小化与 model concretization；其 operator registry 启发了独立的 shape propagation。不同之处是本项目用 `Replicate`、`Shard`、`Partial` 的显式 relation shape semantics 替代 TrainVerify 的 DFG、lineage 和 generic sub-shape ratio 推导，也不引入其多 backend、stage partition 或 value-level验证结构。

## 当前限制与 CLI

本阶段不实现 tensor value、算术展开、relation value encoding、equivalence proof、candidate filtering、lemma、graph alignment 或真实 IR parser。

在 `smt` 环境中运行：

```bash
python scripts/reduce_shapes.py input/matmul_shard_to_partial/matmul_shard_to_partial.json
python scripts/reduce_shapes.py input/matmul_shard_to_partial_large/matmul_shard_to_partial_large.json
```

命令打印 original/reduced shape、目标值及 `Shape reduction: PASS`；shape 约束不可满足或遇到未支持算子时打印失败原因并返回退出码 1。
