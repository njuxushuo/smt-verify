# 单机—分布式程序等价验证项目概述（当前阶段：SMT 验证部分）

## 1. 项目背景

本项目关注一个核心问题：

> **如何形式化证明单机程序与其分布式并行实现之间的语义等价性？**

在大模型训练和推理中，同一个逻辑模型通常会被转换为包含数据并行（DP）、张量并行（TP）、流水线并行（PP）、专家并行（EP）以及集合通信（如 AllReduce、AllGather、ReduceScatter、AllToAll）的分布式实现。分布式程序中的中间 tensor 往往不再与单机程序中的 tensor 逐元素直接相等，而是以不同的分布状态存在，例如：

- Replicate：每个 rank 都保存完整 tensor；
- Shard(dim)：tensor 沿某个维度被不同 rank 分片持有；
- Partial：每个 rank 持有最终 tensor 的部分贡献，例如矩阵乘法 reduction 维上的部分和；
- 后续还可能扩展更多 layout / permutation 类关系。

因此，验证目标不应简单表述为“单机每一步输出都等于分布式每一步输出”，而应建立一套**单机 tensor 与 distributed tensor 之间的关系语义**，并让这种关系沿着计算图逐步传播。

当前整体研究思路是：将程序级等价验证拆解为许多较短的局部 stage。对于每个 stage，已知输入 tensor 之间的 relation，经过单机和分布式两侧的局部计算后，推导并验证输出 tensor 之间的 relation。如果某个 relation 能通过 SMT 被严格证明，则将该结果缓存成可复用 lemma，后续遇到相同或足够相似的计算模式时直接复用，而无需再次进行完整 SMT 验证。

这样，系统逐渐形成：

```text
单机 / 分布式图对齐
        ↓
维护输入 tensor relation
        ↓
局部 stage
        ↓
候选输出 relation
        ↓
SMT 认证
        ↓
生成 certified lemma
        ↓
后续直接复用
```

当前项目的长期目标，是构建一个“**关系传播 + SMT 认证 + lemma 复用**”的单机—分布式程序等价验证系统。

---

## 2. 相关工作与重要参考

以下工作与本项目关系密切，建议理解其核心思想与边界。

### 2.1 TrainVerify

论文：

- https://arxiv.org/pdf/2506.15961

官方仓库：

- https://github.com/microsoft/TrainVerify

TrainVerify 的目标是验证原始单机模型与分布式并行执行计划之间的 arithmetic equivalence。它的总体流程包括：

1. 获取单机和并行模型的 execution plan；
2. 转换为 symbolic SSA DAG；
3. 利用 lineage 将大图划分为多个独立 stage；
4. 对每个 stage 进行 shape reduction；
5. 在 reduced shape 上构造 symbolic tensor；
6. 根据 operator 语义进行 symbolic execution；
7. 使用 Z3 验证单机侧和分布式侧的输出关系是否成立。

TrainVerify 对本项目最重要的参考价值主要有两点：

- **shape reduction**：先把原始大 tensor 的各维度符号化，传播 shape 约束，再求一个满足这些约束的尽可能小的 concrete shape，以显著降低后续 symbolic execution 的规模；
- **element-wise symbolic execution + SMT equivalence checking**：在 reduced shape 上，把 tensor 元素建模成 Z3 symbolic variables，根据算子数学语义展开，并通过检查“前提 ∧ 非目标”是否 UNSAT 来证明目标恒成立。

当前 SMT 模块可以重点参考 TrainVerify 以下代码：

- `Verdict/verdict/stage/rxshape.py`
- `Verdict/verdict/stage/run.py`
- `Verdict/z3_backend/backend.py`
- `Verdict/z3_backend/core.py`
- `Verdict/verdict/operators/registered_ops.py`

需要注意：本项目不是简单复现 TrainVerify。TrainVerify 更偏向直接验证单机与完整分布式 stage 的等价，而本项目希望将验证进一步抽象为 **tensor relation propagation**，并把经过 SMT 证明的 relation transfer 缓存成 lemma。

---

### 2.2 GraphGuard / Iterative Relation Inference

论文：

- https://arxiv.org/abs/2508.09505

该工作强调通过高层 relation 和 iterative rewriting 来验证分布式实现对顺序模型的 refinement。它说明了一个重要事实：

> 分布式程序与单机程序之间不一定需要在每个中间点直接相等，而可以维护更高层的中间关系。

本项目借鉴这一思想，但希望把这些 relation 的局部传播规则通过 SMT 自动认证，并形成可复用 lemma。

---

### 2.3 Scalify

论文：

- https://arxiv.org/pdf/2509.10694

Scalify 使用 equality saturation、relational reasoning、symbolic bijection inference 等方法验证生产级分布式计算图。它对本项目的重要启发是：

> distributed tensor state / layout relation 应作为一等语义对象进行传播，而不是只比较 tensor equality。

本项目当前的 relation domain 可先以类似 DTensor 的形式表示，例如：

```text
Replicate
Shard(dim)
Partial(reduce_op)
```

后续可以根据真实分布式程序的需要扩展更多 relation。

---

### 2.4 Emerge

论文：

- https://arxiv.org/abs/2603.21851

Emerge 研究如何从两个实现中自动发现 candidate relation / rewrite rule，再通过 abstraction frontier 形成可复用 lemma。

本项目与它的共同点是：

> 都希望把一次证明结果抽象成可以后续复用的 lemma。

不同之处在于，本项目当前不追求自动生成任意复杂表达式，而是把候选限制在有限的 tensor relation domain 中，并重点通过 SMT 对 relation transfer 进行形式化认证。

---

## 3. 当前整体方法设想

### 3.1 Tensor relation 建模

对单机 tensor `T_s` 和分布式程序中不同 rank 上的 tensor `{T_r}`，使用 relation 描述它们之间的对应关系。

第一版重点考虑：

```text
Replicate
Shard(dim)
Partial(sum)
```

它们应具备严格语义，而不只是字符串标签。

例如：

**Replicate**

```text
对任意 rank r：
T_r == T_s
```

**Shard(dim=d)**

```text
每个 rank 保存 T_s 在第 d 维上的一个合法、不重叠分片，
所有 rank 的 shard 可以按顺序重构 T_s。
```

**Partial(sum)**

```text
所有 rank 的 local tensor shape 与 global tensor 一致，
并且：
T_s = Σ_r T_r
```

未来可以增加 permutation、interleaved shard、layout mapping 等更复杂关系，但当前无需一开始覆盖所有情况。

---

### 3.2 Stage 的基本思想

验证过程按局部 stage 进行。

一个 stage 可以理解为：

- 单机侧一小段局部计算；
- 分布式侧与之对应的一小段局部计算；
- 已知输入 relation；
- 待验证输出 relation。

重要原则是：

> **只要已经能在某个中间点建立合法 relation，就可以结束当前 stage，不必等待两边恢复成完全相等。**

例如，单机和分布式两侧都执行完一个 matmul 后，分布式输出可能只是 `Partial`。此时只要能够证明：

```text
single_output  ~ Partial(distributed_output)
```

就可以结束当前 stage。

后续的 AllReduce 可以作为新的 stage，单独证明：

```text
Partial → Replicate
```

这样能够显著缩短每个 lemma 的长度，提高 SMT 求解效率和 lemma 复用性。

---

## 4. 当前阶段的任务范围

**当前仅需要实现 SMT 相关部分。**

暂时不要实现或过度设计以下内容：

- 整图级单机 / 分布式 graph alignment；
- 如何自动从整张图中切 stage；
- 如何自动选择最优 stage 边界；
- 完整 lemma library 管理系统；
- LLM / Emerge 风格的自动 lemma synthesis；
- 高层前端框架集成。

当前需要实现的核心能力可以抽象成：

> 给定一个已经准备好的局部 stage、已知输入 relation，以及一个或多个候选输出 relation，使用 SMT 判断候选 relation 是否成立。

因此，当前 SMT 模块的输入可以假设已经由上层准备好。

---

## 5. SMT 部分的整体框架

### 5.1 Stage 验证问题的形式

对于一个 stage，设：

- `Gs`：单机侧局部计算；
- `Gd`：分布式侧局部计算；
- `Rin`：输入 tensor relation；
- `Rout_candidate`：待验证输出 relation。

需要证明：

```text
Rin
AND semantics(Gs)
AND semantics(Gd)
=> Rout_candidate
```

实际 SMT 检查：

```text
Rin
AND semantics(Gs)
AND semantics(Gd)
AND NOT Rout_candidate
```

如果 Z3 返回：

```text
UNSAT
```

则证明候选 relation 恒成立。

如果返回：

```text
SAT
```

则当前候选 relation 不成立，并应尽量保留 counterexample 以便调试或供上层尝试其他候选。

还应区分：

```text
PROVED
DISPROVED
UNKNOWN / TIMEOUT
```

避免把 solver timeout 与“关系不成立”混淆。

---

## 6. Shape Reduction

这是当前 SMT 实现中非常重要的组成部分，建议重点参考 TrainVerify。

真实模型中的 tensor shape 通常很大，不能直接对每个原始 tensor 元素进行 symbolic execution。

因此，对每个 stage，首先只处理 shape。

### 6.1 符号化 shape

例如原始：

```text
A: [4096, 11008]
```

先抽象为：

```text
A: [a0, a1]
```

每个维度使用整数 symbolic variable。

### 6.2 收集 shape 约束

约束主要来自：

1. operator 语义；
2. 原始 tensor 之间的 shape 比例；
3. shard / replicate / partial relation；
4. world size 和 partition；
5. tensor 非空；
6. shard 维度整除等合法性条件。

例如 matmul：

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

需要保持：

```text
A[1] == B[0]
C[0] == A[0]
C[1] == B[1]
```

如果某个维度被 8-way shard，还要保证：

```text
K % 8 == 0
```

### 6.3 求 reduced shape

在满足所有约束的前提下，寻找尽可能小的合法 shape。

第一版可以采用与 TrainVerify 类似的目标：

```text
minimize sum(all tensor dimensions)
```

例如原始：

```text
[4096, 11008] × [11008, 4096]
```

可能被缩小为：

```text
[1, 8] × [8, 1]
```

前提是所有 operator / relation / partition constraint 仍成立。

Shape reduction 的目的不是证明一个“具体小 shape case”，而是保留当前 stage 中与结构和 relation 有关的关键约束，使后续 symbolic execution 可以在更小的问题规模上完成。

---

## 7. Reduced Shape 上的 Symbolic Tensor

得到 reduced shape 后，再真正创建 symbolic tensor。

例如：

```text
A: [2, 4]
```

则创建：

```text
A_00, A_01, ..., A_13
```

每个元素是一个 Z3 symbolic expression，通常可先使用 `Real`。

输入 tensor 元素可以作为自由变量。

对于单机 tensor 和 distributed tensor，如果输入 relation 已知，则通过 relation constraints 将它们关联起来，而不是把所有 tensor 视为完全独立。

---

## 8. Relation Encoding

需要为每种 relation 定义严格的 SMT encoding。

当前至少实现：

### Replicate

在 reduced shape 上逐元素表达：

```text
for every rank r:
    T_dist[r][...] == T_single[...]
```

### Shard(dim)

根据 rank 和 local index 计算 global index，并表达：

```text
T_dist[r][local_index]
==
T_single[global_index]
```

第一版可以优先考虑 regular contiguous shard。

### Partial(sum)

逐元素表达：

```text
T_single[index]
==
sum(T_dist[r][index] for all ranks)
```

relation encoding 既用于：

- 输入 relation，作为 solver 的前提；
- 输出 candidate relation，作为待证明目标。

---

## 9. Operator Symbolic Semantics

需要在 reduced shape 上展开 operator 数学语义。

优先支持常见、容易形式化的算子，例如：

```text
Add
Mul
MatMul
Linear
Reshape
Transpose
Slice
Concat
Reduce / Sum
```

通信算子优先支持：

```text
AllReduce
AllGather
ReduceScatter
```

例如 MatMul：

```text
C[i,j] = Σ_k A[i,k] * B[k,j]
```

AllReduce(sum)：

```text
Y_r[i] = Σ_q X_q[i]
```

Reshape / Transpose 等可以视为 index mapping。

这一部分建议参考 TrainVerify 中 `registered_ops.py` 的 operator symbolic implementation，但当前文档不预设具体代码架构。

---

## 10. Candidate Relation 与 Shape Filtering

上层可能会提供一个或多个候选输出 relation。

当前 relation domain 是有限的，因此理论上可以依次尝试：

```text
Replicate
Shard(0)
Shard(1)
...
Partial
```

但不应无脑将所有候选都送给 SMT。

首先利用：

- output shape；
- local shape；
- operator 类型；
- world size；
- 已知 placement；

做低成本过滤。

例如 global output shape 与 local output shape 不同，则很多 `Replicate` 或 `Partial` 候选可以直接排除。

只有 shape 上可能成立的候选才进入 SMT。

---

## 11. Lemma 的当前理解

当前阶段可以允许 lemma 使用**具体 shape**，无需一开始做 shape 参数化。

原因是，同一个大模型的不同层之间经常反复出现相同的：

- operator pattern；
- tensor shape；
- world size；
- placement；
- communication pattern。

因此第一次 SMT 验证成功后，可以记录类似：

```text
Single pattern:
    MatMul

Distributed pattern:
    Local MatMul

Input shapes:
    concrete shapes

Input relations:
    concrete placement relations

World size:
    concrete value

Output relation:
    Partial

Status:
    SMT_PROVED
```

后续遇到完全相同的 pattern，可直接复用。

注意：

> lemma 使用 concrete shape，与 SMT 内部使用 symbolic shape reduction 并不矛盾。

外部缓存的 lemma 可以是具体 shape；内部证明时，仍然可以暂时将各维度符号化、推导约束、reduce 到小 shape，再进行 symbolic execution。

未来如果需要跨 shape 泛化，再进一步研究参数化 lemma。

---

## 12. 当前 SMT 模块的预期工作流

可以先把目标理解为如下逻辑流程：

```text
Input:
    Single-side stage
    Distributed-side stage
    Known input relations
    Candidate output relations
    Original tensor shapes
    World / rank information

        ↓

[1] Build symbolic shape variables

        ↓

[2] Collect shape constraints
    - operator constraints
    - relation constraints
    - sharding constraints
    - partition constraints
    - validity constraints

        ↓

[3] Shape-based candidate filtering

        ↓

[4] Shape reduction
    Find minimum valid reduced shapes

        ↓

[5] Create symbolic tensor elements

        ↓

[6] Encode known input relations

        ↓

[7] Symbolically execute single-side operators

        ↓

[8] Symbolically execute distributed-side operators

        ↓

[9] Encode candidate output relation

        ↓

[10] SMT check

    Given AND NOT Goal

        ↓

UNSAT
    → candidate relation PROVED
    → produce certified local result / lemma

SAT
    → candidate relation DISPROVED
    → retain counterexample if possible

UNKNOWN / TIMEOUT
    → report separately
```

---

## 13. 当前阶段最重要的实现目标

当前不要求一步完成完整系统。

最重要的是先建立一个**可信、可扩展的 Stage SMT Verifier 原型**，证明以下技术链条是可行的：

```text
relation-aware shape constraints
        ↓
shape reduction
        ↓
element-wise symbolic tensor
        ↓
operator symbolic semantics
        ↓
input relation constraints
        ↓
candidate output relation
        ↓
Z3 proof
```

优先验证几个具有代表性的局部 pattern 即可。

当前阶段的关键评价标准不是覆盖多少算子，而是：

1. relation 语义是否定义清楚且严格；
2. shape reduction 是否正确保留 stage 结构；
3. reduced tensor symbolic execution 是否正确；
4. SMT proof / counterexample 是否可靠；
5. 后续是否容易扩展更多 operator 和 relation；
6. 是否能够为未来的 lemma reuse 提供稳定的验证结果。

---

## 14. 当前阶段与 TrainVerify 的关系

实现时可以大胆参考 TrainVerify，但需要保持项目目标上的区别。

TrainVerify 可以帮助理解：

```text
symbolic shape
→ shape constraints
→ reduced shape
→ symbolic tensor
→ operator semantics
→ Z3 equivalence
```

本项目需要进一步加入：

```text
Replicate / Shard / Partial relation semantics
→ relation-aware stage verification
→ output relation inference / certification
→ future lemma reuse
```

因此，当前最合适的开发策略是：

> **先理解并复现 TrainVerify 中一个最小可工作的 shape reduction + symbolic execution + Z3 proof 路径，再在此基础上将“single-vs-parallel equality / lineage”替换或扩展为本项目需要的 distributed tensor relation semantics。**

---

## 15. 当前不要提前假设的内容

在当前阶段，不要自行假设以下内容已经最终确定：

- 最终完整 IR；
- Stage 数据结构；
- Lemma 数据库存储形式；
- graph alignment 算法；
- 多 rank 的最终抽象方式；
- 是否需要完全照搬 TrainVerify 的类层级；
- 是否必须支持所有 PyTorch operator；
- 是否一开始就支持任意 world size；
- 是否一开始就做 shape-parametric lemma。

这些内容后续会逐步确定。

当前任务重点只有一个：

> **把每个局部 stage 的 SMT 认证机制做对、做清楚、做成可扩展原型。**

