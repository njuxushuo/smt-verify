# 从零阅读 FC2 的验证过程与 demo.txt

本文对照以下实际文件，不是另外构造的示例：

- [输入 JSON](../input/mlp_dp2_tp4_index_value/mlp_dp2_tp4_03_fc2_partial.json)
- [验证报告 demo.txt](../input/mlp_dp2_tp4_index_value/mlp_dp2_tp4_03_fc2_partial/demo.txt)
- [主流程代码](../src/verification/verifier.py)

下面的报告行号对应当前 459 行版本。重新生成报告后，公式内部的临时编号可能变化；
章节标题和数学含义比临时编号更重要。报告是程序完成验证后整理生成的审计记录，
不是每一行都对应运行时打印的一步。

## 1. 先认识这道证明题

FC2 是 MLP 第二个线性变换。在这个阶段中，前面的激活函数已经计算完毕，结果作为 Z 输入。
本阶段仅有矩阵乘法，没有 GELU、bias、AllReduce 或 ReduceScatter。

单机计算 `Y = Z @ W2`；分布式 rank r 计算 `Pr = Zr @ W2_r`。

| 张量 | 单机 shape | 每个 rank 的 shape |
| --- | --- | --- |
| 激活 Z | `[4,128,1024]` | `[2,128,256]` |
| 权重 W2 | `[1024,256]` | `[256,256]` |
| 输出 Y / P | `[4,128,256]` | `[2,128,256]` |

激活使用 `[b,s,h]`：batch、sequence、feature。本例中，输入 feature 为 1024，输出 feature 为 256。
为了避免同一个 h 指代不同位置，下文用 k 表示矩阵乘法的求和索引，用 j 表示输出特征索引。
因此 `Z[b,s,k]` 与 `W2[k,j]` 相乘，对 k 求和得到 `Y[b,s,j]`。

DP=2 把全局 4 个 batch 分成两份；每个 DP 副本中的 TP=4 又把 1024 个求和项分成四份。

| rank | 负责的全局 batch | Z 的 feature / W2 的行区间 |
| --- | --- | --- |
| 0 | `[0,2)` | `[0,256)` |
| 1 | `[0,2)` | `[256,512)` |
| 2 | `[0,2)` | `[512,768)` |
| 3 | `[0,2)` | `[768,1024)` |
| 4 | `[2,4)` | `[0,256)` |
| 5 | `[2,4)` | `[256,512)` |
| 6 | `[2,4)` | `[512,768)` |
| 7 | `[2,4)` | `[768,1024)` |

例如 rank 1 本地的 `Z1[1,7,3]` 就是单机的 `Z[1,7,259]`；rank 5 本地的同一坐标
则对应 `Z[3,7,259]`。权重 `W2_1[3,9]` 与 `W2_5[3,9]` 都对应 `W2[259,9]`。

要证明的是：

```text
Y[0:2, :, :] = P0 + P1 + P2 + P3
Y[2:4, :, :] = P4 + P5 + P6 + P7
```

每个 P 都只是局部贡献，不要求 `P0=Y[0:2,:,:]`。这个加法属于待证明的输出关系，
并不意味着这个 stage 实际执行了 AllReduce。后续通信是另外一个 stage。

## 2. 实际代码先后顺序

```text
app.cli.main
  -> app.bootstrap.default_registry：准备算子实现
  -> verification.verifier.verify_file
       -> stage.codec.load_stage：读取 JSON，建立数据对象
       -> verification.verifier.verify_stage
            -> stage.validation.validate_stage：校验并规范化关系
            -> reduction.rules.check_budget：检查资源限制
            -> reduction.reducer.reduce_domains：建立符号输出索引域
            -> smt.context.SymbolicContext：准备符号输入环境
            -> relation.encoding.encode_inputs：建立输入等式并精确代入
            -> verification.executor.execute_stage：建立两侧输出表达式
            -> verification.obligation.build_obligation：构造违反输出关系的条件
            -> smt.solver.check：调用 Z3
       <- VerificationResult
  -> app.reporting.write_report：生成 demo.txt
```

core 的模型被以上模块共用。relation 不执行 matmul，ops 不猜测 Relation；
verification 负责协调它们。报告第 3 节打印的是规范化数据对象，第 5 节才展示符号等式。

## 3. 报告第 1—7 行：结果摘要

| 行 | 输出 | 解释 |
| --- | --- | --- |
| 1 | `Stage: mlp_dp2_tp4_03_fc2_partial` | 任务名称，不用于决定结果。 |
| 2 | `Verification: PROVED` | 最后得出的结果在开头先展示，不是初始化时就认定正确。 |
| 3 | `ORIGINAL_SHAPES_EXACT_REAL` | 验证原始 shape 下的精确实数运算。 |
| 4 | `Z3 version: 4.15.1` | 记录本次生成报告使用的求解器版本。 |
| 5 | 实数和 GELU 说明 | 所有报告共享的模型说明；本例没有实际使用 GELU。 |
| 6 | 浮点和整程序范围说明 | 结论不覆盖具体 GPU 浮点误差，也不是完整模型训练正确性。 |
| 7 | 输入 SHA256 | 对规范化输入内容计算的摘要，防止拿旧报告对应已修改的 JSON。 |

SHA256 不是证明证书，只用于识别本次报告的输入。

## 4. 报告第 9—66 行：JSON 解析成什么对象

对应 [stage/codec.py](../src/stage/codec.py)、[stage/model.py](../src/stage/model.py)
以及 [core](../src/core)。

### 第 11 行：StageSpec

`StageSpec(name=..., world_size=8)` 表示整个证明任务对象。解析并不执行数学计算，
而是把 JSON 转成结构清晰的 Python 对象。

### 第 12—21 行：Group 与 Family

- 第 12、13 行：`tp_0=(0,1,2,3)`、`tp_1=(4,5,6,7)`，用于解释输出 Partial 的两个求和范围。
- 第 14—17 行：DP 的四个组 `[0,4]`、`[1,5]`、`[2,6]`、`[3,7]`。
- 第 18—20 行：family 引用哪些具体组。SP 和 TP 在这套配置中复用实际组。
- 第 21 行：这种复用没有增加 rank 数量，不能将 DP、TP、SP 的大小再次全部相乘。

拓扑出现一个 family，不意味着本例所有张量都要有该 family 的 term；FC2 没有 SP 分片项。

### 第 22—26 行：单机程序

`inputs=('Z','W2')` 和 `outputs=('Y',)` 是切分边界。
`TensorSpec` 只保存 shape 等事实，不创建装满数字的数组。
`elements=524288` 是 `4*128*1024` 的计数，不表示分配了这么多 Z3 变量。

第 26 行：

```text
OpCall 0: matmul inputs=('Z', 'W2') outputs=('Y',) attrs={}
```

表示一个矩阵乘法调用。W2 的 `[1024,256]` 被同样用于每个 batch、sequence 位置。
`attrs={}` 表示该算子不需要额外参数。

### 第 27—66 行：八个 rank 程序

每五行描述一个 rank：边界、Zr、W2_r、Pr、matmul 调用。shape 都相同，但后续 Relation
说明它们代表不同的全局片段。`Z3` 在这里是 rank 3 的激活名字，和求解器 Z3 不是一回事。

## 5. 报告第 68—83 行：先确认问题合法

对应 [stage/validation.py](../src/stage/validation.py)、[stage/dataflow.py](../src/stage/dataflow.py)、
[relation/validation.py](../src/relation/validation.py) 和算子的 `infer()`。

| 报告行 | 检查什么 | 例子 |
| --- | --- | --- |
| 70 | 每个 family 的组完整覆盖 world 且组内不重复 | TP 两组不能漏掉 rank 7。 |
| 71—79 | 单赋值数据流、输入输出边界、算子 shape | 不能使用还没产生的张量；不能重复覆盖输出名。 |
| 80 | 两条输入关系合法 | Z 和 W2 都绑定全部 rank，区间不能越界。 |
| 81 | 一条输出目标合法 | Y 与全部 Pr 的目标映射、Partial 参与组相容。 |
| 82—83 | 上述检查的汇总 | 含未知字段、重复 JSON key、extent 和映射域检查。 |

单机 matmul 的求和维为 1024；每个 rank 的求和维为 256。两侧分别检查自己能否合法相乘，
不会要求单机和本地求和维相同。校验通过只表示“题目写得合法”，还没有证明“答案正确”。

## 6. 报告第 85—181 行：将 Relation 读成数学含义

对应 [relation/normalization.py](../src/relation/normalization.py)。

### 第 87—123 行：Z 的两个 term

第 96—104 行描述 DP batch 映射；第 105—113 行描述 TP feature 映射。
它们组合为一个完整映射，不分别生成两条全局值相等关系。

定义 `d=rank//4`、`t=rank%4`，则：

```text
Phi_Z,r(b,s,k) = (b+2*d, s, k+256*t)
Zr[b,s,k] = Z[b+2*d, s, k+256*t]
```

第 114 行说其他维度保持恒等。本例即 sequence 维 `s -> s`。
第 115 行的 `D_r[i] = S[Phi_r(i)]` 中：S 表示单机张量，D 表示本地张量，i 是一个完整坐标元组。

第 116—123 行的 `single-image intervals` 是“本地所有元素对应的单机区域”。
例如第 121 行 rank 5 的 `(((2,4),),((0,128),),((256,512),))` 表示
`Z[2:4,0:128,256:512]`。多层括号用于允许一个维度由多个区间组成，本例每维只有一段。

### 第 124—151 行：W2 的一个 term

W2 只沿第 0 维分片：

```text
Phi_W,r(k,j) = (k+256*t, j)
W2_r[k,j] = W2[k+256*t,j]
```

rank 0 和 rank 4 对应相同权重行，因此它们的权重分片相等。这个事实已经由全 rank 映射表达，
不需要另写 DP 恒等 term。报告中 Z 和 W2 都简写为 `Phi_r`，但它们各有自己的映射，不是同一个函数。

### 第 152—181 行：Y 的候选输出关系

第 152 行 `OUTPUT GOAL (never assumed)` 强调：这是目标，不能被当作已知等式。
第 161—169 行仍是 batch 映射；第 170 行是组内 Partial(sum)。组合起来：

```text
Y[b,s,j]   = P0[b,s,j]+P1[b,s,j]+P2[b,s,j]+P3[b,s,j]
Y[b+2,s,j] = P4[b,s,j]+P5[b,s,j]+P6[b,s,j]+P7[b,s,j]
```

这里 b 的范围都是 `[0,2)`。第 172 行公式中的 `Phi^-1(g)` 表示“给定单机位置 g，找到
某个 rank 对应的本地位置”。对第二组，单机 batch 3 的局部位置是 batch 1，所以需要减去 2。
第 173 行提醒 Partial 不等于复制，不能要求各 Pr 自己就等于完整 Y。

## 7. 报告第 183—196 行：到底做了什么 reduction

对应 [reduction/reducer.py](../src/reduction/reducer.py)。当前实现没有把物理 shape 缩小。

| 行 | 输出 | 含义 |
| --- | --- | --- |
| 185 | `SYMBOLIC_INDEX_DOMAIN` | 用符号输出坐标代表原始输出域。 |
| 186 | 原 shape、区间不变 | batch=4、sequence=128、求和维=1024 都保留。 |
| 187—188 | 任意整数坐标加边界 | 不是选择第一个元素，也不是随机取样。 |
| 189 | Partial 每组建立域 | 一个代表 rank 用来枚举组的共同目标域，求和仍包含组内全部 rank。 |
| 190 | 原始 K 项全部保留 | 单机 1024 项、本地 256 项，不改成 K=1 或 K=2。 |
| 192 | `131072 ... -> 2 ... domains` | 用两个符号查询域覆盖 131072 个输出等式。 |
| 193—194 | `Physical shape reduction: NONE` | 没有最小化 shape 的 Optimize，也没有小 shape 推广假设。 |
| 195—196 | 两组的边界 | 每组查询任意 `b in [0,2), s in [0,128), j in [0,256)`。 |

每组覆盖 `2*128*256=65536` 个位置，两组共 131072 个位置，正好是单机 Y 的元素数。
查询的是“是否存在任意一个位置失败”；若没有，就对整组所有位置成立。

## 8. 报告第 198—384 行：符号输入、等式与代入

对应 [smt/context.py](../src/smt/context.py) 和 [relation/encoding.py](../src/relation/encoding.py)。

### 符号输入不是随机数组

代码为每个原始输入创建一个从索引到实数的函数。比如 `input_0(b,s,k)` 表示单机 Z 的
一个尚不知道具体数值的元素。函数没有预先固定为 0 或某组训练权重。

报告第 343—361 行给出名称对应表：

```text
input_0 = single.Z        input_1 = single.W2
input_2 = rank0.Z0        input_3 = rank0.W2_0
input_4 = rank1.Z1        input_5 = rank1.W2_1
...
input_16 = rank7.Z7       input_17 = rank7.W2_7
```

18 个符号函数代表 18 个输入 tensor，不是只创建了 18 个标量元素。
调用同一函数的同一坐标，必须得到同一个值；不同坐标的值可以任意变化。

### 逐行解释 Rin[1]（第 211—220 行）

```lisp
(forall ((rin0_r1_i0 Int) (rin0_r1_i1 Int) (rin0_r1_i2 Int))
  (=> (and (>= rin0_r1_i0 0)
           (< rin0_r1_i0 2)
           (>= rin0_r1_i1 0)
           (< rin0_r1_i1 128)
           (>= rin0_r1_i2 0)
           (< rin0_r1_i2 256))
      (= (input_4 rin0_r1_i0 rin0_r1_i1 rin0_r1_i2)
         (input_0 rin0_r1_i0 rin0_r1_i1 (+ 256 rin0_r1_i2)))))
```

- `Rin` 是 input relation 的缩写；这里第 0 条输入关系是 Z，rank 是 1。
- `forall`：对于所有这些索引，而不是只选一个。
- `Int`：索引类型为整数；元素值仍是 Real。
- `i0/i1/i2`：本地 batch、sequence、feature 索引，可分别记为 b、s、k。
- `and`：后面的六个边界条件都成立。
- `=>`：如果索引位于合法范围，那么后面的等式成立。
- `=`：值相等。
- `input_4(...)`：Z1 的元素。
- `input_0(...,(+ 256 k))`：单机 Z 中 feature 偏移 256 的元素。

整个公式就是 `Z1[b,s,k] = Z[b,s,k+256]`，对所有合法本地坐标成立。
第 241—280 行的 rank 4—7 还会出现 `(+ 2 ...i0)`，表示 DP batch 偏移。

第 201—280 行 `Rin[0]` 到 `Rin[7]` 是 Z 的八个 rank；第 281—342 行 `Rin[8]` 到
`Rin[15]` 是 W2 的八个 rank。W2 只有两个索引，求和维偏移加在 `i0` 上。

### 第 362—378 行：为什么叫 elimination

代码随后不再把 Z1 当作完全独立的输入，而是直接定义：

```text
读取 Z1[b,s,k] 时，返回 Z[b,s,k+256] 的符号表达式。
读取 W2_1[k,j] 时，返回 W2[k+256,j] 的符号表达式。
```

这和普通代数中的“已知 a=b+1，所以把式子里的 a 换成 b+1”一样。
第 363 行等输出中的 `:=` 是“用右边定义左边”，不是待证明目标。

因此最后的 solver 查询没有再原样加入上述 forall Rin：约束已经通过代入被严格执行。
单机 Z、W2 仍然是任意值，因此证明仍覆盖所有满足输入关系的可能输入。
没有把输入固定成某一组数值，也没有假设输出关系成立。

第 379—384 行是通用的代入可靠性说明。其中“选择 Partial pivot 再减去其他贡献”针对
输入本身是 Partial 的情况，例如后续 AllReduce stage；FC2 的两个输入都不是 Partial，
所以此例实际只使用索引相等代入。不能把这段通用说明误读为提前假定 Pr 的和正确。

## 9. 报告第 386—401 行：真正的符号执行

对应 [verification/executor.py](../src/verification/executor.py)、[ops/matmul.py](../src/ops/matmul.py)
和 [smt/tensor.py](../src/smt/tensor.py)。

第 388 行执行单机 matmul，第 389—396 行分别处理 rank 0—7 的 matmul。
这里“执行”是构造数学表达式，不是在 GPU 上执行浮点乘法。

单机输出在任意坐标的表达式：

```text
Y[b,s,j] = Z[b,s,0]*W2[0,j] + ... + Z[b,s,1023]*W2[1023,j]
```

rank 1 本地输出先按算子定义展开，再代入输入关系：

```text
P1[b,s,j] = sum(k=0..255, Z1[b,s,k] * W2_1[k,j])
          = sum(k=0..255, Z[b,s,k+256] * W2[k+256,j])
          = sum(q=256..511, Z[b,s,q] * W2[q,j])
```

四个 rank 的求和范围分别是：

```text
P0：  0..255       P1：256..511
P2：512..767       P3：768..1023
```

它们没有重叠，也没有遗漏，四段相加就是完整的 0..1023 求和。
rank 4—7 对单机 batch `b+2` 做同样的事情。

`lazy element expressions` 的意思是：代码先保存“如何求某个输出元素”的函数，只有构造证明查询
时才展开那个符号坐标的表达式，不为全部输出位置各建立一份庞大的表达式。
matmul 的 Python 循环仍真实展开全部 K 项；延迟的是输出坐标枚举，不是求和项。

第 399—401 行是报告通用的算子语义提示。本例没有执行其中提到的 GELU 和通信操作。
报告的 `collective matching` 标题也是通用标题；本例没有通信事件需要配对。

## 10. 报告第 403—454 行：求解器在问什么

对应 [verification/obligation.py](../src/verification/obligation.py) 和
[smt/solver.py](../src/smt/solver.py)。

第 405 行是 TP 组 `[0,1,2,3]` 的查询。`goal0_i0/i1/i2` 就是任意本地输出坐标 b、s、j。
第 407—412 行 Guard 限制这些索引合法，代码再要求输出关系不成立：

```text
0 <= b < 2 AND 0 <= s < 128 AND 0 <= j < 256
AND Y[b,s,j] != P0[b,s,j]+P1[b,s,j]+P2[b,s,j]+P3[b,s,j]
```

求解器尝试为索引和仍然自由的输入函数找一组取值，使这些条件同时满足。
若找到，就得到反例；若任何取值都不行，就说明目标对所有允许输入成立。

### 为什么第 413 行是 false

`Simplified disequality: false` 是调用 `z3.simplify(left != right)` 的实际结果。
第 9 节的四段求和组合后，与单机的求和表达式完全一致，相当于尝试证明：

```text
a + b + c + d != (a + b) + (c + d)
```

精确实数代数把这个不等式化简成 false。这不是预先写死的结果，也不是把输出目标加入前提。
本例的主要证明工作已经在输入代入、算子表达式展开和 Z3 符号化简中完成，最后的求解查询因此很简单。

### 第 415—429 行 SMT-LIB 逐项解释

| 内容 | 意义 |
| --- | --- |
| `BEGIN SMT-LIB` / `END SMT-LIB` | 报告分隔符，内部是完整实际查询。 |
| `; benchmark generated...` | 注释，不参与逻辑。 |
| `(set-info :status unknown)` | 生成查询时的元信息，不是 solver 最后返回的 UNKNOWN。 |
| `(declare-fun goal0_i0 () Int)` 等 | 声明自由整数变量；零参数函数等价于一个常量符号。 |
| 第一个 `(assert ...)` | 要求 b、s、j 满足 Guard。 |
| `let`、`$x156` 等 | 自动生成的局部缩写，减少重复，不是张量名。 |
| `(assert false)` | 加入已经化简的“输出不相等”条件。false 不可能成立。 |
| `(check-sat)` | 检查所有 assert 是否有共同解。 |

由于不等式已化简，输入函数不再出现在最终查询中。这是表达式消去的结果，不是漏掉了输入值。
Guard 与 false 不可能同时满足，因此返回 UNSAT。

第 430—454 行是第二个 TP 组 `[4,5,6,7]`。本地索引范围相同，但关系编码在单机侧加了
batch 偏移 2。其不等式也化简为 false，所以第二组同样 UNSAT。

## 11. 第 456—459 行：最终能得出什么结论

两个输出域都 UNSAT，`verify_stage()` 返回 PROVED。

这句话完整的含义是：对任意满足 JSON 输入映射关系的实数 Z 和 W2，
8 个 rank 按 JSON 所写的局部 matmul 执行后，每个 TP 组的局部贡献之和等于单机对应 batch 的输出。

它不表示：输入关系已经由后端证明、真实捕获程序与此 JSON 自动一致、浮点舍入逐位相同，
或者前后所有 stage 的整个模型已经证明。输入假设的来源和跨 stage 衔接仍需前端负责。

其他结果也有不同含义：

| 结果 | 含义 |
| --- | --- |
| PROVED | 所有输出查询域均 UNSAT。 |
| DISPROVED | 存在符合前提、违反目标的输入与位置，报告会增加反例章节。 |
| UNKNOWN | 至少一项求解未判定，例如超时；不能当作成功。 |
| INVALID_INPUT | JSON、shape、映射或通信结构不合法，不能建立这道证明题。 |
| UNSUPPORTED | 超出当前语义或资源限制，不代表关系错误。 |

本例没有“第 8 节 Counterexample”，因为没有找到反例。不是缺少输出。

## 12. 实际打开代码时的阅读顺序

1. `verification/verifier.py`：掌握总体调用顺序。
2. `stage/codec.py` 和 `stage/validation.py`：看输入如何被解释、检查。
3. `relation/normalization.py` 和 `relation/encoding.py`：看 Phi、Partial、输入代入与输出等式。
4. `ops/matmul.py`：看完整 K 项如何展开。
5. `reduction/reducer.py` 和 `verification/obligation.py`：看任意受界坐标与反例条件。
6. `smt/solver.py`：看实际 Z3 调用和三种求解结果。
7. `app/reporting.py`：看上述数据如何整理成 demo.txt。

复现单例并生成临时报告，可在仓库根目录执行：

```bash
/data0/xs/miniconda3/envs/TrainVerify/bin/python -m src.app.cli verify \
  input/mlp_dp2_tp4_index_value/mlp_dp2_tp4_03_fc2_partial.json \
  --report /tmp/fc2-demo.txt
```

该命令不修改输入 JSON，不需要 GPU，也不读取训练权重。
