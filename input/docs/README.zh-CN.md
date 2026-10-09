# Stage JSON 输入规范

[English](README.md)

本文是 JSON 样例的目标格式合同，全部文件现已集中在 `input/`。每个文件描述一个局部证明任务：
单机子图、各 rank 子图、作为前提的输入关系、待证明的输出关系。重构后的后端已支持本文列明的
关系和算子，采用原始 shape 上的符号索引域验证。

精简后保留 10 个 JSON，包括全部 5 个受保护的 MLP 阶段。从原 34 个文件中移除了 6 个反例、
2 个 shape 非法用例和 16 个正例变体。
关系表示已同步到本规范；第 8 节记录集合核查结果。
新解析入口为 `src/stage/codec.py`，旧测试及脚本已替换。全部 10 个 JSON 已生成新验证器的 demo.txt。
文件名不决定证明结论；AutoVerify 前端模型仍需适配。输入关系消元及索引域约简的可靠性说明见
[当前架构](../../docs/architecture.md)，不要把它与物理 shape 缩小混淆。

## 1. 固定约定

- 使用 UTF-8 JSON；不允许注释、重复对象 key 或未知字段。
- 除明确说明外，下文所有字段都必须提供。
- 输入不包含 `schema/version/relation_format/semantics/source/coverage/scope/placements/mesh/`
  `index_transform/default`。语义统一由本文规定，不作为每个文件自行切换的配置。
- 张量数值采用精确实数语义，不表示浮点逐位等价。GELU 使用所有程序共享的、确定性的逐元素
  一元未解释函数；不是对具体浮点 GELU 内核的建模。
- 张量维度从 0 开始。shape 是非空的正整数列表；标量、动态维度、tuple 值、存储别名、
  原地修改暂不在本规范范围内。
- rank 为 `0 .. world_size-1`。对象键使用规范十进制字符串，如 `"0"`；map 条目中的 rank 使用整数。
- 每个 stage 文件自包含，不通过张量名称引用另一个文件。跨文件同名边界方便衔接，但不自动证明衔接正确。

## 2. 顶层字段

| key | 类型 | 含义 |
| --- | --- | --- |
| `name` | 非空字符串 | 用例标识，用于定位和报告，不从名称推断计算语义。 |
| `topology` | 对象 | world size 和明确的通信组、组族。 |
| `single` | 程序对象 | 单机参考子图。 |
| `distributed` | 对象 | 仅包含 `ranks`，即各 rank 的程序。 |
| `input_relations` | 绑定数组 | 输入证明前提，可以为空。 |
| `output_relations` | 非空绑定数组 | 输出候选结论，必须证明，不能当作值假设。 |

`distributed.ranks` 必须恰好包含全部 world ranks。单机和并行侧的名字、算子数量、内部 shape
可以不同。不能在未检查 rank 特有参数和 shape 的情况下，直接复制 rank 0 的程序。

## 3. 拓扑

| key | 含义 |
| --- | --- |
| `topology.world_size` | 参与的 rank 数，正整数。 |
| `topology.groups` | 以具体 group ID 为键的对象。 |
| `groups.<id>.ranks` | 非空、有序、无重复的 world rank 列表；顺序影响 gather/scatter。 |
| `topology.families` | 以 family ID 为键的对象。 |
| `families.<id>.groups` | 具体 group ID 的非空列表。 |

每个 family 内的组互斥并完整覆盖 world。多个逻辑 family 可以引用同一个具体 group ID，
例如本例 SP 和 TP 复用通信组。这是同一组划分的角色别名，不是新增 mesh 轴或独立归约维度，
不能将共享组重复计入归约。不同 family 不要求正交。`dp/tp/sp/mesh_axis_0` 都是标识，
名字本身不决定数学语义。旧用例中未命名的 mesh 轴转换为 `mesh_axis_N`，不凭空赋予 TP/DP 含义。
family 数量不决定张量维数。

8 卡 MLP 使用 TP 组 `[0,1,2,3]`、`[4,5,6,7]`，DP 组 `[0,4]`、`[1,5]`、`[2,6]`、`[3,7]`。
逻辑 `sp` 和 `tp` family 都引用已有的 `tp_0`、`tp_1` 组。序列映射归入 sp，
特征或权重映射、局部乘积的 Partial(sum) 归入 tp，batch 映射归入 dp。
world size 仍为 8，不能按 DP×TP×SP 相乘。SP 的 AllGather/ReduceScatter 中，
`attrs.group` 仍引用实际共享组；组名中的 tp 不意味着序列布局也必须标注为 TP。

## 4. 程序与算子

| key | 含义 |
| --- | --- |
| `tensors` | 当前程序作用域中的张量名到声明的映射。 |
| `tensors.<name>.shape` | 完整原始 shape，包含 batch 维；不是 shape reduction 后的大小。 |
| `inputs` | 自由边界输入，名称不重复。没有额外约束时，权重也是自由张量。 |
| `outputs` | 该 stage 导出的张量，名称不重复。 |
| `ops` | 按依赖顺序排列的算子。张量引用可以构成 DAG，不要求是一条链。 |
| `ops[].type` | 算子名称，按固定算子合同解释。 |
| `ops[].inputs` | 当前程序中的有序输入张量名。 |
| `ops[].outputs` | 本算子新产生的有序张量名。 |
| `ops[].attrs` | 算子参数；没有参数时填写 `{}`。 |

每个张量要么是输入，要么恰好被一个算子产生。算子引用必须已声明，使用前必须可用；stage
输出必须在结束时可用。单机恒等程序可使用 `ops: []`，同一个张量同时出现在 inputs 和 outputs。
普通局部算子不能直接引用另一 rank 的张量。

| 算子 | attrs | 含义 |
| --- | --- | --- |
| `matmul` | `{}` | rank >= 2 的矩阵乘法，支持 batch 广播。 |
| `add`、`mul` | `{}` | 逐元素运算，按末尾维度对齐进行广播。 |
| `gelu` | `{}` | 当前实数模型中的共享逐元素未解释函数。 |
| `transpose` | `dim0`、`dim1` | 交换两个维度；允许合法负轴，按输入 rank 规范化。 |
| `reshape` | `shape` | 正整数目标 shape，必须等于输出声明；保持 row-major 的线性元素顺序。 |
| `expand` | `shape` | 广播到声明的正整数目标 shape。 |
| `all_gather` | `group`、`axis` | 按 group 的 rank 顺序沿 axis 拼接，各成员都收到拼接结果。 |
| `all_reduce` | `group`、`reduce_op` | 组内逐元素归约，当前仅允许 sum。 |
| `reduce_scatter` | `group`、`axis`、`reduce_op` | 先逐元素求和，再沿 axis 连续等分，按组成员顺序分配。 |

通信算子的 `group` 引用具体组，不是 family。`axis` 非负且从 0 开始；MLP 激活为 `[b,s,h]`，
所以 `axis: 1` 是序列维。这些是同步、函数式的后端抽象，不是直接照搬 c10d 签名。
对每个具体组，按程序顺序将各 rank 的第 n 次通信配对，要求 type、attrs 相同且 shape 兼容。
当前样例每个组在每个 stage 中最多执行一次通信；非均匀通信、重叠组调度和异步句柄需要另行定义。
squeeze/unsqueeze、slice/concat 等尚未注册的算子返回 UNSUPPORTED；边界的多段索引映射已独立支持。

## 5. 张量绑定与 term

| key | 含义 |
| --- | --- |
| `single_tensor` | single 中的张量名。 |
| `distributed_tensors` | 每个 rank 字符串到本地张量名的对象，必须覆盖所有 rank。 |
| `relation.terms` | 非恒等索引组成部分及可选 Partial 的数组，允许为空。 |
| `terms[].family` | topology 中存在的 family ID。 |
| `terms[].expr` | 一个 `index_phi` 或 `partial` 表达式。 |

`relation` 对象只包含 `terms`；每个 term 只包含 `family` 和 `expr`。
这两层的类型由所在位置确定，不再包含 `kind`。只保留必填的 `expr.kind`，
用于区分 `index_phi` 与 `partial`。例如关系本体可以写为：

```json
{
  "terms": [
    {"family": "tp", "expr": {"kind": "partial", "op": "sum"}}
  ]
}
```

输入绑定只能引用真正的程序输入；输出绑定只能引用 stage 输出。所有导出张量都必须有输出目标。
未绑定关系的输入在跨程序之间保持不受约束，不能因名字相同就假定相等。同一列表内每个单机张量
最多绑定一次，多 family 使用 terms 表达。第一阶段支持范围内，每个 family 最多出现一次，
每个索引维度最多一个 term；term 顺序不影响数学含义。这是支持范围限制，不意味着实际布局永远满足该限制。

明确的 `"relation": {"terms": []}` 表示完整逐元素相等：每个本地张量都与单机张量 shape、值相同。
它不表示信息缺失。没有输入绑定，才表示未给出该跨程序关系；不接受 `relation: null`。
`output_relations: []` 非法，但输出绑定中的 `terms: []` 是有效且需要证明的相等目标。

每个张量不必为 topology 中的每个 family 都设置 term。MLP 的 W1/W2 只保留 TP IndexPhi：
其中已经明确列出全部 8 个 rank，包括两个 DP 副本中重复的权重分片。没有 Partial 时，
本地元素与映射位置的单机元素相等，已经能推出这些副本相等。不应仅为标注复制而添加空 DP term，
也不应为此任意指定一个张量维度。topology 中的 DP family 仍然保留。

该规则同样适用于 batched-matmul 输入和复制加分片的 transpose 等纯索引关系，
不再生成只表示恒等的 IndexPhi term；完全复制张量使用 `terms: []`。
AllGather 输出和后续 FC1 的输入 X 只保留 DP batch 映射：完整的全 rank 关系已经包含序列恒等
和组内复制，不再为了记录前序通信添加 SP 恒等项；通信过程记录在 ops 中。
Partial 只约束组内归约，不隐式添加不同副本的局部贡献相等条件；这类独立条件需要未来显式的
值关系扩展，不能通过恒等项隐式引入这种条件。

### IndexPhi

```json
{
  "kind": "index_phi",
  "dim": 1,
  "maps": [
    {
      "rank": 0,
      "local_extent": 32,
      "single_extent": 128,
      "pieces": [
        {"local": {"start": 0, "stop": 32}, "single": {"start": 0, "stop": 32}}
      ]
    }
  ]
}
```

以上仅展示一个 rank 条目；maps 必须列出所有 rank，不能把这个片段直接当作多 rank 的完整关系。

| key | 含义 |
| --- | --- |
| `expr.kind` | 固定为 `index_phi`。 |
| `expr.dim` | 必填非负张量维度，不是 mesh 轴。 |
| `expr.maps` | 非空数组，每个 rank 恰好一个映射条目。 |
| `maps[].rank` | 当前条目对应的 world rank。 |
| `maps[].local_extent` | 本地张量在 expr.dim 的长度，必须匹配 shape。 |
| `maps[].single_extent` | 单机张量在 expr.dim 的长度，必须匹配 shape。 |
| `maps[].pieces` | 完整覆盖本地维度的区间映射。 |
| `pieces[].local.start/stop` | 本地半开区间 `[start,stop)`。 |
| `pieces[].single.start/stop` | 对应的单机半开区间。 |

每段两端长度相等，映射为 `single.start + i - local.start`。本地段必须无重叠、无空缺地覆盖
整个 local_extent；单机段不能越界，同一 rank 内目标段不能重叠。不同 rank 可以映射到相同位置。
多段 pieces 支持非连续布局，各 rank 也可以有不同长度。
区间端点是非负整数，每段必须满足 `start < stop`。

`dim` 放在 expr 层，不放在各 rank 条目中。没有 default 字段。maps 必须列出全部 rank，
包括在该维恰好采用恒等映射的 rank。只有至少一个 rank 在该维存在非恒等映射或长度变化时，
才生成这个 term；若所有 rank 在整个维度上均为恒等，则省略整个 term。
空 `maps` 不再是合法的恒等简写。

### Fused QKV 与多段映射

假设融合输出特征轴依次为 Q `[0,8)`、K `[8,16)`、V `[16,24)`，TP=2，
每个 rank 本地融合宽度为 12，则可以描述为：

| rank | 本地区间 | 单机区间 |
| --- | --- | --- |
| 0 | `[0,4)`、`[4,8)`、`[8,12)` | `[0,4)`、`[8,12)`、`[16,20)` |
| 1 | `[0,4)`、`[4,8)`、`[8,12)` | `[4,8)`、`[12,16)`、`[20,24)` |

这是一个 IndexPhi term，每个 rank 三个 pieces，不是三个 term。它描述静态边界对应，
不等于实现了 concat/slice 算子，也不能直接把三个独立单机张量绑定到一个融合本地张量。
pieces 必须根据实际 QKV 排列填写，不能通过名字推断布局。跨维置换不由这种映射直接表达，
需要支持的算子或未来的表达扩展。

### Partial

```json
{"kind": "partial", "op": "sum"}
```

当前 op 仅支持 sum。Partial 不带 dim：它在完整索引映射组装后组合元素值，归约范围由外层
term 的 family 决定。一个绑定最多包含一个 Partial term；多 family 连续归约需要未来显式的
组合合同，当前不能猜测。

## 6. 完整关系如何组合

1. 收集所有 IndexPhi；第一阶段要求各 term 作用于不同维度，显式维度冲突时拒绝。
2. 没有 term 的维度采用全局恒等，检查单机、本地长度相等；所有绑定张量的维数一致。
3. 得到每个 rank 完整的本地到单机映射 Phi_r。不能为每个 IndexPhi 单独生成一条全局值相等断言。
4. 没有 Partial 时，对每个 rank 和本地元素要求 `D_r[l] = S[Phi_r(l)]`。
5. 有 Partial 时，对其 family 的每个组 G，要求在共同的目标域上成立
   `S[g] = sum(D_r[Phi_r^-1(g)] for r in G)`。组内成员必须映射到同一个目标域，在该域上有唯一逆映射。
   不同组可以对应不同的 DP batch 区域。缺失贡献应拒绝，不能默认为零。
6. 有 Partial 时，不能因映射相同或 family 名字而隐式增加值相等约束。无 Partial 时，相同映射
   已通过第 4 条推出复制；有 Partial 时只约束明确指定的组内求和。

必须覆盖完整本地域，但不强制覆盖完整单机目标域，slice 关系只约束其映射到的位置。
因此不再设置 coverage。未知 family、缺 rank、extent 错误、映射冲突、缺失输出目标属于输入错误，
不是 SMT 反例。非正交拓扑也必须满足映射组合和 Partial 目标域的限制。

后端检查“所有输入关系成立，且至少一条输出关系不成立”是否可满足，算子语义通过表达式替换或等式
约束加入。输出关系不能当作值前提，空输出目标非法。整程序正确性仍需要解释初始假设从何而来，
结构模拟成功不构成其依据。

## 7. MLP 样例及编写步骤

建议先阅读这些完整文件：

| 示例 | 文件 |
| --- | --- |
| 分片矩阵乘法到 Partial | [MLP FC2](../mlp_dp2_tp4_index_value/mlp_dp2_tp4_03_fc2_partial.json) |
| 两个维度的显式映射 | [MLP FC1/GELU](../mlp_dp2_tp4_index_value/mlp_dp2_tp4_02_fc1_gelu.json) |
| 索引映射加 Partial | [Shard/Partial transpose](../mesh/mesh_shard_partial_transpose_proved/mesh_shard_partial_transpose_proved.json) |
| 8 卡序列 AllGather | [MLP 阶段 01](../mlp_dp2_tp4_index_value/mlp_dp2_tp4_sp_01_allgather.json) |
| 完整 MLP 样例集 | [阶段索引](../mlp_dp2_tp4_index_value/README.md) |

MLP 主链为 AllGather -> FC1/GELU -> FC2 Partial -> ReduceScatter；AllReduce 是替代结尾，
不是第五个连续阶段。相邻边界的名字、shape、关系保持一致。全局激活为 `[4,128,256]`，FFN 宽度
1024，每个 DP 副本处理两个 batch 元素。DP 项提供 batch 映射，SP 项提供 sequence 映射，
TP 项提供 feature/weight 映射或 Partial。组共享不意味着每个张量都需要同时列出 SP 和 TP term。
权重通过覆盖所有 rank 的 TP 映射体现 DP 复制，因此 W1/W2 不设 DP term。
所有独立 matmul 样例（包括小型基线）已删除，由受保护的 MLP 阶段代表其核心计算。
MLP 之外只保留 Add、Mul、Transpose、Reshape、Expand 各一个代表，优先覆盖核心证明任务，
不单独保留参数组合。细分广播、负轴、squeeze/unsqueeze 等处理路径在后端实现时需要另加单元测试。

FC1/GELU 仍是人为组合的多算子样例，不表示 AutoVerify 一定将这两个算子切入同一个 stage。

MLP 的 AllReduce 输出只保留 DP batch 映射 term。同一个 TP 组的全部 rank 映射到相同的
单机 batch 区域，因此映射后的值相等已经表达 TP 复制。AllReduce 不指定张量轴，不需要
额外的 TP `dim: 1` 恒等项。其输入仍保留 TP Partial(sum)，实际通信组仍记录在算子 attrs 中。

新增样例时：

1. 先声明有序通信组和完整 family；不要让后端通过 mesh 坐标猜测 rank 映射。
2. 声明两侧程序、边界和内部 shape，以及完整算子参数。切分边界应提升为自由输入，不带入其原图生产操作。
3. 绑定输入关系，仅对非恒等维度添加带 expr.dim 的 IndexPhi，并写出全部 rank 的区间。
   完整相等使用 `terms: []`。需要归约时添加至多一个 Partial，不在恒等项中隐藏额外值约束。
4. 为每个输出声明待证关系。当前 input 集合只保留正向目标，未来负向回归应单独维护在测试 fixture 中；
   序列化时不能擅自修复候选关系。
5. 检查 shape、数据流、映射覆盖、组成员和顺序、阶段边界，再交给验证器。
   JSON 合法不等于证明成功。大 shape 可以保留，但不能只缩小数字而不调整区间并论证缩减的可靠性。

## 8. 支持边界与迁移核查

| 当前目标支持 | 尚未定义或不支持 |
| --- | --- |
| 静态正整数 shape、精确实数值 | 动态路由或 shape、浮点内核等价、别名、tuple 和标量值 |
| 每个变化维度一个 IndexPhi，各自使用不同 family | 同 family 多个索引 term；多 family 分片同一维 |
| 非均匀静态分片、多段单位步长区间 | 跨维映射、同 rank 重复目标索引、动态 gather |
| 无 Partial 或一个 Partial(sum) | 多 Partial family、隐式嵌套归约、额外副本值约束 |
| 显式组，包括非正交 family | 不经检查就假定任意非正交布局都可组合 |

同维多 family 分片会出现在 FSDP+TP、CP+SP，需定义有序嵌套映射或最终规范化映射，
不能简单相加偏移。当前每个 family 至多一个 term 是支持范围限制，不是对所有实际布局的断言。

2026-10-09 集合精简后核查：10 个文件含 24 条绑定、33 个 term；目前没有样例使用 `terms: []`，
但空 terms 仍是规范支持的合法关系。已无空 maps 的 IndexPhi。
当前 10 个输入已通过新后端验证，证明范围为 ORIGINAL_SHAPES_EXACT_REAL。
曾删除的 6 个反例与 2 个非法 shape 是历史清理记录；新的负向回归位于 tests 中。

- 精简时未改变存留 JSON 内容，5 个受保护的 MLP JSON 逐字节核对不变；文档另行同步更新。
- 五个 MLP 文件只使用非空的单维映射和 Partial，阶段边界一致，各阶段目标已在实数/共享函数模型下
  通过 SMT；阶段输入前提仍需前端提供可靠依据。

新后端不再接收旧 placement。AutoVerify 原生 RelationCodec v3 仍不同：dim 在各 rank map 内，
有 identity default，逐个 IndexPhi 独立校验，尚未实现这里的 term 组合。新合同还明确了空 terms
的相等含义，并允许逻辑 SP/TP 共享组。转换必须保留真实 family、映射和证明前提，
不能因格式相似就宣称可以无损导出，前端导出适配仍待实现。本次不缩小物理 shape：保留全部原始
求和项和索引范围，用受界符号输出索引覆盖每个合法元素，因此 UNSAT 不依赖小 shape 推广。
demo.txt 保存这些边界和实际 SMT 查询。默认上限为 K=8192、10000 个算子调用和 64 层惰性依赖；
超限、不支持或求解器 UNKNOWN 都不会当作 PROVED。
