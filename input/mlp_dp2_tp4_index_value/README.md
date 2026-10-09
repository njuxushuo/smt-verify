# MLP DP2/TP4/SP fixtures

Full field definitions and relation composition:
[English](../docs/README.md) /
[中文](../docs/README.zh-CN.md).

These five files are manually designed stages, not captured AutoVerify exports.
The target contract is shared with `input/`; all fixtures now omit identity-only
terms, as listed in the contract's audit. The existing backend
still needs adaptation. No SMT proof is claimed.

| File | Distributed computation | Output relation |
| --- | --- | --- |
| [01 AllGather](mlp_dp2_tp4_sp_01_allgather.json) | Gather SP sequence shards on shared TP groups | DP batch map; sequence identity and replicas are implied |
| [02 FC1/GELU](mlp_dp2_tp4_02_fc1_gelu.json) | Column-parallel matmul, GELU | DP batch map + TP feature map |
| [03 FC2](mlp_dp2_tp4_03_fc2_partial.json) | Row-parallel local matmul | DP batch map + TP Partial(sum) |
| [04 ReduceScatter](mlp_dp2_tp4_sp_04_reduce_scatter.json) | Sum TP partials, scatter SP sequence | DP batch map + SP sequence map |
| [04 AllReduce](mlp_dp2_tp4_04_allreduce.json) | Sum within TP | DP batch map; equal images imply TP replicas |

The SP chain is 01 -> 02 -> 03 -> ReduceScatter. AllReduce is an alternative
ending after 03. Without SP, enter at 02 with the full per-DP sequence.
Adjacent boundaries have identical names, shapes and relations. W1/W2 are
additional stage input assumptions.
The AllGather output and FC1 input X have only a DP term. The input of AllGather
and output of ReduceScatter retain SP terms for their actual sequence shards.

World size is 8, DP=2, TP=4. SP reuses TP groups. Activations use [b,s,h]:
Logical `sp` and `tp` families reference the same concrete groups. Sequence
relations use `sp`, feature/weight and Partial relations use `tp`; operation
attributes retain the actual group IDs. This does not create additional ranks.
global X/Y [4,128,256], global Z [4,128,1024], W1 [256,1024], W2 [1024,256].
Local sequence shards are [2,32,256]; local feature shards are [2,128,256].
DP splits global batch into [0,2) and [2,4); weights are replicated.
W1 and W2 have only a TP IndexPhi term: its eight explicit rank maps also
establish equal slices across DP replicas. No empty DP weight term is needed.
Activation relations keep DP terms where they actually map the batch dimension.

本目录的 01、02、03、ReduceScatter 构成主链，AllReduce 是替代结尾。
激活按角色区分 DP batch、SP sequence、TP feature/Partial；SP 和 TP 引用相同通信组。
W1/W2 只保留覆盖全部 rank 的 TP term，不添加冗余 DP 复制项。
IndexPhi 的 dim 位于 expr 层，只记录非恒等维度；完整相等使用 terms: []，不使用空 maps 占位。
精确实数和共享逐元素 GELU 抽象、通信顺序及完整字段含义以链接的规范为准。
