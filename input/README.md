# Stage input fixtures / 输入样例

Format: [中文规范](docs/README.zh-CN.md) / [English contract](docs/README.md).
Inventory and removal evidence: [精简记录](CLEANUP.md).

All input JSON files are now under this directory. There are **10 retained
positive-target cases**, reduced from 34 by removing six concrete-counterexample
cases, two invalid-shape cases and sixteen positive-case variants. This means
intended proof targets, not a claim that the new backend has proved them.

| Category / 类别 | Count / 数量 | Purpose / 用途 |
| --- | ---: | --- |
| [MLP DP2/TP4/SP](mlp_dp2_tp4_index_value/README.md) | 5 | Protected stage chain and alternative collective ending; all five JSON files preserved unchanged. |
| Add | 1 | Sharded activation plus sharded one-dimensional bias. |
| Mul | 1 | Elementwise multiplication with broadcasting and sharded operands. |
| Transpose | 1 | Axis remapping with index/Partial composition. |
| Reshape | 1 | Row-major layout transformation preserving the shard relation. |
| Expand | 1 | Broadcasting a singleton dimension while preserving the shard relation. |

The protected MLP JSON files were compared byte-for-byte after cleanup. Their
README only has corrected links. Other retained JSON contents also remain
unchanged. Redundancy decisions are about development coverage, not formal
logical equivalence between different proof tasks. Standalone matmul cases are
represented by the MLP stages. Rank/shape variants, negative-axis normalization,
squeeze/unsqueeze and specialized broadcast patterns are left to future operator
unit tests; this is a small representative collection, not exhaustive coverage.

The existing loader/verifier and tests still expect the legacy format; some
legacy tests also reference removed fixtures. Adapt them during backend work.
All historical `demo.txt` files have been removed. There are no new-format proof
reports; do not regenerate them with the old exporter. Future negative regression
cases should live in a separate test fixture collection.

当前从 34 个 JSON 精简到 10 个：删除了 6 个已找到具体反例的用例、2 个 shape 非法用例，
以及 16 个正例变体。按核心计算和关系变换归类，不再单独保留不同 shape、rank 数和参数形式的相似样例。
独立 matmul 由 MLP 代表；负轴、squeeze/unsqueeze 等细分覆盖以后放入算子单元测试。
每项依据和保留清单见精简记录。MLP 目录内 5 个 JSON 全部保留且内容不变。
前后端尚未适配新协议，未执行新 SMT 验证，不把历史 PROVED 报告当作当前证明结果。
