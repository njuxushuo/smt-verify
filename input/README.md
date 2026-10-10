# Stage input fixtures / 输入样例

Format: [中文规范](docs/README.zh-CN.md) / [English contract](docs/README.md).
Architecture and proof method: [implementation guide](../docs/architecture.md).

All constructed input JSON files belong under this directory. There are **11
positive-target cases**: ten retained cases from the previous cleanup and one
new ordered DP/CP/SP composition case. These are intended proof targets.
The backend now proves all eleven over original
shapes in the declared exact-real/shared-GELU abstraction.

| Category / 类别 | Count / 数量 | Purpose / 用途 |
| --- | ---: | --- |
| [MLP DP2/TP4/SP](mlp_dp2_tp4_index_value/README.md) | 5 | Protected stage chain and alternative collective ending; all five JSON files preserved unchanged. |
| Add | 1 | Sharded activation plus sharded one-dimensional bias. |
| Mul | 1 | Elementwise multiplication with broadcasting and sharded operands. |
| Transpose | 1 | Axis remapping with index/Partial composition. |
| Reshape | 1 | Row-major layout transformation preserving the shard relation. |
| Expand | 1 | Broadcasting a singleton dimension while preserving the shard relation. |
| [DP2/CP2/SP2 Zigzag](dp2_cp2_sp2_zigzag_allgather/dp2_cp2_sp2_zigzag_allgather.json) | 1 | Same-dimension SP then CP mapping chain; AllGather removes the SP level. |

The protected MLP JSON files were compared byte-for-byte after cleanup. Their
README only has corrected links. Other retained JSON contents also remain
unchanged. Redundancy decisions are about development coverage, not formal
logical equivalence between different proof tasks. Standalone matmul cases are
represented by the MLP stages. Rank/shape variants, negative-axis normalization,
squeeze/unsqueeze and specialized broadcast patterns require separate operator
tests; this is a small representative collection, not exhaustive coverage.
Negative-axis and broadcasting regressions are included in the new tests;
squeeze/unsqueeze operators are not implemented yet.

The backend and tests now use the new contract. Generate fresh reports with
`python -m src.app.cli batch input --write-reports`. Each standalone case has a
sibling `demo.txt`; the five MLP files have `<json-stem>/demo.txt` subdirectories
to avoid overwriting one another. Every report records input hashes, parsed
models, original shape/index bounds, relation equations, substitutions, execution
and actual SMT queries. New negative regressions are in tests, not this directory.

原 34 个 JSON 精简到 10 个后，新增 DP2/CP2/SP2 Zigzag AllGather，共 11 个。
输入 term 中 SP→CP 按顺序组合 sequence，DP 单独映射 batch；输出保留 DP、CP。
按核心计算和关系变换归类，不再单独保留不同 shape、rank 数和参数形式的相似样例。
独立 matmul 由 MLP 代表；负轴、squeeze/unsqueeze 等细分覆盖以后放入算子单元测试。
MLP 目录内 5 个 JSON 全部保留且内容不变。后端已支持有序映射，11 个输入已验证并生成报告；
前端导出仍需适配。结论限于给定输入前提和实数/共享 GELU 抽象，不是浮点内核证明。
