# smt-verify

一个处于早期研究与原型阶段的 SMT 验证项目，目标是逐步验证单机程序与其分布式实现之间的语义等价性。

当前阶段只关注局部 stage 的 relation-aware SMT 认证：给定单机 stage、分布式 stage、已知输入 relation 和候选输出 relation，判断候选关系能否被严格证明。项目暂不实现整图对齐、自动 stage 切分、真实计算图 IR 绑定或完整 lemma 系统。

## 目录

- `docs/`：项目背景、设计讨论、方案文档，以及论文和相关仓库的阅读笔记。
- `src/`：后续 SMT verifier 的正式实现；当前不预设内部模块结构。
- `input/`：供 verifier 直接读取的结构化 stage 输入；具体中间描述格式待后续确定。
- `scripts/`：运行、输入生成或转换、调试等辅助脚本。
- `examples/`：少量可读的验证案例及其预期行为。
- `tests/`：后续单元测试和功能测试。

项目背景和当前约束见 [`docs/distributed_equivalence_smt_project_overview.md`](docs/distributed_equivalence_smt_project_overview.md)。
