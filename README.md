# smt-verify

一个处于早期研究与原型阶段的 SMT 验证项目，目标是逐步验证单机程序与其分布式实现之间的语义等价性。

当前阶段只关注局部 stage 的 relation-aware SMT 认证：给定单机 stage、分布式 stage、已知输入 relation 和候选输出 relation，判断候选关系能否被严格证明。项目暂不实现整图对齐、自动 stage 切分、真实计算图 IR 绑定或完整 lemma 系统。

## 当前输入协议

全部输入已集中到 `input/`，按核心证明内容精简为 10 个 JSON：完整保留 5 个 MLP 阶段，另保留 Add、Mul、Transpose、Reshape、Expand 各一个代表，采用按 family 组织的 IndexPhi/Partial 草案；
恒等关系使用 `terms: []`，详情见规范中的核查清单。
字段和组合语义见 [中文规范](input/docs/README.zh-CN.md) / [English contract](input/docs/README.md)。
保留清单与删除依据见 [输入精简记录](input/CLEANUP.md)。
现有 loader、验证器及依赖旧 fixture 的测试尚未适配；旧 `demo.txt` 已全部清理，当前没有新协议的求解报告。

## 目录

- `docs/`：项目背景、设计讨论、方案文档，以及论文和相关仓库的阅读笔记。
- `src/`：后续 SMT verifier 的正式实现；当前不预设内部模块结构。
- `input/`：Stage 输入、MLP 样例及中英文规范。
- `scripts/`：运行、输入生成或转换、调试等辅助脚本。
- `tests/`：后续单元测试和功能测试。

项目背景和当前约束见 [`docs/distributed_equivalence_smt_project_overview.md`](docs/distributed_equivalence_smt_project_overview.md)。

以下是旧协议的报告导出命令，需待 loader 和 verifier 适配后才能用于当前 JSON；目前不要重新生成报告：

```bash
conda run -n smt python scripts/export_input_demos.py
```
