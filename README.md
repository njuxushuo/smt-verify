# smt-verify

一个可独立调用的分布式张量 stage 关系证明后端。读取单机程序、各 rank 程序、显式拓扑、
输入 IndexPhi/Partial 关系和候选输出关系，通过 Z3 判断是否存在违反目标的输入。

## 已实现范围

- 严格 JSON 解析、shape/SSA/边界/通信配对及分段映射校验。
- matmul、add、mul、gelu、transpose、reshape、expand、all_gather、all_reduce、reduce_scatter。
- 精确实数语义；GELU 使用两侧共享的一元未解释函数，不表示 GPU 浮点内核等价。
- 输入关系通过完整索引映射和 Partial 消元精确参数化；不根据文件名选择结论。
- 保留原始 shape 和完整 matmul 求和范围，用任意受界符号索引替代输出元素枚举。
- PROVED / DISPROVED / UNKNOWN / INVALID_INPUT / UNSUPPORTED，以及原始索引反例和可审计报告。

现有 10 个 JSON 已全部在原始 shape 上得到 PROVED；这仍是给定输入假设和实数/共享函数模型下的条件性结论。
所有输入文件保持不变。没有执行物理 shape 缩小，也不依赖小 shape 到大 shape 的未经证明推广。
集成测试之外还包含错误目标、缺失前提、QKV 多段映射、通信顺序、API 校验和依赖方向回归。

## 安装和运行

要求 Python 3.10+ 和 Z3：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m src.app.cli verify input/mlp_dp2_tp4_index_value/mlp_dp2_tp4_03_fc2_partial.json
python -m src.app.cli batch input --write-reports
```

也可使用 `make check`、`make demos`，或安装后的 `smt-verify verify ...`。
现有环境可运行：
`make check PYTHON=/data0/xs/miniconda3/envs/TrainVerify/bin/python`。

旧命令 `python scripts/verify_stage.py FILE.json` 和
`python scripts/export_input_demos.py --input-root input` 已作为新入口的薄包装。
`--timeout-ms` 是每个 Z3 查询的超时，不限制全部 Python 表达式构造时间。
构造阶段限制 contraction 长度、操作数和依赖深度，超限返回 UNSUPPORTED。

## 报告

每个 JSON 对应一份 `demo.txt`：独立用例目录直接保存于 JSON 旁；MLP 的五个 JSON 共用目录，
因此报告保存于 `input/mlp_dp2_tp4_index_value/<json-stem>/demo.txt`，不移动或覆盖 JSON。

报告包含解析模型、原始 shape、所有索引映射、校验项、输入逻辑等式、精确替换过程、
符号执行轨迹、约简索引边界、完整可重放 SMT-LIB 查询和最终结论。报告不等于独立检查的形式化证明证书。

## 设计文档

- [roadmap：项目目的和模块方案](roadmap.md)
- [当前架构与证明方法](docs/architecture.md)
- [输入规范：中文](input/docs/README.zh-CN.md) / [English](input/docs/README.md)
- [MLP 样例与报告](input/mlp_dp2_tp4_index_value/README.md)

`src/` 包含 core、smt、relation、ops、stage、reduction、verification、app 八个模块。
关系与算子互不依赖，实例由 app/bootstrap 显式组装。旧 placement、强制小 shape 缩减、
旧 lemma 入口已从生产代码移除；仓库早期的阶段文档和需求文档保留为历史背景，不能作为新接口说明。
AutoVerify 前端原生 Relation v3 仍需显式适配，不能假定与本协议直接兼容。
