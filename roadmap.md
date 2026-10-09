# smt-verify roadmap

## sec1 项目目的

实现可独立调用的分布式张量 stage 关系证明后端。输入是一份自包含 JSON：单机子图、
所有 rank 的子图、通信拓扑、输入 Relation 和候选输出 Relation。验证任务是
`Rin AND Sem(single, distributed) => Rout`，不是比较两次随机运行的结果。

前端负责图捕获、对齐、切分及候选生成；后端负责严格解析和校验、算子符号执行、
IndexPhi/Partial 编码、SMT 判定、反例和可审计报告。没有 term 表示逐元素相等；
非恒等维度使用显式分段索引映射，Partial(sum) 表示指定 family 的组内贡献求和。
DP/SP/TP 是拓扑和布局角色，不通过名字猜测数学语义。

第一阶段支持现有 10 个 JSON 所需的 matmul、add、mul、gelu、transpose、reshape、expand、
all_gather、all_reduce、reduce_scatter。采用精确实数模型，GELU 是两侧共享的一元未解释函数。
输出区分 PROVED、DISPROVED、UNKNOWN、INVALID_INPUT、UNSUPPORTED；不把不支持当成反例。
不宣称 GPU 浮点逐位等价或整程序正确性。lemma 泛化及缓存复用后续实现。

规模控制必须保持原证明义务。禁止仅凭较小 shape 的 UNSAT 推广原 shape。
优先以任意受界符号输出索引替代元素枚举，保留原始 shape 与完整 contraction sum，
作为可说明等价性的索引域约简。若未来真正缩小 shape，必须另附适用规则和可靠性依据。
每个输入旁生成 demo.txt，记录解析对象、校验、索引与值等式、约简条件、执行轨迹、
实际 SMT 查询及结论，便于人工核查。

## sec2 项目结构方案

| 模块 | 文件和职责 |
| --- | --- |
| core | tensor.py：TensorSpec/Ref；program.py：OpCall/Program；topology.py：Group/Family；errors.py：共享错误分类 |
| smt | tensor.py：惰性符号张量；context.py：符号和共享函数；solver.py：Z3 包装 |
| relation | model.py、codec.py、validation.py、normalization.py、encoding.py：解析、校验、组合和约束生成 |
| ops | api.py、registry.py、broadcast.py、matmul.py、elementwise.py、view.py、collectives.py：算子合同与语义 |
| stage | model.py、codec.py、validation.py、dataflow.py：任务输入、引用和执行前检查 |
| verification | executor.py、collectives.py、obligation.py、verifier.py、result.py、counterexample.py：执行协调、证明义务及结果 |
| reduction | model.py、rules.py、reducer.py：精确的索引域约简及适用条件 |
| app | bootstrap.py、cli.py、reporting.py：组件组装、CLI 与人类可读报告 |

core 不依赖 Z3 或上层模块。smt 不解释具体关系或算子。relation 与 ops 互不依赖，
也不导入 stage/verification/app。stage 聚合 core 和 relation 并委托 ops 校验。
verification/reduction 使用下层接口，app 位于最上层。__init__.py 不注册全局可变组件。
用静态依赖测试固定方向；算子测试、关系测试、通信测试、反例测试、10 输入集成测试独立组织。

## 实施与验收

1. 提交本路线图及当前输入合同作为检查点。
2. 重建基础模型、严格 codec、关系组合与合法性校验。
3. 实现局部算子和跨 rank collective 协调；建立原始范围的符号证明流程。
4. 用小型反例和非法输入验证拒绝路径，用全部现有输入验证正确路径。
5. 为全部输入生成 demo.txt，独立审核并处理发现，回归测试。

新实现不得改写输入目标以获得成功，不得根据文件名选择证明结论。

## 实现状态（2026-10-09）

- 已创建前置检查点 e994b19；旧实现和旧测试可在该提交中查看。
- 八个模块已实现；10 个输入的原始 shape 目标均 PROVED，输入 JSON 未改动。
- 采用精确索引域约简，不缩小 shape，不做未经证明的 reduced-shape lifting。
- 重新编写正反例、算子、通信、codec、依赖规则及报告重放测试。
- 每个输入都有独立 demo.txt；MLP 共享目录下使用同名子目录存放报告。
- 独立子智能体审核发现 API 负维度校验、深依赖限制和报告路径硬链接覆盖问题，均已修复并增加回归。
  当前工具不提供名为 gpt 6 astra ultra 的模型覆盖项，审核使用继承当前模型的子智能体，未冒称指定模型。
- 最终 41 项测试通过，10 个输入全部 PROVED，10 份报告中的 44 条 SMT 查询重放均 UNSAT。
  审核和验证记录见 [review-20261009.md](docs/review-20261009.md)。
