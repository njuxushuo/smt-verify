# Stage 1 输入格式

阶段一规定 SMT verifier 原型的 Stage 输入契约。它只描述和校验局部 stage，不执行 operator，也不进行 SMT、shape reduction 或真实计算图解析。

## 顶层结构

每个文件是 UTF-8 JSON，必须包含 `name`、`world_size`、`single`、`distributed`、`input_relations` 和 `output_relation`。`mesh` 可省略以兼容旧的 1-D 输入：

```json
{
  "name": "case_name",
  "world_size": 2,
  "mesh": {"shape": [2]},
  "single": {"tensors": {}, "ops": []},
  "distributed": {"ranks": {}},
  "input_relations": [],
  "output_relation": {}
}
```

`mesh.shape` 必须是非空正整数列表，且其乘积必须等于 `world_size`。省略 `mesh` 时 loader 将其规范化为 `[world_size]`。`distributed.ranks` 必须恰好列出从 `"0"` 到 `"world_size - 1"` 的所有 flat rank；flat rank 与 mesh coordinate 使用 row-major 映射。

## Tensor 与 operator

Tensor 由当前 scope 的名字和正整数 shape 描述，例如 `"A": {"shape": [4, 8]}`。shape 至少一维；不支持 `None`、`-1`、布尔值或动态维度。

每个 operator 是按顺序解释的对象：

```json
{"type": "matmul", "inputs": ["A", "B"], "outputs": ["C"]}
```

`inputs` 和 `outputs` 必须引用本 scope 已声明的 tensor。可选的 `attrs` 必须是 JSON object；省略时等价于 `{}`，因此原有 Add/Mul/MatMul JSON 保持兼容。Transpose、Reshape、Squeeze、Unsqueeze 与 Expand 的参数均由 `attrs` 携带，并由对应 operator semantics 校验必需字段、类型和未知字段。当前识别 `add`、`mul`、`matmul`、`transpose`、`reshape`、`squeeze`、`unsqueeze`、`expand`、`sum`、`all_reduce`、`all_gather` 和 `reduce_scatter`。

## Relation

Relation 关联一个单机 tensor 与按 flat rank 顺序排列的 local tensor，并为每个 mesh axis 声明一个 placement：

```json
{
  "single_tensor": "A",
  "distributed_tensors": ["A0", "A1"],
  "placements": [
    {"type": "replicate"},
    {"type": "shard", "dim": 1}
  ]
}
```

每个 placement 支持三种类型：

- `replicate`：不含 `dim` 或 `reduce_op`；它复制由其他 mesh axis 的 Shard 确定的 global shard，并不一定表示完整 tensor。
- `shard`：必须有合法的 `dim`，该 tensor 维按对应 mesh axis extent 做规则连续等分。
- `partial`：不含 `dim`，必须是 `"reduce_op": "sum"`；只沿该 Partial mesh axis（或多个 Partial axes 的笛卡尔积）分组求和。

`placements` 长度必须等于 mesh rank。同一 tensor dimension 当前最多由一个 mesh axis Shard。Replicate 与 Partial 不改变 local shape；每个 tensor dimension 的 local extent 等于 single extent 除以该维的 composite shard factor。

旧 JSON 没有 `mesh` 时仍可使用顶层 `type`、`dim`、`reduce_op` relation shorthand，loader 会将其转换为单 placement。显式多维 mesh 必须使用 `placements`，且新旧 relation 字段不能混用。

`input_relations` 是可为空的 relation 列表，同一个单机 tensor 最多出现一次。`output_relation` 是一个单独的候选 relation。

## 完整示例

```json
{
  "name": "matmul_shard_to_partial",
  "world_size": 2,
  "single": {
    "tensors": {"A": {"shape": [4, 8]}, "B": {"shape": [8, 4]}, "C": {"shape": [4, 4]}},
    "ops": [{"type": "matmul", "inputs": ["A", "B"], "outputs": ["C"]}]
  },
  "distributed": {
    "ranks": {
      "0": {"tensors": {"A0": {"shape": [4, 4]}, "B0": {"shape": [4, 4]}, "C0": {"shape": [4, 4]}}, "ops": [{"type": "matmul", "inputs": ["A0", "B0"], "outputs": ["C0"]}]},
      "1": {"tensors": {"A1": {"shape": [4, 4]}, "B1": {"shape": [4, 4]}, "C1": {"shape": [4, 4]}}, "ops": [{"type": "matmul", "inputs": ["A1", "B1"], "outputs": ["C1"]}]}
    }
  },
  "input_relations": [
    {"single_tensor": "A", "distributed_tensors": ["A0", "A1"], "type": "shard", "dim": 1},
    {"single_tensor": "B", "distributed_tensors": ["B0", "B1"], "type": "shard", "dim": 0}
  ],
  "output_relation": {"single_tensor": "C", "distributed_tensors": ["C0", "C1"], "type": "partial", "reduce_op": "sum"}
}
```

上面的完整示例刻意保留 legacy 1-D relation 写法，用于验证 backward compatibility。多维格式的可运行样例见 [`../input/mesh/mesh_shard_shard_transpose_proved/mesh_shard_shard_transpose_proved.json`](../input/mesh/mesh_shard_shard_transpose_proved/mesh_shard_shard_transpose_proved.json)。
每个输入 case 位于以 JSON stem 命名的独立目录中；同目录 `demo.txt` 记录其真实 shape reduction、symbolic execution、relation encoding 和验证结果。运行 `conda run -n smt python scripts/export_input_demos.py` 可批量重建全部报告。静态非法 case 的报告会记录 `REJECTED` 和具体 validation error，不会伪造未执行的后续阶段。

## 当前限制

输入采用顺序 operator 列表而不是 DAG；每个 rank 必须显式列出；不支持 symbolic shape、跨 rank tensor 引用或真实 IR 前端。Stage 1 只负责结构、数据流和 concrete operator/relation 合法性，Z3 reduction 和 relation value encoding 仍属于后续阶段。
