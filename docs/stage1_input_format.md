# Stage 1 输入格式

阶段一规定 SMT verifier 原型的 Stage 输入契约。它只描述和校验局部 stage，不执行 operator，也不进行 SMT、shape reduction 或真实计算图解析。

## 顶层结构

每个文件是 UTF-8 JSON，必须包含 `name`、`world_size`、`single`、`distributed`、`input_relations` 和 `output_relation`：

```json
{
  "name": "case_name",
  "world_size": 2,
  "single": {"tensors": {}, "ops": []},
  "distributed": {"ranks": {}},
  "input_relations": [],
  "output_relation": {}
}
```

`world_size` 至少为 1，且 `distributed.ranks` 必须恰好列出从 `"0"` 到 `"world_size - 1"` 的所有 rank。

## Tensor 与 operator

Tensor 由当前 scope 的名字和正整数 shape 描述，例如 `"A": {"shape": [4, 8]}`。shape 至少一维；不支持 `None`、`-1`、布尔值或动态维度。

每个 operator 是按顺序解释的对象：

```json
{"type": "matmul", "inputs": ["A", "B"], "outputs": ["C"]}
```

`inputs` 和 `outputs` 必须引用本 scope 已声明的 tensor。当前只识别 `add`、`mul`、`matmul`、`transpose`、`reshape`、`sum`、`all_reduce`、`all_gather` 和 `reduce_scatter`；不包含这些算子的属性或数学语义。

## Relation

Relation 关联一个单机 tensor 与按 rank 顺序排列的 local tensor：

```json
{
  "single_tensor": "A",
  "distributed_tensors": ["A0", "A1"],
  "type": "shard",
  "dim": 1
}
```

支持三类 relation：

- `replicate`：不含 `dim` 或 `reduce_op`，每个 local shape 与单机 shape 完全相同。
- `shard`：必须有合法的 `dim`，使用规则连续等大小切分；全局该维可被 world size 整除，且每个 local shape 与切分结果一致。
- `partial`：不含 `dim`，必须是 `"reduce_op": "sum"`，每个 local shape 与单机 shape 完全相同。其后续语义为单机 tensor 等于各 rank local tensor 之和。

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

完整的可运行样例见 [`../input/matmul_shard_to_partial.json`](../input/matmul_shard_to_partial.json)。

## 当前限制

输入采用顺序 operator 列表而不是 DAG；每个 rank 必须显式列出；不支持 symbolic shape、额外 operator 属性、跨 rank tensor 引用、真实 IR 前端、Z3 或 relation 的 SMT 编码。
