# Stage 3: Program-level symbolic execution

Stage 3 turns a validated `StageSpec` and its `ReducedShapeResult` into the symbolic computation of the single program and every distributed rank. It deliberately does not encode tensor relations or prove single/distributed equivalence.

## Inputs, producers, and order

A program input is a declared tensor with no operator producer. Inputs retain `ProgramSpec.tensors` declaration order and each reduced element becomes a `z3.Real` named with its execution scope, such as `single__A__v0` or `rank0__A0__v0`.

Programs use single-assignment semantics: a tensor may have only one producer. Operations execute strictly in `ProgramSpec.ops` order. An input that is produced later is therefore a use-before-produce error, not a new free input.

## Symbolic tensors and results

`SymbolicTensor` stores a positive, concrete reduced shape and an immutable row-major flattened tuple of Z3 arithmetic expressions. Its checked `at(...)` method maps multidimensional indices to this storage. Inputs store free Real variables; intermediate and output tensors store expressions returned by their producer operator and never acquire independent output variables.

`SymbolicProgramResult` contains the detected input names and every symbolic tensor in one program. `SymbolicStageResult` contains the single result and one result for each distributed rank.

## Operator semantics

The existing `OperatorSemantics` abstraction in `src/operators.py` now owns both shape constraints and `symbolic_execute(...)`. The supported operators are:

- `matmul`: two-dimensional matrix multiplication using `z3.Sum` over the shared dimension.
- `add`: same-shape elementwise addition.
- `mul`: same-shape elementwise multiplication.

`src/symbolic_executor.py` only coordinates producer checks, ordered execution, reduced-shape validation, and registry dispatch through `get_operator()`; it contains no operator-specific mathematical formulas.

## Current boundary

No Replicate, Shard, Partial, collective, value-relation, or equivalence constraint is created here. Stage 3 supplies symbolic single and rank-local computations for a later relation-value verification stage.

## CLI

Run the complete Stage 1 → Stage 2 → Stage 3 chain with:

```bash
python scripts/inspect_symbolic.py input/matmul_shard_to_partial.json
```

The command reports inputs, produced tensors, and representative output expressions. It exits nonzero for Stage input, shape-reduction, or symbolic-execution errors.
