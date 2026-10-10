# Optional shape reduction and finite-element verification

The input JSON contract is unchanged. The default verifier still uses original
shapes, function-valued inputs, symbolic output indices, and full contraction
expansion. Enable the second route explicitly:

```bash
python -m src.app.cli verify input/mlp_dp2_tp4_index_value/mlp_dp2_tp4_03_fc2_partial.json --shape-reduce --report /tmp/fc2-demo.txt
python -m src.app.cli batch input --shape-reduce --write-reports
```

The Python API accepts `verify_stage(..., shape_reduce=True)` and
`verify_file(..., shape_reduce=True)`. `--max-reduced-elements` (default 50000)
bounds the sum of all candidate tensor volumes before element allocation.
Batch reports use `demo.shape-reduced.txt`, beside the existing `demo.txt`.

## Ownership

- `smt/shapes.py`: named dimension variables, finite scale choices, named constraints,
  Z3 optimization. It knows no stage, operator, family or relation semantics.
- `ops/shape_rules.py`: operator-owned `shape_constraints` implementations.
  Operators may expose `reduced_attrs(call, output_shape)` for parameter updates
  and `contraction_axes` metadata for scale selection and expansion budgets.
  Registry aliases do not require changes to the reduction coordinator.
- `relation/shape_rules.py`: matching coordinate-space scales and exact rescaling
  of every supplied IndexPhi level. Partial membership and value rules are unchanged.
- `reduction/shapes.py`: orchestration and construction of an independent StageSpec.
  Operators are accessed through the injected registry; no operator implementations
  are imported here. Original objects and JSON files are never mutated.
- `smt/elements.py`: independently named Real input elements, concrete lookup only.
- `verification/shape_mode.py`: reduced obligations, original certification, fallback.
- `smt/blocks.py`: sound, deliberately incomplete abstraction of finite sums.

Relation elimination, map composition, operator value semantics, collective
scheduling and the solver wrapper are shared between both routes.

## Candidate constraints

Each axis has an integer reduced length and integral block scale with
`reduced * scale = original`. Choices are enumerated from exact divisors.
Original singleton axes stay singleton; all other axes conservatively remain
at least 2. Very large extents whose square root exceeds 100000 are pinned,
rather than spending unbounded time enumerating divisors.

Matmul aligns M/K/N and batch broadcast roles. Elementwise and expand preserve
original singleton roles. Transpose retains its axis permutation. Reshape
matches original contiguous axis groups and constrains their reduced products;
its target attrs are rebuilt from the candidate output shape. Collectives retain
groups and order, equal member shapes, and gather/scatter length equations.

Related local/global axes share a scale. Every intermediate extent and every
piece endpoint must be divisible by it: no rounding or independently moved
offsets. This policy is narrower than arbitrary segment-length optimization,
but preserves all currently supplied mapping chains, including CP/SP zigzag.
Unmapped axes, Partial image compatibility, rank coverage and SSA constraints
are checked again on the resulting candidate. Global gaps are not filled.
Input and output shape relations constrain the optimizer; output VALUE equations
are never premises. The objective is the sum of all declared tensor dimensions.
A feasible model is required, but global optimality is not a correctness premise.

## Finite-element proof

All candidate input tensors allocate scalar Reals. Equality-bound local elements
reference global elements through the same IndexPhi implementation. Partial inputs
retain independent non-pivot contributions, with the pivot equal to the single
value minus their sum. Intermediate tensors remain cached expressions built by
the existing operators. Every mapped output coordinate is enumerated and its
disequality included in its group's SMT query. Unmapped global regions are not
silently added to the proof scope.

## Why a small proof does not imply a large proof

There is no assumed general TrainVerify-style lifting theorem in this backend.
Even legal shapes and K>=2 are insufficient for arbitrary operator compositions.
After reduced UNSAT, the verifier therefore constructs an independent obligation
over ORIGINAL indices, ORIGINAL relations and ORIGINAL operators.

For each matmul, the contraction is partitioned into equal blocks. The common
block width is the gcd of the candidate contraction scales; it divides all
original contraction lengths. For a block beginning at `start`, evaluate its
summand symbolically at `start + q`, without enumerating q. A fresh Real atom
represents the sum of this body for `0 <= q < width`. Two blocks share an atom
only if their simplified bodies and widths are identical. All other atoms are
independent. Each block definition is recorded in the report.

For any assignment of original inputs and output indices, interpret each atom
as its actual finite sum. Identical bodies have identical sums, so this extends
the assignment consistently. Thus every original counterexample is also an
abstract counterexample. UNSAT of the abstract disequality proves original
equivalence. SAT is only inconclusive. This is a checked overapproximation,
not an assumption that one matrix size represents all sizes, and not proof reuse.

Opaque sums nested within opaque sums are rejected, including cached nested
atoms: otherwise a hidden dependence on an outer summation index could be lost.
Noncanonical algebraically equal bodies may also prevent certification. These
limitations lead to explicit exact fallback, never an unsupported proof claim.
The common nonnested TP matmul cases use roughly reduced-K many atoms instead
of expanding original-K products, while still covering arbitrary original output
indices. The other original operators continue to use their exact semantics.

## Result and resource behavior

`SHAPE_REDUCED_ELEMENTWISE` means reduced UNSAT AND original certificate UNSAT.
`SYMBOLIC_INDEX_FALLBACK` records the reason and the exact original result.
Reduced SAT, reduced UNKNOWN, shape optimization failure, element budget excess,
and inconclusive certification all require original verification. A reduced SAT
is not reported as an original counterexample. Original UNKNOWN remains UNKNOWN.
Invalid original input is rejected before candidate construction. A generated
candidate invariant failure is UNKNOWN with mode SHAPE_REDUCTION_ERROR, retaining
the attempted plan and diagnostic; it is not mislabeled INVALID_INPUT or hidden
by exact fallback. Unexpected implementation errors are not broadly swallowed.

The timeout applies per optimizer/solver invocation, not to all Python work.
Operator count, contraction expansion and dependency-depth limits still apply.
The certificate checks its actual original-K/block-width expansion count too;
coprime contraction scales can force width=1 and trigger the expansion budget.
Reports include candidate constraints and shapes, the reduced JSON and finite
equations, reduced queries, opaque block definitions, original queries and any
fallback reason. SMT-LIB queries can be replayed; the translation from operators
to block definitions remains part of the trusted implementation.
