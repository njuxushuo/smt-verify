# Stage 4.5: Original-shape-aware reduction and Stage input boundaries

Stage 4.5 clarifies two preconditions for later local verification. It does not implement a verifier, a solver-status API, or any cross-stage analysis.

## Original-structure-preserving reduction

Reduced dimensions are still symbolic positive integers optimized by Stage 2. For tensor dimension `d`, `relations.py` computes the composite shard factor `F_d` from the mesh placements and requires:

```text
single_reduced[d] % F_d == 0
local_reduced[rank,d] * F_d == single_reduced[d]
```

Replicate and Partial contribute factor one; a regular Shard contributes its mesh-axis extent. The current milestone rejects repeated Shard placements on the same tensor dimension. The covered operator structure remains MatMul matrix/batch equations, Add/Mul broadcast patterns, View axis mappings, Reshape contiguous factorization groups, and Expand unary broadcast patterns. This is not a claim about physical stride/layout, memory aliasing, uneven/interleaved shards, or unsupported operators.

Concrete legality is checked during `validate_stage()`: each implemented operator receives its original shapes and attrs before reduction. This includes exact View output metadata, negative-axis bounds, positive resolved Reshape/Expand targets, singleton requirements, and element-count checks. Thus an invalid original program cannot be repaired by reduction.

## Input boundary contract

`Rin` is a Stage proof premise, not a satisfiability obligation. This project does not perform an `exists X. Rin(X)` or input-premise SAT pre-check. A later local verifier may only construct `Rin ∧ ¬Rout`.

Static Stage validation uses `program_analysis.py` to identify producer-free program inputs and rejects an `input_relations` entry that names a produced tensor on the single program or any rank. The relation encoder now only resolves symbolic tensors and delegates value encoding. Output relations may still reference inputs, which supports identity stages.

## Current boundary

No cross-stage provenance, output-to-next-input continuity, relation metadata matching, scheduler, lineage manager, stage splitting, or whole-graph logic is introduced here. Those responsibilities belong to a future whole-graph verifier.
