# Stage 4.5: Original-shape-aware reduction and Stage input boundaries

Stage 4.5 clarifies two preconditions for later local verification. It does not implement a verifier, a solver-status API, or any cross-stage analysis.

## Original-structure-preserving reduction

Reduced dimensions are still symbolic positive integers optimized by Stage 2, but relation reduction is no longer an arbitrary minimum satisfying only relation names. For each rank and dimension, `relations.py` retains the original local-to-single ratio using integer cross multiplication:

```text
local_reduced * original_single == single_reduced * original_local
```

This avoids Z3 Real division. Replicate and Partial have original ratio one; regular Shard preserves its original equal-size ratio and retains reduced shard divisibility. The current covered structure is tensor rank, regular Replicate/Shard/Partial shape ratios, concrete MatMul/Add/Mul legality, and their existing reduced structural equations. This is not a claim of general preservation for future broadcast, reshape, sum, layout, or uneven-shard semantics.

Concrete legality is checked during `validate_stage()`: MatMul is 2-D with matching shared/output dimensions; Add and Mul require exact same shapes. Thus an invalid original program cannot be repaired by reduction.

## Input boundary contract

`Rin` is a Stage proof premise, not a satisfiability obligation. This project does not perform an `exists X. Rin(X)` or input-premise SAT pre-check. A later local verifier may only construct `Rin ∧ ¬Rout`.

Static Stage validation uses `program_analysis.py` to identify producer-free program inputs and rejects an `input_relations` entry that names a produced tensor on the single program or any rank. The relation encoder now only resolves symbolic tensors and delegates value encoding. Output relations may still reference inputs, which supports identity stages.

## Current boundary

No cross-stage provenance, output-to-next-input continuity, relation metadata matching, scheduler, lineage manager, stage splitting, or whole-graph logic is introduced here. Those responsibilities belong to a future whole-graph verifier.
