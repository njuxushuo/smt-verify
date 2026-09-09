# Stage 4: Relation semantics and value constraints

Stage 4 encodes the value relationship between a single symbolic tensor and its distributed rank-local tensors. It consumes `StageSpec` and the Stage 3 `SymbolicStageResult`, then returns separate input-premise and output-candidate Z3 constraints. It does not perform a proof or expose a verifier status API.

## RelationSemantics

[`src/relations.py`](../src/relations.py) is the single source of relation-specific symbolic semantics. `RelationSemantics` supplies both `shape_constraints(...)` and `value_constraints(...)`, with the registry providing `replicate`, `shard`, and `partial` implementations.

- **Replicate** requires equal shapes and encodes every `local[r][i] == single[i]`.
- **Shard(dim)** requires regular contiguous equal-size one-axis shards. It constrains non-shard dimensions to match, requires `world_size * local_dim == single_dim` and divisibility, then maps local index `i` to global index `rank * local_dim + i` on the shard axis.
- **Partial(sum)** requires equal shapes and encodes `single[i] == Sum(local[0][i], ..., local[P-1][i])`.

## Shape and value orchestration

`shape_constraints.py` resolves symbolic dimension tuples and delegates relation shape semantics through `get_relation()`. It no longer contains concrete Replicate/Shard/Partial formulas or a separate shard divisibility pass.

`relation_encoder.py` resolves Stage 3 tensors, delegates value constraints through the same registry, and returns:

```python
EncodedRelations(
    input_constraints=(...),
    output_constraints=(...),
)
```

Input relations are required to name true program inputs on the single program and every rank. This prevents produced/intermediate tensors from becoming unsound proof premises. Output relations may name any symbolic tensor that exists, including an input for identity stages.

## Boundary with Stage 5

The encoder does not call a solver, decide `PROVED`/`DISPROVED`, return counterexamples, generate candidates, or implement collective operations. A future Stage 5 verifier may combine these separate constraint tuples as `Rin ∧ ¬Rout`.

## CLI

Run the Stage 1 → 2 → 3 → 4 chain in the `smt` environment:

```bash
python scripts/inspect_relations.py input/matmul_shard_to_partial.json
```

The command prints simplified relation constraints and exits nonzero for input, shape, symbolic-execution, or relation-encoding errors.
