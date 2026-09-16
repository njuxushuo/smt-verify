# Stage 4: Relation semantics and value constraints

Stage 4 encodes the value relationship between a single symbolic tensor and its distributed rank-local tensors. It consumes `StageSpec` and the Stage 3 `SymbolicStageResult`, then returns separate input-premise and output-candidate Z3 constraints. It does not perform a proof or expose a verifier status API.

## Composite relation semantics

[`src/semantics/relations.py`](../src/semantics/relations.py) is the single source of relation semantics. `CompositeRelationSemantics` composes exactly one placement per mesh axis; Replicate, Shard, and Partial are not independently encoded whole-tensor relations.

- **Shard(dim)** maps a local index to `mesh_coordinate[axis] * local_extent[dim] + local_index[dim]` on its tensor dimension.
- **Replicate** preserves the shard mapping established by other axes. Ranks that differ only on a Replicate axis encode equal replicas of the same global shard.
- **Partial(sum)** sums only ranks whose coordinates differ on Partial axes while all non-Partial coordinates remain fixed. Multiple Partial axes use their Cartesian product.

Flat ranks remain the storage/API representation. [`src/stage/mesh.py`](../src/stage/mesh.py) provides the row-major flat-rank/coordinate conversion and coordinate-group enumeration.

## Shape and value orchestration

`shape_constraints.py` resolves symbolic dimension tuples and delegates to the composite semantics. For tensor dimension `d`, shard factor `F_d` is the product of mesh extents whose placement is `Shard(d)`. Reduction requires `single'[d] % F_d == 0` and `local'[rank,d] * F_d == single'[d]`.

`relation_encoder.py` resolves Stage 3 tensors by flat rank, passes `stage.mesh` to the composite value semantics, and returns:

```python
EncodedRelations(
    input_constraints=(...),
    output_constraints=(...),
)
```

Static Stage validation requires input relations to name true program inputs on the single program and every rank. This prevents produced/intermediate tensors from becoming unsound proof premises before symbolic execution. The encoder only resolves tensors and emits value constraints. Output relations may name any symbolic tensor that exists, including an input for identity stages.

## Boundary with Stage 5

The encoder does not call a solver, decide `PROVED`/`DISPROVED`, return counterexamples, generate candidates, or implement collective operations. A future Stage 5 verifier may combine these separate constraint tuples as `Rin ∧ ¬Rout`.

## CLI

Run the Stage 1 → 2 → 3 → 4 chain in the `smt` environment:

```bash
python scripts/inspect_relations.py input/matmul_shard_to_partial/matmul_shard_to_partial.json
```

The command prints simplified relation constraints and exits nonzero for input, shape, symbolic-execution, or relation-encoding errors.
