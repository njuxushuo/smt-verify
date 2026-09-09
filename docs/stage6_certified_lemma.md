# Stage 6: Certified Lemma Materialization MVP

## Input and candidate boundary

Stage 6 continues to accept an existing Stage JSON. Its single `output_relation` is one already-chosen candidate; no new candidate input format exists.

If a future external generator or scheduler wants to consider several candidates, it must emit independent JSON files. Each file independently runs Stage 1 through Stage 6. Stage 6 does not enumerate, schedule, filter, rank, generate, match, or reuse candidates.

## Materialization rule

`certify_stage()` calls the unique Stage 5 `verify_stage()` API once. It constructs a `CertifiedLemma` only when the result is `PROVED`:

```text
PROVED      -> materialize one concrete lemma
DISPROVED   -> no lemma
UNKNOWN     -> no lemma
```

No second reduction, symbolic execution, relation encoding, or SMT query is implemented in Stage 6.

## Concrete artifact and identity

The persisted lemma includes input relations, single and rank-local operator sequences, output relation, original single/rank shapes, reduced proof shapes, world size, and source Stage name. It is concrete: tensor names, shape values, rank-specific operators, and world size are retained.

The deterministic ID is `lemma_<first 16 hex chars of sha256>`. The SHA-256 payload is canonical JSON with sorted keys and compact separators `(',', ':')`; it contains world size, input/output relations, single/distributed operator sequences, and original shapes. It intentionally excludes `source_stage_name` and `reduced_shapes`, so a renamed but semantically identical Stage has the same ID.

## Persistence

`save_lemma()` writes canonical JSON to `<lemma-dir>/<lemma_id>.json`, creating the directory when necessary. Re-saving byte-identical canonical content succeeds idempotently. Existing content with the same ID but different bytes raises `LemmaStoreError`; it is never overwritten.

## Current scope

The current lemma is certified, concrete, not generalized, and not reusable yet. Lemma generalization, alpha-renaming, symbolic-shape or world-size generalization, matching, reuse, cache-based SMT skipping, provenance, and whole-graph reasoning remain future work.
