# Refactored stage verifier

The implementation follows [roadmap.md](../roadmap.md). Input documents live
under `input/`; their contract is in [English](../input/docs/README.md) and
[Chinese](../input/docs/README.zh-CN.md). The eleven checked-in JSON files were not
changed to make the new verifier succeed.

## Modules and dependency direction

| Module | Responsibilities | Allowed project dependencies |
| --- | --- | --- |
| core | Program/tensor/topology facts, strict JSON primitives, typed errors | core only; no Z3 |
| smt | Lazy indexed tensors, function context, solver invocation | core, smt |
| relation | Own models/codecs, structural validation, map normalization, value encoding | core, smt, relation |
| ops | Local and collective operator math, shape/attribute checking, explicit registry | core, smt, ops |
| stage | Task model/codec, SSA and boundary validation; delegates to passed registry | core, relation, stage |
| reduction | Exact original-domain index reduction and budget checks | core, relation, stage, reduction |
| verification | Collective scheduling, execution, proof obligations, result/counterexample | lower layers, verification |
| app | Registry composition, CLI and report I/O | application and lower layers |

`tests/test_architecture.py` enforces these import directions and prohibits
registration through `__init__.py`. Operators do not inspect relations; relations
do not execute operators. Both use lazy `SymbolicTensor` values. A collective's
mathematics lives in `ops/collectives.py`; matching calls and coordinating rank
progress lives in `verification/collectives.py` and `executor.py`.

The old placement-based `semantics`, forced small-shape `shape`, independent-rank
`symbolic`, and provisional `lemma` production paths were removed. Legacy source
is retained in the checkpoint commit. Lemma generalization/reuse is deferred,
not emulated by labelling an ordinary result file a proof certificate.

## Exact input relation encoding

Every single input and unbound local input starts as an arbitrary function from
integer coordinates to a Real value. There are no random samples and no implicit
equality between same-named inputs. Functions are only interpreted on valid
tensor indices. All dimensions, maps and collective orders use the original JSON.

For an equal input relation, substitute `D_r[i] := S[Phi_r(i)]`. For Partial(sum),
choose one pivot per concrete group, keep the other rank contributions freely
symbolic, and substitute

```
D_p[i] := S[Phi_p(i)] - sum(D_r[Phi_r^-1(Phi_p(i))] for r != p in G)
```

Validation establishes injectivity per local map, identical group image domains,
and non-overlapping tensor bindings. These substitutions parameterize exactly
the models of Rin: any satisfying input restricts to an assignment of free
functions, and any such assignment reconstructs a satisfying original input.
No replica-value conditions are added in the Partial case. This is why the
solver query does not need to retain quantified Rin after substitution.
The original universal Rin equations and substitutions are printed in demo.txt.

## Ordered IndexPhi normalization

IndexPhi terms on the same tensor dimension compose in their supplied order:
`[SP, CP]` means `CP(SP(i))`. Each dimension has an independent chain. No reordering
by family name or inferred extent is allowed, even when domains have equal sizes.
Each map level is validated on its own input/output domains; for every rank,
the first source is the actual local extent, adjacent extents must match, and
the last target is the actual single extent. Partial is not a coordinate stage.

`relation/normalization.py` composes pure interval translations without Z3 or
operator dependencies. It intersects each preceding image interval with the
next source partition, splits crossing pieces, sorts by original local indices,
and merges only adjacent equal translations. The sweep is linear after sorting
because validated maps are injective. It does not enumerate tensor elements.

The internal `(rank, dim) -> pieces` interface still contains exactly one final
map. Existing SMT forward/inverse encoding and Partial image checks use that
map, not the uncomposed terms. Original terms and order remain in the binding
and codec output for audit. Reports show intermediate extents, level order and
the final maps separately. Old single-level relations keep their semantics.
Repeated family IDs and multiple Partial terms remain outside the supported subset.

## What reduction means here

The original backend minimized physical dimensions and relied on a restricted
lifting argument. This implementation deliberately does not generalize a proof
from one small shape. It reduces output-element enumeration to bounded symbolic
indices while retaining original shapes and every original matmul contraction term.

For each output rank, or each Partial group, introduce arbitrary integer indices
`i0,...,in` with `0 <= id < local_shape[d]`. Evaluate both output expressions at
the corresponding mapped coordinate and ask if their disequality is satisfiable.
The representative rank in a Partial group enumerates its whole image; all group
members contribute via inverse maps. Validated map image equality makes this
domain enumeration complete even when members store pieces in different orders.

`UNSAT` across every domain proves all output elements at the original shapes.
`SAT` supplies original-shape output coordinates plus function interpretations
and reconstructed input samples. Symbolic coordinate bounds, full-K sum rules,
normalization checks and actual SMT-LIB queries appear in every report.

This is called `SYMBOLIC_INDEX_DOMAIN`, with proof scope
`ORIGINAL_SHAPES_EXACT_REAL`. Physical shape reduction is explicitly reported as
NONE. No smaller activation tensor is substituted into the original problem.

## Arithmetic, resources and outcomes

Values use Z3 Real arithmetic. GELU is one shared unary uninterpreted function,
so the proof establishes layout preservation for any identical pointwise
function; it does not certify a floating-point GELU implementation.

The initial registry implements matmul, add, mul, gelu, transpose, reshape,
expand, all_gather, all_reduce and reduce_scatter. Unknown operators/reductions
are UNSUPPORTED; malformed shape, topology, dataflow and rank maps are INVALID_INPUT.
Logical family aliases do not multiply participants. Group member order controls
gather/scatter layout. Missing or inconsistent collective members and impossible
cross-group ordering are rejected.

Default limits are 8192 contraction terms, 10000 stage operator calls, and lazy
dependency depth 64. Exceeding these limits is UNSUPPORTED, never PROVED. The
per-query Z3 timeout defaults to 30000 ms and UNKNOWN retains its reason. The
timeout bounds solver work, not all Python expression construction. Deep recursion
is caught as UNSUPPORTED. These are correctness-focused prototype limits, not a
claim of bounded memory for arbitrary adversarial programs.

PROVED requires every domain UNSAT; a SAT domain returns DISPROVED immediately.
UNKNOWN domains cannot silently become successful proofs. Invalid input and
unsupported semantics are not counterexamples. Reports are audit artifacts,
not independently checked formal proof certificates.

## Tests and reports

Run `python -m unittest discover -s tests -v`. Negative tests cover wrong output
relations, missing premises, QKV range errors, Partial without replica equality,
collective order, absent members and deadlock. Direct API validation and deep
dependency limits have regressions from independent review. Report tests replay
both SAT and UNSAT SMT-LIB queries and check that report writes cannot overwrite inputs.

`python -m src.app.cli batch input --write-reports` generates all eleven reports.
Normally a report is beside its JSON. For the five MLP JSON files sharing a parent,
reports go to `<parent>/<json-stem>/demo.txt`, preserving every input path.
Each report records the canonical input hash, parsed models, original shapes,
validation checks, full relation maps, universal input equations, substitution
rules, execution order, domain bounds, actual SMT query and final proof scope.
