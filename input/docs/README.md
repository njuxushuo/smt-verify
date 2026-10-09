# Stage JSON input contract

[Chinese / 中文](README.zh-CN.md)

This is the target contract for the JSON fixtures, all located in `input/`.
It specifies one local equivalence obligation: single-device
program, rank-local programs, assumed input relations, and candidate output
relations. The refactored backend implements this contract for the supported
operators below, using original-shape symbolic index-domain verification.

There are 10 retained JSON files, including all five protected MLP stages.
Six counterexample cases, two shape-invalid cases and sixteen positive-case
variants were removed from the previous 34-file collection. Their relation representation is
aligned with this contract; section 8 records the collection audit.
The entry point is `src/stage/codec.py`; legacy tests and scripts have been
replaced. All ten JSON files have fresh demo.txt reports from the new verifier.
Names do not control results. AutoVerify's producer model still needs adaptation.
See [architecture](../../docs/architecture.md) for exact input elimination and
the distinction between index-domain reduction and physical shape shrinking.

## 1. Fixed conventions

- UTF-8 JSON, with no comments, duplicate object keys, or unknown fields.
- All fields described below are required unless explicitly noted otherwise.
- No `schema`, `version`, `relation_format`, `semantics`, `source`, `coverage`,
  `scope`, `placements`, `mesh`, `index_transform`, or `default` fields occur in
  the input. Interpretation is fixed by this document, not per-file switches.
- Numeric tensor values use exact real arithmetic. This does not establish
  floating-point bitwise equivalence. GELU is one shared, deterministic unary
  uninterpreted function, applied elementwise in every program.
- Tensor dimensions are zero-based. Shapes are nonempty lists of positive
  integers. Scalars, dynamic dimensions, tuple values, aliasing and mutation
  are outside this initial contract.
- All rank IDs are integers `0 .. world_size-1`. Object keys use their canonical
  decimal string representations, such as `"0"`; rank-map entries use integers.
- Every stage file is self-contained. No tensor name refers into another file.
  Equal boundary names across files make chaining easier but do not prove it.

## 2. Top-level fields

| Key | Type | Meaning |
| --- | --- | --- |
| `name` | nonempty string | Human-readable case identifier. Never infer semantics from it. |
| `topology` | object | World size and explicit group/family definitions. |
| `single` | program | The single-device reference subgraph. |
| `distributed` | object | Contains exactly `ranks`, the rank-to-program mapping. |
| `input_relations` | array of bindings | Proof premises relating true boundary inputs. May be empty. |
| `output_relations` | nonempty array of bindings | Candidate conclusions; they must be proved, not assumed. |

`distributed.ranks` contains exactly one program for every world rank. The
single and distributed programs need not have identical names, operations or
internal tensor shapes. A backend must not clone rank 0 without checking all
rank-specific parameters and shapes.

## 3. Topology

| Key | Meaning |
| --- | --- |
| `topology.world_size` | Positive number of participating ranks. |
| `topology.groups` | Object keyed by concrete group ID. |
| `groups.<id>.ranks` | Nonempty, ordered list of unique world ranks. Order is semantically significant for gathering/scattering. |
| `topology.families` | Object keyed by family ID. |
| `families.<id>.groups` | Nonempty list of concrete group IDs. |

Within each family, groups are disjoint and cover the whole world exactly.
A concrete group ID may be referenced by multiple logical families when they
reuse the same communication groups, as SP and TP do in these MLP fixtures.
These are aliases of a group partition, not additional mesh axes or independent
reduction dimensions; never count a shared group twice. Different families need not be
orthogonal. Their names (`dp`, `tp`, `mesh_axis_0`) are labels, not algorithms.
The old unnamed mesh axes are migrated to `mesh_axis_N` without inventing TP/DP
semantics. Family count does not determine tensor dimensionality.

In the eight-rank MLP fixtures, TP groups are `[0,1,2,3]` and `[4,5,6,7]`, while
DP groups are `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`. Logical `sp` and `tp` families
both reference the existing `tp_0` and `tp_1` groups. Sequence mappings use `sp`;
feature/weight mappings and local-product Partial(sum) use `tp`. DP batch
mappings use `dp`. World size remains 8, not DP times TP times SP.
Collective `attrs.group` still names the actual shared group, even for an SP
AllGather or ReduceScatter; its `tp_0` name does not change the layout role.

## 4. Programs and operators

| Key | Meaning |
| --- | --- |
| `tensors` | Object from scope-local tensor name to a tensor declaration. |
| `tensors.<name>.shape` | Full shape, including batch dimensions; this is the original shape, not a reduced one. |
| `inputs` | Unique names of free boundary tensors. A weight is a free tensor unless separately constrained. |
| `outputs` | Unique names of tensors exported by this stage. |
| `ops` | Operations in dependency order. Tensor references express a DAG, not necessarily a chain. |
| `ops[].type` | Operator name, interpreted using the fixed operator contract. |
| `ops[].inputs` | Ordered input tensor names in the current program. |
| `ops[].outputs` | Ordered names of newly produced tensors. |
| `ops[].attrs` | Operator-specific parameters; `{}` when no attributes are needed. |

Every tensor is either a declared input or produced exactly once. Inputs and
outputs of an operation must exist in `tensors`; operands must be available
before use. Stage outputs must be available at the end. An identity single
program has `ops: []` and may name the same tensor in `inputs` and `outputs`.
Ordinary operators cannot refer to tensors on another rank.

| Operator | `attrs` | Interpretation |
| --- | --- | --- |
| `matmul` | `{}` | Rank >= 2 matrix multiplication, with batch broadcasting. |
| `add`, `mul` | `{}` | Elementwise operation with trailing-dimension broadcasting. |
| `gelu` | `{}` | Shared elementwise uninterpreted function in the current real-valued model. |
| `transpose` | `dim0`, `dim1` | Swap two dimensions; legal negative axes are normalized against input rank. |
| `reshape` | `shape` | Positive concrete target shape, equal to declared output shape; preserves row-major linear element order. |
| `expand` | `shape` | Broadcast to the declared positive target shape. |
| `all_gather` | `group`, `axis` | Concatenate group inputs along `axis`, in group rank order; every member receives the concatenation. |
| `all_reduce` | `group`, `reduce_op` | Elementwise group reduction; currently only `sum`. |
| `reduce_scatter` | `group`, `axis`, `reduce_op` | Sum group inputs, then divide into equal contiguous slices along `axis`, assigned in group rank order. |

Collective `group` references one concrete group, not a family. `axis` is
zero-based and nonnegative; MLP activations are `[b,s,h]`, so `axis: 1` is
sequence. Collectives are synchronous and functional abstractions, not direct
c10d signatures. Every member must execute matching collective calls: pair the
n-th call on each concrete group in program order, with equal type/attrs and
compatible shapes. Current examples contain at most one collective per group
per stage. Uneven collectives, overlapping-group schedules and asynchronous
handles need additional rules before use. Squeeze/unsqueeze, slice/concat
operators and other operators not listed in the implemented registry currently
return UNSUPPORTED. Multi-piece boundary maps are supported independently.

## 5. Tensor bindings and terms

| Key | Meaning |
| --- | --- |
| `single_tensor` | Tensor name in `single`. |
| `distributed_tensors` | Object mapping every rank string to its local tensor name. |
| `relation.terms` | Array of nonidentity index components and an optional Partial component. May be empty. |
| `terms[].family` | Existing family ID in topology. |
| `terms[].expr` | One `index_phi` or `partial` expression. |

The `relation` object contains only `terms`; each term contains only `family`
and `expr`. Their types follow from their positions, so neither object has a
`kind` field. Only `expr.kind` is required to distinguish `index_phi` from
`partial`. For example, a relation payload is:

```json
{
  "terms": [
    {"family": "tp", "expr": {"kind": "partial", "op": "sum"}}
  ]
}
```

Input bindings refer only to declared program inputs; output bindings refer to
declared stage outputs. Every exported tensor must have an output goal. A
boundary input without a relation remains unconstrained across programs: names
alone never imply equality. Each single tensor is bound at most once per list;
use `terms` for multiple family components. In the initial supported subset, a
family occurs at most once within a binding, and each index dimension has at
most one term. Term order has no mathematical significance. These are support
limits, not claims that more general layouts never occur.

An explicit `"relation": {"terms": []}` means full elementwise equality:
every local tensor has the same shape and values as the single tensor.
It is not missing information. A missing input binding gives no cross-program
relation; `relation: null` is not accepted. `output_relations: []` is invalid,
but an output binding with `terms: []` is a valid, nontrivial equality goal.

A tensor need not have a term for every topology family. For W1/W2 in the MLP,
keep only the TP IndexPhi: it explicitly maps all eight ranks, including the
repeated weight slices in the two DP replicas. Without Partial, equality to the
mapped single elements already implies equality of those replicas. Do not add
an empty DP term or choose an arbitrary tensor dimension merely to label this
replication. The DP family remains in topology.

This also applies to other pure index relations, such as the batched-matmul
inputs and transpose relations combining replication with sharding. Do not emit identity-only
IndexPhi terms: a fully replicated tensor uses `terms: []`.
The AllGather output and the following FC1 input X keep only the DP batch map:
sequence identity and group-local replication already follow from that complete
all-rank relation. Do not add an SP identity term just to record the preceding
collective; communication history belongs in `ops`. Partial constrains group
reductions only. Equal local contributions across replica groups are not
implicitly added; such an independent condition requires a future explicit
value-relation extension; no identity term may silently introduce such a constraint.

### IndexPhi

```json
{
  "kind": "index_phi",
  "dim": 1,
  "maps": [
    {
      "rank": 0,
      "local_extent": 32,
      "single_extent": 128,
      "pieces": [
        {"local": {"start": 0, "stop": 32}, "single": {"start": 0, "stop": 32}}
      ]
    }
  ]
}
```

This fragment shows one rank entry; the `maps` array must include every
world rank exactly once. It is not a complete multi-rank relation on its own.

| Key | Meaning |
| --- | --- |
| `expr.kind` | Exactly `index_phi`. |
| `expr.dim` | Required nonnegative tensor dimension. It is not a mesh axis. |
| `expr.maps` | Nonempty array containing exactly one entry for every rank. |
| `maps[].rank` | World rank to which this map applies. |
| `maps[].local_extent` | Local tensor size at `expr.dim`; must match the tensor declaration. |
| `maps[].single_extent` | Single tensor size at `expr.dim`; must match its declaration. |
| `maps[].pieces` | Pieces covering the entire local dimension. |
| `pieces[].local.start/stop` | Half-open source interval `[start, stop)` in local indices. |
| `pieces[].single.start/stop` | Half-open target interval in single-tensor indices. |

Each piece has equal source/target length and maps `i` to
`single.start + i - local.start`. Source pieces cover the local dimension
without gaps or overlap; target pieces stay within bounds and do not overlap
within one rank. Different ranks may map to the same single positions. Several
pieces support noncontiguous layouts; ranks may have unequal local extents.
Bounds are nonnegative integers, with `start < stop` for every piece.

`dim` belongs to the expression, not each rank entry. There is no `default`.
Every rank must be present, including ranks whose map happens to be identity.
Emit a term only if at least one rank has a nonidentity map or a changed extent
on that dimension; omit the entire term if all ranks map the full dimension
identically. Empty `maps` is no longer an accepted identity shorthand.

### Fused QKV and multiple pieces

For a fused output-feature axis with Q `[0,8)`, K `[8,16)`, V `[16,24)` and
TP=2, a local fused tensor of width 12 can use these maps:

| Rank | Local intervals | Single intervals |
| --- | --- | --- |
| 0 | `[0,4)`, `[4,8)`, `[8,12)` | `[0,4)`, `[8,12)`, `[16,20)` |
| 1 | `[0,4)`, `[4,8)`, `[8,12)` | `[4,8)`, `[12,16)`, `[20,24)` |

This is one IndexPhi term with three pieces per rank, not three terms. It
describes static boundary correspondence. It neither implements concat/slice
operator semantics nor supports binding three independent single tensors to
one fused local tensor. Actual QKV packing determines the pieces; do not infer
the layout from the tensor name. Cross-dimension permutations are outside this
map representation and must be handled by supported operators or a future extension.

### Partial

```json
{"kind": "partial", "op": "sum"}
```

`op` is currently restricted to `sum`. Partial has no `dim`: it combines values
after the complete index map has been assembled. The surrounding term's family
defines the participating groups. At most one Partial term is supported per
binding; multiple reductions across families require an explicit future
composition contract and must not be guessed.

## 6. Composing the relation

1. Gather all IndexPhi maps. Different terms must affect different dimensions;
   overlapping explicit dimensions are rejected in this initial contract.
2. For dimensions with no term, use global identity and check equal
   local/single sizes. All bound tensors have the same number of dimensions.
3. The result is a complete local-to-single map `Phi_r` for each rank. There is
   no separate global equality assertion for each IndexPhi term.
4. Without Partial, require `D_r[l] = S[Phi_r(l)]` for every local element and rank.
5. With Partial, for each group G of its family require
   `S[g] = sum(D_r[Phi_r^-1(g)] for r in G)` on the common mapped target domain.
   All members must have the same target domain and a unique inverse there.
   Different groups may cover different DP batch slices. Missing contributors
   are errors, never implicit zero values.
6. Do not infer additional value equalities from matching maps or family names
   in the presence of Partial. Without Partial, matching maps imply replication
   through step 4. With Partial, only the specified group sums are constrained.

Full local-domain coverage is mandatory; global target coverage is not. A slice
relation constrains only its image. There is deliberately no `coverage` switch.
Unknown family IDs, missing ranks, invalid extents, conflicting maps and missing
output goals are validation errors, not proof failures. Nonorthogonal topology
does not remove the restrictions on map composition or Partial target domains.

The backend checks the conjunction of input relations and the negation of the
conjunction of output goals, with operator semantics substituted or conjoined.
Output goals must never become value premises. Empty output goals are invalid.
Input assumptions need an external justification for whole-program correctness;
successful structural simulation does not supply it.

## 7. MLP examples and authoring workflow

Start with these complete files:

| Example | File |
| --- | --- |
| Sharded matmul to Partial | [MLP FC2](../mlp_dp2_tp4_index_value/mlp_dp2_tp4_03_fc2_partial.json) |
| Two explicit index dimensions | [MLP FC1/GELU](../mlp_dp2_tp4_index_value/mlp_dp2_tp4_02_fc1_gelu.json) |
| Index map plus Partial | [Shard/Partial transpose](../mesh/mesh_shard_partial_transpose_proved/mesh_shard_partial_transpose_proved.json) |
| Eight-rank sequence AllGather | [MLP stage 01](../mlp_dp2_tp4_index_value/mlp_dp2_tp4_sp_01_allgather.json) |
| Complete MLP suite | [Stage index](../mlp_dp2_tp4_index_value/README.md) |

The MLP chain is AllGather -> FC1/GELU -> FC2 Partial -> ReduceScatter. AllReduce
is an alternative ending, not a fifth sequential stage. Adjacent boundaries
use the same tensor names, shapes and relations. The global activation shape is
`[4,128,256]`, FFN width is 1024, and each DP replica handles two batch elements.
DP contributes a batch map; SP contributes a sequence map; TP contributes a
feature/weight map or Partial. Families sharing groups do not imply that every
tensor needs both an SP and a TP term.
Weights are replicated across DP groups through their all-rank TP mappings;
their bindings have no DP term. All standalone matmul fixtures, including the
small baseline, were removed in favor of the protected MLP stages. Outside MLP,
one representative each is retained for Add, Mul, Transpose, Reshape and Expand.
This prioritizes core proof tasks over parameter combinations; specialized
broadcast cases and negative-axis/squeeze/unsqueeze checks need separate unit
tests when those backend paths are implemented.

FC1/GELU remains a manually combined multi-operator fixture. It is not evidence
that AutoVerify always cuts these operators into the same stage.

The MLP AllReduce output has only the DP batch-map term. All ranks in each TP
group map to the same single batch slice, so mapped value equality already
establishes TP replication. AllReduce does not select a tensor axis: no TP
`dim: 1` identity term is needed. Its input still includes TP Partial(sum), and
the actual collective group remains in the operator attributes.

To add a fixture:

1. Declare ordered groups and complete families; keep mesh coordinates out of
   the proof input and write rank maps explicitly.
2. Declare both programs, all boundary/internal tensor shapes, operations and
   their actual attributes. Promote cut inputs to free tensors, not their old
   producer operations.
3. Bind input relations. Add only nonidentity index terms with expression-level
   `dim` and all-rank intervals. Use `terms: []` for full equality. Add at most
   one Partial where needed; never hide additional value constraints in identity terms.
4. Bind every output to a candidate relation. The current input collection keeps
   positive targets only; future negative regression cases should be maintained
   separately in test fixtures. Never silently repair a candidate during serialization.
5. Check shape/dataflow legality, map completeness, group membership/order and
   stage-to-stage boundaries. Then use the verifier; valid JSON alone is
   not a proof. Keep large-shape fixtures, but never arbitrarily shrink shapes
   without transforming index ranges and proving the reduction is sound.

## 8. Support limits and migration audit

| Supported target | Not yet defined or supported |
| --- | --- |
| Static positive shapes; exact real values | Dynamic routing/shapes, IEEE kernel equivalence, aliases, tuple/scalar values |
| One IndexPhi per changed dimension; distinct families | One family with multiple index terms; multiple families sharding the same dimension |
| Uneven static shards and several unit-stride pieces | Cross-dimension maps, duplicate target indices within a rank, dynamic gathers |
| No Partial or one Partial(sum) | Several Partial families, implicit nested reduction or additional replica constraints |
| Explicit groups, including nonorthogonal families | Assuming arbitrary nonorthogonal maps/reductions compose without checks |

The excluded same-dimension case is relevant to FSDP+TP and CP+SP. It requires
ordered/nested maps or a normalized final map, not silently adding offsets.
Retaining the current one-family/one-term limit is an implementation boundary,
not a claim that all distributed layouts obey it.

Audit on 2026-10-09 after collection cleanup: 10 files contain 24 bindings and
33 terms. None currently uses `terms: []`, but empty terms remain valid under
this contract. No empty-map IndexPhi terms remain.
The ten retained inputs now pass the new backend with proof scope
ORIGINAL_SHAPES_EXACT_REAL. The six removed false goals and two malformed files
are historical cleanup facts; new negative cases live in tests.

- Retained JSON contents were preserved during pruning. The protected five MLP
  JSON files were compared byte-for-byte. Documentation was updated separately.
- Five MLP files use nonempty, dimension-specific maps and Partial only. Their
  boundary relations are consistent and their stage goals are SMT-proved under
  the declared real/UF model. Stage input assumptions still need frontend justification.

The backend no longer accepts legacy placements. Current AutoVerify RelationCodec
v3 differs: it stores dim per rank map, has identity defaults, independently
validates each IndexPhi, and does not implement this term composition. The new
contract additionally gives empty terms explicit equality semantics and permits
logical SP/TP group sharing. Conversion must preserve actual family IDs, maps,
and proof assumptions; format similarity is not a lossless-export guarantee.
The producer adapter remains future work. Physical shape reduction is not used:
all original contraction terms and index domains are retained. A symbolic output
index represents every valid coordinate, so UNSAT does not rely on small-shape
lifting. demo.txt records these bounds and the actual SMT queries. Default
resource caps are K=8192, 10000 operator calls and lazy dependency depth 64;
unsupported limits and solver UNKNOWN never become PROVED.
