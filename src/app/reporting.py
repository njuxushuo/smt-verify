"""Deterministic, human-readable proof reports with replayable SMT queries."""

import hashlib
import json
from pathlib import Path

import z3

from ..relation.model import IndexPhi
from ..stage.codec import stage_to_dict


def _section(lines, title):
    lines.extend(("", title, "=" * len(title)))


def _relation(lines, rel, role, number):
    b = rel.binding
    lines.append(f"{role}[{number}]: single.{b.single_tensor}{rel.single_shape}")
    for r, name in sorted(b.distributed_tensors.items()):
        lines.append(f"  rank{r}.{name}{rel.local_shapes[r]}")
    for term in b.relation.terms:
        e = term.expr
        if isinstance(e, IndexPhi):
            lines.append(f"  family={term.family}, IndexPhi dim={e.dim}")
            for m in e.maps:
                text = "; ".join(
                    f"local [{p.local_start},{p.local_stop}) -> single [{p.single_start},{p.single_stop})"
                    for p in m.pieces
                )
                lines.append(f"    rank {m.rank}: {text}")
        else:
            lines.append(
                f"  family={term.family}, Partial({e.op}), groups={rel.reduction_groups}"
            )
    lines.append("  Other dimensions: identity; matching extents checked.")
    if rel.reduction_groups:
        lines.append(
            "  Equation: S[g] = sum(D_r[Phi_r^-1(g)] for r in G), separately for each G."
        )
        lines.append(
            "  Partial alone DOES NOT assert D_r = S or replica contribution equality."
        )
    else:
        lines.append(
            "  Equation: D_r[i] = S[Phi_r(i)] for every valid local i and every rank."
        )
    if not b.relation.terms:
        lines.append(
            "  terms=[] is FULL ELEMENTWISE EQUALITY, not an empty assumption."
        )
    for r in sorted(rel.local_shapes):
        lines.append(f"  rank {r} single-image intervals by dimension: {rel.image(r)}")


def render_report(result):
    lines = [
        f"Stage: {result.name}",
        f"Verification: {result.status.value}",
        f"Proof scope: {result.scope}",
        f"Z3 version: {z3.get_version_string()}",
        "Arithmetic: exact real; GELU: shared unary uninterpreted function.",
        "No claim of IEEE floating-point kernel equivalence or whole-program correctness.",
    ]
    if result.reason:
        lines.append(f"Reason: {result.reason}")
    if result.validated is None:
        lines.append("Input could not be validated; no proof obligation was accepted.")
        return "\n".join(lines) + "\n"
    v = result.validated
    stage = v.stage
    canonical = json.dumps(stage_to_dict(stage), sort_keys=True, separators=(",", ":"))
    lines.append(
        "Canonical input SHA256: " + hashlib.sha256(canonical.encode()).hexdigest()
    )
    _section(lines, "1. Parsed models / JSON parsing")
    lines.append(
        f"StageSpec(name={stage.name!r}, world_size={stage.topology.world_size})"
    )
    for name, members in stage.topology.groups.items():
        lines.append(f"Group {name}: ordered ranks={members}")
    for name, groups in stage.topology.families.items():
        lines.append(f"Family {name}: groups={groups}")
    lines.append(
        "Logical SP/TP families may alias actual groups; they are not extra rank axes."
    )
    for rank, p in [(None, stage.single), *sorted(stage.distributed.items())]:
        label = "single" if rank is None else f"rank{rank}"
        lines.append(f"{label}: inputs={p.inputs}, outputs={p.outputs}")
        for name, t in p.tensors.items():
            lines.append(
                f"  TensorSpec {name}: original shape={t.shape}, elements={t.numel}"
            )
        for i, op in enumerate(p.ops):
            lines.append(
                f"  OpCall {i}: {op.type} inputs={op.inputs} outputs={op.outputs} attrs={dict(op.attrs)}"
            )
        if not p.ops:
            lines.append(
                "  No operators: identity program using the declared input as output."
            )
    _section(lines, "2. Validation checks")
    lines.extend(v.checks)
    lines.append(
        "Checks include unknown fields, duplicate JSON keys, boundary references, SSA order, operator attrs/shapes,"
    )
    lines.append(
        "full rank coverage, extents, local interval coverage, disjoint target pieces and Partial image compatibility."
    )
    _section(lines, "3. Normalized input assumptions and output goals")
    for role, rels in [
        ("INPUT PREMISE", v.inputs),
        ("OUTPUT GOAL (never assumed)", v.outputs),
    ]:
        for n, rel in enumerate(rels):
            _relation(lines, rel, role, n)
    if result.reduction is not None:
        plan = result.reduction
        _section(lines, "4. Shape/index-domain reduction and its justification")
        lines.append("Method: " + plan.method)
        lines.extend(plan.notes)
        lines.append(
            f"{plan.original_output_elements} output relation elements -> {len(plan.domains)} symbolic-index domains."
        )
        lines.append(
            "Physical shape reduction: NONE. No optimizer picks smaller dimensions; original shapes are retained."
        )
        lines.append(
            "Reduction constraints are exact original index bounds, not sample-size constraints."
        )
        for n, domain in enumerate(plan.domains):
            guards = " AND ".join(
                f"0 <= goal{n}_i{d} < {size}" for d, size in enumerate(domain.shape)
            )
            lines.append(
                f"  Domain {n}: relation={domain.relation_index}, group={domain.group}: {guards}"
            )
    if result.inputs is not None:
        _section(lines, "5. Input equalities and exact elimination")
        lines.append(
            "The following universal equalities define Rin before elimination:"
        )
        lines.extend(
            f"Rin[{i}] SMT-LIB:\n{equation}"
            for i, equation in enumerate(result.inputs.equations)
        )
        lines.append("Free function symbols corresponding to original input tensors:")
        # Allocated in encode_inputs: all single inputs followed by each rank, in program order.
        i = 0
        for rank, p in [(None, stage.single), *sorted(stage.distributed.items())]:
            for name in p.inputs:
                label = "single" if rank is None else f"rank{rank}"
                lines.append(f"  input_{i}: {label}.{name}")
                i += 1
        lines.append("Substitutions applied to the local input environment:")
        lines.extend("  " + s for s in result.inputs.substitutions)
        lines.append(
            "Justification: equality inputs are determined by S and Phi. For each Partial group,"
        )
        lines.append(
            "choose every non-pivot contribution freely and define the pivot as S minus their sum."
        )
        lines.append(
            "Each original solution restricts to these free functions, and each such assignment reconstructs a solution."
        )
        lines.append(
            "Unique local preimages and disjoint input bindings were checked. Functions outside valid domains are irrelevant."
        )
        lines.append(
            "Thus Rin is enforced by exact parameterization, not dropped or replaced with sampled numeric inputs."
        )
        lines.append(
            "No extra replica equalities are inserted. Unbound program inputs remain independent free functions."
        )
    if result.execution is not None:
        _section(lines, "6. Symbolic execution and collective matching")
        lines.extend(result.execution.trace)
        lines.append(
            "Local outputs are lazy element expressions; no concrete tensor values are used."
        )
        lines.append(
            "MatMul: Out[batch,i,j] = Sum_k Left[broadcast(batch),i,k]*Right[broadcast(batch),k,j], all original k."
        )
        lines.append(
            "Add/Mul broadcast indices; Transpose permutes indices; Reshape uses row-major ravel/unravel; Expand broadcasts."
        )
        lines.append(
            "GELU uses shared_gelu(x). AllGather concatenates in actual group order; AllReduce sums;"
        )
        lines.append(
            "ReduceScatter sums then assigns contiguous axis chunks in actual group order."
        )
    _section(lines, "7. Actual solver obligations after substitution")
    for n, (obligation, outcome) in enumerate(result.queries):
        lines.append(f"Query {n}: {obligation.name}; result={outcome.status}")
        lines.append(
            "Meaning: EXISTS valid original output index where single and distributed results disagree."
        )
        lines.append("Guard: " + obligation.guard.sexpr())
        lines.append("Simplified disequality: " + obligation.violation.sexpr())
        lines.append(
            "A false disequality means exact symbolic simplification made the two expressions identical."
        )
        lines.append("BEGIN SMT-LIB (complete query)")
        lines.append(outcome.query.rstrip())
        lines.append("END SMT-LIB")
        if outcome.reason:
            lines.append("Solver reason: " + outcome.reason)
    if result.counterexample:
        _section(lines, "8. Counterexample in original-shape coordinates")
        lines.append(json.dumps(result.counterexample, indent=2))
    _section(lines, "Final result")
    lines.append(f"Verification: {result.status.value}")
    if result.status.value == "PROVED":
        lines.append(
            "Every output domain was UNSAT over original shapes, conditional on the stated input relations."
        )
    elif result.status.value == "DISPROVED":
        lines.append(
            "An original-domain counterexample exists in the declared real/UF abstraction."
        )
    else:
        lines.append(result.reason or "No complete proof conclusion.")
    return "\n".join(lines) + "\n"


def write_report(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(result), encoding="utf-8")
    return path


def demo_path(input_path):
    """Keep existing inputs in place, with one unambiguous report per case."""
    path = Path(input_path)
    directory = (
        path.parent if path.parent.name == path.stem else path.parent / path.stem
    )
    return directory / "demo.txt"
