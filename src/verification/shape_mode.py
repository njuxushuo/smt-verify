"""Optional candidate proof followed by a separate original-shape certificate."""

from dataclasses import dataclass

import z3

from ..core.errors import InvalidInput, Unsupported
from ..relation.encoding import encode_inputs, output_equality
from ..reduction.reducer import reduce_domains
from ..reduction.rules import check_budget
from ..reduction.shapes import reduce_shapes
from ..smt.blocks import BlockContext
from ..smt.elements import ElementContext, coordinates
from ..smt.solver import check
from ..stage.validation import validate_stage
from .collectives import validate_schedule
from .executor import execute_stage
from .obligation import ProofObligation, build_obligation
from .result import Status, VerificationResult


@dataclass
class ShapeAttempt:
    plan: object = None
    reduced: object = None
    certificate: object = None
    fallback_reason: str | None = None
    block_definitions: tuple = ()


def element_proof(stage, registry, timeout_ms):
    result = VerificationResult(stage.name, Status.UNKNOWN, scope="REDUCED_SHAPES_ONLY")
    result.validated = validate_stage(stage, registry)
    context = ElementContext()
    result.inputs = encode_inputs(
        stage.single, stage.distributed, result.validated.inputs, context
    )
    result.execution = execute_stage(stage, result.inputs.tensors, registry, context)
    domains = reduce_domains(result.validated.outputs).domains
    for domain in domains:
        relation = result.validated.outputs[domain.relation_index]
        violations = []
        for index in coordinates(domain.shape):
            left, right = output_equality(
                relation, domain.group, index, result.execution.tensors
            )
            violations.append(left != right)
        violation = z3.simplify(z3.Or(*violations))
        obligation = ProofObligation(
            f"finite output[{domain.relation_index}] group={domain.group}",
            domain,
            (),
            z3.BoolVal(True),
            None,
            None,
            violation,
        )
        outcome = check((), violation, timeout_ms)
        result.queries.append((obligation, outcome))
        if outcome.status != "UNSAT":
            result.status = (
                Status.DISPROVED if outcome.status == "SAT" else Status.UNKNOWN
            )
            result.reason = (
                "Reduced-instance result only; original confirmation required."
            )
            return result
    result.status = Status.PROVED
    return result


def certify_original(
    stage, registry, timeout_ms, width, max_contraction, max_symbolic_ops
):
    result = VerificationResult(stage.name, Status.UNKNOWN)
    result.validated = validate_stage(stage, registry)
    result.reduction = reduce_domains(result.validated.outputs)
    context = BlockContext(width)
    check_budget(
        stage,
        max_contraction,
        max_symbolic_ops,
        registry=registry,
        contraction_terms=context.contraction_terms,
    )
    result.inputs = encode_inputs(
        stage.single, stage.distributed, result.validated.inputs, context
    )
    result.execution = execute_stage(stage, result.inputs.tensors, registry, context)
    for n, domain in enumerate(result.reduction.domains):
        obligation = build_obligation(
            n,
            domain,
            result.validated.outputs[domain.relation_index],
            result.execution.tensors,
        )
        outcome = check((obligation.guard,), obligation.violation, timeout_ms)
        result.queries.append((obligation, outcome))
        if outcome.status != "UNSAT":
            result.reason = "Original block-sum certificate inconclusive; abstract SAT is not a counterexample."
            return result, context
    result.status = Status.PROVED
    return result, context


def verify_reduced(
    stage,
    registry,
    *,
    timeout_ms,
    max_elements,
    max_contraction,
    max_symbolic_ops,
    exact,
):
    attempt = ShapeAttempt()
    original = validate_stage(stage, registry)
    events = validate_schedule(stage, registry)
    try:
        if (
            sum(len(p.ops) for p in (stage.single, *stage.distributed.values()))
            > max_symbolic_ops
        ):
            raise Unsupported("stage exceeds symbolic operation budget")
        attempt.plan = reduce_shapes(stage, registry, events, timeout_ms, max_elements)
        check_budget(
            attempt.plan.stage, max_contraction, max_symbolic_ops, registry=registry
        )
        attempt.reduced = element_proof(attempt.plan.stage, registry, timeout_ms)
        if attempt.reduced.status == Status.PROVED:
            certificate, context = certify_original(
                stage,
                registry,
                timeout_ms,
                attempt.plan.block_size,
                max_contraction,
                max_symbolic_ops,
            )
            attempt.certificate = certificate
            attempt.block_definitions = tuple(
                (str(atom), width, body)
                for (width, body), atom in context.atoms.items()
            )
            if certificate.status == Status.PROVED:
                certificate.shape_attempt = attempt
                certificate.mode = "SHAPE_REDUCED_ELEMENTWISE"
                # Avoid a recursive result/report graph.
                attempt.certificate = None
                return certificate
            attempt.fallback_reason = certificate.reason
        else:
            attempt.fallback_reason = attempt.reduced.reason
    except Unsupported as exc:
        attempt.fallback_reason = str(exc)
    except InvalidInput as exc:
        # The original already passed validation; do not mislabel a generated bug.
        result = VerificationResult(stage.name, Status.UNKNOWN)
        result.validated = original
        result.reason = f"Internal shape candidate/certificate invariant failure: {exc}"
        result.mode = "SHAPE_REDUCTION_ERROR"
        result.shape_attempt = attempt
        return result
    result = exact()
    result.shape_attempt = attempt
    result.mode = "SYMBOLIC_INDEX_FALLBACK"
    return result
