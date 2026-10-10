from ..core.errors import InvalidInput, Unsupported
from ..relation.encoding import encode_inputs
from ..reduction.reducer import reduce_domains
from ..reduction.rules import check_budget
from ..smt.context import SymbolicContext
from ..smt.solver import check
from ..stage.codec import load_stage
from ..stage.validation import validate_stage
from .counterexample import extract
from .executor import execute_stage
from .obligation import build_obligation
from .result import Status, VerificationResult
from .shape_mode import verify_reduced


def verify_stage(
    stage,
    registry,
    *,
    timeout_ms=30000,
    max_contraction=8192,
    max_symbolic_ops=10000,
    shape_reduce=False,
    max_reduced_elements=50000,
):
    if type(timeout_ms) is not int or timeout_ms < 1:
        raise InvalidInput("timeout_ms must be a positive integer")
    if (
        type(shape_reduce) is not bool
        or type(max_reduced_elements) is not int
        or max_reduced_elements < 1
    ):
        raise InvalidInput("shape_reduce must be boolean and element budget positive")
    result = VerificationResult(stage.name, Status.UNKNOWN)
    if shape_reduce:
        result.mode = "SHAPE_REDUCTION_NOT_STARTED"
    try:
        if shape_reduce:
            return verify_reduced(
                stage,
                registry,
                timeout_ms=timeout_ms,
                max_elements=max_reduced_elements,
                max_contraction=max_contraction,
                max_symbolic_ops=max_symbolic_ops,
                exact=lambda: verify_stage(
                    stage,
                    registry,
                    timeout_ms=timeout_ms,
                    max_contraction=max_contraction,
                    max_symbolic_ops=max_symbolic_ops,
                ),
            )
        result.validated = validate_stage(stage, registry)
        check_budget(stage, max_contraction, max_symbolic_ops, registry=registry)
        result.reduction = reduce_domains(result.validated.outputs)
        context = SymbolicContext()
        result.inputs = encode_inputs(
            stage.single, stage.distributed, result.validated.inputs, context
        )
        result.execution = execute_stage(
            stage, result.inputs.tensors, registry, context
        )
        unknown = []
        for n, domain in enumerate(result.reduction.domains):
            obligation = build_obligation(
                n,
                domain,
                result.validated.outputs[domain.relation_index],
                result.execution.tensors,
            )
            outcome = check((obligation.guard,), obligation.violation, timeout_ms)
            result.queries.append((obligation, outcome))
            if outcome.status == "SAT":
                result.status = Status.DISPROVED
                result.counterexample = extract(
                    obligation, outcome.model, context, result.inputs
                )
                return result
            if outcome.status == "UNKNOWN":
                unknown.append(outcome.reason)
        result.status = Status.UNKNOWN if unknown else Status.PROVED
        result.reason = "; ".join(unknown) if unknown else None
    except InvalidInput as e:
        result.status = Status.INVALID_INPUT
        result.reason = str(e)
    except Unsupported as e:
        result.status = Status.UNSUPPORTED
        result.reason = str(e)
    except RecursionError:
        result.status = Status.UNSUPPORTED
        result.reason = (
            "symbolic dependency expansion exceeded the interpreter recursion budget"
        )
    return result


def verify_file(path, registry, **options):
    try:
        stage = load_stage(path)
    except (InvalidInput, Unsupported) as e:
        status = (
            Status.INVALID_INPUT if isinstance(e, InvalidInput) else Status.UNSUPPORTED
        )
        result = VerificationResult(str(path), status, reason=str(e))
        if options.get("shape_reduce") is True:
            result.mode = "SHAPE_REDUCTION_NOT_STARTED"
        return result
    return verify_stage(stage, registry, **options)
