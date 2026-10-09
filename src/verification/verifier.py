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


def verify_stage(
    stage, registry, *, timeout_ms=30000, max_contraction=8192, max_symbolic_ops=10000
):
    if type(timeout_ms) is not int or timeout_ms < 1:
        raise InvalidInput("timeout_ms must be a positive integer")
    result = VerificationResult(stage.name, Status.UNKNOWN)
    try:
        result.validated = validate_stage(stage, registry)
        check_budget(stage, max_contraction, max_symbolic_ops)
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
    except InvalidInput as e:
        return VerificationResult(str(path), Status.INVALID_INPUT, reason=str(e))
    except Unsupported as e:
        return VerificationResult(str(path), Status.UNSUPPORTED, reason=str(e))
    return verify_stage(stage, registry, **options)
