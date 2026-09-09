"""End-to-end local Stage verification over reduced symbolic shapes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import z3

from .relation_encoder import EncodedRelations, encode_stage_relations
from .shape_model import ReducedShapeResult
from .shape_reducer import reduce_shapes
from .stage_model import StageSpec
from .symbolic_executor import execute_stage
from .symbolic_tensor import SymbolicStageResult


class VerificationStatus(str, Enum):
    """The three possible outcomes of the Stage proof query."""

    PROVED = "PROVED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Counterexample:
    """A model restricted to free variables of true program inputs."""

    single_inputs: dict[str, tuple[str, ...]]
    distributed_inputs: dict[int, dict[str, tuple[str, ...]]]
    failed_output_constraints: tuple[int, ...]


@dataclass(frozen=True)
class VerificationResult:
    """The result of reducing, executing, encoding, and checking one Stage."""

    stage_name: str
    status: VerificationStatus
    reduced_shapes: ReducedShapeResult
    input_constraint_count: int
    output_constraint_count: int
    counterexample: Counterexample | None = None
    reason_unknown: str | None = None


class VerificationError(ValueError):
    """Raised when the verifier cannot construct its fixed proof query."""


def _validate_timeout(timeout_ms: int | None) -> None:
    if timeout_ms is None:
        return
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise VerificationError("timeout_ms must be a positive integer or None")


def _serialize_input_tensors(
    symbolic: SymbolicStageResult,
    model: z3.ModelRef,
) -> tuple[dict[str, tuple[str, ...]], dict[int, dict[str, tuple[str, ...]]]]:
    """Serialize exactly the free tensors identified as program inputs."""

    single_inputs = {
        name: tuple(
            str(model.eval(value, model_completion=True))
            for value in symbolic.single.tensors[name].values
        )
        for name in symbolic.single.input_names
    }
    distributed_inputs = {
        rank: {
            name: tuple(
                str(model.eval(value, model_completion=True))
                for value in program.tensors[name].values
            )
            for name in program.input_names
        }
        for rank, program in symbolic.distributed.items()
    }
    return single_inputs, distributed_inputs


def _extract_counterexample(
    symbolic: SymbolicStageResult,
    encoded: EncodedRelations,
    model: z3.ModelRef,
) -> Counterexample:
    single_inputs, distributed_inputs = _serialize_input_tensors(symbolic, model)
    failed = tuple(
        index
        for index, constraint in enumerate(encoded.output_constraints)
        if z3.is_false(model.eval(constraint, model_completion=True))
    )
    return Counterexample(
        single_inputs=single_inputs,
        distributed_inputs=distributed_inputs,
        failed_output_constraints=failed,
    )


def verify_stage(
    stage: StageSpec,
    *,
    timeout_ms: int | None = None,
) -> VerificationResult:
    """Check the fixed query ``Rin ∧ ¬And(Rout)`` for one reduced Stage."""

    _validate_timeout(timeout_ms)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    if not encoded.output_constraints:
        raise VerificationError("stage output relation produced no constraints")

    solver = z3.Solver()
    if timeout_ms is not None:
        solver.set(timeout=timeout_ms)
    solver.add(*encoded.input_constraints)
    solver.add(z3.Not(z3.And(*encoded.output_constraints)))
    outcome = solver.check()

    common = {
        "stage_name": stage.name,
        "reduced_shapes": reduced,
        "input_constraint_count": len(encoded.input_constraints),
        "output_constraint_count": len(encoded.output_constraints),
    }
    if outcome == z3.unsat:
        return VerificationResult(status=VerificationStatus.PROVED, **common)
    if outcome == z3.sat:
        return VerificationResult(
            status=VerificationStatus.DISPROVED,
            counterexample=_extract_counterexample(symbolic, encoded, solver.model()),
            **common,
        )
    return VerificationResult(
        status=VerificationStatus.UNKNOWN,
        reason_unknown=solver.reason_unknown(),
        **common,
    )
