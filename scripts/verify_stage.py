"""Verify one Stage candidate relation with the local SMT verifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.relation_encoder import RelationEncodingError
from src.shape_model import ShapeReductionError
from src.stage_loader import StageInputError, load_stage
from src.symbolic_tensor import SymbolicExecutionError
from src.verifier import VerificationError, VerificationResult, VerificationStatus, verify_stage


def _print_counterexample(result: VerificationResult) -> None:
    assert result.counterexample is not None
    counterexample = result.counterexample
    print("Counterexample:")
    print("  Single inputs:")
    for name, values in counterexample.single_inputs.items():
        print(f"    {name}: {values}")
    print("  Distributed inputs:")
    for rank, inputs in counterexample.distributed_inputs.items():
        print(f"    Rank {rank}:")
        for name, values in inputs.items():
            print(f"      {name}: {values}")
    print(f"  Failed output constraints: {counterexample.failed_output_constraints}")


def _print_result(result: VerificationResult) -> None:
    print(f"Stage: {result.stage_name}")
    print("Verification mode: reduced-shape with supported-fragment lifting")
    print(f"Reduced objective: {result.reduced_shapes.objective_value}")
    print(f"Input constraints: {result.input_constraint_count}")
    print(f"Output constraints: {result.output_constraint_count}")
    smt_result = {
        VerificationStatus.PROVED: "UNSAT",
        VerificationStatus.DISPROVED: "SAT",
        VerificationStatus.UNKNOWN: "UNKNOWN",
    }[result.status]
    print(f"SMT result: {smt_result}")
    print(f"Verification: {result.status.value}")
    if result.status is VerificationStatus.DISPROVED:
        _print_counterexample(result)
    elif result.status is VerificationStatus.UNKNOWN:
        print(f"Reason unknown: {result.reason_unknown}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_json", help="path to a validated Stage JSON input")
    parser.add_argument("--timeout-ms", type=int, default=None, help="positive Z3 timeout in ms")
    arguments = parser.parse_args(argv)
    try:
        stage = load_stage(arguments.stage_json)
        _print_result(verify_stage(stage, timeout_ms=arguments.timeout_ms))
    except (
        StageInputError,
        ShapeReductionError,
        SymbolicExecutionError,
        RelationEncodingError,
        VerificationError,
    ) as exc:
        print("Verification: FAIL")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
