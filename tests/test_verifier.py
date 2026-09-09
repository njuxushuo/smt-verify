from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_loader import load_stage
from src.verifier import VerificationError, VerificationStatus, verify_stage


STANDARD_FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"
WRONG_FIXTURE = ROOT / "input" / "matmul_shard_wrong_replicate.json"


def test_correct_candidate_is_proved_with_expected_constraint_counts() -> None:
    result = verify_stage(load_stage(STANDARD_FIXTURE))

    assert result.status is VerificationStatus.PROVED
    assert result.counterexample is None
    assert result.reason_unknown is None
    assert result.input_constraint_count == 8
    assert result.output_constraint_count == 1


def test_wrong_candidate_is_disproved_with_input_only_counterexample() -> None:
    result = verify_stage(load_stage(WRONG_FIXTURE))

    assert result.status is VerificationStatus.DISPROVED
    assert result.counterexample is not None
    assert set(result.counterexample.single_inputs) == {"A", "B"}
    assert set(result.counterexample.distributed_inputs) == {0, 1}
    assert set(result.counterexample.distributed_inputs[0]) == {"A0", "B0"}
    assert set(result.counterexample.distributed_inputs[1]) == {"A1", "B1"}
    assert result.counterexample.failed_output_constraints


@pytest.mark.parametrize("timeout_ms", [0, -1, True, False, "100"])
def test_invalid_timeout_is_rejected(timeout_ms: object) -> None:
    with pytest.raises(VerificationError, match="timeout_ms"):
        verify_stage(load_stage(STANDARD_FIXTURE), timeout_ms=timeout_ms)  # type: ignore[arg-type]
