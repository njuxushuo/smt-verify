from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shape.reducer import reduce_shapes
from src.stage.loader import StageInputError, load_stage
from src.symbolic.executor import execute_stage
from src.verification.relation_encoder import encode_stage_relations
from src.verification.verifier import VerificationStatus, verify_stage


BROADCAST_INPUTS = ROOT / "input" / "broadcast"
POSITIVE_FIXTURES = (
    BROADCAST_INPUTS / "add_rank_mismatch.json",
    BROADCAST_INPUTS / "add_singleton.json",
    BROADCAST_INPUTS / "mul_4d.json",
)
INVALID_FIXTURE = BROADCAST_INPUTS / "add_invalid.json"
WRONG_RELATION_FIXTURE = BROADCAST_INPUTS / "add_wrong_partial.json"


@pytest.mark.parametrize("fixture", POSITIVE_FIXTURES)
def test_broadcast_example_runs_complete_pipeline_and_is_proved(fixture: Path) -> None:
    stage = load_stage(fixture)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)

    assert encoded.input_constraints
    assert encoded.output_constraints
    assert verification.status is VerificationStatus.PROVED
    assert verification.counterexample is None


def test_singleton_example_reduction_preserves_broadcast_pattern() -> None:
    stage = load_stage(BROADCAST_INPUTS / "add_singleton.json")
    reduced = reduce_shapes(stage)

    assert reduced.single["A"][1] == 1
    assert reduced.single["B"][0] == 1
    assert reduced.single["A"][2] == reduced.single["B"][2]
    for rank in range(stage.world_size):
        assert reduced.distributed[rank][f"A{rank}"][1] == 1
        assert reduced.distributed[rank][f"B{rank}"][0] == 1
        assert (
            reduced.distributed[rank][f"A{rank}"][2]
            == reduced.distributed[rank][f"B{rank}"][2]
        )


def test_invalid_broadcast_example_fails_during_load_validation() -> None:
    with pytest.raises(StageInputError, match="not broadcast-compatible"):
        load_stage(INVALID_FIXTURE)


def test_wrong_relation_example_runs_complete_pipeline_and_is_disproved() -> None:
    stage = load_stage(WRONG_RELATION_FIXTURE)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)

    assert encoded.input_constraints
    assert encoded.output_constraints
    assert verification.status is VerificationStatus.DISPROVED
    assert verification.counterexample is not None
    assert verification.counterexample.failed_output_constraints
