from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shape.reducer import reduce_shapes
from src.stage.loader import load_stage
from src.symbolic.executor import execute_stage
from src.verification.relation_encoder import encode_stage_relations
from src.verification.verifier import VerificationStatus, verify_stage


VIEW_INPUTS = ROOT / "input" / "view"
POSITIVE_FIXTURES = (
    VIEW_INPUTS / "transpose_shard_axis_1_to_2_proved.json",
    VIEW_INPUTS / "transpose_negative_axes_proved.json",
    VIEW_INPUTS / "reshape_shard_dim_0_proved.json",
    VIEW_INPUTS / "squeeze_shard_dim_0_proved.json",
    VIEW_INPUTS / "unsqueeze_shard_dim_0_proved.json",
    VIEW_INPUTS / "expand_shard_dim_0_proved.json",
)
WRONG_TRANSPOSE_FIXTURE = VIEW_INPUTS / "transpose_wrong_shard_axis_disproved.json"


@pytest.mark.parametrize("fixture", POSITIVE_FIXTURES)
def test_view_example_runs_complete_pipeline_and_is_proved(fixture: Path) -> None:
    stage = load_stage(fixture)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)

    assert encoded.input_constraints
    assert encoded.output_constraints
    assert verification.status is VerificationStatus.PROVED
    assert verification.counterexample is None


def test_wrong_transpose_shard_axis_reaches_smt_and_is_disproved() -> None:
    stage = load_stage(WRONG_TRANSPOSE_FIXTURE)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)

    assert encoded.input_constraints
    assert encoded.output_constraints
    assert verification.status is VerificationStatus.DISPROVED
    assert verification.counterexample is not None
    assert verification.counterexample.failed_output_constraints
