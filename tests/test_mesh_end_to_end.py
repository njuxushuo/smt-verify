from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shape.reducer import reduce_shapes
from src.lemma import certify_stage
from src.stage.loader import load_stage
from src.symbolic.executor import execute_stage
from src.verification.relation_encoder import encode_stage_relations
from src.verification.verifier import VerificationStatus, verify_stage


MESH_INPUTS = ROOT / "input" / "mesh"
PROVED_FIXTURES = (
    MESH_INPUTS
    / "mesh_replicate_shard_transpose_proved"
    / "mesh_replicate_shard_transpose_proved.json",
    MESH_INPUTS
    / "mesh_shard_shard_transpose_proved"
    / "mesh_shard_shard_transpose_proved.json",
    MESH_INPUTS
    / "mesh_shard_partial_transpose_proved"
    / "mesh_shard_partial_transpose_proved.json",
)
WRONG_FIXTURE = (
    MESH_INPUTS
    / "mesh_shard_shard_wrong_output_disproved"
    / "mesh_shard_shard_wrong_output_disproved.json"
)


@pytest.mark.parametrize("fixture", PROVED_FIXTURES)
def test_mesh_example_runs_complete_pipeline_and_is_proved(fixture: Path) -> None:
    stage = load_stage(fixture)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)
    certification = certify_stage(stage)

    assert stage.mesh.shape == (2, 2)
    assert encoded.input_constraints
    assert encoded.output_constraints
    assert verification.status is VerificationStatus.PROVED
    assert verification.counterexample is None
    assert certification.lemma is not None
    assert certification.lemma.mesh == stage.mesh


def test_shape_valid_wrong_mesh_mapping_reaches_smt_and_is_disproved() -> None:
    stage = load_stage(WRONG_FIXTURE)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)
    certification = certify_stage(stage)

    assert reduced.single == {"X": (2, 2), "Y": (2, 2)}
    for rank in range(4):
        assert reduced.distributed[rank] == {
            f"X{rank}": (1, 1),
            f"Y{rank}": (1, 1),
        }
    assert encoded.input_constraints
    assert encoded.output_constraints
    assert verification.status is VerificationStatus.DISPROVED
    assert verification.counterexample is not None
    assert verification.counterexample.failed_output_constraints
    assert certification.lemma is None
