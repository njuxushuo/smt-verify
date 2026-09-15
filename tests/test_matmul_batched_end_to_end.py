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


MATMUL_INPUTS = ROOT / "input" / "matmul_batched"
POSITIVE_FIXTURES = (
    MATMUL_INPUTS / "matmul_3d_2d_shard_replicate_to_shard.json",
    MATMUL_INPUTS / "matmul_4d_shard_shard_to_shard.json",
    MATMUL_INPUTS / "matmul_batch_broadcast_shard_replicate_to_shard.json",
    MATMUL_INPUTS / "matmul_batched_contraction_shards_to_partial.json",
)
INVALID_FIXTURE = MATMUL_INPUTS / "matmul_invalid_batch_broadcast_rejected.json"
WRONG_RELATION_FIXTURE = (
    MATMUL_INPUTS
    / "matmul_batched_contraction_shards_wrong_replicate_disproved.json"
)


@pytest.mark.parametrize("fixture", POSITIVE_FIXTURES)
def test_batched_matmul_example_runs_complete_pipeline_and_is_proved(
    fixture: Path,
) -> None:
    stage = load_stage(fixture)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)

    assert encoded.input_constraints
    assert encoded.output_constraints
    assert verification.status is VerificationStatus.PROVED
    assert verification.counterexample is None


def test_batched_matmul_contraction_shards_retain_local_k_relation() -> None:
    stage = load_stage(
        MATMUL_INPUTS / "matmul_batched_contraction_shards_to_partial.json"
    )
    reduced = reduce_shapes(stage)

    for rank in range(stage.world_size):
        local = reduced.distributed[rank]
        assert reduced.single["A"][-1] == stage.world_size * local[f"A{rank}"][-1]
        assert reduced.single["B"][-2] == stage.world_size * local[f"B{rank}"][-2]
        assert local[f"A{rank}"][-1] == local[f"B{rank}"][-2]


def test_invalid_batch_broadcast_example_fails_during_load_validation() -> None:
    with pytest.raises(StageInputError, match="not broadcast-compatible"):
        load_stage(INVALID_FIXTURE)


def test_wrong_batched_matmul_relation_is_disproved() -> None:
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
