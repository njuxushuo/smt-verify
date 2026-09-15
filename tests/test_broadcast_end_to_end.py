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
    BROADCAST_INPUTS / "add_rank_mismatch_shard_replicate_to_shard.json",
    BROADCAST_INPUTS / "add_singleton_shard_replicate_to_shard.json",
    BROADCAST_INPUTS / "mul_4d_shard_replicate_to_shard.json",
    BROADCAST_INPUTS / "mul_4d_shard_shard_to_shard.json",
)
INVALID_FIXTURE = BROADCAST_INPUTS / "add_incompatible_shapes_rejected.json"
WRONG_RELATION_FIXTURE = (
    BROADCAST_INPUTS / "add_replicate_inputs_wrong_partial_disproved.json"
)
SHARDED_BIAS_FIXTURE = BROADCAST_INPUTS / "add_hidden_sharded_bias_to_shard.json"


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
    stage = load_stage(BROADCAST_INPUTS / "add_singleton_shard_replicate_to_shard.json")
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


def test_sharded_bias_broadcast_runs_complete_pipeline_and_is_proved() -> None:
    stage = load_stage(SHARDED_BIAS_FIXTURE)

    assert stage.single.tensors["A"].shape == (2, 3, 8)
    assert stage.single.tensors["B"].shape == (8,)
    assert stage.single.tensors["C"].shape == (2, 3, 8)
    for rank in range(stage.world_size):
        assert stage.distributed[rank].tensors[f"A{rank}"].shape == (2, 3, 4)
        assert stage.distributed[rank].tensors[f"B{rank}"].shape == (4,)
        assert stage.distributed[rank].tensors[f"C{rank}"].shape == (2, 3, 4)

    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)

    for rank in range(stage.world_size):
        local = reduced.distributed[rank]
        assert reduced.single["A"][2] == stage.world_size * local[f"A{rank}"][2]
        assert reduced.single["B"][0] == stage.world_size * local[f"B{rank}"][0]
        assert reduced.single["C"][2] == stage.world_size * local[f"C{rank}"][2]
        assert local[f"A{rank}"][2] == local[f"B{rank}"][0]
        assert local[f"C{rank}"][2] == local[f"A{rank}"][2]

    assert encoded.input_constraints
    assert encoded.output_constraints
    assert verification.status is VerificationStatus.PROVED
    assert verification.counterexample is None


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
