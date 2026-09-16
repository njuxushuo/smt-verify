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
    VIEW_INPUTS / "transpose_shard_axis_1_to_2_proved" / "transpose_shard_axis_1_to_2_proved.json",
    VIEW_INPUTS / "transpose_negative_axes_proved" / "transpose_negative_axes_proved.json",
    VIEW_INPUTS / "reshape_shard_dim_0_proved" / "reshape_shard_dim_0_proved.json",
    VIEW_INPUTS / "squeeze_shard_dim_0_proved" / "squeeze_shard_dim_0_proved.json",
    VIEW_INPUTS / "unsqueeze_shard_dim_0_proved" / "unsqueeze_shard_dim_0_proved.json",
    VIEW_INPUTS / "expand_shard_dim_0_proved" / "expand_shard_dim_0_proved.json",
)
WRONG_TRANSPOSE_FIXTURE = VIEW_INPUTS / "transpose_wrong_shard_axis_disproved" / "transpose_wrong_shard_axis_disproved.json"
EXPAND_ON_SHARD_AXIS_FIXTURE = (
    VIEW_INPUTS
    / "expand_on_shard_axis_wrong_replicate_disproved"
    / "expand_on_shard_axis_wrong_replicate_disproved.json"
)


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


def test_expand_on_shard_axis_reaches_smt_and_disproves_replicate() -> None:
    stage = load_stage(EXPAND_ON_SHARD_AXIS_FIXTURE)

    assert stage.single.tensors["X"].shape == (4, 2, 6)
    assert stage.single.tensors["Y"].shape == (4, 2, 6)
    assert stage.input_relations[0].placements[0].type == "shard"
    assert stage.input_relations[0].placements[0].dim == 1
    assert stage.output_relation.placements[0].type == "replicate"
    for rank in range(stage.world_size):
        assert stage.distributed[rank].tensors[f"X{rank}"].shape == (4, 1, 6)
        assert stage.distributed[rank].tensors[f"Y{rank}"].shape == (4, 2, 6)

    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)

    assert reduced.single["X"] == (1, 2, 1)
    assert reduced.single["Y"] == (1, 2, 1)
    assert reduced.objective_value == 22
    for rank in range(stage.world_size):
        assert reduced.distributed[rank][f"X{rank}"] == (1, 1, 1)
        assert reduced.distributed[rank][f"Y{rank}"] == (1, 2, 1)
    assert encoded.input_constraints
    assert encoded.output_constraints
    assert len(encoded.input_constraints) == 2
    assert len(encoded.output_constraints) == 4
    assert verification.status is VerificationStatus.DISPROVED
    assert verification.counterexample is not None
    assert verification.counterexample.failed_output_constraints
