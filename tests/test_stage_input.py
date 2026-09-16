from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage.loader import StageInputError, load_stage


FIXTURE = ROOT / "input" / "matmul_shard_to_partial" / "matmul_shard_to_partial.json"


def _fixture_data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _load_data(tmp_path: Path, data: dict):
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_stage(path)


def _mesh_relation_case() -> dict:
    return {
        "name": "mesh_relation_case",
        "world_size": 4,
        "mesh": {"shape": [2, 2]},
        "single": {"tensors": {"X": {"shape": [4, 6]}}, "ops": []},
        "distributed": {
            "ranks": {
                str(rank): {
                    "tensors": {f"X{rank}": {"shape": [2, 3]}},
                    "ops": [],
                }
                for rank in range(4)
            }
        },
        "input_relations": [],
        "output_relation": {
            "single_tensor": "X",
            "distributed_tensors": [f"X{rank}" for rank in range(4)],
            "placements": [
                {"type": "shard", "dim": 0},
                {"type": "shard", "dim": 1},
            ],
        },
    }


def test_loads_standard_matmul_shard_to_partial_case():
    stage = load_stage(FIXTURE)

    assert stage.name == "matmul_shard_to_partial"
    assert stage.single.ops[0].attrs == {}
    assert all(program.ops[0].attrs == {} for program in stage.distributed.values())
    assert stage.world_size == 2
    assert stage.mesh.shape == (2,)
    assert sorted(stage.distributed) == [0, 1]
    assert len(stage.input_relations) == 2
    assert stage.input_relations[0].placements[0].type == "shard"
    assert stage.input_relations[0].placements[0].dim == 1
    assert stage.input_relations[1].placements[0].type == "shard"
    assert stage.input_relations[1].placements[0].dim == 0
    assert stage.output_relation.placements[0].type == "partial"
    assert stage.output_relation.placements[0].reduce_op == "sum"


def test_rejects_non_object_operator_attrs(tmp_path: Path) -> None:
    data = _fixture_data()
    data["single"]["ops"][0]["attrs"] = []

    with pytest.raises(StageInputError, match=r"single\.ops\[0\]\.attrs: expected object"):
        _load_data(tmp_path, data)


def test_rejects_world_size_rank_mismatch(tmp_path: Path):
    data = _fixture_data()
    del data["distributed"]["ranks"]["1"]

    with pytest.raises(StageInputError, match="rank count mismatch"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("invalid_dim", [0, -1])
def test_rejects_non_positive_shape_dimension(tmp_path: Path, invalid_dim: int):
    data = _fixture_data()
    data["single"]["tensors"]["A"]["shape"][0] = invalid_dim

    with pytest.raises(StageInputError, match="dimension must be greater than 0"):
        _load_data(tmp_path, data)


def test_rejects_unknown_operator_tensor_reference(tmp_path: Path):
    data = _fixture_data()
    data["single"]["ops"][0]["inputs"][1] = "missing"

    with pytest.raises(StageInputError, match="unknown tensor"):
        _load_data(tmp_path, data)


def test_rejects_unsupported_operator(tmp_path: Path):
    data = _fixture_data()
    data["single"]["ops"][0]["type"] = "relu"

    with pytest.raises(StageInputError, match="unsupported operator"):
        _load_data(tmp_path, data)


def test_rejects_shard_dim_out_of_range(tmp_path: Path):
    data = _fixture_data()
    data["input_relations"][0]["dim"] = 2

    with pytest.raises(StageInputError, match="shard dim 2 is out of range"):
        _load_data(tmp_path, data)


def test_rejects_incorrect_shard_local_shape(tmp_path: Path):
    data = _fixture_data()
    data["distributed"]["ranks"]["0"]["tensors"]["A0"]["shape"] = [4, 5]

    with pytest.raises(StageInputError, match="local tensor on rank 0"):
        _load_data(tmp_path, data)


def test_rejects_non_divisible_shard_dimension(tmp_path: Path):
    data = _fixture_data()
    data["single"]["tensors"]["A"]["shape"] = [4, 7]

    with pytest.raises(StageInputError, match="not divisible by shard factor"):
        _load_data(tmp_path, data)


def test_rejects_partial_shape_mismatch(tmp_path: Path):
    data = _fixture_data()
    data["distributed"]["ranks"]["1"]["tensors"]["C1"]["shape"] = [2, 4]

    with pytest.raises(StageInputError, match="local tensor on rank 1"):
        _load_data(tmp_path, data)


def test_rejects_replicate_shape_mismatch(tmp_path: Path):
    data = _fixture_data()
    data["input_relations"][0] = {
        "single_tensor": "A",
        "distributed_tensors": ["A0", "A1"],
        "type": "replicate",
    }

    with pytest.raises(StageInputError, match="local tensor on rank 0"):
        _load_data(tmp_path, data)


def test_rejects_distributed_tensor_rank_reference_error(tmp_path: Path):
    data = _fixture_data()
    data["input_relations"][0]["distributed_tensors"] = ["A1", "A0"]

    with pytest.raises(StageInputError, match="does not exist on rank 0"):
        _load_data(tmp_path, data)


def test_rejects_duplicate_input_relation_for_single_tensor(tmp_path: Path):
    data = _fixture_data()
    data["input_relations"].append(deepcopy(data["input_relations"][0]))

    with pytest.raises(StageInputError, match="duplicate input relation"):
        _load_data(tmp_path, data)


def test_loads_explicit_multidimensional_mesh_and_placements(tmp_path: Path) -> None:
    stage = _load_data(tmp_path, _mesh_relation_case())

    assert stage.mesh.shape == (2, 2)
    assert tuple(placement.type for placement in stage.output_relation.placements) == (
        "shard",
        "shard",
    )
    assert tuple(placement.dim for placement in stage.output_relation.placements) == (0, 1)


def test_rejects_mesh_size_world_size_mismatch(tmp_path: Path) -> None:
    data = _mesh_relation_case()
    data["mesh"]["shape"] = [2, 3]

    with pytest.raises(StageInputError, match="does not match world_size"):
        _load_data(tmp_path, data)


def test_rejects_multid_mesh_legacy_relation_shorthand(tmp_path: Path) -> None:
    data = _mesh_relation_case()
    data["output_relation"].pop("placements")
    data["output_relation"]["type"] = "shard"
    data["output_relation"]["dim"] = 0

    with pytest.raises(StageInputError, match="must define placements"):
        _load_data(tmp_path, data)


def test_rejects_mixed_legacy_and_canonical_relation_fields(tmp_path: Path) -> None:
    data = _mesh_relation_case()
    data["output_relation"]["type"] = "shard"

    with pytest.raises(StageInputError, match="cannot be combined"):
        _load_data(tmp_path, data)


def test_rejects_placement_count_different_from_mesh_rank(tmp_path: Path) -> None:
    data = _mesh_relation_case()
    data["output_relation"]["placements"].pop()

    with pytest.raises(StageInputError, match="expected 2 placements"):
        _load_data(tmp_path, data)


def test_rejects_repeated_same_tensor_dimension_sharding(tmp_path: Path) -> None:
    data = _mesh_relation_case()
    data["output_relation"]["placements"][1]["dim"] = 0
    for rank in range(4):
        data["distributed"]["ranks"][str(rank)]["tensors"][f"X{rank}"]["shape"] = [1, 6]

    with pytest.raises(StageInputError, match="sharded by multiple mesh axes"):
        _load_data(tmp_path, data)


def test_rejects_unknown_placement_fields(tmp_path: Path) -> None:
    data = _mesh_relation_case()
    data["output_relation"]["placements"][0]["axis_name"] = "dp"

    with pytest.raises(StageInputError, match="unknown placement fields"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("shape", [[], [2, 0], [2, True]])
def test_rejects_invalid_mesh_shape(tmp_path: Path, shape: list[object]) -> None:
    data = _mesh_relation_case()
    data["mesh"]["shape"] = shape

    with pytest.raises(StageInputError, match="mesh shape"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    ("placement", "message"),
    [
        ({"type": "unknown"}, "unsupported placement"),
        ({"type": "replicate", "dim": 0}, "must not define"),
        ({"type": "shard", "dim": True}, "dim must be an integer"),
        ({"type": "partial", "reduce_op": "max"}, "must be 'sum'"),
    ],
)
def test_rejects_invalid_placement_schema(
    tmp_path: Path, placement: dict[str, object], message: str
) -> None:
    data = _mesh_relation_case()
    data["output_relation"]["placements"][0] = placement

    with pytest.raises(StageInputError, match=message):
        _load_data(tmp_path, data)
