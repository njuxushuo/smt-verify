from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_loader import StageInputError, load_stage


FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"


def _fixture_data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _load_data(tmp_path: Path, data: dict):
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_stage(path)


def test_loads_standard_matmul_shard_to_partial_case():
    stage = load_stage(FIXTURE)

    assert stage.name == "matmul_shard_to_partial"
    assert stage.world_size == 2
    assert sorted(stage.distributed) == [0, 1]
    assert len(stage.input_relations) == 2
    assert stage.input_relations[0].type == "shard"
    assert stage.input_relations[0].dim == 1
    assert stage.input_relations[1].type == "shard"
    assert stage.input_relations[1].dim == 0
    assert stage.output_relation.type == "partial"
    assert stage.output_relation.reduce_op == "sum"


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

    with pytest.raises(StageInputError, match="shard tensor on rank 0"):
        _load_data(tmp_path, data)


def test_rejects_non_divisible_shard_dimension(tmp_path: Path):
    data = _fixture_data()
    data["single"]["tensors"]["A"]["shape"] = [4, 7]

    with pytest.raises(StageInputError, match="not divisible by world_size"):
        _load_data(tmp_path, data)


def test_rejects_partial_shape_mismatch(tmp_path: Path):
    data = _fixture_data()
    data["distributed"]["ranks"]["1"]["tensors"]["C1"]["shape"] = [2, 4]

    with pytest.raises(StageInputError, match="partial tensor on rank 1"):
        _load_data(tmp_path, data)


def test_rejects_replicate_shape_mismatch(tmp_path: Path):
    data = _fixture_data()
    data["input_relations"][0] = {
        "single_tensor": "A",
        "distributed_tensors": ["A0", "A1"],
        "type": "replicate",
    }

    with pytest.raises(StageInputError, match="replicate tensor on rank 0"):
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
