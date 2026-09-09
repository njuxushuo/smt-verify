from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_loader import StageInputError, load_stage


STANDARD_FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"


def _write_and_load(tmp_path: Path, data: dict):
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_stage(path)


def _add_stage_data(input_relation: dict | None = None, ops: list[dict] | None = None) -> dict:
    operation_list = (
        [{"type": "add", "inputs": ["A", "B"], "outputs": ["C"]}]
        if ops is None
        else ops
    )
    rank_operations = [
        {
            "type": operation["type"],
            "inputs": [f"{name}0" for name in operation["inputs"]],
            "outputs": [f"{name}0" for name in operation["outputs"]],
        }
        for operation in operation_list
    ]
    rank_one_operations = [
        {
            "type": operation["type"],
            "inputs": [f"{name}1" for name in operation["inputs"]],
            "outputs": [f"{name}1" for name in operation["outputs"]],
        }
        for operation in operation_list
    ]
    return {
        "name": "boundary_case",
        "world_size": 2,
        "single": {
            "tensors": {name: {"shape": [1]} for name in ("A", "B", "C", "D")},
            "ops": operation_list,
        },
        "distributed": {
            "ranks": {
                "0": {
                    "tensors": {f"{name}0": {"shape": [1]} for name in ("A", "B", "C", "D")},
                    "ops": rank_operations,
                },
                "1": {
                    "tensors": {f"{name}1": {"shape": [1]} for name in ("A", "B", "C", "D")},
                    "ops": rank_one_operations,
                },
            }
        },
        "input_relations": [] if input_relation is None else [input_relation],
        "output_relation": {"single_tensor": "C", "distributed_tensors": ["C0", "C1"], "type": "replicate"},
    }


def test_true_input_relation_passes_static_validation() -> None:
    assert load_stage(STANDARD_FIXTURE).name == "matmul_shard_to_partial"


def test_produced_single_tensor_as_input_relation_is_rejected(tmp_path: Path) -> None:
    relation = {"single_tensor": "C", "distributed_tensors": ["C0", "C1"], "type": "replicate"}

    with pytest.raises(StageInputError, match="input relation.*C.*program input"):
        _write_and_load(tmp_path, _add_stage_data(relation))


def test_produced_rank_local_tensor_as_input_relation_is_rejected(tmp_path: Path) -> None:
    relation = {"single_tensor": "A", "distributed_tensors": ["C0", "C1"], "type": "replicate"}

    with pytest.raises(StageInputError, match="C0.*program input"):
        _write_and_load(tmp_path, _add_stage_data(relation))


def test_multiple_producers_are_rejected_during_static_validation(tmp_path: Path) -> None:
    operations = [
        {"type": "add", "inputs": ["A", "B"], "outputs": ["C"]},
        {"type": "mul", "inputs": ["A", "B"], "outputs": ["C"]},
    ]

    with pytest.raises(StageInputError, match="C.*multiple producers"):
        _write_and_load(tmp_path, _add_stage_data(ops=operations))


def test_use_before_produce_is_rejected_during_static_validation(tmp_path: Path) -> None:
    operations = [
        {"type": "add", "inputs": ["C", "A"], "outputs": ["D"]},
        {"type": "add", "inputs": ["A", "B"], "outputs": ["C"]},
    ]

    with pytest.raises(StageInputError, match="C.*not available"):
        _write_and_load(tmp_path, _add_stage_data(ops=operations))


def test_output_relation_can_still_reference_input_tensor(tmp_path: Path) -> None:
    data = _add_stage_data(ops=[])
    data["output_relation"] = {
        "single_tensor": "A",
        "distributed_tensors": ["A0", "A1"],
        "type": "replicate",
    }

    assert _write_and_load(tmp_path, data).name == "boundary_case"
