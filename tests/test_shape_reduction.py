from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shape_constraints import ShapeReductionError, UnsupportedShapeSemanticsError
from src.shape_reducer import reduce_shapes
from src.stage_loader import load_stage


STANDARD_FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"
LARGE_FIXTURE = ROOT / "input" / "matmul_shard_to_partial_large.json"


def _load_data(tmp_path: Path, data: dict):
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_stage(path)


def _relation_only_case(relation_type: str) -> dict:
    relation = {
        "single_tensor": "X",
        "distributed_tensors": ["X0", "X1"],
        "type": relation_type,
    }
    if relation_type == "partial":
        relation["reduce_op"] = "sum"
    return {
        "name": f"{relation_type}_case",
        "world_size": 2,
        "single": {"tensors": {"X": {"shape": [8, 4]}}, "ops": []},
        "distributed": {
            "ranks": {
                "0": {"tensors": {"X0": {"shape": [8, 4]}}, "ops": []},
                "1": {"tensors": {"X1": {"shape": [8, 4]}}, "ops": []},
            }
        },
        "input_relations": [],
        "output_relation": relation,
    }


def _elementwise_case(operator: str) -> dict:
    return {
        "name": f"{operator}_case",
        "world_size": 1,
        "single": {
            "tensors": {"A": {"shape": [8, 4]}, "B": {"shape": [8, 4]}, "C": {"shape": [8, 4]}},
            "ops": [{"type": operator, "inputs": ["A", "B"], "outputs": ["C"]}],
        },
        "distributed": {
            "ranks": {
                "0": {
                    "tensors": {"A0": {"shape": [8, 4]}, "B0": {"shape": [8, 4]}, "C0": {"shape": [8, 4]}},
                    "ops": [{"type": operator, "inputs": ["A0", "B0"], "outputs": ["C0"]}],
                }
            }
        },
        "input_relations": [],
        "output_relation": {
            "single_tensor": "C",
            "distributed_tensors": ["C0"],
            "type": "replicate",
        },
    }


def test_standard_matmul_shard_to_partial_reduction():
    result = reduce_shapes(load_stage(STANDARD_FIXTURE))

    assert result.single["A"] == (1, 2)
    assert result.single["B"] == (2, 1)
    assert result.single["C"] == (1, 1)
    for rank in (0, 1):
        assert result.distributed[rank][f"A{rank}"] == (1, 1)
        assert result.distributed[rank][f"B{rank}"] == (1, 1)
        assert result.distributed[rank][f"C{rank}"] == (1, 1)
    assert result.objective_value == 20


def test_large_shape_case_reduces_to_rank_aware_minimum():
    result = reduce_shapes(load_stage(LARGE_FIXTURE))

    assert result.single["A"] == (1, 8)
    assert result.single["B"] == (8, 1)
    assert result.single["C"] == (1, 1)
    for rank in range(8):
        assert result.distributed[rank][f"A{rank}"] == (1, 1)
        assert result.distributed[rank][f"B{rank}"] == (1, 1)
        assert result.distributed[rank][f"C{rank}"] == (1, 1)


def test_matmul_k_dimensions_match_on_single_and_all_ranks():
    result = reduce_shapes(load_stage(STANDARD_FIXTURE))

    assert result.single["A"][1] == result.single["B"][0]
    for rank in (0, 1):
        assert result.distributed[rank][f"A{rank}"][1] == result.distributed[rank][f"B{rank}"][0]


def test_replicate_shape_relation_reduces_to_ones(tmp_path: Path):
    result = reduce_shapes(_load_data(tmp_path, _relation_only_case("replicate")))

    assert result.single["X"] == (1, 1)
    assert result.distributed[0]["X0"] == (1, 1)
    assert result.distributed[1]["X1"] == (1, 1)


def test_partial_shape_relation_reduces_to_ones(tmp_path: Path):
    result = reduce_shapes(_load_data(tmp_path, _relation_only_case("partial")))

    assert result.single["X"] == (1, 1)
    assert result.distributed[0]["X0"] == (1, 1)
    assert result.distributed[1]["X1"] == (1, 1)


@pytest.mark.parametrize("operator", ["add", "mul"])
def test_elementwise_operator_shape_semantics(tmp_path: Path, operator: str):
    result = reduce_shapes(_load_data(tmp_path, _elementwise_case(operator)))

    assert result.single["A"] == result.single["B"] == result.single["C"] == (1, 1)


def test_unsupported_operator_fails_during_shape_reduction(tmp_path: Path):
    data = _elementwise_case("add")
    data["single"]["ops"][0]["type"] = "reshape"

    stage = _load_data(tmp_path, data)
    with pytest.raises(UnsupportedShapeSemanticsError, match="reshape"):
        reduce_shapes(stage)


def test_rank_three_matmul_is_rejected_by_shape_reduction(tmp_path: Path):
    data = _elementwise_case("add")
    data["single"] = {
        "tensors": {"A": {"shape": [2, 3, 4]}, "B": {"shape": [2, 4, 5]}, "C": {"shape": [2, 3, 5]}},
        "ops": [{"type": "matmul", "inputs": ["A", "B"], "outputs": ["C"]}],
    }
    data["distributed"]["ranks"]["0"] = {
        "tensors": {"A0": {"shape": [2, 3, 4]}, "B0": {"shape": [2, 4, 5]}, "C0": {"shape": [2, 3, 5]}},
        "ops": [{"type": "matmul", "inputs": ["A0", "B0"], "outputs": ["C0"]}],
    }
    data["output_relation"] = {
        "single_tensor": "C",
        "distributed_tensors": ["C0"],
        "type": "replicate",
    }

    stage = _load_data(tmp_path, data)
    with pytest.raises(ShapeReductionError, match="only 2-D tensors"):
        reduce_shapes(stage)


def test_conflicting_operator_and_relation_shape_constraints_are_unsat(tmp_path: Path):
    data = {
        "name": "unsat_add_relations",
        "world_size": 2,
        "single": {
            "tensors": {"A": {"shape": [4, 8]}, "B": {"shape": [4, 8]}, "C": {"shape": [4, 8]}},
            "ops": [{"type": "add", "inputs": ["A", "B"], "outputs": ["C"]}],
        },
        "distributed": {
            "ranks": {
                "0": {"tensors": {"A0": {"shape": [4, 4]}, "B0": {"shape": [4, 8]}, "C0": {"shape": [4, 8]}}, "ops": [{"type": "add", "inputs": ["A0", "B0"], "outputs": ["C0"]}]},
                "1": {"tensors": {"A1": {"shape": [4, 4]}, "B1": {"shape": [4, 8]}, "C1": {"shape": [4, 8]}}, "ops": [{"type": "add", "inputs": ["A1", "B1"], "outputs": ["C1"]}]},
            }
        },
        "input_relations": [
            {"single_tensor": "A", "distributed_tensors": ["A0", "A1"], "type": "shard", "dim": 1},
            {"single_tensor": "B", "distributed_tensors": ["B0", "B1"], "type": "replicate"},
        ],
        "output_relation": {"single_tensor": "C", "distributed_tensors": ["C0", "C1"], "type": "replicate"},
    }

    stage = _load_data(tmp_path, data)
    with pytest.raises(ShapeReductionError, match="unsatisfiable"):
        reduce_shapes(stage)
