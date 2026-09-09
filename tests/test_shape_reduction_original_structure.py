from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import z3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shape_constraints import build_shape_constraints, create_symbolic_shapes
from src.shape_model import TensorRef
from src.shape_reducer import reduce_shapes
from src.stage_loader import StageInputError, load_stage
from src.stage_model import ProgramSpec, RelationSpec, StageSpec, TensorSpec


STANDARD_FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"
LARGE_FIXTURE = ROOT / "input" / "matmul_shard_to_partial_large.json"


def _operator_case(operator: str, shapes: dict[str, list[int]]) -> dict:
    return {
        "name": f"{operator}_concrete_case",
        "world_size": 1,
        "single": {
            "tensors": {name: {"shape": shape} for name, shape in shapes.items()},
            "ops": [{"type": operator, "inputs": ["A", "B"], "outputs": ["C"]}],
        },
        "distributed": {
            "ranks": {
                "0": {
                    "tensors": {f"{name}0": {"shape": shape} for name, shape in shapes.items()},
                    "ops": [{"type": operator, "inputs": ["A0", "B0"], "outputs": ["C0"]}],
                }
            }
        },
        "input_relations": [
            {"single_tensor": "A", "distributed_tensors": ["A0"], "type": "replicate"},
            {"single_tensor": "B", "distributed_tensors": ["B0"], "type": "replicate"},
        ],
        "output_relation": {
            "single_tensor": "C",
            "distributed_tensors": ["C0"],
            "type": "replicate",
        },
    }


def _load_data(tmp_path: Path, data: dict):
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_stage(path)


def test_valid_original_matmul_passes_static_validation() -> None:
    assert load_stage(STANDARD_FIXTURE).name == "matmul_shard_to_partial"


def test_invalid_original_matmul_shared_dimension_is_rejected(tmp_path: Path) -> None:
    data = _operator_case("matmul", {"A": [4, 8], "B": [7, 4], "C": [4, 4]})

    with pytest.raises(StageInputError, match="matmul input dimensions do not match"):
        _load_data(tmp_path, data)


def test_invalid_original_matmul_output_shape_is_rejected(tmp_path: Path) -> None:
    data = _operator_case("matmul", {"A": [4, 8], "B": [8, 4], "C": [4, 5]})

    with pytest.raises(StageInputError, match="matmul output shape"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("operator", ["add", "mul"])
def test_valid_original_elementwise_operators_pass_static_validation(
    tmp_path: Path, operator: str
) -> None:
    assert _load_data(tmp_path, _operator_case(operator, {"A": [2, 3], "B": [2, 3], "C": [2, 3]}))


@pytest.mark.parametrize("operator", ["add", "mul"])
def test_invalid_original_elementwise_operators_are_rejected(
    tmp_path: Path, operator: str
) -> None:
    data = _operator_case(operator, {"A": [2, 3], "B": [2, 4], "C": [2, 3]})

    with pytest.raises(StageInputError, match=f"{operator} tensors must have the same shape"):
        _load_data(tmp_path, data)


def _ratio_stage(
    relation_type: str,
    single_shape: tuple[int, ...],
    local_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    dim: int | None = None,
    reduce_op: str | None = None,
) -> StageSpec:
    relation = RelationSpec("X", ("X0", "X1"), relation_type, dim, reduce_op)
    return StageSpec(
        name=f"{relation_type}_ratio",
        world_size=2,
        single=ProgramSpec({"X": TensorSpec("X", single_shape)}, ()),
        distributed={
            0: ProgramSpec({"X0": TensorSpec("X0", local_shapes[0])}, ()),
            1: ProgramSpec({"X1": TensorSpec("X1", local_shapes[1])}, ()),
        },
        input_relations=(relation,),
        output_relation=relation,
    )


@pytest.mark.parametrize(
    ("relation_type", "single_shape", "local_shapes", "dim", "reduce_op"),
    [
        ("replicate", (6, 10), ((6, 10), (6, 10)), None, None),
        ("partial", (6, 10), ((6, 10), (6, 10)), None, "sum"),
        ("shard", (8, 3), ((4, 3), (4, 3)), 0, None),
        ("shard", (3, 8), ((3, 4), (3, 4)), 1, None),
    ],
)
def test_relation_reduction_preserves_original_ratio_by_cross_multiplication(
    relation_type: str,
    single_shape: tuple[int, ...],
    local_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    dim: int | None,
    reduce_op: str | None,
) -> None:
    stage = _ratio_stage(relation_type, single_shape, local_shapes, dim, reduce_op)
    shapes = create_symbolic_shapes(stage)
    constraints = build_shape_constraints(stage, shapes)
    solver = z3.Solver()
    solver.add(*constraints)

    reduced_single = shapes.shapes[TensorRef("single", None, "X")]
    for rank in range(2):
        reduced_local = shapes.shapes[TensorRef("distributed", rank, f"X{rank}")]
        for index, (original_single, original_local) in enumerate(
            zip(single_shape, local_shapes[rank])
        ):
            solver.push()
            solver.add(
                z3.Not(
                    reduced_local[index] * original_single
                    == reduced_single[index] * original_local
                )
            )
            assert solver.check() == z3.unsat
            solver.pop()


def test_standard_and_large_original_structure_reduction_objectives_are_preserved() -> None:
    assert reduce_shapes(load_stage(STANDARD_FIXTURE)).objective_value == 20
    assert reduce_shapes(load_stage(LARGE_FIXTURE)).objective_value == 68
