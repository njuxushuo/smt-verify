from __future__ import annotations

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
from src.stage_loader import load_stage
from src.stage_model import OpSpec, ProgramSpec, RelationSpec, StageSpec, TensorSpec


STANDARD_FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"
LARGE_FIXTURE = ROOT / "input" / "matmul_shard_to_partial_large.json"


def _solver_for(stage: StageSpec) -> tuple[z3.Solver, object]:
    shapes = create_symbolic_shapes(stage)
    solver = z3.Solver()
    solver.add(*build_shape_constraints(stage, shapes))
    return solver, shapes


def _original_shape(stage: StageSpec, ref: TensorRef) -> tuple[int, ...]:
    if ref.scope == "single":
        return stage.single.tensors[ref.name].shape
    assert ref.rank is not None
    return stage.distributed[ref.rank].tensors[ref.name].shape


def test_each_reduced_dimension_is_explicitly_bounded_by_its_original_extent() -> None:
    stage = load_stage(STANDARD_FIXTURE)
    solver, shapes = _solver_for(stage)

    for ref, reduced_shape in shapes.shapes.items():
        for reduced_dimension, original_dimension in zip(reduced_shape, _original_shape(stage, ref)):
            solver.push()
            solver.add(reduced_dimension > original_dimension)
            assert solver.check() == z3.unsat
            solver.pop()


def test_matmul_contraction_dimension_uses_original_k_minimum_basis() -> None:
    stage = load_stage(STANDARD_FIXTURE)
    solver, shapes = _solver_for(stage)
    reduced_k = shapes.shapes[TensorRef("single", None, "A")][1]

    solver.add(reduced_k < 2)
    assert solver.check() == z3.unsat


def _k_one_stage() -> StageSpec:
    single = ProgramSpec(
        tensors={
            "A": TensorSpec("A", (2, 1)),
            "B": TensorSpec("B", (1, 3)),
            "C": TensorSpec("C", (2, 3)),
        },
        ops=(OpSpec("matmul", ("A", "B"), ("C",)),),
    )
    local = ProgramSpec(
        tensors={
            "A0": TensorSpec("A0", (2, 1)),
            "B0": TensorSpec("B0", (1, 3)),
            "C0": TensorSpec("C0", (2, 3)),
        },
        ops=(OpSpec("matmul", ("A0", "B0"), ("C0",)),),
    )
    inputs = (
        RelationSpec("A", ("A0",), "replicate"),
        RelationSpec("B", ("B0",), "replicate"),
    )
    return StageSpec(
        name="matmul_k_one",
        world_size=1,
        single=single,
        distributed={0: local},
        input_relations=inputs,
        output_relation=RelationSpec("C", ("C0",), "replicate"),
    )


def test_original_k_one_keeps_reduced_k_one_feasible() -> None:
    stage = _k_one_stage()
    solver, shapes = _solver_for(stage)
    reduced_k = shapes.shapes[TensorRef("single", None, "A")][1]

    assert solver.check() == z3.sat
    solver.add(reduced_k != 1)
    assert solver.check() == z3.unsat


@pytest.mark.parametrize(
    ("relation", "single_shape", "local_shapes"),
    [
        ("replicate", (6, 10), ((6, 10), (6, 10))),
        ("partial", (6, 10), ((6, 10), (6, 10))),
        ("shard0", (8, 3), ((4, 3), (4, 3))),
        ("shard1", (3, 8), ((3, 4), (3, 4))),
    ],
)
def test_stage_four_point_five_original_ratio_constraints_remain_implied(
    relation: str,
    single_shape: tuple[int, ...],
    local_shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    relation_type = "shard" if relation.startswith("shard") else relation
    dimension = 0 if relation == "shard0" else 1 if relation == "shard1" else None
    reduce_op = "sum" if relation == "partial" else None
    relation_spec = RelationSpec("X", ("X0", "X1"), relation_type, dimension, reduce_op)
    stage = StageSpec(
        name=f"{relation}_ratio",
        world_size=2,
        single=ProgramSpec({"X": TensorSpec("X", single_shape)}, ()),
        distributed={
            0: ProgramSpec({"X0": TensorSpec("X0", local_shapes[0])}, ()),
            1: ProgramSpec({"X1": TensorSpec("X1", local_shapes[1])}, ()),
        },
        input_relations=(relation_spec,),
        output_relation=relation_spec,
    )
    solver, shapes = _solver_for(stage)
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


def test_standard_and_large_reduction_objectives_remain_regressions() -> None:
    assert reduce_shapes(load_stage(STANDARD_FIXTURE)).objective_value == 28
    assert reduce_shapes(load_stage(LARGE_FIXTURE)).objective_value == 100
