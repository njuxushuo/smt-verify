from __future__ import annotations

import sys
from pathlib import Path

import pytest
import z3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantics.relations import COMPOSITE_RELATION, ConcreteRelationError
from src.stage.model import DeviceMeshSpec, PlacementSpec, RelationSpec
from src.symbolic.tensor import SymbolicTensor, create_symbolic_input_tensor


MESH = DeviceMeshSpec((2, 2))


def _relation(*placements: PlacementSpec, ranks: int = 4) -> RelationSpec:
    return RelationSpec(
        single_tensor="X",
        distributed_tensors=tuple(f"X{rank}" for rank in range(ranks)),
        placements=placements,
    )


def _tensors(
    single_shape: tuple[int, ...], local_shape: tuple[int, ...], count: int = 4
) -> tuple[SymbolicTensor, tuple[SymbolicTensor, ...]]:
    single = create_symbolic_input_tensor("X", single_shape, "single")
    locals_ = tuple(
        create_symbolic_input_tensor(f"X{rank}", local_shape, f"rank{rank}")
        for rank in range(count)
    )
    return single, locals_


def _assert_implied(constraints: list[z3.BoolRef], conclusion: z3.BoolRef) -> None:
    solver = z3.Solver()
    solver.add(*constraints, z3.Not(conclusion))
    assert solver.check() == z3.unsat


@pytest.mark.parametrize(
    ("placements", "single_shape", "local_shape"),
    [
        ((PlacementSpec("replicate"), PlacementSpec("replicate")), (4, 6), (4, 6)),
        ((PlacementSpec("replicate"), PlacementSpec("shard", dim=1)), (4, 6), (4, 3)),
        ((PlacementSpec("shard", dim=0), PlacementSpec("replicate")), (4, 6), (2, 6)),
        ((PlacementSpec("shard", dim=0), PlacementSpec("shard", dim=1)), (4, 6), (2, 3)),
        ((PlacementSpec("shard", dim=0), PlacementSpec("partial", reduce_op="sum")), (4, 6), (2, 6)),
        ((PlacementSpec("partial", reduce_op="sum"), PlacementSpec("shard", dim=1)), (4, 6), (4, 3)),
        ((PlacementSpec("replicate"), PlacementSpec("partial", reduce_op="sum")), (4, 6), (4, 6)),
        ((PlacementSpec("partial", reduce_op="sum"), PlacementSpec("replicate")), (4, 6), (4, 6)),
    ],
)
def test_composite_concrete_shapes(
    placements: tuple[PlacementSpec, ...],
    single_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> None:
    COMPOSITE_RELATION.validate_concrete_shapes(
        _relation(*placements), single_shape, (local_shape,) * 4, MESH, "test"
    )


def test_composite_concrete_shape_uses_mesh_axis_factor_not_world_size() -> None:
    relation = _relation(PlacementSpec("replicate"), PlacementSpec("shard", dim=1))
    COMPOSITE_RELATION.validate_concrete_shapes(
        relation, (4, 6), ((4, 3),) * 4, MESH, "test"
    )
    with pytest.raises(ConcreteRelationError, match=r"expected \(4, 3\)"):
        COMPOSITE_RELATION.validate_concrete_shapes(
            relation, (4, 6), ((4, 1),) * 4, MESH, "test"
        )


def test_reduced_shape_constraints_use_per_dimension_shard_factors() -> None:
    relation = _relation(
        PlacementSpec("shard", dim=0), PlacementSpec("shard", dim=1)
    )
    single = (z3.Int("s0"), z3.Int("s1"))
    locals_ = tuple(
        (z3.Int(f"l{rank}_0"), z3.Int(f"l{rank}_1")) for rank in range(4)
    )
    constraints = COMPOSITE_RELATION.shape_constraints(
        relation,
        single,
        locals_,
        (4, 6),
        ((2, 3),) * 4,
        MESH,
        "test",
    )

    for rank in range(4):
        _assert_implied(constraints, locals_[rank][0] * 2 == single[0])
        _assert_implied(constraints, locals_[rank][1] * 2 == single[1])
    _assert_implied(constraints, single[0] % 2 == 0)
    _assert_implied(constraints, single[1] % 2 == 0)


def test_replicate_shard_value_mapping_replicates_each_global_shard() -> None:
    relation = _relation(PlacementSpec("replicate"), PlacementSpec("shard", dim=1))
    single, locals_ = _tensors((2, 4), (2, 2))
    constraints = COMPOSITE_RELATION.value_constraints(
        relation, single, locals_, MESH, "test"
    )

    _assert_implied(constraints, locals_[0].at((1, 1)) == single.at((1, 1)))
    _assert_implied(constraints, locals_[2].at((1, 1)) == single.at((1, 1)))
    _assert_implied(constraints, locals_[1].at((0, 0)) == single.at((0, 2)))
    _assert_implied(constraints, locals_[3].at((0, 0)) == single.at((0, 2)))


def test_shard_shard_value_mapping_uses_both_mesh_coordinates() -> None:
    relation = _relation(
        PlacementSpec("shard", dim=0), PlacementSpec("shard", dim=1)
    )
    single, locals_ = _tensors((4, 6), (2, 3))
    constraints = COMPOSITE_RELATION.value_constraints(
        relation, single, locals_, MESH, "test"
    )

    _assert_implied(constraints, locals_[0].at((1, 2)) == single.at((1, 2)))
    _assert_implied(constraints, locals_[1].at((1, 0)) == single.at((1, 3)))
    _assert_implied(constraints, locals_[2].at((0, 2)) == single.at((2, 2)))
    _assert_implied(constraints, locals_[3].at((1, 2)) == single.at((3, 5)))


def test_shard_partial_sums_only_within_each_shard_group() -> None:
    relation = _relation(
        PlacementSpec("shard", dim=0), PlacementSpec("partial", reduce_op="sum")
    )
    single, locals_ = _tensors((4, 2), (2, 2))
    constraints = COMPOSITE_RELATION.value_constraints(
        relation, single, locals_, MESH, "test"
    )

    _assert_implied(
        constraints,
        single.at((0, 1)) == locals_[0].at((0, 1)) + locals_[1].at((0, 1)),
    )
    _assert_implied(
        constraints,
        single.at((2, 1)) == locals_[2].at((0, 1)) + locals_[3].at((0, 1)),
    )
    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        single.at((0, 1))
        != sum(local.at((0, 1)) for local in locals_)
    )
    assert solver.check() == z3.sat


def test_replicate_partial_builds_one_sum_per_replica_group() -> None:
    relation = _relation(
        PlacementSpec("replicate"), PlacementSpec("partial", reduce_op="sum")
    )
    single, locals_ = _tensors((2,), (2,))
    constraints = COMPOSITE_RELATION.value_constraints(
        relation, single, locals_, MESH, "test"
    )

    _assert_implied(
        constraints, single.at((1,)) == locals_[0].at((1,)) + locals_[1].at((1,))
    )
    _assert_implied(
        constraints, single.at((1,)) == locals_[2].at((1,)) + locals_[3].at((1,))
    )


def test_multiple_partial_axes_sum_cartesian_product() -> None:
    mesh = DeviceMeshSpec((2, 2, 2))
    relation = _relation(
        PlacementSpec("partial", reduce_op="sum"),
        PlacementSpec("partial", reduce_op="sum"),
        PlacementSpec("shard", dim=0),
        ranks=8,
    )
    single, locals_ = _tensors((4,), (2,), count=8)
    constraints = COMPOSITE_RELATION.value_constraints(
        relation, single, locals_, mesh, "test"
    )

    _assert_implied(
        constraints,
        single.at((0,))
        == z3.Sum([locals_[rank].at((0,)) for rank in (0, 2, 4, 6)]),
    )
    _assert_implied(
        constraints,
        single.at((2,))
        == z3.Sum([locals_[rank].at((0,)) for rank in (1, 3, 5, 7)]),
    )
