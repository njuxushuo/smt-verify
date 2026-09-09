from __future__ import annotations

import sys
from pathlib import Path

import pytest
import z3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relation_encoder import encode_stage_relations
from src.relations import (
    PartialRelation,
    RelationEncodingError,
    ReplicateRelation,
    ShardRelation,
    get_relation,
)
from src.shape_model import ReducedShapeResult
from src.shape_reducer import reduce_shapes
from src.stage_loader import load_stage
from src.stage_loader import StageInputError
from src.stage_model import OpSpec, ProgramSpec, RelationSpec, StageSpec, TensorSpec
from src.stage_validator import validate_stage
from src.symbolic_executor import execute_stage
from src.symbolic_tensor import SymbolicTensor, create_symbolic_input_tensor


STANDARD_FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"
LARGE_FIXTURE = ROOT / "input" / "matmul_shard_to_partial_large.json"


def _relation(
    relation_type: str,
    dim: int | None = None,
    reduce_op: str | None = None,
) -> RelationSpec:
    return RelationSpec(
        single_tensor="X",
        distributed_tensors=("X0", "X1"),
        type=relation_type,
        dim=dim,
        reduce_op=reduce_op,
    )


def _tensor(name: str, shape: tuple[int, ...]) -> SymbolicTensor:
    return create_symbolic_input_tensor(name, shape, "test")


def _assert_implied(constraints: list[z3.BoolRef], conclusion: z3.BoolRef) -> None:
    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(z3.Not(conclusion))
    assert solver.check() == z3.unsat


def _program(
    shapes: dict[str, tuple[int, ...]], ops: tuple[OpSpec, ...] = ()
) -> ProgramSpec:
    return ProgramSpec(
        tensors={name: TensorSpec(name=name, shape=shape) for name, shape in shapes.items()},
        ops=ops,
    )


def test_relation_registry_returns_implemented_semantics() -> None:
    assert isinstance(get_relation("replicate"), ReplicateRelation)
    assert isinstance(get_relation("shard"), ShardRelation)
    assert isinstance(get_relation("partial"), PartialRelation)


def test_stage_two_shape_reduction_results_are_preserved() -> None:
    standard = reduce_shapes(load_stage(STANDARD_FIXTURE))
    large = reduce_shapes(load_stage(LARGE_FIXTURE))

    assert standard.single == {"A": (1, 4), "B": (4, 1), "C": (1, 1)}
    assert standard.objective_value == 28
    assert large.single == {"A": (1, 16), "B": (16, 1), "C": (1, 1)}
    assert large.objective_value == 100


def test_replicate_value_constraints_imply_each_rank_matches_single() -> None:
    single = _tensor("X", (2, 2))
    local_zero = _tensor("X0", (2, 2))
    local_one = _tensor("X1", (2, 2))
    constraints = get_relation("replicate").value_constraints(
        _relation("replicate"), single, (local_zero, local_one), 2, "test"
    )

    _assert_implied(constraints, local_zero.at((1, 0)) == single.at((1, 0)))
    _assert_implied(constraints, local_one.at((0, 1)) == single.at((0, 1)))


def test_shard_dim_zero_maps_contiguous_rows() -> None:
    single = _tensor("X", (4, 2))
    local_zero = _tensor("X0", (2, 2))
    local_one = _tensor("X1", (2, 2))
    constraints = get_relation("shard").value_constraints(
        _relation("shard", dim=0), single, (local_zero, local_one), 2, "test"
    )

    _assert_implied(constraints, local_zero.at((0, 0)) == single.at((0, 0)))
    _assert_implied(constraints, local_zero.at((1, 1)) == single.at((1, 1)))
    _assert_implied(constraints, local_one.at((0, 0)) == single.at((2, 0)))
    _assert_implied(constraints, local_one.at((1, 1)) == single.at((3, 1)))


def test_shard_dim_one_maps_contiguous_columns_on_every_row() -> None:
    single = _tensor("X", (2, 4))
    local_zero = _tensor("X0", (2, 2))
    local_one = _tensor("X1", (2, 2))
    constraints = get_relation("shard").value_constraints(
        _relation("shard", dim=1), single, (local_zero, local_one), 2, "test"
    )

    _assert_implied(constraints, local_zero.at((0, 0)) == single.at((0, 0)))
    _assert_implied(constraints, local_zero.at((0, 1)) == single.at((0, 1)))
    _assert_implied(constraints, local_one.at((0, 0)) == single.at((0, 2)))
    _assert_implied(constraints, local_one.at((0, 1)) == single.at((0, 3)))
    _assert_implied(constraints, local_zero.at((1, 1)) == single.at((1, 1)))
    _assert_implied(constraints, local_one.at((1, 1)) == single.at((1, 3)))


def test_partial_sum_value_constraints() -> None:
    single = _tensor("X", (2,))
    local_zero = _tensor("X0", (2,))
    local_one = _tensor("X1", (2,))
    constraints = get_relation("partial").value_constraints(
        _relation("partial", reduce_op="sum"), single, (local_zero, local_one), 2, "test"
    )

    _assert_implied(constraints, single.at((0,)) == local_zero.at((0,)) + local_one.at((0,)))
    _assert_implied(constraints, single.at((1,)) == local_zero.at((1,)) + local_one.at((1,)))


@pytest.mark.parametrize(
    ("relation", "single_shape", "local_shapes"),
    [
        (_relation("replicate"), (2,), ((1,), (2,))),
        (_relation("partial", reduce_op="sum"), (2,), ((2,), (1,))),
        (_relation("shard", dim=0), (4, 2), ((2, 3), (2, 3))),
        (_relation("shard", dim=0), (4, 2), ((1, 2), (1, 2))),
    ],
)
def test_relation_value_shape_mismatches_are_rejected(
    relation: RelationSpec,
    single_shape: tuple[int, ...],
    local_shapes: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(RelationEncodingError):
        get_relation(relation.type).value_constraints(
            relation,
            _tensor("X", single_shape),
            tuple(_tensor(f"X{rank}", shape) for rank, shape in enumerate(local_shapes)),
            2,
            "test",
        )


@pytest.mark.parametrize(
    "relation",
    [
        _relation("replicate"),
        _relation("shard", dim=0),
        _relation("partial", reduce_op="sum"),
    ],
)
def test_relation_value_constraints_reject_wrong_rank_count(relation: RelationSpec) -> None:
    with pytest.raises(RelationEncodingError, match="expected 2 local tensors"):
        get_relation(relation.type).value_constraints(
            relation, _tensor("X", (2,)), (_tensor("X0", (2,)),), 2, "test"
        )


def test_partial_value_constraints_reject_non_sum_reduce_op() -> None:
    with pytest.raises(RelationEncodingError, match="reduce_op"):
        get_relation("partial").value_constraints(
            _relation("partial", reduce_op="max"),
            _tensor("X", (1,)),
            (_tensor("X0", (1,)), _tensor("X1", (1,))),
            2,
            "test",
        )


def _produced_relation_stage() -> tuple[StageSpec, ReducedShapeResult]:
    single = _program(
        {"A": (1,), "B": (1,), "C": (1,)},
        (OpSpec(type="add", inputs=("A", "B"), outputs=("C",)),),
    )
    distributed = {
        rank: _program(
            {f"A{rank}": (1,), f"B{rank}": (1,), f"C{rank}": (1,)},
            (
                OpSpec(
                    type="add",
                    inputs=(f"A{rank}", f"B{rank}"),
                    outputs=(f"C{rank}",),
                ),
            ),
        )
        for rank in range(2)
    }
    produced = RelationSpec("C", ("C0", "C1"), "replicate")
    stage = StageSpec(
        name="produced_input_relation",
        world_size=2,
        single=single,
        distributed=distributed,
        input_relations=(produced,),
        output_relation=produced,
    )
    reduced = ReducedShapeResult(
        stage_name=stage.name,
        single={"A": (1,), "B": (1,), "C": (1,)},
        distributed={
            rank: {f"A{rank}": (1,), f"B{rank}": (1,), f"C{rank}": (1,)}
            for rank in range(2)
        },
        objective_value=0,
    )
    return stage, reduced


def test_input_relation_must_reference_true_program_inputs() -> None:
    stage, reduced = _produced_relation_stage()

    with pytest.raises(StageInputError, match="input relation.*C.*program input"):
        validate_stage(stage)


def test_output_relation_can_reference_an_input_tensor() -> None:
    stage = StageSpec(
        name="identity_output_relation",
        world_size=2,
        single=_program({"X": (1,)}),
        distributed={0: _program({"X0": (1,)}), 1: _program({"X1": (1,)})},
        input_relations=(),
        output_relation=RelationSpec("X", ("X0", "X1"), "replicate"),
    )
    reduced = ReducedShapeResult(
        stage_name=stage.name,
        single={"X": (1,)},
        distributed={0: {"X0": (1,)}, 1: {"X1": (1,)}},
        objective_value=0,
    )

    encoded = encode_stage_relations(stage, execute_stage(stage, reduced))
    assert len(encoded.input_constraints) == 0
    assert len(encoded.output_constraints) == 2


def _standard_encoded():
    stage = load_stage(STANDARD_FIXTURE)
    symbolic = execute_stage(stage, reduce_shapes(stage))
    return stage, symbolic, encode_stage_relations(stage, symbolic)


def test_standard_case_relation_constraint_counts() -> None:
    _, _, encoded = _standard_encoded()
    assert len(encoded.input_constraints) == 8
    assert len(encoded.output_constraints) == 1


def test_standard_case_input_premises_imply_partial_output_candidate() -> None:
    _, _, encoded = _standard_encoded()
    solver = z3.Solver()
    solver.add(*encoded.input_constraints)
    solver.add(z3.Not(z3.And(*encoded.output_constraints)))
    assert solver.check() == z3.unsat


def test_wrong_replicate_output_candidate_is_not_implied() -> None:
    stage, symbolic, encoded = _standard_encoded()
    wrong_candidate = RelationSpec("C", ("C0", "C1"), "replicate")
    wrong_constraints = get_relation("replicate").value_constraints(
        wrong_candidate,
        symbolic.single.tensors["C"],
        (symbolic.distributed[0].tensors["C0"], symbolic.distributed[1].tensors["C1"]),
        stage.world_size,
        "wrong_output_relation",
    )
    solver = z3.Solver()
    solver.add(*encoded.input_constraints)
    solver.add(z3.Not(z3.And(*wrong_constraints)))
    assert solver.check() == z3.sat
