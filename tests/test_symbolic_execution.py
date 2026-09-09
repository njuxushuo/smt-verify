from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import z3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators import AddOperator, MulOperator
from src.program_analysis import ProgramAnalysisError, find_program_inputs
from src.shape_model import UnsupportedShapeSemanticsError
from src.shape_reducer import reduce_shapes
from src.stage_loader import load_stage
from src.stage_model import OpSpec, ProgramSpec, TensorSpec
from src.symbolic_executor import execute_program, execute_stage
from src.symbolic_tensor import (
    SymbolicExecutionError,
    SymbolicTensor,
    create_symbolic_input_tensor,
)


FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"


def _program(
    shapes: dict[str, tuple[int, ...]], ops: tuple[OpSpec, ...]
) -> ProgramSpec:
    return ProgramSpec(
        tensors={name: TensorSpec(name=name, shape=shape) for name, shape in shapes.items()},
        ops=ops,
    )


def _assert_equivalent(actual: z3.ArithRef, expected: z3.ArithRef) -> None:
    solver = z3.Solver()
    solver.add(actual != expected)
    assert solver.check() == z3.unsat


def test_symbolic_tensor_uses_row_major_indexing() -> None:
    values = tuple(z3.Real(f"x{index}") for index in range(6))
    tensor = SymbolicTensor(shape=(2, 3), values=values)

    assert tensor.at((0, 0)) is values[0]
    assert tensor.at((0, 2)) is values[2]
    assert tensor.at((1, 0)) is values[3]
    assert tensor.at((1, 2)) is values[5]
    with pytest.raises(IndexError):
        tensor.at((0,))
    with pytest.raises(IndexError):
        tensor.at((-1, 0))
    with pytest.raises(IndexError):
        tensor.at((2, 0))


def test_symbolic_tensor_validates_numel_and_positive_dimensions() -> None:
    values = tuple(z3.Real(f"x{index}") for index in range(6))
    assert SymbolicTensor(shape=(2, 3), values=values).numel == 6
    with pytest.raises(ValueError, match="values length"):
        SymbolicTensor(shape=(2, 3), values=values[:-1])
    with pytest.raises(ValueError, match="positive integer"):
        SymbolicTensor(shape=(0, 3), values=())
    with pytest.raises(ValueError, match="positive integer"):
        SymbolicTensor(shape=(-1, 3), values=())


def test_find_program_inputs_preserves_declared_order() -> None:
    program = _program(
        {"A": (1, 2), "B": (2, 1), "C": (1, 1), "D": (1, 1)},
        (
            OpSpec(type="matmul", inputs=("A", "B"), outputs=("C",)),
            OpSpec(type="add", inputs=("C", "C"), outputs=("D",)),
        ),
    )

    assert find_program_inputs(program) == ("A", "B")


def test_multiple_producers_are_rejected() -> None:
    program = _program(
        {"A": (1,), "B": (1,), "C": (1,)},
        (
            OpSpec(type="add", inputs=("A", "B"), outputs=("C",)),
            OpSpec(type="mul", inputs=("A", "B"), outputs=("C",)),
        ),
    )

    with pytest.raises(ProgramAnalysisError, match="C.*multiple producers"):
        find_program_inputs(program)


def test_use_before_produce_is_rejected_without_reordering() -> None:
    program = _program(
        {"A": (1,), "B": (1,), "C": (1,), "D": (1,)},
        (
            OpSpec(type="add", inputs=("C", "A"), outputs=("D",)),
            OpSpec(type="add", inputs=("A", "B"), outputs=("C",)),
        ),
    )

    with pytest.raises(SymbolicExecutionError, match="C.*not available"):
        execute_program(
            program,
            {"A": (1,), "B": (1,), "C": (1,), "D": (1,)},
            scope_prefix="single",
            context="single",
        )


def test_free_input_variable_names_and_sorts() -> None:
    single = create_symbolic_input_tensor("A", (1, 2), "single")
    rank_zero = create_symbolic_input_tensor("A0", (1, 2), "rank0")

    assert [value.decl().name() for value in single.values] == ["single__A__v0", "single__A__v1"]
    assert [value.decl().name() for value in rank_zero.values] == [
        "rank0__A0__v0",
        "rank0__A0__v1",
    ]
    assert all(value.sort() == z3.RealSort() for value in (*single.values, *rank_zero.values))


def test_standard_stage_matmul_symbolic_semantics() -> None:
    result = execute_stage((stage := load_stage(FIXTURE)), reduce_shapes(stage))

    assert result.single.input_names == ("A", "B")
    assert result.distributed[0].input_names == ("A0", "B0")
    assert result.distributed[1].input_names == ("A1", "B1")
    single_a = result.single.tensors["A"]
    single_b = result.single.tensors["B"]
    expected = z3.Sum(
        [
            single_a.at((0, index)) * single_b.at((index, 0))
            for index in range(single_a.shape[1])
        ]
    )
    _assert_equivalent(result.single.tensors["C"].at((0, 0)), expected)


@pytest.mark.parametrize("rank", [0, 1])
def test_standard_stage_rank_local_matmul_symbolic_semantics(rank: int) -> None:
    result = execute_stage((stage := load_stage(FIXTURE)), reduce_shapes(stage))
    program = result.distributed[rank]
    local_a = program.tensors[f"A{rank}"]
    local_b = program.tensors[f"B{rank}"]
    expected = z3.Sum(
        [
            local_a.at((0, index)) * local_b.at((index, 0))
            for index in range(local_a.shape[1])
        ]
    )
    _assert_equivalent(program.tensors[f"C{rank}"].at((0, 0)), expected)


@pytest.mark.parametrize(
    ("operator", "expected_value"),
    [
        ("add", lambda left, right: left + right),
        ("mul", lambda left, right: left * right),
    ],
)
def test_elementwise_symbolic_semantics(operator: str, expected_value) -> None:
    program = _program(
        {"A": (2, 2), "B": (2, 2), "C": (2, 2)},
        (OpSpec(type=operator, inputs=("A", "B"), outputs=("C",)),),
    )
    result = execute_program(
        program,
        {"A": (2, 2), "B": (2, 2), "C": (2, 2)},
        scope_prefix="single",
        context="single",
    )
    for actual, left, right in zip(
        result.tensors["C"].values,
        result.tensors["A"].values,
        result.tensors["B"].values,
    ):
        _assert_equivalent(actual, expected_value(left, right))


def test_sequential_expression_propagation_does_not_create_output_inputs() -> None:
    program = _program(
        {"A": (1, 2), "B": (2, 1), "C": (1, 1), "D": (1, 1)},
        (
            OpSpec(type="matmul", inputs=("A", "B"), outputs=("C",)),
            OpSpec(type="add", inputs=("C", "C"), outputs=("D",)),
        ),
    )
    result = execute_program(
        program,
        {"A": (1, 2), "B": (2, 1), "C": (1, 1), "D": (1, 1)},
        scope_prefix="single",
        context="single",
    )

    assert result.input_names == ("A", "B")
    left, right = result.tensors["A"], result.tensors["B"]
    expected = 2 * (left.at((0, 0)) * right.at((0, 0)) + left.at((0, 1)) * right.at((1, 0)))
    _assert_equivalent(result.tensors["D"].at((0, 0)), expected)


def test_add_symbolic_execution_supports_z3_algebraic_reasoning() -> None:
    left = create_symbolic_input_tensor("A", (1,), "single")
    right = create_symbolic_input_tensor("B", (1,), "single")
    operator = AddOperator()
    forward = operator.symbolic_execute((left, right), ((1,),), "test")[0]
    reverse = operator.symbolic_execute((right, left), ((1,),), "test")[0]

    _assert_equivalent(forward.values[0], reverse.values[0])


def test_add_and_mul_symbolic_execution_remain_distinct() -> None:
    left = create_symbolic_input_tensor("A", (1,), "single")
    right = create_symbolic_input_tensor("B", (1,), "single")
    add_result = AddOperator().symbolic_execute((left, right), ((1,),), "test")[0]
    mul_result = MulOperator().symbolic_execute((left, right), ((1,),), "test")[0]
    solver = z3.Solver()
    solver.add(add_result.values[0] != mul_result.values[0])
    assert solver.check() == z3.sat


@pytest.mark.parametrize(
    "reduced_shapes",
    [
        {"A": (1,), "B": (1,)},
        {"A": (1,), "B": (1,), "C": (1,), "extra": (1,)},
        {"A": (1, 1), "B": (1,), "C": (1,)},
    ],
)
def test_reduced_shape_mismatch_is_rejected(reduced_shapes: dict[str, tuple[int, ...]]) -> None:
    program = _program(
        {"A": (1,), "B": (1,), "C": (1,)},
        (OpSpec(type="add", inputs=("A", "B"), outputs=("C",)),),
    )
    with pytest.raises(SymbolicExecutionError):
        execute_program(program, reduced_shapes, scope_prefix="single", context="single")


def test_declared_but_unsupported_operator_still_fails_shape_reduction(tmp_path: Path) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["single"]["ops"][0]["type"] = "reshape"
    path = tmp_path / "reshape_stage.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnsupportedShapeSemanticsError, match="reshape"):
        reduce_shapes(load_stage(path))
