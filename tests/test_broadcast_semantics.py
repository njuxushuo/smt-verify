from __future__ import annotations

import sys
from pathlib import Path

import pytest
import z3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantics.broadcast import (
    BroadcastShapeError,
    broadcast_index,
    infer_broadcast_shape,
)
from src.semantics.operators import AddOperator, ConcreteShapeError, MulOperator
from src.symbolic.tensor import create_symbolic_input_tensor


def _assert_equivalent(actual: z3.ArithRef, expected: z3.ArithRef) -> None:
    solver = z3.Solver()
    solver.add(actual != expected)
    assert solver.check() == z3.unsat


def _symbolic_shape(prefix: str, rank: int) -> tuple[z3.ArithRef, ...]:
    return tuple(z3.Int(f"{prefix}{axis}") for axis in range(rank))


def test_add_same_shape_regression() -> None:
    shape = (2, 3, 4)

    assert infer_broadcast_shape(shape, shape) == shape
    AddOperator().validate_concrete_shapes((shape, shape), (shape,), {}, "test")


def test_add_rank_mismatch_broadcast() -> None:
    assert infer_broadcast_shape((2, 3, 4), (4,)) == (2, 3, 4)
    AddOperator().validate_concrete_shapes(((2, 3, 4), (4,)), ((2, 3, 4),), {}, "test")


def test_add_singleton_broadcast() -> None:
    assert infer_broadcast_shape((2, 1, 4), (1, 3, 4)) == (2, 3, 4)
    AddOperator().validate_concrete_shapes(
        ((2, 1, 4), (1, 3, 4)), ((2, 3, 4),), {}, "test"
    )


def test_mul_four_dimensional_broadcast() -> None:
    left_shape = (2, 3, 4, 5)
    right_shape = (1, 3, 1, 5)

    assert infer_broadcast_shape(left_shape, right_shape) == left_shape
    MulOperator().validate_concrete_shapes((left_shape, right_shape), (left_shape,), {}, "test")


def test_invalid_broadcast_is_rejected() -> None:
    with pytest.raises(BroadcastShapeError, match="not broadcast-compatible"):
        infer_broadcast_shape((2, 3, 4), (5,))
    with pytest.raises(ConcreteShapeError, match="not broadcast-compatible"):
        AddOperator().validate_concrete_shapes(
            ((2, 3, 4), (5,)), ((2, 3, 4),), {}, "test"
        )


def test_wrong_broadcast_output_shape_is_rejected() -> None:
    with pytest.raises(ConcreteShapeError, match="does not match broadcast shape"):
        AddOperator().validate_concrete_shapes(
            ((2, 3, 4), (4,)), ((2, 3, 1),), {}, "test"
        )


def test_reduced_constraints_preserve_original_singleton_pattern() -> None:
    left = _symbolic_shape("a", 3)
    right = _symbolic_shape("b", 3)
    output = _symbolic_shape("c", 3)
    constraints = AddOperator().shape_constraints(
        (left, right),
        (output,),
        ((32, 1, 4096), (1, 128, 4096)),
        ((32, 128, 4096),),
        {},
        "test",
    )

    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        z3.Or(
            left[1] != 1,
            right[0] != 1,
            left[2] != right[2],
            output[0] != left[0],
            output[1] != right[1],
            output[2] != left[2],
        )
    )
    assert solver.check() == z3.unsat


def test_reduced_constraints_preserve_missing_leading_dimensions() -> None:
    left = _symbolic_shape("a", 3)
    right = _symbolic_shape("b", 1)
    output = _symbolic_shape("c", 3)
    constraints = MulOperator().shape_constraints(
        (left, right),
        (output,),
        ((32, 128, 4096), (4096,)),
        ((32, 128, 4096),),
        {},
        "test",
    )

    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        z3.Or(
            output[0] != left[0],
            output[1] != left[1],
            output[2] != left[2],
            left[2] != right[0],
        )
    )
    assert solver.check() == z3.unsat


def test_broadcast_index_projects_trailing_and_singleton_dimensions() -> None:
    output_index = (1, 2, 3)
    output_shape = (2, 3, 4)

    assert broadcast_index(output_index, (4,), output_shape) == (3,)
    assert broadcast_index(output_index, (2, 1, 4), output_shape) == (1, 0, 3)
    assert broadcast_index(output_index, (1, 3, 4), output_shape) == (0, 2, 3)


def test_add_symbolic_execution_uses_broadcast_index_projection() -> None:
    left = create_symbolic_input_tensor("A", (2, 1, 4), "single")
    right = create_symbolic_input_tensor("B", (1, 3, 4), "single")
    output = AddOperator().symbolic_execute(
        (left, right), ((2, 3, 4),), {}, "test"
    )[0]

    _assert_equivalent(
        output.at((1, 2, 3)),
        left.at((1, 0, 3)) + right.at((0, 2, 3)),
    )


def test_mul_symbolic_execution_uses_broadcast_index_projection() -> None:
    left = create_symbolic_input_tensor("A", (2, 1, 4), "single")
    right = create_symbolic_input_tensor("B", (1, 3, 4), "single")
    output = MulOperator().symbolic_execute(
        (left, right), ((2, 3, 4),), {}, "test"
    )[0]

    _assert_equivalent(
        output.at((1, 2, 3)),
        left.at((1, 0, 3)) * right.at((0, 2, 3)),
    )


def test_broadcast_index_rejects_incompatible_input_shape() -> None:
    with pytest.raises(BroadcastShapeError, match="not broadcast-compatible"):
        broadcast_index((1, 2, 3), (5,), (2, 3, 4))
