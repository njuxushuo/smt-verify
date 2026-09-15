from __future__ import annotations

import sys
from pathlib import Path

import pytest
import z3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantics.operators import (
    ConcreteShapeError,
    MatMulOperator,
    infer_matmul_output_shape,
)
from src.symbolic.tensor import create_symbolic_input_tensor


def _symbolic_shape(prefix: str, rank: int) -> tuple[z3.ArithRef, ...]:
    return tuple(z3.Int(f"{prefix}{axis}") for axis in range(rank))


@pytest.mark.parametrize(
    ("left", "right", "output"),
    [
        ((3, 4), (4, 5), (3, 5)),
        ((2, 3, 4), (4, 5), (2, 3, 5)),
        ((3, 4), (2, 4, 5), (2, 3, 5)),
        ((2, 3, 4), (2, 4, 5), (2, 3, 5)),
        ((2, 3, 4, 5), (2, 3, 5, 6), (2, 3, 4, 6)),
        ((2, 3, 4, 5), (1, 3, 5, 6), (2, 3, 4, 6)),
        ((2, 3, 4, 5), (3, 5, 6), (2, 3, 4, 6)),
    ],
)
def test_matmul_valid_shapes_support_batch_broadcast(
    left: tuple[int, ...],
    right: tuple[int, ...],
    output: tuple[int, ...],
) -> None:
    assert infer_matmul_output_shape(left, right) == output
    MatMulOperator().validate_concrete_shapes((left, right), (output,), {}, "test")


def test_matmul_rejects_invalid_batch_broadcast() -> None:
    with pytest.raises(ConcreteShapeError, match="not broadcast-compatible"):
        MatMulOperator().validate_concrete_shapes(
            ((2, 3, 4, 5), (4, 3, 5, 6)),
            ((2, 3, 4, 6),),
            {},
            "test",
        )


def test_matmul_rejects_invalid_contraction_dimension() -> None:
    with pytest.raises(ConcreteShapeError, match="input dimensions do not match"):
        MatMulOperator().validate_concrete_shapes(
            ((2, 3, 4, 5), (1, 3, 6, 7)),
            ((2, 3, 4, 7),),
            {},
            "test",
        )


def test_matmul_rejects_wrong_declared_output_shape() -> None:
    with pytest.raises(ConcreteShapeError, match="output shape"):
        MatMulOperator().validate_concrete_shapes(
            ((2, 3, 4, 5), (1, 3, 5, 6)),
            ((2, 3, 4, 7),),
            {},
            "test",
        )


def test_matmul_rejects_output_rank_below_two() -> None:
    with pytest.raises(ConcreteShapeError, match="output must have rank at least 2"):
        MatMulOperator().validate_concrete_shapes(
            ((3, 4), (4, 5)),
            ((3,),),
            {},
            "test",
        )


@pytest.mark.parametrize(
    ("left", "right", "output"),
    [
        ((4,), (4,), ()),
        ((4,), (4, 5), (5,)),
        ((3, 4), (4,), (3,)),
    ],
)
def test_matmul_rejects_rank_one_forms(
    left: tuple[int, ...],
    right: tuple[int, ...],
    output: tuple[int, ...],
) -> None:
    with pytest.raises(ConcreteShapeError, match="rank at least 2"):
        MatMulOperator().validate_concrete_shapes((left, right), (output,), {}, "test")


def test_matmul_reduced_constraints_preserve_batch_singleton_pattern() -> None:
    left = _symbolic_shape("a", 4)
    right = _symbolic_shape("b", 4)
    output = _symbolic_shape("c", 4)
    constraints = MatMulOperator().shape_constraints(
        (left, right),
        (output,),
        ((2, 1, 4, 5), (1, 3, 5, 6)),
        ((2, 3, 4, 6),),
        {},
        "test",
    )

    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        z3.Or(
            left[1] != 1,
            right[0] != 1,
            output[0] != left[0],
            output[1] != right[1],
            left[-1] != right[-2],
            output[-2] != left[-2],
            output[-1] != right[-1],
        )
    )
    assert solver.check() == z3.unsat


def test_matmul_reduced_constraints_preserve_missing_leading_batch_dimension() -> None:
    left = _symbolic_shape("a", 4)
    right = _symbolic_shape("b", 3)
    output = _symbolic_shape("c", 4)
    constraints = MatMulOperator().shape_constraints(
        (left, right),
        (output,),
        ((2, 3, 4, 5), (3, 5, 6)),
        ((2, 3, 4, 6),),
        {},
        "test",
    )

    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        z3.Or(
            output[0] != left[0],
            left[1] != right[0],
            output[1] != left[1],
        )
    )
    assert solver.check() == z3.unsat


def test_matmul_reduced_constraints_preserve_k_minimum_two() -> None:
    left = _symbolic_shape("a", 3)
    right = _symbolic_shape("b", 3)
    output = _symbolic_shape("c", 3)
    constraints = MatMulOperator().shape_constraints(
        (left, right),
        (output,),
        ((2, 3, 8), (2, 8, 5)),
        ((2, 3, 5),),
        {},
        "test",
    )

    solver = z3.Solver()
    solver.add(*constraints, left[-1] < 2)
    assert solver.check() == z3.unsat


def test_matmul_reduced_constraints_allow_k_one_when_original_k_is_one() -> None:
    left = _symbolic_shape("a", 3)
    right = _symbolic_shape("b", 3)
    output = _symbolic_shape("c", 3)
    constraints = MatMulOperator().shape_constraints(
        (left, right),
        (output,),
        ((2, 3, 1), (2, 1, 5)),
        ((2, 3, 5),),
        {},
        "test",
    )

    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(*(dimension == 1 for shape in (left, right, output) for dimension in shape))
    assert solver.check() == z3.sat


def test_matmul_reduced_constraints_do_not_require_m_or_n_two() -> None:
    left = _symbolic_shape("a", 3)
    right = _symbolic_shape("b", 3)
    output = _symbolic_shape("c", 3)
    constraints = MatMulOperator().shape_constraints(
        (left, right),
        (output,),
        ((2, 7, 8), (2, 8, 9)),
        ((2, 7, 9),),
        {},
        "test",
    )

    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(left[-2] == 1, right[-1] == 1, left[-1] == 2)
    assert solver.check() == z3.sat


def test_matmul_symbolic_execution_projects_broadcast_batch_indices() -> None:
    left = create_symbolic_input_tensor("A", (2, 1, 2, 3), "single")
    right = create_symbolic_input_tensor("B", (1, 4, 3, 2), "single")
    output = MatMulOperator().symbolic_execute(
        (left, right),
        ((2, 4, 2, 2),),
        {},
        "test",
    )[0]
    expected = z3.Sum(
        [
            left.at((1, 0, 0, shared)) * right.at((0, 3, shared, 1))
            for shared in range(3)
        ]
    )

    solver = z3.Solver()
    solver.add(output.at((1, 3, 0, 1)) != expected)
    assert solver.check() == z3.unsat
