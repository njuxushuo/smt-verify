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
    ExpandOperator,
    ReshapeOperator,
    SqueezeOperator,
    TransposeOperator,
    UnsqueezeOperator,
    ViewShapeError,
    canonicalize_axis,
    expand_index,
    infer_reshape_groups,
    ravel_index,
    unravel_index,
)
from src.symbolic.tensor import create_symbolic_input_tensor


def _symbolic_shape(prefix: str, rank: int) -> tuple[z3.ArithRef, ...]:
    return tuple(z3.Int(f"{prefix}{axis}") for axis in range(rank))


def _assert_equivalent(actual: z3.ArithRef, expected: z3.ArithRef) -> None:
    solver = z3.Solver()
    solver.add(actual != expected)
    assert solver.check() == z3.unsat


@pytest.mark.parametrize(
    ("axis", "expected"),
    [(-1, 3), (-2, 2), (-4, 0)],
)
def test_canonicalize_regular_negative_axis(axis: int, expected: int) -> None:
    assert canonicalize_axis(axis, 4) == expected


@pytest.mark.parametrize("axis", [-5, 4])
def test_canonicalize_regular_axis_rejects_out_of_range(axis: int) -> None:
    with pytest.raises(ViewShapeError, match="out of range"):
        canonicalize_axis(axis, 4)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [(-4, 0), (-1, 3), (3, 3)],
)
def test_canonicalize_unsqueeze_axis(axis: int, expected: int) -> None:
    assert canonicalize_axis(axis, 3, allow_end=True) == expected


@pytest.mark.parametrize("axis", [-5, 4])
def test_canonicalize_unsqueeze_axis_rejects_out_of_range(axis: int) -> None:
    with pytest.raises(ViewShapeError, match="out of range"):
        canonicalize_axis(axis, 3, allow_end=True)


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "attrs"),
    [
        ((2, 3, 4), (2, 4, 3), {"dim0": 1, "dim1": 2}),
        ((2, 3, 4), (2, 4, 3), {"dim0": -1, "dim1": -2}),
        ((2, 3, 4), (2, 3, 4), {"dim0": 1, "dim1": 1}),
        ((2, 3, 4, 5), (5, 3, 4, 2), {"dim0": 0, "dim1": 3}),
    ],
)
def test_transpose_valid_concrete_shapes(input_shape, output_shape, attrs) -> None:
    TransposeOperator().validate_concrete_shapes(
        (input_shape,), (output_shape,), attrs, "test"
    )


def test_transpose_rejects_invalid_axis_and_wrong_output() -> None:
    operator = TransposeOperator()
    with pytest.raises(ConcreteShapeError, match="out of range"):
        operator.validate_concrete_shapes(
            ((2, 3, 4),), ((2, 4, 3),), {"dim0": 1, "dim1": 3}, "test"
        )
    with pytest.raises(ConcreteShapeError, match="output shape"):
        operator.validate_concrete_shapes(
            ((2, 3, 4),), ((2, 3, 4),), {"dim0": 1, "dim1": 2}, "test"
        )


def test_transpose_reduction_is_only_axis_permutation() -> None:
    input_shape = _symbolic_shape("x", 4)
    output_shape = _symbolic_shape("y", 4)
    constraints = TransposeOperator().shape_constraints(
        (input_shape,),
        (output_shape,),
        ((2, 3, 4, 5),),
        ((5, 3, 4, 2),),
        {"dim0": 0, "dim1": -1},
        "test",
    )
    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(*(dimension == 1 for dimension in (*input_shape, *output_shape)))
    assert solver.check() == z3.sat


def test_transpose_symbolic_index_mapping() -> None:
    input_tensor = create_symbolic_input_tensor("X", (2, 3, 4), "single")
    output = TransposeOperator().symbolic_execute(
        (input_tensor,), ((2, 4, 3),), {"dim0": 1, "dim1": -1}, "test"
    )[0]

    _assert_equivalent(output.at((1, 3, 2)), input_tensor.at((1, 2, 3)))


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [((2, 3, 4), (6, 4)), ((2, 3, 4), (2, 12)), ((2, 3, 4), (24,))],
)
def test_reshape_valid_concrete_shapes(input_shape, output_shape) -> None:
    ReshapeOperator().validate_concrete_shapes(
        (input_shape,), (output_shape,), {"shape": list(output_shape)}, "test"
    )


def test_reshape_rejects_invalid_numel_and_attrs_output_mismatch() -> None:
    operator = ReshapeOperator()
    with pytest.raises(ConcreteShapeError, match="element counts"):
        operator.validate_concrete_shapes(
            ((2, 3, 4),), ((5, 4),), {"shape": [5, 4]}, "test"
        )
    with pytest.raises(ConcreteShapeError, match="attrs shape"):
        operator.validate_concrete_shapes(
            ((2, 3, 4),), ((6, 4),), {"shape": [2, 12]}, "test"
        )


def test_infer_reshape_groups_preserves_contiguous_factorization() -> None:
    assert infer_reshape_groups((2, 3, 4), (6, 4)) == (
        ((0, 1), (0,)),
        ((2,), (1,)),
    )
    assert infer_reshape_groups((2, 3, 4), (2, 12)) == (
        ((0,), (0,)),
        ((1, 2), (1,)),
    )
    assert infer_reshape_groups((2, 3, 4), (24,)) == (
        ((0, 1, 2), (0,)),
    )
    assert infer_reshape_groups((1,), (1, 1)) == (
        ((0,), (0, 1)),
    )


def test_reshape_reduced_groups_are_preserved() -> None:
    input_shape = _symbolic_shape("x", 3)
    output_shape = _symbolic_shape("y", 2)
    constraints = ReshapeOperator().shape_constraints(
        (input_shape,),
        (output_shape,),
        ((2, 3, 4),),
        ((6, 4),),
        {"shape": [6, 4]},
        "test",
    )
    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        z3.Or(
            input_shape[0] * input_shape[1] != output_shape[0],
            input_shape[2] != output_shape[1],
        )
    )
    assert solver.check() == z3.unsat


def test_row_major_index_helpers_and_reshape_symbolic_mapping() -> None:
    assert ravel_index((4, 2), (6, 4)) == 18
    assert unravel_index(18, (2, 3, 4)) == (1, 1, 2)

    input_tensor = create_symbolic_input_tensor("X", (2, 3, 4), "single")
    output = ReshapeOperator().symbolic_execute(
        (input_tensor,), ((6, 4),), {"shape": [6, 4]}, "test"
    )[0]
    _assert_equivalent(output.at((4, 2)), input_tensor.at((1, 1, 2)))


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "attrs"),
    [
        ((2, 1, 3), (2, 3), {"dim": 1}),
        ((2, 3, 1), (2, 3), {"dim": -1}),
        ((1, 2, 3), (2, 3), {}),
        ((1, 2, 1, 3), (2, 3), {}),
    ],
)
def test_squeeze_supports_explicit_and_omitted_dims(input_shape, output_shape, attrs) -> None:
    SqueezeOperator().validate_concrete_shapes(
        (input_shape,), (output_shape,), attrs, "test"
    )


def test_squeeze_rejects_non_singleton_dim_and_wrong_output() -> None:
    operator = SqueezeOperator()
    with pytest.raises(ConcreteShapeError, match="must have extent 1"):
        operator.validate_concrete_shapes(
            ((2, 3, 1),), ((2, 3),), {"dim": 1}, "test"
        )
    with pytest.raises(ConcreteShapeError, match="output shape"):
        operator.validate_concrete_shapes(
            ((2, 1, 3),), ((2, 1),), {"dim": 1}, "test"
        )


def test_squeeze_reduction_preserves_singletons_and_omitted_axis_identity() -> None:
    input_shape = _symbolic_shape("x", 4)
    output_shape = _symbolic_shape("y", 2)
    constraints = SqueezeOperator().shape_constraints(
        (input_shape,),
        (output_shape,),
        ((1, 2, 1, 3),),
        ((2, 3),),
        {},
        "test",
    )
    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        z3.Or(
            input_shape[0] != 1,
            input_shape[2] != 1,
            input_shape[1] < 2,
            input_shape[3] < 2,
            output_shape[0] != input_shape[1],
            output_shape[1] != input_shape[3],
        )
    )
    assert solver.check() == z3.unsat


def test_squeeze_symbolic_mapping_for_explicit_and_omitted_dims() -> None:
    explicit_input = create_symbolic_input_tensor("A", (2, 1, 3), "single")
    explicit = SqueezeOperator().symbolic_execute(
        (explicit_input,), ((2, 3),), {"dim": 1}, "test"
    )[0]
    _assert_equivalent(explicit.at((1, 2)), explicit_input.at((1, 0, 2)))

    omitted_input = create_symbolic_input_tensor("B", (1, 2, 1, 3), "single")
    omitted = SqueezeOperator().symbolic_execute(
        (omitted_input,), ((2, 3),), {}, "test"
    )[0]
    _assert_equivalent(omitted.at((1, 2)), omitted_input.at((0, 1, 0, 2)))


@pytest.mark.parametrize(
    ("dim", "output_shape"),
    [(0, (1, 2, 3)), (-1, (2, 3, 1)), (-3, (1, 2, 3))],
)
def test_unsqueeze_supports_positive_and_negative_boundaries(dim, output_shape) -> None:
    UnsqueezeOperator().validate_concrete_shapes(
        ((2, 3),), (output_shape,), {"dim": dim}, "test"
    )


def test_unsqueeze_rejects_invalid_axis_and_wrong_output() -> None:
    operator = UnsqueezeOperator()
    with pytest.raises(ConcreteShapeError, match="out of range"):
        operator.validate_concrete_shapes(
            ((2, 3),), ((2, 3, 1),), {"dim": 3}, "test"
        )
    with pytest.raises(ConcreteShapeError, match="output shape"):
        operator.validate_concrete_shapes(
            ((2, 3),), ((2, 1, 3),), {"dim": -1}, "test"
        )


def test_unsqueeze_reduction_and_symbolic_mapping() -> None:
    input_shape = _symbolic_shape("x", 2)
    output_shape = _symbolic_shape("y", 3)
    constraints = UnsqueezeOperator().shape_constraints(
        (input_shape,),
        (output_shape,),
        ((2, 3),),
        ((2, 1, 3),),
        {"dim": 1},
        "test",
    )
    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        z3.Or(
            output_shape[0] != input_shape[0],
            output_shape[1] != 1,
            output_shape[2] != input_shape[1],
        )
    )
    assert solver.check() == z3.unsat

    input_tensor = create_symbolic_input_tensor("X", (2, 3), "single")
    output = UnsqueezeOperator().symbolic_execute(
        (input_tensor,), ((2, 1, 3),), {"dim": 1}, "test"
    )[0]
    _assert_equivalent(output.at((1, 0, 2)), input_tensor.at((1, 2)))


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [
        ((2, 1, 4), (2, 3, 4)),
        ((4,), (2, 3, 4)),
        ((2, 3, 4), (2, 3, 4)),
    ],
)
def test_expand_valid_trailing_aligned_shapes(input_shape, output_shape) -> None:
    ExpandOperator().validate_concrete_shapes(
        (input_shape,), (output_shape,), {"shape": list(output_shape)}, "test"
    )


def test_expand_rejects_non_singleton_change_and_attrs_mismatch() -> None:
    operator = ExpandOperator()
    with pytest.raises(ConcreteShapeError, match="non-singleton"):
        operator.validate_concrete_shapes(
            ((2, 2, 4),), ((2, 3, 4),), {"shape": [2, 3, 4]}, "test"
        )
    with pytest.raises(ConcreteShapeError, match="attrs shape"):
        operator.validate_concrete_shapes(
            ((2, 1, 4),), ((2, 3, 4),), {"shape": [1, 3, 4]}, "test"
        )


def test_expand_reduction_preserves_original_singleton_pattern() -> None:
    input_shape = _symbolic_shape("x", 3)
    output_shape = _symbolic_shape("y", 3)
    constraints = ExpandOperator().shape_constraints(
        (input_shape,),
        (output_shape,),
        ((2, 1, 4),),
        ((2, 3, 4),),
        {"shape": [2, 3, 4]},
        "test",
    )
    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(
        z3.Or(
            input_shape[0] != output_shape[0],
            input_shape[1] != 1,
            input_shape[2] != output_shape[2],
        )
    )
    assert solver.check() == z3.unsat

    solver = z3.Solver()
    solver.add(*constraints, output_shape[1] == 2)
    assert solver.check() == z3.sat


def test_expand_symbolic_broadcast_index_mapping() -> None:
    input_tensor = create_symbolic_input_tensor("X", (2, 1, 4), "single")
    output = ExpandOperator().symbolic_execute(
        (input_tensor,), ((2, 3, 4),), {"shape": [2, 3, 4]}, "test"
    )[0]

    assert expand_index((1, 2, 3), input_tensor.shape, output.shape) == (1, 0, 3)
    _assert_equivalent(output.at((1, 2, 3)), input_tensor.at((1, 0, 3)))


@pytest.mark.parametrize(
    ("operator", "input_shape", "output_shape", "attrs", "message"),
    [
        (TransposeOperator(), (2, 3), (3, 2), {"dim0": 0}, "missing required"),
        (TransposeOperator(), (2, 3), (3, 2), {"dim0": 0, "dim1": "1"}, "integer"),
        (TransposeOperator(), (2, 3), (3, 2), {"dim0": 0, "dim1": 1, "x": 1}, "unknown"),
        (ReshapeOperator(), (2, 3), (6,), {}, "missing required"),
        (ReshapeOperator(), (2, 3), (6,), {"shape": [6], "x": 1}, "unknown"),
        (ReshapeOperator(), (2, 3), (6,), {"shape": [-1]}, "positive integer"),
        (SqueezeOperator(), (2, 1), (2,), {"dim": True}, "integer"),
        (SqueezeOperator(), (2, 1), (2,), {"x": 1}, "unknown"),
        (UnsqueezeOperator(), (2, 3), (2, 1, 3), {}, "missing required"),
        (UnsqueezeOperator(), (2, 3), (2, 1, 3), {"dim": 1, "x": 1}, "unknown"),
        (ExpandOperator(), (1, 3), (2, 3), {}, "missing required"),
        (ExpandOperator(), (1, 3), (2, 3), {"shape": [2, 3], "x": 1}, "unknown"),
    ],
)
def test_view_attrs_schema_is_validated_during_concrete_validation(
    operator, input_shape, output_shape, attrs, message
) -> None:
    with pytest.raises(ConcreteShapeError, match=message):
        operator.validate_concrete_shapes(
            (input_shape,), (output_shape,), attrs, "test"
        )
