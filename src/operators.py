"""Operator-specific Stage shape semantics and the shared operator registry."""

from __future__ import annotations

import z3

from .shape_model import ShapeReductionError, UnsupportedShapeSemanticsError
from .symbolic_tensor import SymbolicExecutionError, SymbolicTensor


class ConcreteShapeError(ValueError):
    """Raised when original concrete operator shapes are not legal."""


DECLARED_OPERATOR_TYPES = frozenset(
    {
        "add",
        "mul",
        "matmul",
        "transpose",
        "reshape",
        "sum",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
    }
)


class OperatorSemantics:
    name: str

    def validate_concrete_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> None:
        raise NotImplementedError

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_input_shapes: tuple[tuple[int, ...], ...],
        original_output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> list[z3.BoolRef]:
        raise NotImplementedError

    def symbolic_execute(
        self,
        input_tensors: tuple[SymbolicTensor, ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> tuple[SymbolicTensor, ...]:
        raise NotImplementedError


def _same_shape_constraints(
    left: tuple[z3.ArithRef, ...],
    right: tuple[z3.ArithRef, ...],
    context: str,
) -> list[z3.BoolRef]:
    if len(left) != len(right):
        raise ShapeReductionError(f"{context}: tensors must have the same rank")
    return [left[index] == right[index] for index in range(len(left))]


def _validate_concrete_same_shapes(
    operator_name: str,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    context: str,
) -> None:
    if len(input_shapes) != 2 or len(output_shapes) != 1:
        raise ConcreteShapeError(
            f"{context}: {operator_name} requires exactly two inputs and one output"
        )
    left, right = input_shapes
    (output,) = output_shapes
    if left != right or left != output:
        raise ConcreteShapeError(f"{context}: {operator_name} tensors must have the same shape")


class MatMulOperator(OperatorSemantics):
    name = "matmul"

    def validate_concrete_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> None:
        if len(input_shapes) != 2 or len(output_shapes) != 1:
            raise ConcreteShapeError(
                f"{context}: matmul requires exactly two inputs and one output"
            )
        left, right = input_shapes
        (output,) = output_shapes
        if len(left) != 2 or len(right) != 2 or len(output) != 2:
            raise ConcreteShapeError(f"{context}: matmul currently supports only 2-D tensors")
        if left[1] != right[0]:
            raise ConcreteShapeError(f"{context}: matmul input dimensions do not match")
        if output != (left[0], right[1]):
            raise ConcreteShapeError(
                f"{context}: matmul output shape {output} does not match {(left[0], right[1])}"
            )

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_input_shapes: tuple[tuple[int, ...], ...],
        original_output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> list[z3.BoolRef]:
        if len(input_shapes) != 2 or len(output_shapes) != 1:
            raise ShapeReductionError(
                f"{context}: matmul requires exactly two inputs and one output"
            )
        left, right = input_shapes
        (output,) = output_shapes
        if len(left) != 2 or len(right) != 2 or len(output) != 2:
            raise ShapeReductionError(f"{context}: matmul currently supports only 2-D tensors")
        if len(original_input_shapes) != 2 or len(original_output_shapes) != 1:
            raise ShapeReductionError(
                f"{context}: matmul requires original shapes for two inputs and one output"
            )
        original_left, original_right = original_input_shapes
        (original_output,) = original_output_shapes
        if len(original_left) != 2 or len(original_right) != 2 or len(original_output) != 2:
            raise ShapeReductionError(f"{context}: original matmul shapes must be 2-D")
        return [
            left[1] == right[0],
            output[0] == left[0],
            output[1] == right[1],
            left[1] >= min(2, original_left[1]),
        ]

    def symbolic_execute(
        self,
        input_tensors: tuple[SymbolicTensor, ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> tuple[SymbolicTensor, ...]:
        if len(input_tensors) != 2 or len(output_shapes) != 1:
            raise SymbolicExecutionError(
                f"{context}: matmul requires exactly two inputs and one output"
            )
        left, right = input_tensors
        (output_shape,) = output_shapes
        if len(left.shape) != 2 or len(right.shape) != 2 or len(output_shape) != 2:
            raise SymbolicExecutionError(f"{context}: matmul currently supports only 2-D tensors")
        if left.shape[1] != right.shape[0]:
            raise SymbolicExecutionError(f"{context}: matmul input dimensions do not match")
        expected_shape = (left.shape[0], right.shape[1])
        if output_shape != expected_shape:
            raise SymbolicExecutionError(
                f"{context}: matmul output shape {output_shape} does not match {expected_shape}"
            )

        rows, shared = left.shape
        columns = right.shape[1]
        values = tuple(
            z3.Sum([left.at((row, shared_index)) * right.at((shared_index, column))
                    for shared_index in range(shared)])
            for row in range(rows)
            for column in range(columns)
        )
        return (SymbolicTensor(shape=output_shape, values=values),)


class AddOperator(OperatorSemantics):
    name = "add"

    def validate_concrete_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> None:
        _validate_concrete_same_shapes("add", input_shapes, output_shapes, context)

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_input_shapes: tuple[tuple[int, ...], ...],
        original_output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> list[z3.BoolRef]:
        if len(input_shapes) != 2 or len(output_shapes) != 1:
            raise ShapeReductionError(
                f"{context}: add requires exactly two inputs and one output"
            )
        left, right = input_shapes
        (output,) = output_shapes
        return _same_shape_constraints(left, right, context) + _same_shape_constraints(
            left, output, context
        )

    def symbolic_execute(
        self,
        input_tensors: tuple[SymbolicTensor, ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> tuple[SymbolicTensor, ...]:
        if len(input_tensors) != 2 or len(output_shapes) != 1:
            raise SymbolicExecutionError(f"{context}: add requires exactly two inputs and one output")
        left, right = input_tensors
        (output_shape,) = output_shapes
        if left.shape != right.shape or left.shape != output_shape:
            raise SymbolicExecutionError(f"{context}: add tensors must have the same shape")
        return (
            SymbolicTensor(
                shape=output_shape,
                values=tuple(left_value + right_value for left_value, right_value in zip(left.values, right.values)),
            ),
        )


class MulOperator(OperatorSemantics):
    name = "mul"

    def validate_concrete_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> None:
        _validate_concrete_same_shapes("mul", input_shapes, output_shapes, context)

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_input_shapes: tuple[tuple[int, ...], ...],
        original_output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> list[z3.BoolRef]:
        if len(input_shapes) != 2 or len(output_shapes) != 1:
            raise ShapeReductionError(
                f"{context}: mul requires exactly two inputs and one output"
            )
        left, right = input_shapes
        (output,) = output_shapes
        return _same_shape_constraints(left, right, context) + _same_shape_constraints(
            left, output, context
        )

    def symbolic_execute(
        self,
        input_tensors: tuple[SymbolicTensor, ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> tuple[SymbolicTensor, ...]:
        if len(input_tensors) != 2 or len(output_shapes) != 1:
            raise SymbolicExecutionError(f"{context}: mul requires exactly two inputs and one output")
        left, right = input_tensors
        (output_shape,) = output_shapes
        if left.shape != right.shape or left.shape != output_shape:
            raise SymbolicExecutionError(f"{context}: mul tensors must have the same shape")
        return (
            SymbolicTensor(
                shape=output_shape,
                values=tuple(left_value * right_value for left_value, right_value in zip(left.values, right.values)),
            ),
        )


OPERATOR_REGISTRY: dict[str, OperatorSemantics] = {
    "matmul": MatMulOperator(),
    "add": AddOperator(),
    "mul": MulOperator(),
}


def get_operator(name: str, context: str | None = None) -> OperatorSemantics:
    """Return implemented semantics or raise a contextual shape-support error."""

    prefix = f"{context}: " if context else ""
    if name in OPERATOR_REGISTRY:
        return OPERATOR_REGISTRY[name]
    if name in DECLARED_OPERATOR_TYPES:
        raise UnsupportedShapeSemanticsError(
            f"{prefix}shape semantics for operator {name!r} are not implemented"
        )
    raise ShapeReductionError(f"{prefix}unknown operator {name!r}")
