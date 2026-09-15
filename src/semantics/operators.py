"""Operator-specific Stage shape semantics and the shared operator registry."""

from __future__ import annotations

from itertools import product

import z3

from ..shape.model import ShapeReductionError, UnsupportedShapeSemanticsError
from ..symbolic.tensor import SymbolicExecutionError, SymbolicTensor
from .broadcast import (
    BroadcastShapeError,
    broadcast_index,
    broadcast_shape_constraints,
    infer_broadcast_shape,
)


class ConcreteShapeError(ValueError):
    """Raised when original concrete operator shapes are not legal."""


class MatMulShapeError(ValueError):
    """Raised when concrete shapes violate batched MatMul semantics."""


def infer_matmul_output_shape(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    context: str | None = None,
) -> tuple[int, ...]:
    """Infer a rank >= 2 MatMul result, broadcasting only batch dimensions."""

    prefix = f"{context}: " if context else ""
    if len(left_shape) < 2 or len(right_shape) < 2:
        raise MatMulShapeError(f"{prefix}matmul inputs must have rank at least 2")
    if left_shape[-1] != right_shape[-2]:
        raise MatMulShapeError(
            f"{prefix}matmul input dimensions do not match: contraction "
            f"{left_shape[-1]} != {right_shape[-2]}"
        )
    try:
        output_batch_shape = infer_broadcast_shape(
            left_shape[:-2],
            right_shape[:-2],
            f"{context}: matmul batch" if context else "matmul batch",
        )
    except BroadcastShapeError as exc:
        raise MatMulShapeError(str(exc)) from exc
    return output_batch_shape + (left_shape[-2], right_shape[-1])


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
        if len(left) < 2 or len(right) < 2:
            raise ConcreteShapeError(f"{context}: matmul inputs must have rank at least 2")
        if len(output) < 2:
            raise ConcreteShapeError(f"{context}: matmul output must have rank at least 2")
        try:
            expected_output = infer_matmul_output_shape(left, right, context)
        except MatMulShapeError as exc:
            raise ConcreteShapeError(str(exc)) from exc
        if output != expected_output:
            raise ConcreteShapeError(
                f"{context}: matmul output shape {output} does not match {expected_output}"
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
        if len(left) < 2 or len(right) < 2 or len(output) < 2:
            raise ShapeReductionError(f"{context}: matmul tensors must have rank at least 2")
        if len(original_input_shapes) != 2 or len(original_output_shapes) != 1:
            raise ShapeReductionError(
                f"{context}: matmul requires original shapes for two inputs and one output"
            )
        original_left, original_right = original_input_shapes
        (original_output,) = original_output_shapes
        if len(original_left) < 2 or len(original_right) < 2 or len(original_output) < 2:
            raise ShapeReductionError(
                f"{context}: original matmul shapes must have rank at least 2"
            )
        try:
            constraints = broadcast_shape_constraints(
                left[:-2],
                right[:-2],
                output[:-2],
                original_left[:-2],
                original_right[:-2],
                original_output[:-2],
                f"{context}: matmul batch",
            )
        except BroadcastShapeError as exc:
            raise ShapeReductionError(str(exc)) from exc
        constraints.extend(
            (
                left[-1] == right[-2],
                output[-2] == left[-2],
                output[-1] == right[-1],
                left[-1] >= min(2, original_left[-1]),
            )
        )
        return constraints

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
        if len(left.shape) < 2 or len(right.shape) < 2:
            raise SymbolicExecutionError(
                f"{context}: matmul inputs must have rank at least 2"
            )
        if len(output_shape) < 2:
            raise SymbolicExecutionError(
                f"{context}: matmul output must have rank at least 2"
            )
        try:
            expected_shape = infer_matmul_output_shape(left.shape, right.shape, context)
        except MatMulShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
        if output_shape != expected_shape:
            raise SymbolicExecutionError(
                f"{context}: matmul output shape {output_shape} does not match {expected_shape}"
            )

        left_batch_shape = left.shape[:-2]
        right_batch_shape = right.shape[:-2]
        output_batch_shape = output_shape[:-2]
        rows, shared = left.shape[-2:]
        columns = right.shape[-1]
        values: list[z3.ArithRef] = []
        for batch_index in product(
            *(range(extent) for extent in output_batch_shape)
        ):
            left_batch_index = broadcast_index(
                batch_index,
                left_batch_shape,
                output_batch_shape,
                f"{context}: matmul left batch",
            )
            right_batch_index = broadcast_index(
                batch_index,
                right_batch_shape,
                output_batch_shape,
                f"{context}: matmul right batch",
            )
            for row in range(rows):
                for column in range(columns):
                    values.append(
                        z3.Sum(
                            [
                                left.at(left_batch_index + (row, shared_index))
                                * right.at(
                                    right_batch_index + (shared_index, column)
                                )
                                for shared_index in range(shared)
                            ]
                        )
                    )
        return (SymbolicTensor(shape=output_shape, values=tuple(values)),)


class AddOperator(OperatorSemantics):
    name = "add"

    def validate_concrete_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        context: str,
    ) -> None:
        if len(input_shapes) != 2 or len(output_shapes) != 1:
            raise ConcreteShapeError(f"{context}: add requires exactly two inputs and one output")
        left, right = input_shapes
        (output,) = output_shapes
        try:
            expected_output = infer_broadcast_shape(left, right, f"{context}: add")
        except BroadcastShapeError as exc:
            raise ConcreteShapeError(str(exc)) from exc
        if output != expected_output:
            raise ConcreteShapeError(
                f"{context}: add output shape {output} does not match "
                f"broadcast shape {expected_output}"
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
                f"{context}: add requires exactly two inputs and one output"
            )
        left, right = input_shapes
        (output,) = output_shapes
        if len(original_input_shapes) != 2 or len(original_output_shapes) != 1:
            raise ShapeReductionError(
                f"{context}: add requires original shapes for two inputs and one output"
            )
        original_left, original_right = original_input_shapes
        (original_output,) = original_output_shapes
        try:
            return broadcast_shape_constraints(
                left,
                right,
                output,
                original_left,
                original_right,
                original_output,
                f"{context}: add",
            )
        except BroadcastShapeError as exc:
            raise ShapeReductionError(str(exc)) from exc

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
        try:
            expected_output = infer_broadcast_shape(left.shape, right.shape, f"{context}: add")
        except BroadcastShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
        if output_shape != expected_output:
            raise SymbolicExecutionError(
                f"{context}: add output shape {output_shape} does not match "
                f"broadcast shape {expected_output}"
            )
        return (
            SymbolicTensor(
                shape=output_shape,
                values=tuple(
                    left.at(broadcast_index(index, left.shape, output_shape))
                    + right.at(broadcast_index(index, right.shape, output_shape))
                    for index in product(*(range(extent) for extent in output_shape))
                ),
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
        if len(input_shapes) != 2 or len(output_shapes) != 1:
            raise ConcreteShapeError(f"{context}: mul requires exactly two inputs and one output")
        left, right = input_shapes
        (output,) = output_shapes
        try:
            expected_output = infer_broadcast_shape(left, right, f"{context}: mul")
        except BroadcastShapeError as exc:
            raise ConcreteShapeError(str(exc)) from exc
        if output != expected_output:
            raise ConcreteShapeError(
                f"{context}: mul output shape {output} does not match "
                f"broadcast shape {expected_output}"
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
                f"{context}: mul requires exactly two inputs and one output"
            )
        left, right = input_shapes
        (output,) = output_shapes
        if len(original_input_shapes) != 2 or len(original_output_shapes) != 1:
            raise ShapeReductionError(
                f"{context}: mul requires original shapes for two inputs and one output"
            )
        original_left, original_right = original_input_shapes
        (original_output,) = original_output_shapes
        try:
            return broadcast_shape_constraints(
                left,
                right,
                output,
                original_left,
                original_right,
                original_output,
                f"{context}: mul",
            )
        except BroadcastShapeError as exc:
            raise ShapeReductionError(str(exc)) from exc

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
        try:
            expected_output = infer_broadcast_shape(left.shape, right.shape, f"{context}: mul")
        except BroadcastShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
        if output_shape != expected_output:
            raise SymbolicExecutionError(
                f"{context}: mul output shape {output_shape} does not match "
                f"broadcast shape {expected_output}"
            )
        return (
            SymbolicTensor(
                shape=output_shape,
                values=tuple(
                    left.at(broadcast_index(index, left.shape, output_shape))
                    * right.at(broadcast_index(index, right.shape, output_shape))
                    for index in product(*(range(extent) for extent in output_shape))
                ),
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
