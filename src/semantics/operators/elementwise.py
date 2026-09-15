"""Element-wise Add and Mul semantics with standard broadcasting."""

from __future__ import annotations

from itertools import product

import z3

from ...shape.model import ShapeReductionError
from ...symbolic.tensor import SymbolicExecutionError, SymbolicTensor
from ..broadcast import (
    BroadcastShapeError,
    broadcast_index,
    broadcast_shape_constraints,
    infer_broadcast_shape,
)
from . import ConcreteShapeError, OperatorAttrs, OperatorSemantics, require_empty_attrs


class AddOperator(OperatorSemantics):
    name = "add"

    def validate_concrete_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        attrs: OperatorAttrs,
        context: str,
    ) -> None:
        require_empty_attrs(attrs, context, self.name)
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
        attrs: OperatorAttrs,
        context: str,
    ) -> list[z3.BoolRef]:
        try:
            require_empty_attrs(attrs, context, self.name)
        except ConcreteShapeError as exc:
            raise ShapeReductionError(str(exc)) from exc
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
        attrs: OperatorAttrs,
        context: str,
    ) -> tuple[SymbolicTensor, ...]:
        try:
            require_empty_attrs(attrs, context, self.name)
        except ConcreteShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
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
        attrs: OperatorAttrs,
        context: str,
    ) -> None:
        require_empty_attrs(attrs, context, self.name)
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
        attrs: OperatorAttrs,
        context: str,
    ) -> list[z3.BoolRef]:
        try:
            require_empty_attrs(attrs, context, self.name)
        except ConcreteShapeError as exc:
            raise ShapeReductionError(str(exc)) from exc
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
        attrs: OperatorAttrs,
        context: str,
    ) -> tuple[SymbolicTensor, ...]:
        try:
            require_empty_attrs(attrs, context, self.name)
        except ConcreteShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
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
