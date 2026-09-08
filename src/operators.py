"""Operator-specific Stage shape semantics and the shared operator registry."""

from __future__ import annotations

import z3

from .shape_model import ShapeReductionError, UnsupportedShapeSemanticsError


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

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        context: str,
    ) -> list[z3.BoolRef]:
        raise NotImplementedError


def _same_shape_constraints(
    left: tuple[z3.ArithRef, ...],
    right: tuple[z3.ArithRef, ...],
    context: str,
) -> list[z3.BoolRef]:
    if len(left) != len(right):
        raise ShapeReductionError(f"{context}: tensors must have the same rank")
    return [left[index] == right[index] for index in range(len(left))]


class MatMulOperator(OperatorSemantics):
    name = "matmul"

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
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
        return [left[1] == right[0], output[0] == left[0], output[1] == right[1]]


class AddOperator(OperatorSemantics):
    name = "add"

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
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


class MulOperator(OperatorSemantics):
    name = "mul"

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
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
