"""Operator semantic interfaces, implementations, and registry."""

from __future__ import annotations

from typing import TypeAlias

import z3

from ...shape.model import ShapeReductionError, UnsupportedShapeSemanticsError
from ...symbolic.tensor import SymbolicTensor


OperatorAttrs: TypeAlias = dict[str, object]


class ConcreteShapeError(ValueError):
    """Raised when original concrete operator shapes are not legal."""


def require_empty_attrs(attrs: OperatorAttrs, context: str, operator: str) -> None:
    """Validate the no-attribute schema used by existing operators."""

    if not isinstance(attrs, dict):
        raise ConcreteShapeError(f"{context}: {operator} attrs must be an object")
    if attrs:
        raise ConcreteShapeError(
            f"{context}: {operator} does not accept attrs: {sorted(attrs)}"
        )


class OperatorSemantics:
    name: str

    def validate_concrete_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        attrs: OperatorAttrs,
        context: str,
    ) -> None:
        raise NotImplementedError

    def shape_constraints(
        self,
        input_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        output_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_input_shapes: tuple[tuple[int, ...], ...],
        original_output_shapes: tuple[tuple[int, ...], ...],
        attrs: OperatorAttrs,
        context: str,
    ) -> list[z3.BoolRef]:
        raise NotImplementedError

    def symbolic_execute(
        self,
        input_tensors: tuple[SymbolicTensor, ...],
        output_shapes: tuple[tuple[int, ...], ...],
        attrs: OperatorAttrs,
        context: str,
    ) -> tuple[SymbolicTensor, ...]:
        raise NotImplementedError


DECLARED_OPERATOR_TYPES = frozenset(
    {
        "add",
        "mul",
        "matmul",
        "transpose",
        "reshape",
        "squeeze",
        "unsqueeze",
        "expand",
        "sum",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
    }
)


from .elementwise import AddOperator, MulOperator
from .matmul import MatMulOperator, MatMulShapeError, infer_matmul_output_shape
from .view import (
    ExpandOperator,
    ReshapeOperator,
    SqueezeOperator,
    TransposeOperator,
    UnsqueezeOperator,
    ViewShapeError,
    canonicalize_axis,
    expand_index,
    expand_shape_constraints,
    infer_reshape_groups,
    ravel_index,
    unravel_index,
)


OPERATOR_REGISTRY: dict[str, OperatorSemantics] = {
    "matmul": MatMulOperator(),
    "add": AddOperator(),
    "mul": MulOperator(),
    "transpose": TransposeOperator(),
    "reshape": ReshapeOperator(),
    "squeeze": SqueezeOperator(),
    "unsqueeze": UnsqueezeOperator(),
    "expand": ExpandOperator(),
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


__all__ = [
    "AddOperator",
    "ConcreteShapeError",
    "DECLARED_OPERATOR_TYPES",
    "ExpandOperator",
    "MatMulOperator",
    "MatMulShapeError",
    "MulOperator",
    "OPERATOR_REGISTRY",
    "OperatorAttrs",
    "OperatorSemantics",
    "ReshapeOperator",
    "SqueezeOperator",
    "TransposeOperator",
    "UnsqueezeOperator",
    "ViewShapeError",
    "canonicalize_axis",
    "expand_index",
    "expand_shape_constraints",
    "get_operator",
    "infer_matmul_output_shape",
    "infer_reshape_groups",
    "ravel_index",
    "unravel_index",
]
