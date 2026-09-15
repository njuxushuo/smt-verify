"""Logical view and index-remapping operator semantics."""

from __future__ import annotations

from itertools import product
from math import prod

import z3

from ...shape.model import ShapeReductionError
from ...symbolic.tensor import SymbolicExecutionError, SymbolicTensor
from . import ConcreteShapeError, OperatorAttrs, OperatorSemantics


class ViewShapeError(ValueError):
    """Raised when view attributes or shapes are invalid."""


ReshapeGroup = tuple[tuple[int, ...], tuple[int, ...]]


def _message(context: str | None, detail: str) -> str:
    return f"{context}: {detail}" if context else detail


def _validate_attrs(
    attrs: OperatorAttrs,
    required: set[str],
    allowed: set[str],
    operator: str,
    context: str | None,
) -> None:
    if not isinstance(attrs, dict):
        raise ViewShapeError(_message(context, f"{operator} attrs must be an object"))
    missing = sorted(required - attrs.keys())
    if missing:
        raise ViewShapeError(
            _message(context, f"{operator} is missing required attrs: {missing}")
        )
    unknown = sorted(attrs.keys() - allowed)
    if unknown:
        raise ViewShapeError(
            _message(context, f"{operator} has unknown attrs: {unknown}")
        )


def _integer_attr(
    attrs: OperatorAttrs,
    name: str,
    operator: str,
    context: str | None,
) -> int:
    value = attrs[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ViewShapeError(
            _message(context, f"{operator} attr {name!r} must be an integer")
        )
    return value


def _shape_attr(
    attrs: OperatorAttrs,
    operator: str,
    context: str | None,
) -> tuple[int, ...]:
    _validate_attrs(attrs, {"shape"}, {"shape"}, operator, context)
    value = attrs["shape"]
    if not isinstance(value, list) or not value:
        raise ViewShapeError(
            _message(context, f"{operator} attr 'shape' must be a non-empty list")
        )
    for axis, extent in enumerate(value):
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ViewShapeError(
                _message(
                    context,
                    f"{operator} attr 'shape' dimension {axis} must be a positive integer",
                )
            )
    return tuple(value)


def _unary_shapes(
    input_shapes: tuple[tuple[object, ...], ...],
    output_shapes: tuple[tuple[object, ...], ...],
    operator: str,
    context: str | None,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    if len(input_shapes) != 1 or len(output_shapes) != 1:
        raise ViewShapeError(
            _message(context, f"{operator} requires exactly one input and one output")
        )
    return input_shapes[0], output_shapes[0]


def _validate_concrete_shape(
    shape: tuple[int, ...], label: str, context: str | None
) -> None:
    if not shape:
        raise ViewShapeError(_message(context, f"{label} shape must have positive rank"))
    for axis, extent in enumerate(shape):
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ViewShapeError(
                _message(context, f"{label} shape dimension {axis} must be positive")
            )


def canonicalize_axis(axis: int, rank: int, *, allow_end: bool = False) -> int:
    """Canonicalize a normal axis or an insertion axis for a given rank."""

    if not isinstance(axis, int) or isinstance(axis, bool):
        raise ViewShapeError("axis must be an integer")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ViewShapeError("rank must be a non-negative integer")
    lower = -rank - 1 if allow_end else -rank
    upper = rank if allow_end else rank - 1
    if axis < lower or axis > upper:
        raise ViewShapeError(
            f"axis {axis} is out of range [{lower}, {upper}] for rank {rank}"
        )
    if axis < 0:
        return axis + rank + (1 if allow_end else 0)
    return axis


def ravel_index(index: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Convert a checked row-major multi-index to its flat offset."""

    _validate_concrete_shape(shape, "ravel", None)
    if len(index) != len(shape):
        raise ViewShapeError("index rank does not match shape rank")
    offset = 0
    for axis, (coordinate, extent) in enumerate(zip(index, shape)):
        if not isinstance(coordinate, int) or isinstance(coordinate, bool):
            raise ViewShapeError(f"index coordinate {axis} must be an integer")
        if coordinate < 0 or coordinate >= extent:
            raise ViewShapeError(f"index coordinate {axis} is out of range")
        offset = offset * extent + coordinate
    return offset


def unravel_index(offset: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a checked flat offset to a row-major multi-index."""

    _validate_concrete_shape(shape, "unravel", None)
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise ViewShapeError("flat offset must be an integer")
    if offset < 0 or offset >= prod(shape):
        raise ViewShapeError("flat offset is out of range")
    remaining = offset
    reversed_index: list[int] = []
    for extent in reversed(shape):
        remaining, coordinate = divmod(remaining, extent)
        reversed_index.append(coordinate)
    return tuple(reversed(reversed_index))


def infer_reshape_groups(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[ReshapeGroup, ...]:
    """Greedily align contiguous input/output axes by original element count."""

    _validate_concrete_shape(input_shape, "reshape input", None)
    _validate_concrete_shape(output_shape, "reshape output", None)
    if prod(input_shape) != prod(output_shape):
        raise ViewShapeError("reshape input and output element counts do not match")

    input_axis = 0
    output_axis = 0
    groups: list[ReshapeGroup] = []
    while input_axis < len(input_shape) and output_axis < len(output_shape):
        input_start = input_axis
        output_start = output_axis
        input_count = input_shape[input_axis]
        output_count = output_shape[output_axis]
        input_axis += 1
        output_axis += 1

        while input_count != output_count:
            if input_count < output_count:
                if input_axis >= len(input_shape):
                    raise ViewShapeError("cannot align reshape input/output groups")
                input_count *= input_shape[input_axis]
                input_axis += 1
            else:
                if output_axis >= len(output_shape):
                    raise ViewShapeError("cannot align reshape input/output groups")
                output_count *= output_shape[output_axis]
                output_axis += 1

        groups.append(
            (
                tuple(range(input_start, input_axis)),
                tuple(range(output_start, output_axis)),
            )
        )

    if input_axis != len(input_shape):
        remaining_input = tuple(range(input_axis, len(input_shape)))
        if prod(input_shape[input_axis:]) != 1 or not groups:
            raise ViewShapeError("cannot align reshape input/output groups")
        last_input, last_output = groups[-1]
        groups[-1] = (last_input + remaining_input, last_output)
    if output_axis != len(output_shape):
        remaining_output = tuple(range(output_axis, len(output_shape)))
        if prod(output_shape[output_axis:]) != 1 or not groups:
            raise ViewShapeError("cannot align reshape input/output groups")
        last_input, last_output = groups[-1]
        groups[-1] = (last_input, last_output + remaining_output)
    return tuple(groups)


def _z3_product(values: tuple[z3.ArithRef, ...]) -> z3.ArithRef:
    result: z3.ArithRef = z3.IntVal(1)
    for value in values:
        result *= value
    return result


def _validate_expand_pair(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    context: str | None,
) -> None:
    _validate_concrete_shape(input_shape, "expand input", context)
    _validate_concrete_shape(output_shape, "expand output", context)
    if len(output_shape) < len(input_shape):
        raise ViewShapeError(
            _message(context, "expand target rank must be at least the input rank")
        )
    leading = len(output_shape) - len(input_shape)
    for input_axis, input_extent in enumerate(input_shape):
        output_axis = leading + input_axis
        output_extent = output_shape[output_axis]
        if input_extent != output_extent and input_extent != 1:
            raise ViewShapeError(
                _message(
                    context,
                    f"expand cannot change non-singleton dimension {input_axis} "
                    f"from {input_extent} to {output_extent}",
                )
            )


def expand_shape_constraints(
    input_shape: tuple[z3.ArithRef, ...],
    output_shape: tuple[z3.ArithRef, ...],
    original_input_shape: tuple[int, ...],
    original_output_shape: tuple[int, ...],
    context: str | None = None,
) -> list[z3.BoolRef]:
    """Preserve each original unary expansion mode in reduced shapes."""

    _validate_expand_pair(original_input_shape, original_output_shape, context)
    if len(input_shape) != len(original_input_shape):
        raise ViewShapeError(_message(context, "reduced and original input ranks differ"))
    if len(output_shape) != len(original_output_shape):
        raise ViewShapeError(_message(context, "reduced and original output ranks differ"))

    leading = len(original_output_shape) - len(original_input_shape)
    constraints: list[z3.BoolRef] = []
    for input_axis, original_input_extent in enumerate(original_input_shape):
        output_axis = leading + input_axis
        original_output_extent = original_output_shape[output_axis]
        if original_input_extent == original_output_extent:
            constraints.append(input_shape[input_axis] == output_shape[output_axis])
        else:
            constraints.append(input_shape[input_axis] == 1)
    return constraints


def expand_index(
    output_index: tuple[int, ...],
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    context: str | None = None,
) -> tuple[int, ...]:
    """Project an expanded output index onto the trailing-aligned input."""

    _validate_expand_pair(input_shape, output_shape, context)
    if len(output_index) != len(output_shape):
        raise ViewShapeError(_message(context, "output index rank does not match output rank"))
    for axis, (coordinate, extent) in enumerate(zip(output_index, output_shape)):
        if not isinstance(coordinate, int) or isinstance(coordinate, bool):
            raise ViewShapeError(_message(context, f"output index axis {axis} must be an integer"))
        if coordinate < 0 or coordinate >= extent:
            raise ViewShapeError(_message(context, f"output index axis {axis} is out of range"))
    leading = len(output_shape) - len(input_shape)
    return tuple(
        0 if extent == 1 else output_index[leading + axis]
        for axis, extent in enumerate(input_shape)
    )


def _transpose_axes(
    attrs: OperatorAttrs, rank: int, context: str | None
) -> tuple[int, int]:
    _validate_attrs(attrs, {"dim0", "dim1"}, {"dim0", "dim1"}, "transpose", context)
    try:
        dim0 = canonicalize_axis(_integer_attr(attrs, "dim0", "transpose", context), rank)
        dim1 = canonicalize_axis(_integer_attr(attrs, "dim1", "transpose", context), rank)
    except ViewShapeError as exc:
        raise ViewShapeError(_message(context, f"transpose {exc}")) from exc
    return dim0, dim1


def _squeeze_axes(
    input_shape: tuple[int, ...], attrs: OperatorAttrs, context: str | None
) -> tuple[int, ...]:
    _validate_attrs(attrs, set(), {"dim"}, "squeeze", context)
    if "dim" in attrs:
        try:
            axis = canonicalize_axis(
                _integer_attr(attrs, "dim", "squeeze", context), len(input_shape)
            )
        except ViewShapeError as exc:
            raise ViewShapeError(_message(context, f"squeeze {exc}")) from exc
        if input_shape[axis] != 1:
            raise ViewShapeError(
                _message(context, f"squeeze dimension {axis} must have extent 1")
            )
        return (axis,)
    return tuple(axis for axis, extent in enumerate(input_shape) if extent == 1)


class TransposeOperator(OperatorSemantics):
    name = "transpose"

    def validate_concrete_shapes(self, input_shapes, output_shapes, attrs, context) -> None:
        try:
            input_shape, output_shape = _unary_shapes(
                input_shapes, output_shapes, self.name, context
            )
            _validate_concrete_shape(input_shape, "transpose input", context)
            _validate_concrete_shape(output_shape, "transpose output", context)
            if len(input_shape) != len(output_shape):
                raise ViewShapeError(_message(context, "transpose input/output ranks differ"))
            dim0, dim1 = _transpose_axes(attrs, len(input_shape), context)
            expected = list(input_shape)
            expected[dim0], expected[dim1] = expected[dim1], expected[dim0]
            if output_shape != tuple(expected):
                raise ViewShapeError(
                    _message(context, f"transpose output shape {output_shape} does not match {tuple(expected)}")
                )
        except ViewShapeError as exc:
            raise ConcreteShapeError(str(exc)) from exc

    def shape_constraints(
        self, input_shapes, output_shapes, original_input_shapes,
        original_output_shapes, attrs, context,
    ) -> list[z3.BoolRef]:
        try:
            input_shape, output_shape = _unary_shapes(input_shapes, output_shapes, self.name, context)
            original_input, original_output = _unary_shapes(
                original_input_shapes, original_output_shapes, self.name, context
            )
            self.validate_concrete_shapes(
                (original_input,), (original_output,), attrs, context
            )
            if len(input_shape) != len(original_input) or len(output_shape) != len(original_output):
                raise ViewShapeError(_message(context, "transpose reduced/original ranks differ"))
            dim0, dim1 = _transpose_axes(attrs, len(original_input), context)
        except (ViewShapeError, ConcreteShapeError) as exc:
            raise ShapeReductionError(str(exc)) from exc
        permuted = list(input_shape)
        permuted[dim0], permuted[dim1] = permuted[dim1], permuted[dim0]
        return [output_shape[axis] == permuted[axis] for axis in range(len(output_shape))]

    def symbolic_execute(self, input_tensors, output_shapes, attrs, context):
        if len(input_tensors) != 1:
            raise SymbolicExecutionError(f"{context}: transpose requires exactly one input")
        input_tensor = input_tensors[0]
        try:
            self.validate_concrete_shapes(
                (input_tensor.shape,), output_shapes, attrs, context
            )
            dim0, dim1 = _transpose_axes(attrs, len(input_tensor.shape), context)
        except ConcreteShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
        output_shape = output_shapes[0]
        values = []
        for output_index in product(*(range(extent) for extent in output_shape)):
            input_index = list(output_index)
            input_index[dim0], input_index[dim1] = input_index[dim1], input_index[dim0]
            values.append(input_tensor.at(tuple(input_index)))
        return (SymbolicTensor(output_shape, tuple(values)),)


class ReshapeOperator(OperatorSemantics):
    name = "reshape"

    def validate_concrete_shapes(self, input_shapes, output_shapes, attrs, context) -> None:
        try:
            input_shape, output_shape = _unary_shapes(
                input_shapes, output_shapes, self.name, context
            )
            _validate_concrete_shape(input_shape, "reshape input", context)
            _validate_concrete_shape(output_shape, "reshape output", context)
            target = _shape_attr(attrs, self.name, context)
            if target != output_shape:
                raise ViewShapeError(
                    _message(context, f"reshape attrs shape {target} does not match output shape {output_shape}")
                )
            if prod(input_shape) != prod(output_shape):
                raise ViewShapeError(_message(context, "reshape input/output element counts do not match"))
        except ViewShapeError as exc:
            raise ConcreteShapeError(str(exc)) from exc

    def shape_constraints(
        self, input_shapes, output_shapes, original_input_shapes,
        original_output_shapes, attrs, context,
    ) -> list[z3.BoolRef]:
        try:
            input_shape, output_shape = _unary_shapes(input_shapes, output_shapes, self.name, context)
            original_input, original_output = _unary_shapes(
                original_input_shapes, original_output_shapes, self.name, context
            )
            self.validate_concrete_shapes((original_input,), (original_output,), attrs, context)
            if len(input_shape) != len(original_input) or len(output_shape) != len(original_output):
                raise ViewShapeError(_message(context, "reshape reduced/original ranks differ"))
            groups = infer_reshape_groups(original_input, original_output)
        except (ViewShapeError, ConcreteShapeError) as exc:
            raise ShapeReductionError(str(exc)) from exc
        constraints = [
            _z3_product(tuple(input_shape[axis] for axis in input_axes))
            == _z3_product(tuple(output_shape[axis] for axis in output_axes))
            for input_axes, output_axes in groups
        ]
        constraints.append(_z3_product(input_shape) == _z3_product(output_shape))
        return constraints

    def symbolic_execute(self, input_tensors, output_shapes, attrs, context):
        if len(input_tensors) != 1 or len(output_shapes) != 1:
            raise SymbolicExecutionError(
                f"{context}: reshape requires exactly one input and one output"
            )
        input_tensor = input_tensors[0]
        output_shape = output_shapes[0]
        try:
            target = _shape_attr(attrs, self.name, context)
            if len(target) != len(output_shape):
                raise ViewShapeError(_message(context, "reshape target/output ranks differ"))
            _validate_concrete_shape(output_shape, "reshape output", context)
            if input_tensor.numel != prod(output_shape):
                raise ViewShapeError(_message(context, "reshape input/output element counts do not match"))
        except ViewShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
        values = tuple(
            input_tensor.at(
                unravel_index(ravel_index(output_index, output_shape), input_tensor.shape)
            )
            for output_index in product(*(range(extent) for extent in output_shape))
        )
        return (SymbolicTensor(output_shape, values),)


class SqueezeOperator(OperatorSemantics):
    name = "squeeze"

    def validate_concrete_shapes(self, input_shapes, output_shapes, attrs, context) -> None:
        try:
            input_shape, output_shape = _unary_shapes(
                input_shapes, output_shapes, self.name, context
            )
            _validate_concrete_shape(input_shape, "squeeze input", context)
            _validate_concrete_shape(output_shape, "squeeze output", context)
            removed = set(_squeeze_axes(input_shape, attrs, context))
            expected = tuple(
                extent for axis, extent in enumerate(input_shape) if axis not in removed
            )
            if output_shape != expected:
                raise ViewShapeError(
                    _message(context, f"squeeze output shape {output_shape} does not match {expected}")
                )
        except ViewShapeError as exc:
            raise ConcreteShapeError(str(exc)) from exc

    def shape_constraints(
        self, input_shapes, output_shapes, original_input_shapes,
        original_output_shapes, attrs, context,
    ) -> list[z3.BoolRef]:
        try:
            input_shape, output_shape = _unary_shapes(input_shapes, output_shapes, self.name, context)
            original_input, original_output = _unary_shapes(
                original_input_shapes, original_output_shapes, self.name, context
            )
            self.validate_concrete_shapes((original_input,), (original_output,), attrs, context)
            if len(input_shape) != len(original_input) or len(output_shape) != len(original_output):
                raise ViewShapeError(_message(context, "squeeze reduced/original ranks differ"))
            removed = set(_squeeze_axes(original_input, attrs, context))
        except (ViewShapeError, ConcreteShapeError) as exc:
            raise ShapeReductionError(str(exc)) from exc
        constraints: list[z3.BoolRef] = []
        output_axis = 0
        omitted_dim = "dim" not in attrs
        for input_axis, original_extent in enumerate(original_input):
            if input_axis in removed:
                constraints.append(input_shape[input_axis] == 1)
            else:
                constraints.append(output_shape[output_axis] == input_shape[input_axis])
                if omitted_dim:
                    constraints.append(input_shape[input_axis] >= min(2, original_extent))
                output_axis += 1
        return constraints

    def symbolic_execute(self, input_tensors, output_shapes, attrs, context):
        if len(input_tensors) != 1:
            raise SymbolicExecutionError(f"{context}: squeeze requires exactly one input")
        input_tensor = input_tensors[0]
        try:
            self.validate_concrete_shapes(
                (input_tensor.shape,), output_shapes, attrs, context
            )
            removed = set(_squeeze_axes(input_tensor.shape, attrs, context))
        except ConcreteShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
        output_shape = output_shapes[0]
        values = []
        for output_index in product(*(range(extent) for extent in output_shape)):
            output_axis = 0
            input_index = []
            for input_axis in range(len(input_tensor.shape)):
                if input_axis in removed:
                    input_index.append(0)
                else:
                    input_index.append(output_index[output_axis])
                    output_axis += 1
            values.append(input_tensor.at(tuple(input_index)))
        return (SymbolicTensor(output_shape, tuple(values)),)


class UnsqueezeOperator(OperatorSemantics):
    name = "unsqueeze"

    def _axis(self, attrs: OperatorAttrs, rank: int, context: str | None) -> int:
        _validate_attrs(attrs, {"dim"}, {"dim"}, self.name, context)
        try:
            return canonicalize_axis(
                _integer_attr(attrs, "dim", self.name, context), rank, allow_end=True
            )
        except ViewShapeError as exc:
            raise ViewShapeError(_message(context, f"unsqueeze {exc}")) from exc

    def validate_concrete_shapes(self, input_shapes, output_shapes, attrs, context) -> None:
        try:
            input_shape, output_shape = _unary_shapes(
                input_shapes, output_shapes, self.name, context
            )
            _validate_concrete_shape(input_shape, "unsqueeze input", context)
            _validate_concrete_shape(output_shape, "unsqueeze output", context)
            axis = self._axis(attrs, len(input_shape), context)
            expected = input_shape[:axis] + (1,) + input_shape[axis:]
            if output_shape != expected:
                raise ViewShapeError(
                    _message(context, f"unsqueeze output shape {output_shape} does not match {expected}")
                )
        except ViewShapeError as exc:
            raise ConcreteShapeError(str(exc)) from exc

    def shape_constraints(
        self, input_shapes, output_shapes, original_input_shapes,
        original_output_shapes, attrs, context,
    ) -> list[z3.BoolRef]:
        try:
            input_shape, output_shape = _unary_shapes(input_shapes, output_shapes, self.name, context)
            original_input, original_output = _unary_shapes(
                original_input_shapes, original_output_shapes, self.name, context
            )
            self.validate_concrete_shapes((original_input,), (original_output,), attrs, context)
            if len(input_shape) != len(original_input) or len(output_shape) != len(original_output):
                raise ViewShapeError(_message(context, "unsqueeze reduced/original ranks differ"))
            axis = self._axis(attrs, len(original_input), context)
        except (ViewShapeError, ConcreteShapeError) as exc:
            raise ShapeReductionError(str(exc)) from exc
        constraints = [output_shape[axis] == 1]
        constraints.extend(
            output_shape[output_axis] == input_shape[input_axis]
            for input_axis, output_axis in (
                (input_axis, input_axis if input_axis < axis else input_axis + 1)
                for input_axis in range(len(input_shape))
            )
        )
        return constraints

    def symbolic_execute(self, input_tensors, output_shapes, attrs, context):
        if len(input_tensors) != 1:
            raise SymbolicExecutionError(f"{context}: unsqueeze requires exactly one input")
        input_tensor = input_tensors[0]
        try:
            self.validate_concrete_shapes(
                (input_tensor.shape,), output_shapes, attrs, context
            )
            axis = self._axis(attrs, len(input_tensor.shape), context)
        except ConcreteShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
        output_shape = output_shapes[0]
        values = tuple(
            input_tensor.at(output_index[:axis] + output_index[axis + 1 :])
            for output_index in product(*(range(extent) for extent in output_shape))
        )
        return (SymbolicTensor(output_shape, values),)


class ExpandOperator(OperatorSemantics):
    name = "expand"

    def validate_concrete_shapes(self, input_shapes, output_shapes, attrs, context) -> None:
        try:
            input_shape, output_shape = _unary_shapes(
                input_shapes, output_shapes, self.name, context
            )
            target = _shape_attr(attrs, self.name, context)
            if target != output_shape:
                raise ViewShapeError(
                    _message(context, f"expand attrs shape {target} does not match output shape {output_shape}")
                )
            _validate_expand_pair(input_shape, output_shape, context)
        except ViewShapeError as exc:
            raise ConcreteShapeError(str(exc)) from exc

    def shape_constraints(
        self, input_shapes, output_shapes, original_input_shapes,
        original_output_shapes, attrs, context,
    ) -> list[z3.BoolRef]:
        try:
            input_shape, output_shape = _unary_shapes(input_shapes, output_shapes, self.name, context)
            original_input, original_output = _unary_shapes(
                original_input_shapes, original_output_shapes, self.name, context
            )
            self.validate_concrete_shapes((original_input,), (original_output,), attrs, context)
            return expand_shape_constraints(
                input_shape, output_shape, original_input, original_output, context
            )
        except (ViewShapeError, ConcreteShapeError) as exc:
            raise ShapeReductionError(str(exc)) from exc

    def symbolic_execute(self, input_tensors, output_shapes, attrs, context):
        if len(input_tensors) != 1 or len(output_shapes) != 1:
            raise SymbolicExecutionError(
                f"{context}: expand requires exactly one input and one output"
            )
        input_tensor = input_tensors[0]
        output_shape = output_shapes[0]
        try:
            target = _shape_attr(attrs, self.name, context)
            if len(target) != len(output_shape):
                raise ViewShapeError(_message(context, "expand target/output ranks differ"))
            values = tuple(
                input_tensor.at(
                    expand_index(output_index, input_tensor.shape, output_shape, context)
                )
                for output_index in product(*(range(extent) for extent in output_shape))
            )
        except ViewShapeError as exc:
            raise SymbolicExecutionError(str(exc)) from exc
        return (SymbolicTensor(output_shape, values),)
