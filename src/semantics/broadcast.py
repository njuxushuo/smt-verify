"""Shared primitives for standard trailing-dimension broadcasting."""

from __future__ import annotations

import z3


class BroadcastShapeError(ValueError):
    """Raised when concrete or reduced shapes violate broadcast semantics."""


def _message(context: str | None, detail: str) -> str:
    return f"{context}: {detail}" if context else detail


def _validate_concrete_shape(
    shape: tuple[int, ...],
    label: str,
    context: str | None,
) -> None:
    for axis, dimension in enumerate(shape):
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise BroadcastShapeError(
                _message(context, f"{label} shape dimension {axis} must be a positive integer")
            )


def infer_broadcast_shape(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    context: str | None = None,
) -> tuple[int, ...]:
    """Infer the standard trailing-dimension broadcast result for concrete shapes."""

    _validate_concrete_shape(left_shape, "left", context)
    _validate_concrete_shape(right_shape, "right", context)

    reversed_output: list[int] = []
    for offset in range(1, max(len(left_shape), len(right_shape)) + 1):
        left_dimension = left_shape[-offset] if offset <= len(left_shape) else 1
        right_dimension = right_shape[-offset] if offset <= len(right_shape) else 1
        if left_dimension == right_dimension:
            reversed_output.append(left_dimension)
        elif left_dimension == 1:
            reversed_output.append(right_dimension)
        elif right_dimension == 1:
            reversed_output.append(left_dimension)
        else:
            axis = max(len(left_shape), len(right_shape)) - offset
            raise BroadcastShapeError(
                _message(
                    context,
                    "shapes "
                    f"{left_shape} and {right_shape} are not broadcast-compatible "
                    f"at aligned axis {axis}",
                )
            )
    return tuple(reversed(reversed_output))


def broadcast_shape_constraints(
    left_shape: tuple[z3.ArithRef, ...],
    right_shape: tuple[z3.ArithRef, ...],
    output_shape: tuple[z3.ArithRef, ...],
    original_left_shape: tuple[int, ...],
    original_right_shape: tuple[int, ...],
    original_output_shape: tuple[int, ...],
    context: str | None = None,
) -> list[z3.BoolRef]:
    """Preserve the original per-axis broadcast pattern in reduced shapes."""

    expected_output = infer_broadcast_shape(original_left_shape, original_right_shape, context)
    if original_output_shape != expected_output:
        raise BroadcastShapeError(
            _message(
                context,
                f"original output shape {original_output_shape} does not match "
                f"broadcast shape {expected_output}",
            )
        )
    if len(left_shape) != len(original_left_shape):
        raise BroadcastShapeError(_message(context, "reduced and original left ranks differ"))
    if len(right_shape) != len(original_right_shape):
        raise BroadcastShapeError(_message(context, "reduced and original right ranks differ"))
    if len(output_shape) != len(original_output_shape):
        raise BroadcastShapeError(_message(context, "reduced and original output ranks differ"))

    output_rank = len(original_output_shape)
    left_leading = output_rank - len(original_left_shape)
    right_leading = output_rank - len(original_right_shape)
    constraints: list[z3.BoolRef] = []

    for output_axis in range(output_rank):
        left_axis = output_axis - left_leading
        right_axis = output_axis - right_leading
        left_missing = left_axis < 0
        right_missing = right_axis < 0

        if left_missing:
            constraints.append(output_shape[output_axis] == right_shape[right_axis])
            continue
        if right_missing:
            constraints.append(output_shape[output_axis] == left_shape[left_axis])
            continue

        original_left = original_left_shape[left_axis]
        original_right = original_right_shape[right_axis]
        reduced_left = left_shape[left_axis]
        reduced_right = right_shape[right_axis]
        reduced_output = output_shape[output_axis]

        if original_left == original_right:
            constraints.extend(
                (reduced_left == reduced_right, reduced_output == reduced_left)
            )
        elif original_left == 1:
            constraints.extend((reduced_left == 1, reduced_output == reduced_right))
        elif original_right == 1:
            constraints.extend((reduced_right == 1, reduced_output == reduced_left))
        else:
            raise BroadcastShapeError(
                _message(context, "original input shapes are not broadcast-compatible")
            )
    return constraints


def broadcast_index(
    output_index: tuple[int, ...],
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    context: str | None = None,
) -> tuple[int, ...]:
    """Project one output index onto an input under trailing broadcasting."""

    _validate_concrete_shape(input_shape, "input", context)
    _validate_concrete_shape(output_shape, "output", context)
    if len(output_index) != len(output_shape):
        raise BroadcastShapeError(
            _message(context, "output index rank does not match output shape rank")
        )
    for axis, (coordinate, extent) in enumerate(zip(output_index, output_shape)):
        if not isinstance(coordinate, int) or isinstance(coordinate, bool):
            raise BroadcastShapeError(
                _message(context, f"output index at axis {axis} must be an integer")
            )
        if coordinate < 0 or coordinate >= extent:
            raise BroadcastShapeError(
                _message(context, f"output index at axis {axis} is out of range")
            )

    expected_output = infer_broadcast_shape(input_shape, output_shape, context)
    if expected_output != output_shape:
        raise BroadcastShapeError(
            _message(context, f"input shape {input_shape} cannot broadcast to {output_shape}")
        )

    leading_dimensions = len(output_shape) - len(input_shape)
    return tuple(
        0 if input_extent == 1 else output_index[leading_dimensions + input_axis]
        for input_axis, input_extent in enumerate(input_shape)
    )
