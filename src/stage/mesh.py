"""Device-mesh size, rank-coordinate, and coordinate-group helpers."""

from __future__ import annotations

from itertools import product
from math import prod

from .model import DeviceMeshSpec


class DeviceMeshError(ValueError):
    """Raised when a device mesh or coordinate is invalid."""


def _validated_shape(mesh: DeviceMeshSpec) -> tuple[int, ...]:
    if not isinstance(mesh, DeviceMeshSpec):
        raise DeviceMeshError("mesh must be a DeviceMeshSpec")
    if not isinstance(mesh.shape, tuple) or not mesh.shape:
        raise DeviceMeshError("mesh shape must contain at least one dimension")
    for axis, extent in enumerate(mesh.shape):
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise DeviceMeshError(
                f"mesh shape extent at axis {axis} must be a positive integer"
            )
    return tuple(mesh.shape)


def mesh_size(mesh: DeviceMeshSpec) -> int:
    """Return the product of all mesh-axis extents."""

    return prod(_validated_shape(mesh))


def rank_to_coordinate(rank: int, mesh: DeviceMeshSpec) -> tuple[int, ...]:
    """Convert one row-major flat rank into an N-D mesh coordinate."""

    shape = _validated_shape(mesh)
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise DeviceMeshError("rank must be an integer")
    size = prod(shape)
    if rank < 0 or rank >= size:
        raise DeviceMeshError(f"rank {rank} is out of range [0, {size})")

    remaining = rank
    reversed_coordinate: list[int] = []
    for extent in reversed(shape):
        remaining, coordinate = divmod(remaining, extent)
        reversed_coordinate.append(coordinate)
    return tuple(reversed(reversed_coordinate))


def coordinate_to_rank(
    coordinate: tuple[int, ...], mesh: DeviceMeshSpec
) -> int:
    """Convert one checked mesh coordinate into its row-major flat rank."""

    shape = _validated_shape(mesh)
    if not isinstance(coordinate, (tuple, list)) or len(coordinate) != len(shape):
        raise DeviceMeshError(
            f"coordinate must contain exactly {len(shape)} components"
        )
    rank = 0
    for axis, (component, extent) in enumerate(zip(coordinate, shape)):
        if not isinstance(component, int) or isinstance(component, bool):
            raise DeviceMeshError(f"coordinate component {axis} must be an integer")
        if component < 0 or component >= extent:
            raise DeviceMeshError(
                f"coordinate component {component} at axis {axis} is out of range"
            )
        rank = rank * extent + component
    return rank


def enumerate_coordinate_group(
    mesh: DeviceMeshSpec,
    fixed_coordinate: tuple[int, ...],
    varying_axes: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate coordinates obtained by varying selected axes of a base coordinate."""

    shape = _validated_shape(mesh)
    coordinate_to_rank(fixed_coordinate, mesh)
    if not isinstance(varying_axes, (tuple, list)):
        raise DeviceMeshError("varying_axes must be a sequence")
    if any(
        not isinstance(axis, int) or isinstance(axis, bool)
        for axis in varying_axes
    ):
        raise DeviceMeshError("varying_axes must contain integers")
    if len(set(varying_axes)) != len(varying_axes):
        raise DeviceMeshError("varying_axes must not contain duplicates")
    if any(axis < 0 or axis >= len(shape) for axis in varying_axes):
        raise DeviceMeshError("varying mesh axis is out of range")

    axes = tuple(varying_axes)
    coordinates: list[tuple[int, ...]] = []
    for values in product(*(range(shape[axis]) for axis in axes)):
        coordinate = list(fixed_coordinate)
        for axis, value in zip(axes, values):
            coordinate[axis] = value
        coordinates.append(tuple(coordinate))
    return tuple(coordinates)
