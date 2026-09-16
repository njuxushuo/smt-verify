"""Composite placement concrete, reduced-shape, and symbolic value semantics."""

from __future__ import annotations

from itertools import product

import z3

from ..shape.model import ShapeReductionError
from ..stage.mesh import (
    coordinate_to_rank,
    enumerate_coordinate_group,
    mesh_size,
    rank_to_coordinate,
)
from ..stage.model import DeviceMeshSpec, PlacementSpec, RelationSpec
from ..symbolic.tensor import SymbolicTensor


class RelationEncodingError(ValueError):
    """Raised when symbolic relation value encoding is inconsistent."""


class ConcreteRelationError(ValueError):
    """Raised when original concrete relation shapes are not legal."""


DECLARED_PLACEMENT_TYPES = frozenset({"replicate", "shard", "partial"})


def _raise(error_type: type[ValueError], message: str) -> None:
    raise error_type(message)


def validate_placements(
    relation: RelationSpec,
    mesh: DeviceMeshSpec,
    tensor_rank: int,
    context: str,
    error_type: type[ValueError] = ConcreteRelationError,
) -> None:
    """Validate the canonical one-placement-per-mesh-axis schema."""

    if not isinstance(relation.placements, (tuple, list)):
        _raise(error_type, f"{context}.placements: expected list")
    if len(relation.placements) != len(mesh.shape):
        _raise(
            error_type,
            f"{context}.placements: expected {len(mesh.shape)} placements, "
            f"found {len(relation.placements)}",
        )

    shard_dimensions: set[int] = set()
    for axis, placement in enumerate(relation.placements):
        placement_context = f"{context}.placements[{axis}]"
        if not isinstance(placement, PlacementSpec):
            _raise(error_type, f"{placement_context}: expected PlacementSpec")
        if (
            not isinstance(placement.type, str)
            or placement.type not in DECLARED_PLACEMENT_TYPES
        ):
            _raise(
                error_type,
                f"{placement_context}.type: unsupported placement {placement.type!r}",
            )
        if placement.type == "replicate":
            if placement.dim is not None or placement.reduce_op is not None:
                _raise(
                    error_type,
                    f"{placement_context}: replicate must not define dim or reduce_op",
                )
        elif placement.type == "shard":
            if not isinstance(placement.dim, int) or isinstance(placement.dim, bool):
                _raise(error_type, f"{placement_context}: shard dim must be an integer")
            if placement.dim < 0 or placement.dim >= tensor_rank:
                _raise(
                    error_type,
                    f"{placement_context}: shard dim {placement.dim} is out of range "
                    f"for tensor {relation.single_tensor} with rank {tensor_rank}",
                )
            if placement.reduce_op is not None:
                _raise(error_type, f"{placement_context}: shard must not define reduce_op")
            if placement.dim in shard_dimensions:
                _raise(
                    error_type,
                    f"{context}: tensor dimension {placement.dim} is sharded by multiple "
                    "mesh axes",
                )
            shard_dimensions.add(placement.dim)
        else:
            if placement.dim is not None:
                _raise(error_type, f"{placement_context}: partial must not define dim")
            if placement.reduce_op != "sum":
                _raise(error_type, f"{placement_context}: partial reduce_op must be 'sum'")


def shard_factors(
    relation: RelationSpec,
    mesh: DeviceMeshSpec,
    tensor_rank: int,
) -> tuple[int, ...]:
    """Return the composite regular-shard factor for every tensor dimension."""

    factors = [1] * tensor_rank
    for mesh_axis, placement in enumerate(relation.placements):
        if placement.type == "shard":
            assert placement.dim is not None
            factors[placement.dim] *= mesh.shape[mesh_axis]
    return tuple(factors)


class CompositeRelationSemantics:
    """Compose one placement per mesh axis into a whole-tensor relation."""

    def validate_concrete_shapes(
        self,
        relation: RelationSpec,
        single_shape: tuple[int, ...],
        local_shapes: tuple[tuple[int, ...], ...],
        mesh: DeviceMeshSpec,
        context: str,
    ) -> None:
        validate_placements(relation, mesh, len(single_shape), context)
        expected_count = mesh_size(mesh)
        if len(local_shapes) != expected_count:
            raise ConcreteRelationError(
                f"{context}: expected {expected_count} local shapes, found {len(local_shapes)}"
            )
        factors = shard_factors(relation, mesh, len(single_shape))
        for dimension, (extent, factor) in enumerate(zip(single_shape, factors)):
            if extent % factor != 0:
                raise ConcreteRelationError(
                    f"{context}: tensor dimension {dimension} with extent {extent} "
                    f"is not divisible by shard factor {factor}"
                )
        expected_shape = tuple(
            extent // factor for extent, factor in zip(single_shape, factors)
        )
        for rank, local_shape in enumerate(local_shapes):
            if local_shape != expected_shape:
                raise ConcreteRelationError(
                    f"{context}: local tensor on rank {rank} has shape {local_shape}, "
                    f"expected {expected_shape} for composite placements"
                )

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_single_shape: tuple[int, ...],
        original_local_shapes: tuple[tuple[int, ...], ...],
        mesh: DeviceMeshSpec,
        context: str,
    ) -> list[z3.BoolRef]:
        try:
            self.validate_concrete_shapes(
                relation,
                original_single_shape,
                original_local_shapes,
                mesh,
                context,
            )
        except ConcreteRelationError as exc:
            raise ShapeReductionError(str(exc)) from exc
        if len(single_shape) != len(original_single_shape):
            raise ShapeReductionError(f"{context}: original and reduced single ranks differ")
        expected_count = mesh_size(mesh)
        if len(local_shapes) != expected_count:
            raise ShapeReductionError(
                f"{context}: expected {expected_count} local shapes, found {len(local_shapes)}"
            )

        factors = shard_factors(relation, mesh, len(single_shape))
        constraints: list[z3.BoolRef] = []
        for dimension, factor in enumerate(factors):
            constraints.append(single_shape[dimension] % factor == 0)
        for rank, local_shape in enumerate(local_shapes):
            if len(local_shape) != len(single_shape):
                raise ShapeReductionError(
                    f"{context}: local tensor on rank {rank} must have rank {len(single_shape)}"
                )
            constraints.extend(
                local_extent * factor == single_extent
                for local_extent, single_extent, factor in zip(
                    local_shape, single_shape, factors
                )
            )
        return constraints

    def value_constraints(
        self,
        relation: RelationSpec,
        single_tensor: SymbolicTensor,
        local_tensors: tuple[SymbolicTensor, ...],
        mesh: DeviceMeshSpec,
        context: str,
    ) -> list[z3.BoolRef]:
        expected_count = mesh_size(mesh)
        if len(local_tensors) != expected_count:
            raise RelationEncodingError(
                f"{context}: expected {expected_count} local tensors, "
                f"found {len(local_tensors)}"
            )
        try:
            self.validate_concrete_shapes(
                relation,
                single_tensor.shape,
                tuple(tensor.shape for tensor in local_tensors),
                mesh,
                context,
            )
        except ConcreteRelationError as exc:
            raise RelationEncodingError(str(exc)) from exc

        partial_axes = tuple(
            axis
            for axis, placement in enumerate(relation.placements)
            if placement.type == "partial"
        )
        coordinates = tuple(
            rank_to_coordinate(rank, mesh) for rank in range(mesh_size(mesh))
        )
        base_coordinates = tuple(
            coordinate
            for coordinate in coordinates
            if all(coordinate[axis] == 0 for axis in partial_axes)
        )
        reference_shape = local_tensors[0].shape
        constraints: list[z3.BoolRef] = []
        for base_coordinate in base_coordinates:
            group_coordinates = enumerate_coordinate_group(
                mesh, base_coordinate, partial_axes
            )
            group_tensors = tuple(
                local_tensors[coordinate_to_rank(coordinate, mesh)]
                for coordinate in group_coordinates
            )
            for local_index in product(*(range(extent) for extent in reference_shape)):
                global_index = list(local_index)
                for mesh_axis, placement in enumerate(relation.placements):
                    if placement.type == "shard":
                        assert placement.dim is not None
                        global_index[placement.dim] = (
                            base_coordinate[mesh_axis]
                            * reference_shape[placement.dim]
                            + local_index[placement.dim]
                        )
                local_values = [tensor.at(local_index) for tensor in group_tensors]
                combined_value = (
                    local_values[0]
                    if len(local_values) == 1
                    else z3.Sum(local_values)
                )
                constraints.append(
                    single_tensor.at(tuple(global_index)) == combined_value
                )
        return constraints


COMPOSITE_RELATION = CompositeRelationSemantics()


def get_relation(context: str | None = None) -> CompositeRelationSemantics:
    """Return the single canonical composite relation semantics implementation."""

    return COMPOSITE_RELATION


__all__ = [
    "COMPOSITE_RELATION",
    "CompositeRelationSemantics",
    "ConcreteRelationError",
    "DECLARED_PLACEMENT_TYPES",
    "RelationEncodingError",
    "get_relation",
    "shard_factors",
    "validate_placements",
]
