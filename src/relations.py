"""Relation-specific symbolic shape and value semantics."""

from __future__ import annotations

from itertools import product

import z3

from .shape_model import ShapeReductionError
from .stage_model import RelationSpec
from .symbolic_tensor import SymbolicTensor


class RelationEncodingError(ValueError):
    """Raised when symbolic relation value encoding is inconsistent."""


DECLARED_RELATION_TYPES = frozenset({"replicate", "shard", "partial"})


class RelationSemantics:
    name: str

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        raise NotImplementedError

    def value_constraints(
        self,
        relation: RelationSpec,
        single_tensor: SymbolicTensor,
        local_tensors: tuple[SymbolicTensor, ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        raise NotImplementedError


def _validate_shape_local_count(
    local_shapes: tuple[tuple[z3.ArithRef, ...], ...], world_size: int, context: str
) -> None:
    if len(local_shapes) != world_size:
        raise ShapeReductionError(
            f"{context}: expected {world_size} local shapes, found {len(local_shapes)}"
        )


def _same_shape_constraints(
    single_shape: tuple[z3.ArithRef, ...],
    local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
    world_size: int,
    context: str,
) -> list[z3.BoolRef]:
    _validate_shape_local_count(local_shapes, world_size, context)
    constraints: list[z3.BoolRef] = []
    for rank, local_shape in enumerate(local_shapes):
        if len(local_shape) != len(single_shape):
            raise ShapeReductionError(
                f"{context}: local tensor on rank {rank} must have rank {len(single_shape)}"
            )
        constraints.extend(
            local_shape[dimension] == single_shape[dimension]
            for dimension in range(len(single_shape))
        )
    return constraints


def _validate_value_local_count(
    local_tensors: tuple[SymbolicTensor, ...], world_size: int, context: str
) -> None:
    if len(local_tensors) != world_size:
        raise RelationEncodingError(
            f"{context}: expected {world_size} local tensors, found {len(local_tensors)}"
        )


class ReplicateRelation(RelationSemantics):
    name = "replicate"

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        return _same_shape_constraints(single_shape, local_shapes, world_size, context)

    def value_constraints(
        self,
        relation: RelationSpec,
        single_tensor: SymbolicTensor,
        local_tensors: tuple[SymbolicTensor, ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        if relation.dim is not None or relation.reduce_op is not None:
            raise RelationEncodingError(
                f"{context}: replicate relation must not define dim or reduce_op"
            )
        _validate_value_local_count(local_tensors, world_size, context)
        constraints: list[z3.BoolRef] = []
        for rank, local_tensor in enumerate(local_tensors):
            if local_tensor.shape != single_tensor.shape:
                raise RelationEncodingError(
                    f"{context}: replicate tensor on rank {rank} has shape {local_tensor.shape}, "
                    f"expected {single_tensor.shape}"
                )
            constraints.extend(
                local_value == single_value
                for single_value, local_value in zip(single_tensor.values, local_tensor.values)
            )
        return constraints


class ShardRelation(RelationSemantics):
    name = "shard"

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        if not isinstance(relation.dim, int) or isinstance(relation.dim, bool):
            raise ShapeReductionError(f"{context}: shard dim must be an integer")
        if relation.dim < 0 or relation.dim >= len(single_shape):
            raise ShapeReductionError(f"{context}: shard dim {relation.dim} is out of range")
        _validate_shape_local_count(local_shapes, world_size, context)

        constraints = [single_shape[relation.dim] % world_size == 0]
        for rank, local_shape in enumerate(local_shapes):
            if len(local_shape) != len(single_shape):
                raise ShapeReductionError(
                    f"{context}: shard tensor on rank {rank} must have rank {len(single_shape)}"
                )
            for dimension, (single_dimension, local_dimension) in enumerate(
                zip(single_shape, local_shape)
            ):
                if dimension == relation.dim:
                    constraints.append(world_size * local_dimension == single_dimension)
                else:
                    constraints.append(local_dimension == single_dimension)
        return constraints

    def value_constraints(
        self,
        relation: RelationSpec,
        single_tensor: SymbolicTensor,
        local_tensors: tuple[SymbolicTensor, ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        if not isinstance(relation.dim, int) or isinstance(relation.dim, bool):
            raise RelationEncodingError(f"{context}: shard dim must be an integer")
        if relation.dim < 0 or relation.dim >= len(single_tensor.shape):
            raise RelationEncodingError(f"{context}: shard dim {relation.dim} is out of range")
        if relation.reduce_op is not None:
            raise RelationEncodingError(f"{context}: shard relation must not define reduce_op")
        _validate_value_local_count(local_tensors, world_size, context)

        reference_shape = local_tensors[0].shape
        constraints: list[z3.BoolRef] = []
        for rank, local_tensor in enumerate(local_tensors):
            if len(local_tensor.shape) != len(single_tensor.shape):
                raise RelationEncodingError(
                    f"{context}: shard tensor on rank {rank} has rank {len(local_tensor.shape)}, "
                    f"expected {len(single_tensor.shape)}"
                )
            if local_tensor.shape != reference_shape:
                raise RelationEncodingError(
                    f"{context}: shard tensors must have identical local shapes"
                )
            for dimension, (single_extent, local_extent) in enumerate(
                zip(single_tensor.shape, local_tensor.shape)
            ):
                expected_extent = local_extent * world_size if dimension == relation.dim else local_extent
                if single_extent != expected_extent:
                    raise RelationEncodingError(
                        f"{context}: shard shape mismatch at dimension {dimension}"
                    )
            for local_index in product(*(range(extent) for extent in local_tensor.shape)):
                global_index = list(local_index)
                global_index[relation.dim] = (
                    rank * local_tensor.shape[relation.dim] + local_index[relation.dim]
                )
                constraints.append(
                    local_tensor.at(local_index) == single_tensor.at(tuple(global_index))
                )
        return constraints


class PartialRelation(RelationSemantics):
    name = "partial"

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        return _same_shape_constraints(single_shape, local_shapes, world_size, context)

    def value_constraints(
        self,
        relation: RelationSpec,
        single_tensor: SymbolicTensor,
        local_tensors: tuple[SymbolicTensor, ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        if relation.dim is not None:
            raise RelationEncodingError(f"{context}: partial relation must not define dim")
        if relation.reduce_op != "sum":
            raise RelationEncodingError(f"{context}: partial reduce_op must be 'sum'")
        _validate_value_local_count(local_tensors, world_size, context)
        for rank, local_tensor in enumerate(local_tensors):
            if local_tensor.shape != single_tensor.shape:
                raise RelationEncodingError(
                    f"{context}: partial tensor on rank {rank} has shape {local_tensor.shape}, "
                    f"expected {single_tensor.shape}"
                )
        return [
            single_tensor.values[index]
            == z3.Sum([local_tensor.values[index] for local_tensor in local_tensors])
            for index in range(single_tensor.numel)
        ]


RELATION_REGISTRY: dict[str, RelationSemantics] = {
    "replicate": ReplicateRelation(),
    "shard": ShardRelation(),
    "partial": PartialRelation(),
}


def get_relation(name: str, context: str | None = None) -> RelationSemantics:
    """Return registered relation semantics or a contextual shape-support error."""

    prefix = f"{context}: " if context else ""
    if name in RELATION_REGISTRY:
        return RELATION_REGISTRY[name]
    raise ShapeReductionError(f"{prefix}unknown relation {name!r}")
