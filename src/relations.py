"""Relation-specific concrete, reduced-shape, and symbolic value semantics."""

from __future__ import annotations

from itertools import product

import z3

from .shape_model import ShapeReductionError
from .stage_model import RelationSpec
from .symbolic_tensor import SymbolicTensor


class RelationEncodingError(ValueError):
    """Raised when symbolic relation value encoding is inconsistent."""


class ConcreteRelationError(ValueError):
    """Raised when original concrete relation shapes are not legal."""


DECLARED_RELATION_TYPES = frozenset({"replicate", "shard", "partial"})


class RelationSemantics:
    name: str

    def validate_concrete_shapes(
        self,
        relation: RelationSpec,
        single_shape: tuple[int, ...],
        local_shapes: tuple[tuple[int, ...], ...],
        world_size: int,
        context: str,
    ) -> None:
        raise NotImplementedError

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_single_shape: tuple[int, ...],
        original_local_shapes: tuple[tuple[int, ...], ...],
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


def _validate_concrete_local_count(
    local_shapes: tuple[tuple[int, ...], ...], world_size: int, context: str
) -> None:
    if len(local_shapes) != world_size:
        raise ConcreteRelationError(
            f"{context}: expected {world_size} local shapes, found {len(local_shapes)}"
        )


def _validate_value_local_count(
    local_tensors: tuple[SymbolicTensor, ...], world_size: int, context: str
) -> None:
    if len(local_tensors) != world_size:
        raise RelationEncodingError(
            f"{context}: expected {world_size} local tensors, found {len(local_tensors)}"
        )


def _ratio_constraints(
    single_shape: tuple[z3.ArithRef, ...],
    local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
    original_single_shape: tuple[int, ...],
    original_local_shapes: tuple[tuple[int, ...], ...],
    world_size: int,
    context: str,
) -> list[z3.BoolRef]:
    """Preserve each original local-to-single dimension ratio by cross multiplication."""

    _validate_shape_local_count(local_shapes, world_size, context)
    _validate_concrete_local_count(original_local_shapes, world_size, context)
    if len(single_shape) != len(original_single_shape):
        raise ShapeReductionError(f"{context}: original and reduced single ranks differ")

    constraints: list[z3.BoolRef] = []
    for rank, (local_shape, original_local_shape) in enumerate(
        zip(local_shapes, original_local_shapes)
    ):
        if len(local_shape) != len(single_shape):
            raise ShapeReductionError(
                f"{context}: local tensor on rank {rank} must have rank {len(single_shape)}"
            )
        if len(original_local_shape) != len(original_single_shape):
            raise ShapeReductionError(
                f"{context}: original local tensor on rank {rank} must have rank "
                f"{len(original_single_shape)}"
            )
        constraints.extend(
            local_dimension * original_single_dimension
            == single_dimension * original_local_dimension
            for local_dimension, single_dimension, original_local_dimension, original_single_dimension in zip(
                local_shape, single_shape, original_local_shape, original_single_shape
            )
        )
    return constraints


class ReplicateRelation(RelationSemantics):
    name = "replicate"

    def validate_concrete_shapes(
        self,
        relation: RelationSpec,
        single_shape: tuple[int, ...],
        local_shapes: tuple[tuple[int, ...], ...],
        world_size: int,
        context: str,
    ) -> None:
        if relation.dim is not None:
            raise ConcreteRelationError(f"{context}: replicate relation must not define dim")
        if relation.reduce_op is not None:
            raise ConcreteRelationError(f"{context}: replicate relation must not define reduce_op")
        _validate_concrete_local_count(local_shapes, world_size, context)
        for rank, local_shape in enumerate(local_shapes):
            if local_shape != single_shape:
                raise ConcreteRelationError(
                    f"{context}: replicate tensor on rank {rank} has shape {local_shape}, "
                    f"expected {single_shape}"
                )

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_single_shape: tuple[int, ...],
        original_local_shapes: tuple[tuple[int, ...], ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        return _ratio_constraints(
            single_shape,
            local_shapes,
            original_single_shape,
            original_local_shapes,
            world_size,
            context,
        )

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

    def validate_concrete_shapes(
        self,
        relation: RelationSpec,
        single_shape: tuple[int, ...],
        local_shapes: tuple[tuple[int, ...], ...],
        world_size: int,
        context: str,
    ) -> None:
        if not isinstance(relation.dim, int) or isinstance(relation.dim, bool):
            raise ConcreteRelationError(f"{context}: shard dim must be an integer")
        if relation.dim < 0 or relation.dim >= len(single_shape):
            raise ConcreteRelationError(
                f"{context}: shard dim {relation.dim} is out of range for tensor "
                f"{relation.single_tensor} with rank {len(single_shape)}"
            )
        if relation.reduce_op is not None:
            raise ConcreteRelationError(f"{context}: shard relation must not define reduce_op")
        _validate_concrete_local_count(local_shapes, world_size, context)
        if single_shape[relation.dim] % world_size != 0:
            raise ConcreteRelationError(
                f"{context}: shard dimension {single_shape[relation.dim]} of tensor "
                f"{relation.single_tensor} is not divisible by world_size {world_size}"
            )
        expected_shard_dimension = single_shape[relation.dim] // world_size
        expected_shape = tuple(
            expected_shard_dimension if index == relation.dim else extent
            for index, extent in enumerate(single_shape)
        )
        for rank, local_shape in enumerate(local_shapes):
            if local_shape != expected_shape:
                raise ConcreteRelationError(
                    f"{context}: shard tensor on rank {rank} has shape {local_shape}, "
                    f"expected {expected_shape}"
                )

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_single_shape: tuple[int, ...],
        original_local_shapes: tuple[tuple[int, ...], ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        if not isinstance(relation.dim, int) or isinstance(relation.dim, bool):
            raise ShapeReductionError(f"{context}: shard dim must be an integer")
        if relation.dim < 0 or relation.dim >= len(single_shape):
            raise ShapeReductionError(f"{context}: shard dim {relation.dim} is out of range")
        constraints = _ratio_constraints(
            single_shape,
            local_shapes,
            original_single_shape,
            original_local_shapes,
            world_size,
            context,
        )
        constraints.append(single_shape[relation.dim] % world_size == 0)
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

    def validate_concrete_shapes(
        self,
        relation: RelationSpec,
        single_shape: tuple[int, ...],
        local_shapes: tuple[tuple[int, ...], ...],
        world_size: int,
        context: str,
    ) -> None:
        if relation.dim is not None:
            raise ConcreteRelationError(f"{context}: partial relation must not define dim")
        if relation.reduce_op != "sum":
            raise ConcreteRelationError(f"{context}: partial reduce_op must be 'sum'")
        _validate_concrete_local_count(local_shapes, world_size, context)
        for rank, local_shape in enumerate(local_shapes):
            if local_shape != single_shape:
                raise ConcreteRelationError(
                    f"{context}: partial tensor on rank {rank} has shape {local_shape}, "
                    f"expected {single_shape}"
                )

    def shape_constraints(
        self,
        relation: RelationSpec,
        single_shape: tuple[z3.ArithRef, ...],
        local_shapes: tuple[tuple[z3.ArithRef, ...], ...],
        original_single_shape: tuple[int, ...],
        original_local_shapes: tuple[tuple[int, ...], ...],
        world_size: int,
        context: str,
    ) -> list[z3.BoolRef]:
        return _ratio_constraints(
            single_shape,
            local_shapes,
            original_single_shape,
            original_local_shapes,
            world_size,
            context,
        )

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
