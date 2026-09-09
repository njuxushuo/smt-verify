"""Validation rules for the Stage 1 input contract."""

from __future__ import annotations

from .operators import DECLARED_OPERATOR_TYPES
from .relations import DECLARED_RELATION_TYPES
from .stage_loader import StageInputError
from .stage_model import OpSpec, ProgramSpec, RelationSpec, StageSpec, TensorSpec


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_tensor(name: object, tensor: object, context: str) -> None:
    if not isinstance(name, str) or not name:
        raise StageInputError(f"{context}: tensor name must be a non-empty string")
    if not isinstance(tensor, TensorSpec):
        raise StageInputError(f"{context}: expected TensorSpec")
    if tensor.name != name:
        raise StageInputError(f"{context}: tensor name does not match its dictionary key")
    if not isinstance(tensor.shape, (tuple, list)) or not tensor.shape:
        raise StageInputError(f"{context}: shape must contain at least one dimension")
    for dim_index, dim in enumerate(tensor.shape):
        if not _is_integer(dim):
            raise StageInputError(f"{context}.shape[{dim_index}]: dimension must be an integer")
        if dim <= 0:
            raise StageInputError(f"{context}.shape[{dim_index}]: dimension must be greater than 0")


def _validate_op(op: object, tensors: dict[str, TensorSpec], context: str) -> None:
    if not isinstance(op, OpSpec):
        raise StageInputError(f"{context}: expected OpSpec")
    if not isinstance(op.type, str) or op.type not in DECLARED_OPERATOR_TYPES:
        raise StageInputError(f"{context}.type: unsupported operator {op.type!r}")
    for field_name, references in (("inputs", op.inputs), ("outputs", op.outputs)):
        if not isinstance(references, (tuple, list)) or not references:
            raise StageInputError(f"{context}.{field_name}: must contain at least one tensor")
        for index, tensor_name in enumerate(references):
            if not isinstance(tensor_name, str) or not tensor_name:
                raise StageInputError(f"{context}.{field_name}[{index}]: tensor name must be a non-empty string")
            if tensor_name not in tensors:
                raise StageInputError(
                    f"{context}.{field_name}[{index}]: unknown tensor {tensor_name!r}"
                )


def _validate_program(program: object, context: str) -> None:
    if not isinstance(program, ProgramSpec):
        raise StageInputError(f"{context}: expected ProgramSpec")
    if not isinstance(program.tensors, dict) or not program.tensors:
        raise StageInputError(f"{context}.tensors: must be non-empty")
    for name, tensor in program.tensors.items():
        _validate_tensor(name, tensor, f"{context}.tensors[{name!r}]")
    if not isinstance(program.ops, (tuple, list)):
        raise StageInputError(f"{context}.ops: expected ordered list")
    for index, op in enumerate(program.ops):
        _validate_op(op, program.tensors, f"{context}.ops[{index}]")


def _validate_relation(stage: StageSpec, relation: object, context: str) -> None:
    if not isinstance(relation, RelationSpec):
        raise StageInputError(f"{context}: expected RelationSpec")
    if not isinstance(relation.single_tensor, str) or relation.single_tensor not in stage.single.tensors:
        raise StageInputError(f"{context}: unknown single tensor {relation.single_tensor!r}")
    if not isinstance(relation.distributed_tensors, (tuple, list)):
        raise StageInputError(f"{context}.distributed_tensors: expected list")
    if len(relation.distributed_tensors) != stage.world_size:
        raise StageInputError(
            f"{context}.distributed_tensors: expected {stage.world_size} tensors, "
            f"found {len(relation.distributed_tensors)}"
        )
    if not isinstance(relation.type, str) or relation.type not in DECLARED_RELATION_TYPES:
        raise StageInputError(f"{context}.type: unsupported relation {relation.type!r}")

    single_shape = stage.single.tensors[relation.single_tensor].shape
    local_shapes: list[tuple[int, ...]] = []
    for rank, tensor_name in enumerate(relation.distributed_tensors):
        if not isinstance(tensor_name, str) or not tensor_name:
            raise StageInputError(
                f"{context}.distributed_tensors[{rank}]: tensor name must be a non-empty string"
            )
        rank_tensors = stage.distributed[rank].tensors
        if tensor_name not in rank_tensors:
            raise StageInputError(
                f"{context}.distributed_tensors[{rank}]: tensor {tensor_name!r} "
                f"does not exist on rank {rank}"
            )
        local_shapes.append(rank_tensors[tensor_name].shape)

    if relation.type == "replicate":
        if relation.dim is not None:
            raise StageInputError(f"{context}: replicate relation must not define dim")
        if relation.reduce_op is not None:
            raise StageInputError(f"{context}: replicate relation must not define reduce_op")
        for rank, local_shape in enumerate(local_shapes):
            if local_shape != single_shape:
                raise StageInputError(
                    f"{context}: replicate tensor on rank {rank} has shape {local_shape}, "
                    f"expected {single_shape}"
                )
        return

    if relation.type == "shard":
        if not _is_integer(relation.dim):
            raise StageInputError(f"{context}: shard dim must be an integer")
        if relation.dim < 0 or relation.dim >= len(single_shape):
            raise StageInputError(
                f"{context}: shard dim {relation.dim} is out of range for tensor "
                f"{relation.single_tensor} with rank {len(single_shape)}"
            )
        if relation.reduce_op is not None:
            raise StageInputError(f"{context}: shard relation must not define reduce_op")
        if single_shape[relation.dim] % stage.world_size != 0:
            raise StageInputError(
                f"{context}: shard dimension {single_shape[relation.dim]} of tensor "
                f"{relation.single_tensor} is not divisible by world_size {stage.world_size}"
            )
        expected_shard_dim = single_shape[relation.dim] // stage.world_size
        for rank, local_shape in enumerate(local_shapes):
            if len(local_shape) != len(single_shape):
                raise StageInputError(
                    f"{context}: shard tensor on rank {rank} has rank {len(local_shape)}, "
                    f"expected {len(single_shape)}"
                )
            for dim_index, (single_dim, local_dim) in enumerate(zip(single_shape, local_shape)):
                expected_dim = expected_shard_dim if dim_index == relation.dim else single_dim
                if local_dim != expected_dim:
                    raise StageInputError(
                        f"{context}: shard tensor on rank {rank} has dimension {local_dim} "
                        f"at dim {dim_index}, expected {expected_dim}"
                    )
        return

    if relation.dim is not None:
        raise StageInputError(f"{context}: partial relation must not define dim")
    if relation.reduce_op != "sum":
        raise StageInputError(f"{context}: partial reduce_op must be 'sum'")
    for rank, local_shape in enumerate(local_shapes):
        if local_shape != single_shape:
            raise StageInputError(
                f"{context}: partial tensor on rank {rank} has shape {local_shape}, "
                f"expected {single_shape}"
            )


def validate_stage(stage: StageSpec) -> None:
    """Validate a parsed StageSpec, raising StageInputError on failure."""

    if not isinstance(stage, StageSpec):
        raise StageInputError("stage: expected StageSpec")
    if not isinstance(stage.name, str) or not stage.name:
        raise StageInputError("stage.name: must be a non-empty string")
    if not _is_integer(stage.world_size) or stage.world_size < 1:
        raise StageInputError("stage.world_size: must be an integer greater than or equal to 1")
    if not isinstance(stage.distributed, dict):
        raise StageInputError("distributed: expected rank mapping")
    expected_ranks = set(range(stage.world_size))
    found_ranks = set(stage.distributed)
    if found_ranks != expected_ranks:
        raise StageInputError(
            f"rank count mismatch: world_size={stage.world_size}, found ranks={sorted(found_ranks)}"
        )

    _validate_program(stage.single, "single")
    for rank in range(stage.world_size):
        _validate_program(stage.distributed[rank], f"distributed.ranks[{rank}]")

    if not isinstance(stage.input_relations, (tuple, list)):
        raise StageInputError("input_relations: expected list")
    seen_single_tensors: set[str] = set()
    for index, relation in enumerate(stage.input_relations):
        context = f"input_relations[{index}]"
        _validate_relation(stage, relation, context)
        if relation.single_tensor in seen_single_tensors:
            raise StageInputError(
                f"{context}: duplicate input relation for single tensor {relation.single_tensor!r}"
            )
        seen_single_tensors.add(relation.single_tensor)

    _validate_relation(stage, stage.output_relation, "output_relation")
