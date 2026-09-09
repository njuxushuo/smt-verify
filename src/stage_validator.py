"""Validation rules for the Stage 1 input contract."""

from __future__ import annotations

from .operators import ConcreteShapeError, DECLARED_OPERATOR_TYPES, OPERATOR_REGISTRY
from .program_analysis import ProgramAnalysisError, find_program_inputs, validate_program_dataflow
from .relations import ConcreteRelationError, DECLARED_RELATION_TYPES, get_relation
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
        op_context = f"{context}.ops[{index}]"
        _validate_op(op, program.tensors, op_context)
    try:
        validate_program_dataflow(program, context)
    except ProgramAnalysisError as exc:
        raise StageInputError(str(exc)) from exc


def _validate_program_concrete_shapes(program: ProgramSpec, context: str) -> None:
    for index, op in enumerate(program.ops):
        semantics = OPERATOR_REGISTRY.get(op.type)
        if semantics is None:
            continue
        op_context = f"{context}.ops[{index}]"
        input_shapes = tuple(program.tensors[name].shape for name in op.inputs)
        output_shapes = tuple(program.tensors[name].shape for name in op.outputs)
        try:
            semantics.validate_concrete_shapes(input_shapes, output_shapes, op_context)
        except ConcreteShapeError as exc:
            raise StageInputError(str(exc)) from exc


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

    semantics = get_relation(relation.type, context)
    try:
        semantics.validate_concrete_shapes(
            relation, single_shape, tuple(local_shapes), stage.world_size, context
        )
    except ConcreteRelationError as exc:
        raise StageInputError(str(exc)) from exc


def _validate_input_relation_boundary(
    relation: RelationSpec,
    single_input_names: tuple[str, ...],
    distributed_input_names: dict[int, tuple[str, ...]],
    context: str,
) -> None:
    if relation.single_tensor not in single_input_names:
        raise StageInputError(
            f"{context}: input relation tensor {relation.single_tensor!r} must be a program input"
        )
    for rank, local_name in enumerate(relation.distributed_tensors):
        if local_name not in distributed_input_names[rank]:
            raise StageInputError(
                f"{context}: input relation tensor {local_name!r} on rank {rank} "
                "must be a program input"
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
    single_input_names = find_program_inputs(stage.single)
    distributed_input_names = {
        rank: find_program_inputs(stage.distributed[rank]) for rank in range(stage.world_size)
    }

    if not isinstance(stage.input_relations, (tuple, list)):
        raise StageInputError("input_relations: expected list")
    seen_single_tensors: set[str] = set()
    for index, relation in enumerate(stage.input_relations):
        context = f"input_relations[{index}]"
        _validate_relation(stage, relation, context)
        _validate_input_relation_boundary(
            relation, single_input_names, distributed_input_names, context
        )
        if relation.single_tensor in seen_single_tensors:
            raise StageInputError(
                f"{context}: duplicate input relation for single tensor {relation.single_tensor!r}"
            )
        seen_single_tensors.add(relation.single_tensor)

    _validate_relation(stage, stage.output_relation, "output_relation")

    _validate_program_concrete_shapes(stage.single, "single")
    for rank in range(stage.world_size):
        _validate_program_concrete_shapes(stage.distributed[rank], f"distributed.ranks[{rank}]")
