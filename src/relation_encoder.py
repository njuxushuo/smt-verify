"""Stage-level orchestration for symbolic relation value constraints."""

from __future__ import annotations

from dataclasses import dataclass

import z3

from .relations import RelationEncodingError, get_relation
from .stage_model import RelationSpec, StageSpec
from .symbolic_tensor import SymbolicStageResult, SymbolicTensor


@dataclass
class EncodedRelations:
    input_constraints: tuple[z3.BoolRef, ...]
    output_constraints: tuple[z3.BoolRef, ...]


def _resolve_relation_tensors(
    relation: RelationSpec,
    symbolic: SymbolicStageResult,
    context: str,
) -> tuple[SymbolicTensor, tuple[SymbolicTensor, ...]]:
    """Resolve one relation's single and rank-local symbolic tensors."""

    try:
        single_tensor = symbolic.single.tensors[relation.single_tensor]
    except KeyError as exc:
        raise RelationEncodingError(
            f"{context}: single tensor {relation.single_tensor!r} does not exist"
        ) from exc

    rank_count = len(symbolic.distributed)
    if len(relation.distributed_tensors) != rank_count:
        raise RelationEncodingError(
            f"{context}: expected {rank_count} distributed tensor names, "
            f"found {len(relation.distributed_tensors)}"
        )

    local_tensors: list[SymbolicTensor] = []
    for rank in range(rank_count):
        if rank not in symbolic.distributed:
            raise RelationEncodingError(f"{context}: symbolic result is missing rank {rank}")
        local_name = relation.distributed_tensors[rank]
        try:
            local_tensors.append(symbolic.distributed[rank].tensors[local_name])
        except KeyError as exc:
            raise RelationEncodingError(
                f"{context}: tensor {local_name!r} does not exist on rank {rank}"
            ) from exc
    return single_tensor, tuple(local_tensors)


def _encode_one_relation(
    stage: StageSpec,
    relation: RelationSpec,
    symbolic: SymbolicStageResult,
    context: str,
) -> list[z3.BoolRef]:
    """Delegate one resolved Stage relation to its registered value semantics."""

    semantics = get_relation(relation.type, context)
    single_tensor, local_tensors = _resolve_relation_tensors(relation, symbolic, context)
    return semantics.value_constraints(
        relation, single_tensor, local_tensors, stage.world_size, context
    )


def encode_stage_relations(
    stage: StageSpec,
    symbolic: SymbolicStageResult,
) -> EncodedRelations:
    """Encode ordered input premises and the output candidate separately."""

    if symbolic.stage_name != stage.name:
        raise RelationEncodingError(
            f"symbolic result is for stage {symbolic.stage_name!r}, expected {stage.name!r}"
        )
    if set(symbolic.distributed) != set(stage.distributed):
        raise RelationEncodingError(
            "symbolic distributed rank set must exactly match the stage rank set"
        )

    input_constraints: list[z3.BoolRef] = []
    for index, relation in enumerate(stage.input_relations):
        context = f"input_relations[{index}]"
        input_constraints.extend(_encode_one_relation(stage, relation, symbolic, context))

    output_constraints = _encode_one_relation(
        stage, stage.output_relation, symbolic, "output_relation"
    )
    return EncodedRelations(
        input_constraints=tuple(input_constraints),
        output_constraints=tuple(output_constraints),
    )
