"""Load Stage JSON input into the Stage 1 structured representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage_model import OpSpec, ProgramSpec, RelationSpec, StageSpec, TensorSpec


class StageInputError(ValueError):
    """Raised when a Stage input file is malformed or invalid."""


def _required(mapping: dict[str, Any], field: str, context: str) -> Any:
    if field not in mapping:
        raise StageInputError(f"{context}: missing required field '{field}'")
    return mapping[field]


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StageInputError(f"{context}: expected object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise StageInputError(f"{context}: expected list")
    return value


def _string_list(value: Any, context: str) -> tuple[str, ...]:
    values = _list(value, context)
    for index, item in enumerate(values):
        if not isinstance(item, str):
            raise StageInputError(f"{context}[{index}]: expected string")
    return tuple(values)


def _parse_program(value: Any, context: str) -> ProgramSpec:
    raw_program = _mapping(value, context)
    raw_tensors = _mapping(_required(raw_program, "tensors", context), f"{context}.tensors")
    raw_ops = _list(_required(raw_program, "ops", context), f"{context}.ops")

    tensors: dict[str, TensorSpec] = {}
    for name, raw_tensor in raw_tensors.items():
        tensor_context = f"{context}.tensors[{name!r}]"
        raw_tensor_mapping = _mapping(raw_tensor, tensor_context)
        shape = _list(_required(raw_tensor_mapping, "shape", tensor_context), f"{tensor_context}.shape")
        tensors[name] = TensorSpec(name=name, shape=tuple(shape))

    ops: list[OpSpec] = []
    for index, raw_op in enumerate(raw_ops):
        op_context = f"{context}.ops[{index}]"
        raw_op_mapping = _mapping(raw_op, op_context)
        ops.append(
            OpSpec(
                type=_required(raw_op_mapping, "type", op_context),
                inputs=_string_list(
                    _required(raw_op_mapping, "inputs", op_context), f"{op_context}.inputs"
                ),
                outputs=_string_list(
                    _required(raw_op_mapping, "outputs", op_context), f"{op_context}.outputs"
                ),
            )
        )
    return ProgramSpec(tensors=tensors, ops=tuple(ops))


def _parse_relation(value: Any, context: str) -> RelationSpec:
    raw_relation = _mapping(value, context)
    return RelationSpec(
        single_tensor=_required(raw_relation, "single_tensor", context),
        distributed_tensors=_string_list(
            _required(raw_relation, "distributed_tensors", context),
            f"{context}.distributed_tensors",
        ),
        type=_required(raw_relation, "type", context),
        dim=raw_relation.get("dim"),
        reduce_op=raw_relation.get("reduce_op"),
    )


def _parse_distributed(value: Any) -> dict[int, ProgramSpec]:
    raw_distributed = _mapping(value, "distributed")
    raw_ranks = _mapping(
        _required(raw_distributed, "ranks", "distributed"), "distributed.ranks"
    )
    distributed: dict[int, ProgramSpec] = {}
    for raw_rank, raw_program in raw_ranks.items():
        if not isinstance(raw_rank, str):
            raise StageInputError("distributed.ranks: rank key must be a string")
        try:
            rank = int(raw_rank)
        except ValueError as exc:
            raise StageInputError(
                f"distributed.ranks[{raw_rank!r}]: rank key must be an integer string"
            ) from exc
        if str(rank) != raw_rank:
            raise StageInputError(
                f"distributed.ranks[{raw_rank!r}]: rank key must use canonical decimal form"
            )
        if rank in distributed:
            raise StageInputError(f"distributed.ranks: duplicate rank {rank}")
        distributed[rank] = _parse_program(raw_program, f"distributed.ranks[{raw_rank}]")
    return distributed


def _parse_stage(raw_stage: Any) -> StageSpec:
    raw = _mapping(raw_stage, "stage")
    input_relations = _list(_required(raw, "input_relations", "stage"), "input_relations")
    return StageSpec(
        name=_required(raw, "name", "stage"),
        world_size=_required(raw, "world_size", "stage"),
        single=_parse_program(_required(raw, "single", "stage"), "single"),
        distributed=_parse_distributed(_required(raw, "distributed", "stage")),
        input_relations=tuple(
            _parse_relation(relation, f"input_relations[{index}]")
            for index, relation in enumerate(input_relations)
        ),
        output_relation=_parse_relation(
            _required(raw, "output_relation", "stage"), "output_relation"
        ),
    )


def load_stage(path: str | Path) -> StageSpec:
    """Read, parse, and validate a UTF-8 JSON Stage input file."""

    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            raw_stage = json.load(handle)
    except json.JSONDecodeError as exc:
        raise StageInputError(f"invalid JSON in {input_path}: {exc.msg}") from exc
    except OSError as exc:
        raise StageInputError(f"cannot read stage input {input_path}: {exc}") from exc

    try:
        stage = _parse_stage(raw_stage)
        from .stage_validator import validate_stage

        validate_stage(stage)
        return stage
    except StageInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise StageInputError(f"invalid stage input: {exc}") from exc
