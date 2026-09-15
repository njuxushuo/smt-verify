"""Structured data objects for Stage JSON input."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TensorSpec:
    name: str
    shape: tuple[int, ...]


@dataclass
class OpSpec:
    type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, object] = field(default_factory=dict)


@dataclass
class ProgramSpec:
    tensors: dict[str, TensorSpec]
    ops: tuple[OpSpec, ...]


@dataclass
class RelationSpec:
    single_tensor: str
    distributed_tensors: tuple[str, ...]
    type: str
    dim: int | None = None
    reduce_op: str | None = None


@dataclass
class StageSpec:
    name: str
    world_size: int
    single: ProgramSpec
    distributed: dict[int, ProgramSpec]
    input_relations: tuple[RelationSpec, ...]
    output_relation: RelationSpec
