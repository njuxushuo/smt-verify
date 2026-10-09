from dataclasses import dataclass
from typing import Mapping

from ..core.program import Program
from ..core.topology import Topology
from ..relation.model import RelationBinding
from ..relation.normalization import NormalizedRelation


@dataclass(frozen=True)
class StageSpec:
    name: str
    topology: Topology
    single: Program
    distributed: Mapping[int, Program]
    input_relations: tuple[RelationBinding, ...]
    output_relations: tuple[RelationBinding, ...]


@dataclass(frozen=True)
class ValidatedStage:
    stage: StageSpec
    inputs: tuple[NormalizedRelation, ...]
    outputs: tuple[NormalizedRelation, ...]
    checks: tuple[str, ...]
