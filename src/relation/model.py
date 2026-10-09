from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RangeMap:
    local_start: int
    local_stop: int
    single_start: int
    single_stop: int


@dataclass(frozen=True)
class RankMap:
    rank: int
    local_extent: int
    single_extent: int
    pieces: tuple[RangeMap, ...]


@dataclass(frozen=True)
class IndexPhi:
    dim: int
    maps: tuple[RankMap, ...]


@dataclass(frozen=True)
class Partial:
    op: str


@dataclass(frozen=True)
class RelationTerm:
    family: str
    expr: IndexPhi | Partial


@dataclass(frozen=True)
class TensorRelation:
    terms: tuple[RelationTerm, ...]


@dataclass(frozen=True)
class RelationBinding:
    single_tensor: str
    distributed_tensors: Mapping[int, str]
    relation: TensorRelation
