from dataclasses import dataclass


@dataclass(frozen=True)
class IndexDomain:
    relation_index: int
    group: tuple[int, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ReductionPlan:
    method: str
    domains: tuple[IndexDomain, ...]
    original_output_elements: int
    notes: tuple[str, ...]
