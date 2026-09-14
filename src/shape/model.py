"""Data objects for symbolic and reduced Stage shapes."""

from __future__ import annotations

from dataclasses import dataclass

import z3


class ShapeReductionError(ValueError):
    pass


class UnsupportedShapeSemanticsError(ShapeReductionError):
    pass


@dataclass(frozen=True)
class TensorRef:
    scope: str
    rank: int | None
    name: str

    def __post_init__(self) -> None:
        if self.scope not in {"single", "distributed"}:
            raise ValueError("TensorRef scope must be 'single' or 'distributed'")


@dataclass
class SymbolicShapeBook:
    shapes: dict[TensorRef, tuple[z3.ArithRef, ...]]


@dataclass
class ReducedShapeResult:
    stage_name: str
    single: dict[str, tuple[int, ...]]
    distributed: dict[int, dict[str, tuple[int, ...]]]
    objective_value: int
