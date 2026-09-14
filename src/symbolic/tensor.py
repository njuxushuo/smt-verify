"""Symbolic tensor storage and results for Program-level execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import z3


class SymbolicExecutionError(ValueError):
    """Raised when Program-level symbolic execution cannot proceed."""


@dataclass(frozen=True)
class SymbolicTensor:
    """An immutable, row-major flattened tensor of Z3 arithmetic expressions."""

    shape: tuple[int, ...]
    values: tuple[z3.ArithRef, ...]

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("SymbolicTensor shape must contain at least one dimension")
        for index, dimension in enumerate(self.shape):
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise ValueError(
                    f"SymbolicTensor shape[{index}] must be a positive integer"
                )
        if len(self.values) != self.numel:
            raise ValueError(
                f"SymbolicTensor values length {len(self.values)} does not match "
                f"shape element count {self.numel}"
            )

    @property
    def numel(self) -> int:
        return prod(self.shape)

    def at(self, index: tuple[int, ...]) -> z3.ArithRef:
        """Return one tensor element using a checked row-major index."""

        if len(index) != len(self.shape):
            raise IndexError(
                f"SymbolicTensor index rank {len(index)} does not match tensor rank {len(self.shape)}"
            )

        offset = 0
        for dimension, (coordinate, extent) in enumerate(zip(index, self.shape)):
            if coordinate < 0 or coordinate >= extent:
                raise IndexError(
                    f"SymbolicTensor index {coordinate} at dimension {dimension} is out of range "
                    f"for extent {extent}"
                )
            offset = offset * extent + coordinate
        return self.values[offset]


@dataclass
class SymbolicProgramResult:
    input_names: tuple[str, ...]
    tensors: dict[str, SymbolicTensor]


@dataclass
class SymbolicStageResult:
    stage_name: str
    single: SymbolicProgramResult
    distributed: dict[int, SymbolicProgramResult]


def create_symbolic_input_tensor(
    name: str,
    shape: tuple[int, ...],
    scope_prefix: str,
) -> SymbolicTensor:
    """Create row-major Z3 Real free variables for one program input tensor."""

    element_count = prod(shape)
    values = tuple(z3.Real(f"{scope_prefix}__{name}__v{index}") for index in range(element_count))
    return SymbolicTensor(shape=shape, values=values)
