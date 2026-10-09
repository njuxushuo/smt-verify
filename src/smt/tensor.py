from dataclasses import dataclass, field
from typing import Callable

import z3

from ..core.errors import InvalidInput
from ..core.tensor import Shape

Index = tuple[z3.ArithRef, ...]


@dataclass
class SymbolicTensor:
    shape: Shape
    evaluate: Callable[[Index], z3.ArithRef]
    label: str = ""
    depth: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    def at(self, index):
        if len(index) != len(self.shape):
            raise InvalidInput(f"{self.label}: wrong index dimensionality")
        index = tuple(z3.simplify(z3.IntVal(i) if type(i) is int else i) for i in index)
        for i, n in zip(index, self.shape):
            if z3.is_int_value(i) and not 0 <= i.as_long() < n:
                raise InvalidInput(f"{self.label}: out-of-bounds index {i}")
        key = tuple(i.sexpr() for i in index)
        if key not in self._cache:
            self._cache[key] = z3.simplify(self.evaluate(index))
        return self._cache[key]


def bounds(index, shape):
    return z3.And(*(c for i, n in zip(index, shape) for c in (i >= 0, i < n)))
