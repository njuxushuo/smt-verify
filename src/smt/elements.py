"""Finite symbolic inputs; every access must have concrete integer coordinates."""

from itertools import product

import z3

from ..core.errors import Unsupported
from .context import SymbolicContext
from .tensor import SymbolicTensor


def coordinates(shape):
    return product(*(range(n) for n in shape))


class ElementContext(SymbolicContext):
    def __init__(self):
        super().__init__()
        self.elements = {}

    def input(self, ref, shape):
        if ref not in self.free_inputs:
            number = len(self.free_inputs)
            values = {
                i: z3.Real(f"element_{number}_" + "_".join(map(str, i)))
                for i in coordinates(shape)
            }
            self.elements[ref] = values

            def at(index):
                if any(not z3.is_int_value(i) for i in index):
                    raise Unsupported("element encoding requires concrete indices")
                return values[tuple(i.as_long() for i in index)]

            self.free_inputs[ref] = SymbolicTensor(shape, at, ref.label())
        return self.free_inputs[ref]

    def input_equations(self, shape, prefix, equation):
        return tuple(equation(i).sexpr() for i in coordinates(shape))
