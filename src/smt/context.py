import z3

from .tensor import SymbolicTensor


class SymbolicContext:
    def __init__(self):
        self.functions = {}
        self.free_inputs = {}
        self.gelu = z3.Function("shared_gelu", z3.RealSort(), z3.RealSort())

    def input(self, ref, shape):
        if ref not in self.free_inputs:
            # Numeric IDs avoid collisions between user-controlled tensor names.
            function = z3.Function(
                f"input_{len(self.functions)}",
                *([z3.IntSort()] * len(shape)),
                z3.RealSort(),
            )
            self.functions[ref] = function
            self.free_inputs[ref] = SymbolicTensor(
                shape, lambda i, f=function: f(*i), ref.label()
            )
        return self.free_inputs[ref]
