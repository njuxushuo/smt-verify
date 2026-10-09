from ..smt.tensor import SymbolicTensor
from .api import arity, attrs_exact
from .broadcast import broadcast_index, broadcast_shape


class Elementwise:
    collective = False

    def __init__(self, kind):
        self.kind = kind

    def infer(self, call, shapes, group_size=1):
        arity(call, 1 if self.kind == "gelu" else 2)
        attrs_exact(call)
        return shapes[0] if self.kind == "gelu" else broadcast_shape(*shapes)

    def evaluate(self, call, inputs, context):
        shape = self.infer(call, tuple(t.shape for t in inputs))

        def element(index):
            xs = [t.at(broadcast_index(index, t.shape)) for t in inputs]
            if self.kind == "add":
                return xs[0] + xs[1]
            if self.kind == "mul":
                return xs[0] * xs[1]
            return context.gelu(xs[0])

        return SymbolicTensor(shape, element)
