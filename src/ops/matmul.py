import z3

from ..core.errors import InvalidInput, Unsupported
from ..smt.tensor import SymbolicTensor
from .api import arity, attrs_exact, ExecutionLimits
from .broadcast import broadcast_index, broadcast_shape


class MatMul:
    collective = False

    def __init__(self, limits=ExecutionLimits()):
        self.limits = limits

    def infer(self, call, shapes, group_size=1):
        arity(call, 2)
        attrs_exact(call)
        a, b = shapes
        if min(len(a), len(b)) < 2:
            raise Unsupported("matmul requires tensor rank >= 2")
        if a[-1] != b[-2]:
            raise InvalidInput("matmul contraction extents differ")
        return broadcast_shape(a[:-2], b[:-2]) + (a[-2], b[-1])

    def evaluate(self, call, inputs, context):
        a, b = inputs
        shape = self.infer(call, (a.shape, b.shape))
        if a.shape[-1] > self.limits.max_contraction:
            raise Unsupported(
                "matmul contraction exceeds configured exact expansion limit"
            )

        def element(index):
            left = broadcast_index(index[:-2], a.shape[:-2])
            right = broadcast_index(index[:-2], b.shape[:-2])
            return z3.Sum(
                [
                    a.at(left + (index[-2], k)) * b.at(right + (k, index[-1]))
                    for k in range(a.shape[-1])
                ]
            )

        return SymbolicTensor(shape, element)
