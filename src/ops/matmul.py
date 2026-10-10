from ..core.errors import InvalidInput, Unsupported
from ..smt.tensor import SymbolicTensor
from .api import arity, attrs_exact, ExecutionLimits
from .broadcast import broadcast_index, broadcast_shape
from .shape_rules import matmul


class MatMul:
    collective = False
    shape_constraints = staticmethod(matmul)
    contraction_axes = ((0, -1),)

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
        if context.contraction_terms(a.shape[-1]) > self.limits.max_contraction:
            raise Unsupported(
                "matmul contraction exceeds configured exact expansion limit"
            )

        def element(index):
            left = broadcast_index(index[:-2], a.shape[:-2])
            right = broadcast_index(index[:-2], b.shape[:-2])
            return context.contract(
                a.shape[-1],
                lambda k: a.at(left + (index[-2], k)) * b.at(right + (k, index[-1])),
            )

        return SymbolicTensor(shape, element)
