import z3

from ..core.errors import InvalidInput, Unsupported
from ..smt.tensor import SymbolicTensor
from .api import arity, attrs_exact
from .shape_rules import collective as collective_shape


class Collective:
    collective = True

    def shape_constraints(self, *args):
        return collective_shape(self.kind, *args)

    def __init__(self, kind):
        self.kind = kind

    def infer(self, call, shapes, group_size=1):
        arity(call, 1)
        required = (
            ("group",)
            + (() if self.kind == "all_reduce" else ("axis",))
            + (() if self.kind == "all_gather" else ("reduce_op",))
        )
        attrs_exact(call, required)
        if not isinstance(call.attrs["group"], str) or not call.attrs["group"]:
            raise InvalidInput("collective group must be a nonempty ID")
        if self.kind != "all_gather" and call.attrs["reduce_op"] != "sum":
            raise Unsupported("only sum collectives are implemented")
        shape = list(shapes[0])
        if self.kind != "all_reduce":
            dim = call.attrs["axis"]
            if type(dim) is not int or not 0 <= dim < len(shape):
                raise InvalidInput(
                    "collective axis must be a valid nonnegative dimension"
                )
            if self.kind == "all_gather":
                shape[dim] *= group_size
            else:
                if shape[dim] % group_size:
                    raise Unsupported("uneven reduce_scatter is unsupported")
                shape[dim] //= group_size
        return tuple(shape)

    def evaluate_group(self, call, inputs, context):
        if any(t.shape != inputs[0].shape for t in inputs):
            raise Unsupported("collectives require equal member input shapes")
        shape = self.infer(call, (inputs[0].shape,), len(inputs))
        result = []
        for member in range(len(inputs)):

            def element(index, member=member):
                if self.kind == "all_gather":
                    dim = call.attrs["axis"]
                    width = inputs[0].shape[dim]
                    values = []
                    # Clamp branch-local indices to avoid evaluating a concrete
                    # out-of-range index in an unreachable branch.
                    for t in inputs:
                        local = list(index)
                        local[dim] = index[dim] % width
                        values.append(t.at(tuple(local)))
                    owner = index[dim] / width
                    value = values[-1]
                    for i in reversed(range(len(inputs) - 1)):
                        value = z3.If(owner == i, values[i], value)
                    return value
                source = list(index)
                if self.kind == "reduce_scatter":
                    dim = call.attrs["axis"]
                    source[dim] += member * shape[dim]
                return z3.Sum([t.at(tuple(source)) for t in inputs])

            result.append(SymbolicTensor(shape, element))
        return tuple(result)
