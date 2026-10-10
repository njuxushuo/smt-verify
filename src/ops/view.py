from math import prod

from ..core.errors import InvalidInput
from ..smt.tensor import SymbolicTensor
from .api import arity, attrs_exact, axis
from .broadcast import broadcast_index, broadcast_shape
from .shape_rules import view


class View:
    collective = False

    def shape_constraints(self, *args):
        return view(self.kind, *args)

    def reduced_attrs(self, call, output_shape):
        attrs = dict(call.attrs)
        if self.kind in ("reshape", "expand"):
            attrs["shape"] = list(output_shape)
        return attrs

    def __init__(self, kind):
        self.kind = kind

    def infer(self, call, shapes, group_size=1):
        arity(call, 1)
        (shape,) = shapes
        if self.kind == "transpose":
            attrs_exact(call, ("dim0", "dim1"))
            a, b = (
                axis(call.attrs["dim0"], len(shape)),
                axis(call.attrs["dim1"], len(shape)),
            )
            result = list(shape)
            result[a], result[b] = result[b], result[a]
            return tuple(result)
        attrs_exact(call, ("shape",))
        target = call.attrs["shape"]
        if (
            not isinstance(target, tuple)
            or not target
            or any(type(n) is not int or n < 1 for n in target)
        ):
            raise InvalidInput(f"{self.kind}: target must be concrete positive shape")
        if self.kind == "reshape":
            if prod(shape) != prod(target):
                raise InvalidInput("reshape element counts differ")
        elif len(target) < len(shape) or broadcast_shape(shape, target) != target:
            raise InvalidInput("expand target does not broadcast input")
        return target

    def evaluate(self, call, inputs, context):
        (tensor,) = inputs
        shape = self.infer(call, (tensor.shape,))

        def element(index):
            if self.kind == "transpose":
                a, b = (
                    axis(call.attrs["dim0"], len(shape)),
                    axis(call.attrs["dim1"], len(shape)),
                )
                source = list(index)
                source[a], source[b] = source[b], source[a]
                return tensor.at(tuple(source))
            if self.kind == "expand":
                return tensor.at(broadcast_index(index, tensor.shape))
            flat = sum(i * prod(shape[d + 1 :]) for d, i in enumerate(index))
            source = tuple(
                (flat / prod(tensor.shape[d + 1 :])) % n
                for d, n in enumerate(tensor.shape)
            )
            return tensor.at(source)

        return SymbolicTensor(shape, element)
