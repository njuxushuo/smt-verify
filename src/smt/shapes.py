"""Finite-domain dimension optimization, independent of programs and relations."""

from dataclasses import dataclass
from math import isqrt

import z3

from ..core.errors import Unsupported


@dataclass(frozen=True)
class Dimension:
    original: int
    size: object
    scale: object


class ShapeConstraints:
    def __init__(self):
        self.dimensions = {}
        self.constraints = []

    def add(self, source, reason, *expressions):
        for expression in expressions:
            if type(expression) is bool:
                expression = z3.BoolVal(expression)
            self.constraints.append((source, reason, z3.simplify(expression)))

    def shape(self, ref, original):
        result = []
        for axis, extent in enumerate(original):
            key = (ref, axis)
            if key not in self.dimensions:
                number = len(self.dimensions)
                size, scale = z3.Ints(f"dimension_{number} scale_{number}")
                divisors = set()
                if isqrt(extent) > 100000:
                    divisors.add(1)
                else:
                    for d in range(1, isqrt(extent) + 1):
                        if extent % d == 0:
                            divisors.update((d, extent // d))
                choices = sorted(d for d in divisors if extent // d >= min(2, extent))
                self.add(
                    f"{ref.label()}.shape[{axis}]",
                    "Exact integral block scaling; singleton fixed; other axes retain >=2.",
                    z3.Or(*(z3.And(scale == d, size == extent // d) for d in choices)),
                )
                self.dimensions[key] = Dimension(extent, size, scale)
            result.append(self.dimensions[key].size)
        return tuple(result)

    def scale(self, ref, axis):
        return self.dimensions[(ref, axis)].scale

    def solve(self, timeout_ms):
        optimizer = z3.Optimize()
        optimizer.set(timeout=timeout_ms)
        optimizer.add(*(c for _, _, c in self.constraints))
        optimizer.minimize(z3.Sum([d.size for d in self.dimensions.values()]))
        status = optimizer.check()
        if status != z3.sat:
            raise Unsupported(
                f"shape optimization {status}: {optimizer.reason_unknown()}"
            )
        model = optimizer.model()
        if any(
            not z3.is_true(model.eval(c, model_completion=True))
            for _, _, c in self.constraints
        ):
            raise Unsupported("shape optimizer did not return a feasible model")
        return model
