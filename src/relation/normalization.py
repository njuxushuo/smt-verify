from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..core.tensor import Shape
from .model import IndexPhi, RangeMap, RelationBinding


def merged_intervals(pieces):
    result = []
    for start, stop in sorted(pieces):
        if result and result[-1][1] == start:
            result[-1] = (result[-1][0], stop)
        else:
            result.append((start, stop))
    return tuple(result)


@dataclass(frozen=True)
class NormalizedRelation:
    binding: RelationBinding
    single_shape: Shape
    local_shapes: Mapping[int, Shape]
    axes: Mapping[tuple[int, int], tuple[RangeMap, ...]]
    reduction_groups: tuple[tuple[int, ...], ...]

    def pieces(self, rank, dim):
        return self.axes.get(
            (rank, dim),
            (RangeMap(0, self.local_shapes[rank][dim], 0, self.single_shape[dim]),),
        )

    def image(self, rank):
        return tuple(
            merged_intervals(
                (p.single_start, p.single_stop) for p in self.pieces(rank, d)
            )
            for d in range(len(self.single_shape))
        )


def normalize(binding, single_shape, local_shapes, topology):
    axes = {}
    groups = ()
    for term in binding.relation.terms:
        if isinstance(term.expr, IndexPhi):
            for m in term.expr.maps:
                axes[(m.rank, term.expr.dim)] = m.pieces
        else:
            groups = topology.family_groups(term.family)
    return NormalizedRelation(
        binding,
        single_shape,
        MappingProxyType(dict(local_shapes)),
        MappingProxyType(axes),
        groups,
    )
