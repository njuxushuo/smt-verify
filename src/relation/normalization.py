from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..core.errors import InvalidInput
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


def canonicalize_maps(pieces: tuple[RangeMap, ...]) -> tuple[RangeMap, ...]:
    """Sort by source index; merge only identical translations on adjacent ranges."""
    result = []
    for piece in sorted(pieces, key=lambda p: p.local_start):
        if (
            result
            and result[-1].local_stop == piece.local_start
            and result[-1].single_stop == piece.single_start
        ):
            previous = result[-1]
            result[-1] = RangeMap(
                previous.local_start,
                piece.local_stop,
                previous.single_start,
                piece.single_stop,
            )
        else:
            result.append(piece)
    return tuple(result)


def compose_maps(
    first: tuple[RangeMap, ...], second: tuple[RangeMap, ...]
) -> tuple[RangeMap, ...]:
    """Return second(first(i)) for validated, unit-stride injective maps.

    Sweep the first map's image against the second map's source partition.
    Both are disjoint; sorting by intermediate coordinates avoids a quadratic
    cross product even when either map reorders its pieces.
    """
    right = sorted(second, key=lambda p: p.local_start)
    result = []
    cursor = 0
    for left in sorted(first, key=lambda p: p.single_start):
        while cursor < len(right) and right[cursor].local_stop <= left.single_start:
            cursor += 1
        current = cursor
        covered = 0
        while current < len(right) and right[current].local_start < left.single_stop:
            following = right[current]
            start = max(left.single_start, following.local_start)
            stop = min(left.single_stop, following.local_stop)
            if start < stop:
                local_start = left.local_start + start - left.single_start
                global_start = following.single_start + start - following.local_start
                result.append(
                    RangeMap(
                        local_start,
                        local_start + stop - start,
                        global_start,
                        global_start + stop - start,
                    )
                )
                covered += stop - start
            if following.local_stop >= left.single_stop:
                break
            current += 1
        if covered != left.local_stop - left.local_start:
            raise InvalidInput("mapping composition does not cover the preceding image")
        cursor = current
    return canonicalize_maps(tuple(result))


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
    """Compose already validated chains in their supplied term order."""
    axes = {}
    groups = ()
    for term in binding.relation.terms:
        if isinstance(term.expr, IndexPhi):
            for m in term.expr.maps:
                key = (m.rank, term.expr.dim)
                axes[key] = (
                    compose_maps(axes[key], m.pieces)
                    if key in axes
                    else canonicalize_maps(m.pieces)
                )
        else:
            groups = topology.family_groups(term.family)
    return NormalizedRelation(
        binding,
        single_shape,
        MappingProxyType(dict(local_shapes)),
        MappingProxyType(axes),
        groups,
    )
