from ..core.errors import InvalidInput, Unsupported
from .model import (
    IndexPhi,
    Partial,
    RangeMap,
    RankMap,
    RelationBinding,
    RelationTerm,
    TensorRelation,
)
from .normalization import normalize


def validate_relation(binding, single_shape, local_shapes, topology):
    if not isinstance(binding, RelationBinding) or not isinstance(
        binding.relation, TensorRelation
    ):
        raise InvalidInput("expected a typed relation binding")
    world = set(range(topology.world_size))
    if any(type(r) is not int for r in binding.distributed_tensors):
        raise InvalidInput("relation rank IDs must be integers, not booleans")
    if set(binding.distributed_tensors) != world or set(local_shapes) != world:
        raise InvalidInput("relation must bind exactly all ranks")
    if any(len(s) != len(single_shape) for s in local_shapes.values()):
        raise InvalidInput("relation tensor dimensionalities must match")
    families, dims = set(), set()
    partials = 0
    for term in binding.relation.terms:
        if not isinstance(term, RelationTerm):
            raise InvalidInput("expected RelationTerm")
        topology.family_groups(term.family)
        if term.family in families:
            raise Unsupported("repeated family terms are outside the initial contract")
        families.add(term.family)
        e = term.expr
        if isinstance(e, Partial):
            partials += 1
            if e.op != "sum" or partials > 1:
                raise Unsupported("only one Partial(sum) term is supported")
            continue
        if not isinstance(e, IndexPhi):
            raise InvalidInput("expected IndexPhi or Partial")
        if type(e.dim) is not int or not 0 <= e.dim < len(single_shape):
            raise InvalidInput("IndexPhi dimension outside tensor rank")
        if e.dim in dims:
            raise Unsupported(
                "multiple families mapping the same dimension are unsupported"
            )
        dims.add(e.dim)
        if any(not isinstance(m, RankMap) or type(m.rank) is not int for m in e.maps):
            raise InvalidInput("IndexPhi rank maps must have integer rank IDs")
        if len(e.maps) != topology.world_size or {m.rank for m in e.maps} != world:
            raise InvalidInput(
                "IndexPhi maps must include each rank exactly once; empty maps are invalid"
            )
        changed = False
        for m in e.maps:
            if any(
                type(n) is not int or n < 1 for n in (m.local_extent, m.single_extent)
            ):
                raise InvalidInput("IndexPhi extents must be positive integers")
            if (
                m.local_extent != local_shapes[m.rank][e.dim]
                or m.single_extent != single_shape[e.dim]
            ):
                raise InvalidInput("IndexPhi extent differs from tensor shape")
            end = 0
            images = []
            if any(
                not isinstance(p, RangeMap)
                or any(
                    type(n) is not int
                    for n in (
                        p.local_start,
                        p.local_stop,
                        p.single_start,
                        p.single_stop,
                    )
                )
                for p in m.pieces
            ):
                raise InvalidInput("range map bounds must be integers")
            for p in sorted(m.pieces, key=lambda p: p.local_start):
                if not (
                    p.local_start == end < p.local_stop <= m.local_extent
                    and 0 <= p.single_start < p.single_stop <= m.single_extent
                    and p.local_stop - p.local_start == p.single_stop - p.single_start
                ):
                    raise InvalidInput(
                        "invalid, overlapping, or gapped local range map"
                    )
                end = p.local_stop
                images.append((p.single_start, p.single_stop))
                changed |= p.local_start != p.single_start
            if end != m.local_extent:
                raise InvalidInput("pieces do not cover local dimension")
            images.sort()
            if any(a[1] > b[0] for a, b in zip(images, images[1:])):
                raise InvalidInput("overlapping single image within one rank")
            changed |= m.local_extent != m.single_extent
        if not changed:
            raise InvalidInput(
                "identity-only IndexPhi must be omitted; use terms=[] for full equality"
            )
    for shape in local_shapes.values():
        if any(shape[d] != single_shape[d] for d in range(len(shape)) if d not in dims):
            raise InvalidInput(
                "unmapped dimension must have equal single/local extents"
            )
    normalized = normalize(binding, single_shape, local_shapes, topology)
    for group in normalized.reduction_groups:
        if any(normalized.image(r) != normalized.image(group[0]) for r in group):
            raise InvalidInput(
                "Partial group members must cover the same mapped single domain"
            )
    return normalized
