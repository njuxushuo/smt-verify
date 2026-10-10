"""Conservative exact scaling of all levels of an IndexPhi chain."""

from ..core.tensor import TensorRef
from .model import IndexPhi


def constrain(binding, constraints, source):
    single = TensorRef(None, binding.single_tensor)
    rank = len([key for key in constraints.dimensions if key[0] == single])
    for r, name in binding.distributed_tensors.items():
        local = TensorRef(r, name)
        for axis in range(rank):
            constraints.add(
                source,
                "One scale in corresponding coordinate spaces, including implicit identity axes.",
                constraints.scale(single, axis) == constraints.scale(local, axis),
            )
    for term in binding.relation.terms:
        if not isinstance(term.expr, IndexPhi):
            continue
        expr = term.expr
        scale = constraints.scale(single, expr.dim)
        for m in expr.maps:
            endpoints = [m.local_extent, m.single_extent]
            for p in m.pieces:
                endpoints.extend(
                    (p.local_start, p.local_stop, p.single_start, p.single_stop)
                )
            constraints.add(
                source,
                f"{term.family} rank {m.rank}: exactly divisible extents and piece boundaries; no rounded offsets.",
                *(n % scale == 0 for n in endpoints),
            )


def rescale(raw, scales):
    """Transform an encoded binding; retain every family and supplied term order."""
    ref = TensorRef(None, raw["single_tensor"])
    for term in raw["relation"]["terms"]:
        expr = term["expr"]
        if expr["kind"] != "index_phi":
            continue
        scale = scales[(ref, expr["dim"])]
        for m in expr["maps"]:
            for key in ("local_extent", "single_extent"):
                m[key] //= scale
            for p in m["pieces"]:
                for side in ("local", "single"):
                    for key in ("start", "stop"):
                        p[side][key] //= scale
