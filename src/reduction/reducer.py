from math import prod

from .model import IndexDomain, ReductionPlan


def reduce_domains(relations):
    domains = []
    elements = 0
    for number, rel in enumerate(relations):
        groups = rel.reduction_groups or tuple((r,) for r in sorted(rel.local_shapes))
        for group in groups:
            shape = rel.local_shapes[group[0]]
            elements += prod(shape)
            domains.append(IndexDomain(number, group, shape))
    return ReductionPlan(
        "SYMBOLIC_INDEX_DOMAIN",
        tuple(domains),
        elements,
        (
            "Original tensor shapes and all map intervals remain unchanged.",
            "Replace enumeration of output elements by arbitrary integer coordinates i[d].",
            "Each query asserts 0 <= i[d] < original local extent[d]. No coordinate is fixed to a sample.",
            "For Partial, a representative member enumerates the common image; every group member contributes.",
            "MatMul contraction sums retain all original K terms. No reduced-K lifting assumption.",
            "SAT is an original-shape counterexample; UNSAT for all domains proves all original output elements.",
        ),
    )
