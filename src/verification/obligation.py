from dataclasses import dataclass

import z3

from ..relation.encoding import output_equality
from ..smt.tensor import bounds


@dataclass
class ProofObligation:
    name: str
    domain: object
    indices: tuple
    guard: object
    single_value: object
    distributed_value: object
    violation: object


def build_obligation(number, domain, relation, tensors):
    ids = tuple(z3.Int(f"goal{number}_i{d}") for d in range(len(domain.shape)))
    left, right = output_equality(relation, domain.group, ids, tensors)
    return ProofObligation(
        f"output[{domain.relation_index}] group={domain.group}",
        domain,
        ids,
        bounds(ids, domain.shape),
        left,
        right,
        z3.simplify(left != right),
    )
