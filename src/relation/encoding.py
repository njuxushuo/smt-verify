"""Exact input parameterization and output equalities over original index domains."""

from dataclasses import dataclass

import z3

from ..core.tensor import TensorRef
from ..smt.tensor import SymbolicTensor


def _coordinate(value, pieces, inverse=False):
    if type(value) is int:
        value = z3.IntVal(value)

    def endpoints(p):
        return (
            (p.single_start, p.single_stop, p.local_start)
            if inverse
            else (p.local_start, p.local_stop, p.single_start)
        )

    # Callers establish local bounds or membership in the mapped image.
    start, stop, target = endpoints(pieces[-1])
    result = value + target - start
    for p in reversed(pieces[:-1]):
        start, stop, target = endpoints(p)
        result = z3.If(
            z3.And(value >= start, value < stop), value + target - start, result
        )
    return z3.simplify(result)


def forward(relation, rank, index):
    """Apply the normalized final map, never an individual intermediate term."""
    return tuple(_coordinate(i, relation.pieces(rank, d)) for d, i in enumerate(index))


def inverse(relation, rank, index):
    """Invert the composed map on its validated final single-tensor image."""
    return tuple(
        _coordinate(i, relation.pieces(rank, d), True) for d, i in enumerate(index)
    )


@dataclass
class InputEncoding:
    tensors: dict[TensorRef, SymbolicTensor]
    equations: tuple[str, ...]
    substitutions: tuple[str, ...]


def encode_inputs(single, distributed, relations, context):
    tensors = {}
    for rank, program in [(None, single), *sorted(distributed.items())]:
        for name in program.inputs:
            ref = TensorRef(rank, name)
            tensors[ref] = context.input(ref, program.tensors[name].shape)
    equations, substitutions = [], []
    for number, rel in enumerate(relations):
        b = rel.binding
        reference = tensors[TensorRef(None, b.single_tensor)]
        if not rel.reduction_groups:
            for rank, name in sorted(b.distributed_tensors.items()):
                ref = TensorRef(rank, name)
                raw = tensors[ref]
                equations.extend(
                    context.input_equations(
                        raw.shape,
                        f"rin{number}_r{rank}",
                        lambda ids: raw.at(ids)
                        == reference.at(forward(rel, rank, ids)),
                    )
                )
                tensors[ref] = SymbolicTensor(
                    raw.shape,
                    lambda i, rel=rel, rank=rank, reference=reference: reference.at(
                        forward(rel, rank, i)
                    ),
                    ref.label(),
                )
                substitutions.append(
                    f"{ref.label()}[i] := single.{b.single_tensor}[Phi_{rank}(i)]"
                )
        else:
            for group in rel.reduction_groups:
                pivot = group[-1]
                pivot_ref = TensorRef(pivot, b.distributed_tensors[pivot])
                raw = tensors[pivot_ref]
                others = tuple(
                    (rank, tensors[TensorRef(rank, b.distributed_tensors[rank])])
                    for rank in group[:-1]
                )

                def equation(ids):
                    target = forward(rel, pivot, ids)
                    terms = [t.at(inverse(rel, r, target)) for r, t in others] + [
                        raw.at(ids)
                    ]
                    return reference.at(target) == z3.Sum(terms)

                equations.extend(
                    context.input_equations(
                        raw.shape, f"rin{number}_g{pivot}", equation
                    )
                )

                def contribution(
                    i, rel=rel, pivot=pivot, reference=reference, others=others
                ):
                    target = forward(rel, pivot, i)
                    return reference.at(target) - z3.Sum(
                        [t.at(inverse(rel, r, target)) for r, t in others]
                    )

                tensors[pivot_ref] = SymbolicTensor(
                    raw.shape, contribution, pivot_ref.label()
                )
                substitutions.append(
                    f"{pivot_ref.label()}[i] := single.{b.single_tensor}[Phi_{pivot}(i)] - "
                    f"sum(other independent contributions in group {group}); no extra replica equalities"
                )
    return InputEncoding(tensors, tuple(equations), tuple(substitutions))


def output_equality(relation, group, index, tensors):
    rank = group[0]
    target = forward(relation, rank, index)
    b = relation.binding
    left = tensors[TensorRef(None, b.single_tensor)].at(target)
    values = [
        tensors[TensorRef(r, b.distributed_tensors[r])].at(inverse(relation, r, target))
        for r in group
    ]
    right = z3.Sum(values) if relation.reduction_groups else values[0]
    return left, right
