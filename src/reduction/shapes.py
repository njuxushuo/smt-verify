"""Build a reduced candidate; no equivalence conclusion follows from this alone."""

from dataclasses import dataclass
from functools import reduce
from math import gcd, prod

from ..core.errors import Unsupported
from ..core.tensor import TensorRef
from ..relation.shape_rules import constrain, rescale
from ..smt.shapes import ShapeConstraints
from ..stage.codec import stage_from_dict, stage_to_dict


@dataclass
class ShapePlan:
    stage: object
    changes: tuple
    constraints: tuple
    scales: dict
    block_size: int
    original_elements: int
    reduced_elements: int


def reduce_shapes(stage, registry, events, timeout_ms, max_elements):
    c = ShapeConstraints()
    programs = [(None, stage.single), *sorted(stage.distributed.items())]
    shapes = {}
    for rank, program in programs:
        for name, tensor in program.tensors.items():
            ref = TensorRef(rank, name)
            shapes[ref] = c.shape(ref, tensor.shape)
    for rank, program in programs:
        for number, call in enumerate(program.ops):
            op = registry.get(call.type)
            rule = getattr(op, "shape_constraints", None)
            if rule is None:
                raise Unsupported(f"{call.type} has no shape reduction rule")
            group_size = (
                len(stage.topology.groups[call.attrs["group"]]) if op.collective else 1
            )
            refs = [TensorRef(rank, n) for n in call.inputs]
            source = f"{'single' if rank is None else f'rank{rank}'}.ops[{number}] {call.type}"
            rule(
                c,
                source,
                call,
                [program.tensors[n].shape for n in call.inputs],
                [shapes[r] for r in refs],
                shapes[TensorRef(rank, call.outputs[0])],
                group_size,
            )
    for event in events:
        members = []
        for rank, index in event.calls.items():
            name = stage.distributed[rank].ops[index].inputs[0]
            members.append(shapes[TensorRef(rank, name)])
        for member in members[1:]:
            c.add(
                f"{event.group}#{event.ordinal}",
                "Matched collective members have equal input shapes.",
                *(a == b for a, b in zip(members[0], member)),
            )
    for role, bindings in (
        ("input", stage.input_relations),
        ("output", stage.output_relations),
    ):
        for number, binding in enumerate(bindings):
            constrain(binding, c, f"{role}_relations[{number}] (shape only)")
    model = c.solve(timeout_ms)
    scales = {key: model.eval(d.scale).as_long() for key, d in c.dimensions.items()}
    raw = stage_to_dict(stage)
    changes, contractions = [], []
    original_elements = reduced_elements = 0
    for rank, program in programs:
        target = (
            raw["single"] if rank is None else raw["distributed"]["ranks"][str(rank)]
        )
        for name, tensor in program.tensors.items():
            ref = TensorRef(rank, name)
            shape = tuple(model.eval(d).as_long() for d in shapes[ref])
            target["tensors"][name]["shape"] = list(shape)
            changes.append((ref.label(), tensor.shape, shape))
            original_elements += tensor.numel
            reduced_elements += prod(shape)
        for old, new in zip(program.ops, target["ops"]):
            op = registry.get(old.type)
            rewrite = getattr(op, "reduced_attrs", None)
            if rewrite is not None:
                new["attrs"] = rewrite(old, target["tensors"][old.outputs[0]]["shape"])
            for input_number, axis in getattr(op, "contraction_axes", ()):
                ref = TensorRef(rank, old.inputs[input_number])
                contractions.append(
                    scales[(ref, axis % len(program.tensors[ref.name].shape))]
                )
    if reduced_elements > max_elements:
        raise Unsupported(
            f"reduced stage has {reduced_elements} elements, budget {max_elements}"
        )
    for role in ("input_relations", "output_relations"):
        for binding in raw[role]:
            rescale(binding, scales)
    records = tuple(
        (source, reason, expression.sexpr())
        for source, reason, expression in c.constraints
    )
    return ShapePlan(
        stage_from_dict(raw),
        tuple(changes),
        records,
        scales,
        reduce(gcd, contractions) if contractions else 1,
        original_elements,
        reduced_elements,
    )
