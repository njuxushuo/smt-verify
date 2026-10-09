from ..core.errors import InvalidInput, Unsupported
from ..relation.validation import validate_relation
from .dataflow import validate_dataflow
from .model import ValidatedStage


def validate_stage(stage, registry):
    stage.topology.validate()
    if any(type(r) is not int for r in stage.distributed) or set(
        stage.distributed
    ) != set(range(stage.topology.world_size)):
        raise InvalidInput("distributed programs must cover all ranks")
    checks = ["Topology: each family partitions world; group member order preserved."]
    for rank, program in [(None, stage.single), *sorted(stage.distributed.items())]:
        label = "single" if rank is None else f"rank{rank}"
        validate_dataflow(program, label)
        for i, call in enumerate(program.ops):
            op = registry.get(call.type)
            group_size = 1
            if op.collective:
                if rank is None:
                    raise Unsupported("single-reference collectives are not supported")
                group = call.attrs.get("group")
                if not isinstance(group, str) or group not in stage.topology.groups:
                    raise InvalidInput(f"{label}: unknown collective group")
                if rank not in stage.topology.groups[group]:
                    raise InvalidInput(
                        f"{label}: collective rank is not a group member"
                    )
                group_size = len(stage.topology.groups[group])
            inferred = op.infer(
                call, tuple(program.tensors[n].shape for n in call.inputs), group_size
            )
            if (
                len(call.outputs) != 1
                or program.tensors[call.outputs[0]].shape != inferred
            ):
                raise InvalidInput(
                    f"{label}.ops[{i}]: declared output shape differs from {inferred}"
                )
        checks.append(
            f"{label}: SSA, explicit boundaries and {len(program.ops)} operator shapes checked."
        )
    if not stage.output_relations:
        raise InvalidInput("output_relations must not be empty")
    normalized = []
    for role, bindings in [
        ("input", stage.input_relations),
        ("output", stage.output_relations),
    ]:
        result, seen_single, seen_local = [], set(), set()
        for b in bindings:
            if b.single_tensor in seen_single:
                raise InvalidInput(f"duplicate {role} single binding")
            seen_single.add(b.single_tensor)
            boundary = stage.single.inputs if role == "input" else stage.single.outputs
            if b.single_tensor not in boundary:
                raise InvalidInput(f"{role} binding is not on single boundary")
            if set(b.distributed_tensors) != set(stage.distributed):
                raise InvalidInput("binding missing ranks")
            locals_ = {}
            for rank, name in b.distributed_tensors.items():
                program = stage.distributed[rank]
                boundary = program.inputs if role == "input" else program.outputs
                if name not in boundary:
                    raise InvalidInput(f"{role} binding is not on rank{rank} boundary")
                if (rank, name) in seen_local:
                    raise InvalidInput(f"duplicate {role} rank-local tensor binding")
                seen_local.add((rank, name))
                locals_[rank] = program.tensors[name].shape
            result.append(
                validate_relation(
                    b,
                    stage.single.tensors[b.single_tensor].shape,
                    locals_,
                    stage.topology,
                )
            )
        if role == "output":
            if seen_single != set(stage.single.outputs):
                raise InvalidInput("missing single output goal")
            if seen_local != {
                (r, n) for r, p in stage.distributed.items() for n in p.outputs
            }:
                raise InvalidInput("missing distributed output goal")
        normalized.append(tuple(result))
        checks.append(
            f"{role}: {len(result)} bindings; rank/extent/range/domain constraints checked."
        )
    return ValidatedStage(stage, normalized[0], normalized[1], tuple(checks))
