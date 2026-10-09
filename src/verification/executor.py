from dataclasses import dataclass

from ..core.errors import InvalidInput, Unsupported
from ..core.tensor import TensorRef
from .collectives import match_collectives


@dataclass
class Execution:
    tensors: dict
    trace: tuple[str, ...]


def execute_stage(stage, input_tensors, registry, context, max_depth=64):
    tensors = dict(input_tensors)
    trace = []

    def checked_depth(inputs):
        depth = 1 + max((t.depth for t in inputs), default=0)
        if depth > max_depth:
            raise Unsupported(
                f"lazy symbolic dependency depth exceeds {max_depth}; split the stage"
            )
        return depth

    for index, call in enumerate(stage.single.ops):
        ins = tuple(tensors[TensorRef(None, n)] for n in call.inputs)
        depth = checked_depth(ins)
        out = registry.get(call.type).evaluate(call, ins, context)
        out.depth = depth
        ref = TensorRef(None, call.outputs[0])
        out.label = ref.label()
        tensors[ref] = out
        trace.append(
            f"single op {index}: {call.type}{call.inputs} -> {call.outputs} shape={out.shape} attrs={dict(call.attrs)}"
        )
    lookup, events = match_collectives(stage, registry)
    cursor = {r: 0 for r in stage.distributed}
    while any(cursor[r] < len(p.ops) for r, p in stage.distributed.items()):
        progressed = False
        for rank, program in sorted(stage.distributed.items()):
            index = cursor[rank]
            if index == len(program.ops):
                continue
            call = program.ops[index]
            op = registry.get(call.type)
            if not op.collective:
                ins = tuple(tensors[TensorRef(rank, n)] for n in call.inputs)
                depth = checked_depth(ins)
                out = op.evaluate(call, ins, context)
                out.depth = depth
                ref = TensorRef(rank, call.outputs[0])
                out.label = ref.label()
                tensors[ref] = out
                cursor[rank] += 1
                progressed = True
                trace.append(
                    f"rank{rank} op {index}: {call.type}{call.inputs} -> {call.outputs} shape={out.shape} attrs={dict(call.attrs)}"
                )
            else:
                event = lookup[(rank, index)]
                if not all(cursor[r] == i for r, i in event.calls.items()):
                    continue
                members = stage.topology.groups[event.group]
                calls = [stage.distributed[r].ops[event.calls[r]] for r in members]
                inputs = tuple(
                    tensors[TensorRef(r, c.inputs[0])] for r, c in zip(members, calls)
                )
                depth = checked_depth(inputs)
                outputs = op.evaluate_group(call, inputs, context)
                for r, c, out in zip(members, calls, outputs):
                    out.depth = depth
                    ref = TensorRef(r, c.outputs[0])
                    out.label = ref.label()
                    tensors[ref] = out
                    cursor[r] += 1
                progressed = True
                trace.append(
                    f"collective {event.group} #{event.ordinal}: {call.type}, members={members}, attrs={dict(call.attrs)}, "
                    f"inputs={[t.shape for t in inputs]}, outputs={[t.shape for t in outputs]}"
                )
        if not progressed:
            raise InvalidInput(
                "collective schedule cannot make progress: incompatible cross-group ordering"
            )
    return Execution(tensors, tuple(trace))
