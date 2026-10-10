from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter

from ..core.errors import InvalidInput, Unsupported


@dataclass(frozen=True)
class CollectiveEvent:
    group: str
    ordinal: int
    calls: dict[int, int]


def match_collectives(stage, registry):
    grouped = {}
    for rank, program in sorted(stage.distributed.items()):
        for index, call in enumerate(program.ops):
            if registry.get(call.type).collective:
                grouped.setdefault(call.attrs["group"], {}).setdefault(rank, []).append(
                    index
                )
    lookup, events = {}, []
    for group, by_rank in sorted(grouped.items()):
        members = stage.topology.groups[group]
        if set(by_rank) != set(members) or len({len(v) for v in by_rank.values()}) != 1:
            raise InvalidInput(
                f"collective group {group}: missing or mismatched member call count"
            )
        for ordinal in range(len(by_rank[members[0]])):
            calls = {r: by_rank[r][ordinal] for r in members}
            descriptors = []
            shapes = []
            for r, i in calls.items():
                p = stage.distributed[r]
                op = p.ops[i]
                descriptors.append((op.type, dict(op.attrs)))
                shapes.append(p.tensors[op.inputs[0]].shape)
            if any(d != descriptors[0] for d in descriptors):
                raise InvalidInput(
                    f"collective group {group} occurrence {ordinal}: member signatures differ"
                )
            if any(s != shapes[0] for s in shapes):
                raise Unsupported("uneven collective member input shapes")
            event = CollectiveEvent(group, ordinal, calls)
            events.append(event)
            for r, i in calls.items():
                lookup[(r, i)] = event
    return lookup, tuple(events)


def validate_schedule(stage, registry):
    """Check the original call dependencies without evaluating tensor values."""
    lookup, events = match_collectives(stage, registry)
    predecessors = {}
    for rank, program in sorted(stage.distributed.items()):
        previous = None
        for index, _ in enumerate(program.ops):
            event = lookup.get((rank, index))
            node = (
                ("collective", event.group, event.ordinal)
                if event is not None
                else ("local", rank, index)
            )
            parents = predecessors.setdefault(node, set())
            if previous is not None:
                parents.add(previous)
            previous = node
    try:
        tuple(TopologicalSorter(predecessors).static_order())
    except CycleError as exc:
        raise InvalidInput(
            "collective schedule cannot make progress: incompatible cross-group ordering"
        ) from exc
    return events
