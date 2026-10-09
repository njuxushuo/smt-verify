from ..core.errors import InvalidInput


def validate_dataflow(program, label):
    if not program.tensors or not program.outputs:
        raise InvalidInput(f"{label}: tensors and outputs must be nonempty")
    for name, t in program.tensors.items():
        if not isinstance(name, str) or not name or t.name != name:
            raise InvalidInput(f"{label}: tensor name differs from its declaration key")
        if not t.shape or any(type(n) is not int or n < 1 for n in t.shape):
            raise InvalidInput(
                f"{label}: shapes must be nonempty positive integer tuples"
            )
    if len(set(program.inputs)) != len(program.inputs) or len(
        set(program.outputs)
    ) != len(program.outputs):
        raise InvalidInput(f"{label}: duplicate boundary tensor")
    available = set(program.inputs)
    if not available <= program.tensors.keys():
        raise InvalidInput(f"{label}: unknown input")
    for i, call in enumerate(program.ops):
        if not set(call.inputs) <= available:
            raise InvalidInput(f"{label}.ops[{i}]: input not available")
        if (
            not call.outputs
            or not set(call.outputs) <= program.tensors.keys()
            or set(call.outputs) & available
        ):
            raise InvalidInput(f"{label}.ops[{i}]: invalid or overwritten output")
        available.update(call.outputs)
    if available != set(program.tensors) or not set(program.outputs) <= available:
        raise InvalidInput(
            f"{label}: declared tensors are not completely produced or outputs are missing"
        )
