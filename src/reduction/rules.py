from ..core.errors import Unsupported


def check_budget(
    stage, max_contraction, max_symbolic_ops, *, registry=None, contraction_terms=None
):
    operations = 0
    for program in [stage.single, *stage.distributed.values()]:
        operations += len(program.ops)
        for call in program.ops:
            axes = (
                getattr(registry.get(call.type), "contraction_axes", ())
                if registry is not None
                else ((0, -1),)
                if call.type == "matmul"
                else ()
            )
            for input_number, axis in axes:
                length = program.tensors[call.inputs[input_number]].shape[axis]
                terms = contraction_terms(length) if contraction_terms else length
                if terms > max_contraction:
                    raise Unsupported("contraction expansion exceeds configured budget")
    if operations > max_symbolic_ops:
        raise Unsupported("stage exceeds symbolic operation budget")
