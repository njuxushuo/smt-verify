from ..core.errors import Unsupported


def check_budget(stage, max_contraction, max_symbolic_ops):
    operations = 0
    for program in [stage.single, *stage.distributed.values()]:
        operations += len(program.ops)
        for call in program.ops:
            if (
                call.type == "matmul"
                and program.tensors[call.inputs[0]].shape[-1] > max_contraction
            ):
                raise Unsupported(
                    "original contraction exceeds exact expansion budget; shape shrinking is not assumed sound"
                )
    if operations > max_symbolic_ops:
        raise Unsupported("stage exceeds symbolic operation budget")
