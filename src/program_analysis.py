"""Static producer and dataflow analysis for one program."""

from __future__ import annotations

from .stage_model import ProgramSpec


class ProgramAnalysisError(ValueError):
    """Raised when a program violates static single-assignment dataflow."""


def _producer_counts(program: ProgramSpec) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in program.ops:
        for output_name in op.outputs:
            if output_name not in program.tensors:
                raise ProgramAnalysisError(f"unknown output tensor {output_name!r}")
            counts[output_name] = counts.get(output_name, 0) + 1
            if counts[output_name] > 1:
                raise ProgramAnalysisError(f"tensor {output_name!r} has multiple producers")
    return counts


def find_program_inputs(program: ProgramSpec) -> tuple[str, ...]:
    """Return declared tensors without a producer, preserving declaration order."""

    producer_counts = _producer_counts(program)
    return tuple(name for name in program.tensors if name not in producer_counts)


def validate_program_dataflow(program: ProgramSpec, context: str) -> None:
    """Validate ordered single-assignment dataflow without symbolic execution."""

    input_names = find_program_inputs(program)
    available = set(input_names)
    for index, op in enumerate(program.ops):
        op_context = f"{context}.ops[{index}]"
        for input_name in op.inputs:
            if input_name not in available:
                raise ProgramAnalysisError(
                    f"{op_context}: input tensor {input_name!r} is not available or has not been produced"
                )
        for output_name in op.outputs:
            if output_name in available:
                raise ProgramAnalysisError(
                    f"{op_context}: output tensor {output_name!r} would overwrite an existing tensor"
                )
            available.add(output_name)

    if available != set(program.tensors):
        missing = sorted(set(program.tensors) - available)
        raise ProgramAnalysisError(
            f"{context}: declared tensors are neither inputs nor produced: {missing}"
        )
