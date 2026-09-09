"""Program and Stage symbolic-execution orchestration."""

from __future__ import annotations

from .operators import get_operator
from .program_analysis import ProgramAnalysisError, find_program_inputs, validate_program_dataflow
from .shape_model import ReducedShapeResult
from .stage_model import ProgramSpec, StageSpec
from .symbolic_tensor import (
    SymbolicExecutionError,
    SymbolicProgramResult,
    SymbolicStageResult,
    SymbolicTensor,
    create_symbolic_input_tensor,
)


def _validate_reduced_shapes(
    program: ProgramSpec,
    reduced_shapes: dict[str, tuple[int, ...]],
    context: str,
) -> None:
    """Ensure concretized shapes exactly cover and agree with one program."""

    if set(reduced_shapes) != set(program.tensors):
        raise SymbolicExecutionError(
            f"{context}: reduced shape tensors must exactly match declared program tensors"
        )

    for name, tensor in program.tensors.items():
        reduced_shape = reduced_shapes[name]
        if len(reduced_shape) != len(tensor.shape):
            raise SymbolicExecutionError(
                f"{context}: reduced shape for tensor {name!r} has rank {len(reduced_shape)}, "
                f"expected {len(tensor.shape)}"
            )
        for dimension, extent in enumerate(reduced_shape):
            if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
                raise SymbolicExecutionError(
                    f"{context}: reduced shape for tensor {name!r} has non-positive "
                    f"dimension at index {dimension}"
                )


def execute_program(
    program: ProgramSpec,
    reduced_shapes: dict[str, tuple[int, ...]],
    scope_prefix: str,
    context: str,
) -> SymbolicProgramResult:
    """Execute a program in declared operation order over symbolic tensors."""

    _validate_reduced_shapes(program, reduced_shapes, context)
    try:
        validate_program_dataflow(program, context)
        input_names = find_program_inputs(program)
    except ProgramAnalysisError as exc:
        raise SymbolicExecutionError(str(exc)) from exc
    environment: dict[str, SymbolicTensor] = {
        name: create_symbolic_input_tensor(name, reduced_shapes[name], scope_prefix)
        for name in input_names
    }

    for index, op in enumerate(program.ops):
        op_context = f"{context}.ops[{index}]"
        semantics = get_operator(op.type, op_context)
        input_tensors = tuple(environment[name] for name in op.inputs)
        output_shapes = tuple(reduced_shapes[name] for name in op.outputs)
        outputs = semantics.symbolic_execute(input_tensors, output_shapes, op_context)
        if len(outputs) != len(op.outputs):
            raise SymbolicExecutionError(
                f"{op_context}: operator returned {len(outputs)} outputs, expected {len(op.outputs)}"
            )
        for output_name, output_tensor in zip(op.outputs, outputs):
            environment[output_name] = output_tensor

    if set(environment) != set(program.tensors):
        raise SymbolicExecutionError(
            f"{context}: symbolic execution did not produce every declared tensor"
        )
    return SymbolicProgramResult(input_names=input_names, tensors=environment)


def execute_stage(stage: StageSpec, reduced: ReducedShapeResult) -> SymbolicStageResult:
    """Execute the single program and each distributed rank with reduced shapes."""

    if reduced.stage_name != stage.name:
        raise SymbolicExecutionError(
            f"reduced result is for stage {reduced.stage_name!r}, expected {stage.name!r}"
        )
    if set(reduced.distributed) != set(stage.distributed):
        raise SymbolicExecutionError(
            "reduced distributed rank set must exactly match the stage rank set"
        )

    single = execute_program(stage.single, reduced.single, scope_prefix="single", context="single")
    distributed = {
        rank: execute_program(
            program,
            reduced.distributed[rank],
            scope_prefix=f"rank{rank}",
            context=f"distributed.ranks[{rank}]",
        )
        for rank, program in stage.distributed.items()
    }
    return SymbolicStageResult(stage_name=stage.name, single=single, distributed=distributed)
