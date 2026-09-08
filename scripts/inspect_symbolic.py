"""Inspect Program-level symbolic execution for one Stage JSON file."""

from __future__ import annotations

import sys
from pathlib import Path

import z3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shape_model import ShapeReductionError
from src.shape_reducer import reduce_shapes
from src.stage_loader import StageInputError, load_stage
from src.stage_model import ProgramSpec, StageSpec
from src.symbolic_executor import execute_stage
from src.symbolic_tensor import SymbolicProgramResult, SymbolicExecutionError


def _print_program_summary(
    label: str,
    program: ProgramSpec,
    result: SymbolicProgramResult,
) -> None:
    print(f"{label}:")
    print("  Inputs:")
    for name in result.input_names:
        tensor = result.tensors[name]
        print(f"    {name}: shape={list(tensor.shape)}, symbolic_values={tensor.numel}")
    print("  Produced:")
    for name in program.tensors:
        if name not in result.input_names:
            tensor = result.tensors[name]
            print(f"    {name}: shape={list(tensor.shape)}, expressions={tensor.numel}")


def _format_index(shape: tuple[int, ...]) -> str:
    return "[" + ",".join("0" for _ in shape) + "]"


def _print_output_expression(label: str, name: str, result: SymbolicProgramResult) -> None:
    tensor = result.tensors[name]
    index = tuple(0 for _ in tensor.shape)
    expression = z3.simplify(tensor.at(index))
    print(f"  {label}.{name}{_format_index(tensor.shape)}: {expression}")


def _print_success(stage_path: str) -> None:
    stage: StageSpec = load_stage(stage_path)
    reduced = reduce_shapes(stage)
    result = execute_stage(stage, reduced)

    print(f"Stage: {stage.name}")
    print()
    print("Symbolic execution:")
    print()
    _print_program_summary("Single", stage.single, result.single)
    for rank, program in stage.distributed.items():
        print()
        _print_program_summary(f"Rank {rank}", program, result.distributed[rank])

    print()
    print("Output expressions:")
    _print_output_expression("Single", stage.output_relation.single_tensor, result.single)
    for rank, local_name in enumerate(stage.output_relation.distributed_tensors):
        _print_output_expression(f"Rank {rank}", local_name, result.distributed[rank])
    print()
    print("Symbolic execution: PASS")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("Symbolic execution: FAIL")
        print("Error: expected exactly one Stage JSON path")
        return 1
    try:
        _print_success(arguments[0])
    except (StageInputError, ShapeReductionError, SymbolicExecutionError) as exc:
        print("Symbolic execution: FAIL")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
