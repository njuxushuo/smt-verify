"""Run Stage 2 symbolic shape reduction for one Stage JSON file."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shape_constraints import ShapeReductionError
from src.shape_reducer import reduce_shapes
from src.stage_loader import StageInputError, load_stage


def _print_program_shapes(label: str, shapes: dict[str, tuple[int, ...]]) -> None:
    print(f"{label}:")
    for name, shape in shapes.items():
        print(f"  {name}: {list(shape)}")


def _print_success(stage_path: str) -> None:
    stage = load_stage(stage_path)
    result = reduce_shapes(stage)
    print(f"Stage: {stage.name}")
    print()
    print("Original shapes:")
    _print_program_shapes("Single", {name: tensor.shape for name, tensor in stage.single.tensors.items()})
    for rank, program in stage.distributed.items():
        _print_program_shapes(
            f"Rank {rank}", {name: tensor.shape for name, tensor in program.tensors.items()}
        )
    print()
    print("Reduced shapes:")
    _print_program_shapes("Single", result.single)
    for rank, shapes in result.distributed.items():
        _print_program_shapes(f"Rank {rank}", shapes)
    print()
    print(f"Objective value: {result.objective_value}")
    print("Shape reduction: PASS")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("Shape reduction: FAIL")
        print("Error: expected exactly one Stage JSON path")
        return 1
    try:
        _print_success(arguments[0])
    except (StageInputError, ShapeReductionError) as exc:
        print("Shape reduction: FAIL")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
