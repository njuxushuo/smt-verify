"""Print the fixed Stage 1 inspection summary for a Stage JSON file."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage_loader import StageInputError, load_stage
from src.stage_model import RelationSpec


def _format_relation(relation: RelationSpec) -> str:
    if relation.type == "replicate":
        return "Replicate"
    if relation.type == "shard":
        return f"Shard(dim={relation.dim})"
    return f"Partial({relation.reduce_op})"


def _print_summary(stage_path: str) -> None:
    stage = load_stage(stage_path)
    distributed_tensors = sum(len(program.tensors) for program in stage.distributed.values())
    distributed_ops = sum(len(program.ops) for program in stage.distributed.values())
    print(f"Stage: {stage.name}")
    print(f"World size: {stage.world_size}")
    print(f"Single tensors: {len(stage.single.tensors)}")
    print(f"Single ops: {len(stage.single.ops)}")
    print(f"Distributed ranks: {len(stage.distributed)}")
    print(f"Distributed tensors: {distributed_tensors}")
    print(f"Distributed ops: {distributed_ops}")
    print("Input relations:")
    for relation in stage.input_relations:
        print(f"  {relation.single_tensor} -> {_format_relation(relation)}")
    print("Output relation:")
    print(f"  {stage.output_relation.single_tensor} -> {_format_relation(stage.output_relation)}")
    print("Validation: PASS")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("Validation: FAIL")
        print("Error: expected exactly one Stage JSON path")
        return 1
    try:
        _print_summary(arguments[0])
    except StageInputError as exc:
        print("Validation: FAIL")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
