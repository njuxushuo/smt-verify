"""Inspect symbolic relation constraints for one Stage JSON file."""

from __future__ import annotations

import sys
from pathlib import Path

import z3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.relation_encoder import _encode_one_relation, encode_stage_relations
from src.relations import RelationEncodingError
from src.shape_model import ShapeReductionError
from src.shape_reducer import reduce_shapes
from src.stage_loader import StageInputError, load_stage
from src.symbolic_executor import execute_stage
from src.symbolic_tensor import SymbolicExecutionError


def _relation_label(relation_type: str, dim: int | None, reduce_op: str | None) -> str:
    labels = {
        "replicate": "replicate",
        "shard": f"shard(dim={dim})",
        "partial": f"partial({reduce_op})",
    }
    return labels[relation_type]


def _print_constraints(constraints: tuple[z3.BoolRef, ...]) -> None:
    print(f"    constraints: {len(constraints)}")
    for index, constraint in enumerate(constraints):
        print(f"    [{index}] {z3.simplify(constraint)}")


def _print_success(stage_path: str) -> None:
    stage = load_stage(stage_path)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)

    print(f"Stage: {stage.name}")
    print()
    print("Relation encoding:")
    print()
    print("Input relations:")
    offset = 0
    for index, relation in enumerate(stage.input_relations):
        display_constraints = tuple(
            _encode_one_relation(stage, relation, symbolic, f"input_relations[{index}]")
        )
        constraints = encoded.input_constraints[offset : offset + len(display_constraints)]
        offset += len(display_constraints)
        print(
            f"  [{index}] {relation.single_tensor}: "
            f"{_relation_label(relation.type, relation.dim, relation.reduce_op)}"
        )
        _print_constraints(constraints)
        print()

    relation = stage.output_relation
    print("Output relation:")
    print(
        f"  {relation.single_tensor}: "
        f"{_relation_label(relation.type, relation.dim, relation.reduce_op)}"
    )
    _print_constraints(encoded.output_constraints)
    print()
    print(f"Total input constraints: {len(encoded.input_constraints)}")
    print(f"Total output constraints: {len(encoded.output_constraints)}")
    print("Relation encoding: PASS")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("Relation encoding: FAIL")
        print("Error: expected exactly one Stage JSON path")
        return 1
    try:
        _print_success(arguments[0])
    except (StageInputError, ShapeReductionError, SymbolicExecutionError, RelationEncodingError) as exc:
        print("Relation encoding: FAIL")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
