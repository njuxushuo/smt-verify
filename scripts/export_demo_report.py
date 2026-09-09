"""Export the three current Stage examples as one real Stage 1–6 demo report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import z3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lemma import certify_stage, lemma_to_dict
from src.relation_encoder import encode_stage_relations
from src.shape_reducer import reduce_shapes
from src.stage_loader import load_stage
from src.symbolic_executor import execute_stage
from src.verifier import VerificationStatus, verify_stage


DEFAULT_INPUTS = (
    PROJECT_ROOT / "input" / "matmul_shard_to_partial.json",
    PROJECT_ROOT / "input" / "matmul_shard_to_partial_large.json",
    PROJECT_ROOT / "input" / "matmul_shard_wrong_replicate.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "demo_report.txt"
DIVIDER = "=" * 60


def _relation_label(relation) -> str:
    if relation.type == "shard":
        return f"Shard(dim={relation.dim})"
    if relation.type == "partial":
        return f"Partial({relation.reduce_op})"
    return relation.type.capitalize()


def _operation_label(op) -> str:
    return f"{op.type}({', '.join(op.inputs)}) -> {', '.join(op.outputs)}"


def _append_shapes(lines: list[str], single, distributed) -> None:
    lines.append("Single:")
    for name, shape in single.items():
        lines.append(f"  {name}: {list(shape)}")
    for rank, tensors in sorted(distributed.items()):
        lines.append("")
        lines.append(f"Rank {rank}:")
        for name, shape in tensors.items():
            lines.append(f"  {name}: {list(shape)}")


def _append_symbolic_outputs(lines: list[str], label: str, program) -> None:
    lines.append(f"{label} outputs:")
    output_names = [name for name in program.tensors if name not in program.input_names]
    if not output_names:
        lines.append("  None")
        return
    for name in output_names:
        tensor = program.tensors[name]
        for index, value in enumerate(tensor.values):
            lines.append(f"  {name}[{index}] = {value}")


def _smt_result(status: VerificationStatus) -> str:
    return {
        VerificationStatus.PROVED: "UNSAT",
        VerificationStatus.DISPROVED: "SAT",
        VerificationStatus.UNKNOWN: "UNKNOWN",
    }[status]


def _append_counterexample(lines: list[str], verification) -> None:
    lines.append("[7] Counterexample")
    if verification.counterexample is None:
        lines.append("Counterexample: None")
        return
    counterexample = verification.counterexample
    lines.append("Single inputs:")
    for name, values in counterexample.single_inputs.items():
        lines.append(f"  {name} = {values}")
    lines.append("Distributed inputs:")
    for rank, inputs in sorted(counterexample.distributed_inputs.items()):
        lines.append(f"  Rank {rank}:")
        for name, values in inputs.items():
            lines.append(f"    {name} = {values}")
    lines.append("Failed output constraints:")
    lines.append(f"  {counterexample.failed_output_constraints}")


def _append_example(lines: list[str], number: int, input_path: Path) -> tuple[str, VerificationStatus, str | None]:
    stage = load_stage(input_path)
    reduced = reduce_shapes(stage)
    symbolic = execute_stage(stage, reduced)
    encoded = encode_stage_relations(stage, symbolic)
    verification = verify_stage(stage)
    certification = certify_stage(stage)

    lines.extend((DIVIDER, f"Example {number}: {stage.name}", DIVIDER, ""))
    lines.append("[1] Input Stage Summary")
    lines.append(f"Stage name: {stage.name}")
    lines.append(f"World size: {stage.world_size}")
    lines.append("Input relations:")
    for relation in stage.input_relations:
        lines.append(f"  {relation.single_tensor} -> {_relation_label(relation)}")
    lines.append("Single ops:")
    for op in stage.single.ops:
        lines.append(f"  {_operation_label(op)}")
    lines.append("Distributed ops:")
    for rank, program in sorted(stage.distributed.items()):
        lines.append(f"  Rank {rank}:")
        for op in program.ops:
            lines.append(f"    {_operation_label(op)}")
    lines.append(f"Output relation: {stage.output_relation.single_tensor} -> "
                 f"{_relation_label(stage.output_relation)}")
    lines.append("")

    lines.append("[2] Original Shapes")
    _append_shapes(
        lines,
        {name: tensor.shape for name, tensor in stage.single.tensors.items()},
        {
            rank: {name: tensor.shape for name, tensor in program.tensors.items()}
            for rank, program in stage.distributed.items()
        },
    )
    lines.append("")

    lines.append("[3] Reduced Shapes")
    _append_shapes(lines, reduced.single, reduced.distributed)
    lines.append(f"Reduced objective: {reduced.objective_value}")
    lines.append("")

    lines.append("[4] Symbolic Execution")
    _append_symbolic_outputs(lines, "Single", symbolic.single)
    for rank, program in sorted(symbolic.distributed.items()):
        _append_symbolic_outputs(lines, f"Rank {rank}", program)
    lines.append("")

    lines.append("[5] Relation Encoding")
    lines.append(f"Input constraints: {len(encoded.input_constraints)}")
    for index, constraint in enumerate(encoded.input_constraints):
        lines.append(f"  [{index}] {z3.simplify(constraint)}")
    lines.append(f"Output constraints: {len(encoded.output_constraints)}")
    for index, constraint in enumerate(encoded.output_constraints):
        lines.append(f"  [{index}] {z3.simplify(constraint)}")
    lines.append("")

    lines.append("[6] SMT Verification")
    lines.append("SMT query:")
    lines.append("  Rin AND NOT(AND(Rout))")
    lines.append(f"SMT result: {_smt_result(verification.status)}")
    lines.append(f"Verification: {verification.status.value}")
    lines.append(f"Input constraint count: {verification.input_constraint_count}")
    lines.append(f"Output constraint count: {verification.output_constraint_count}")
    lines.append(f"Reduced objective: {verification.reduced_shapes.objective_value}")
    if verification.reason_unknown is not None:
        lines.append(f"Reason unknown: {verification.reason_unknown}")
    lines.append("")

    _append_counterexample(lines, verification)
    lines.append("")

    lines.append("[8] Certified Lemma")
    if certification.lemma is None:
        lines.append("Lemma: NOT GENERATED")
        lemma_id = None
    else:
        lines.append("Lemma: GENERATED")
        lines.append(f"Lemma ID: {certification.lemma.lemma_id}")
        lines.append(
            json.dumps(
                lemma_to_dict(certification.lemma),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        lemma_id = certification.lemma.lemma_id
    lines.extend(("", ""))
    return stage.name, verification.status, lemma_id


def build_demo_report(input_paths: tuple[Path, ...] = DEFAULT_INPUTS) -> str:
    """Collect real results from every existing pipeline API into display text."""

    lines = [
        "SMT-Verify Demo Report",
        "Each example follows:",
        "Input -> Shape Reduction -> Symbolic Execution",
        "-> Relation Encoding -> SMT Verification -> Certified Lemma",
        "",
    ]
    summary = [_append_example(lines, number, path) for number, path in enumerate(input_paths, 1)]
    lines.extend((DIVIDER, "Summary", DIVIDER, ""))
    for name, status, lemma_id in summary:
        lines.append(name)
        lines.append(f"  Verification: {status.value}")
        lines.append(f"  Lemma: {'GENERATED' if lemma_id is not None else 'NOT GENERATED'}")
        if lemma_id is not None:
            lines.append(f"  Lemma ID: {lemma_id}")
        lines.append("")
    return "\n".join(lines)


def export_demo_report(
    output: str | Path = DEFAULT_OUTPUT,
    input_paths: tuple[Path, ...] = DEFAULT_INPUTS,
) -> Path:
    """Write the UTF-8 report using only the existing Stage 1–6 APIs."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_demo_report(input_paths), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="UTF-8 report destination")
    arguments = parser.parse_args(argv)
    output_path = export_demo_report(arguments.output)
    print(f"Demo report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
