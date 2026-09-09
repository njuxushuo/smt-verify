"""Symbolic shape creation and Stage 2 shape constraints."""

from __future__ import annotations

import z3

from .operators import get_operator
from .relations import get_relation
from .shape_model import SymbolicShapeBook, TensorRef
from .stage_model import ProgramSpec, RelationSpec, StageSpec


def _single_ref(name: str) -> TensorRef:
    return TensorRef(scope="single", rank=None, name=name)


def _distributed_ref(rank: int, name: str) -> TensorRef:
    return TensorRef(scope="distributed", rank=rank, name=name)


def create_symbolic_shapes(stage: StageSpec) -> SymbolicShapeBook:
    """Create a stable Z3 integer dimension for every Stage tensor dimension."""

    shapes: dict[TensorRef, tuple[z3.ArithRef, ...]] = {}
    for name, tensor in stage.single.tensors.items():
        ref = _single_ref(name)
        shapes[ref] = tuple(z3.Int(f"single__{name}__d{index}") for index in range(len(tensor.shape)))
    for rank, program in stage.distributed.items():
        for name, tensor in program.tensors.items():
            ref = _distributed_ref(rank, name)
            shapes[ref] = tuple(
                z3.Int(f"rank{rank}__{name}__d{index}") for index in range(len(tensor.shape))
            )
    return SymbolicShapeBook(shapes=shapes)


def _relation_constraints(
    stage: StageSpec,
    relation: RelationSpec,
    shapes: SymbolicShapeBook,
    context: str,
) -> list[z3.BoolRef]:
    """Resolve one Stage relation and delegate its symbolic shape semantics."""

    single_shape = shapes.shapes[_single_ref(relation.single_tensor)]
    local_shapes = tuple(
        shapes.shapes[_distributed_ref(rank, local_name)]
        for rank, local_name in enumerate(relation.distributed_tensors)
    )
    semantics = get_relation(relation.type, context)
    return semantics.shape_constraints(
        relation, single_shape, local_shapes, stage.world_size, context
    )


def _operator_constraints(
    program: ProgramSpec,
    tensor_refs: dict[str, TensorRef],
    shapes: SymbolicShapeBook,
    context: str,
) -> list[z3.BoolRef]:
    """Resolve Stage operators and delegate their shape semantics."""

    constraints: list[z3.BoolRef] = []
    for index, op in enumerate(program.ops):
        op_context = f"{context}.ops[{index}]"
        semantics = get_operator(op.type, op_context)
        input_shapes = tuple(shapes.shapes[tensor_refs[name]] for name in op.inputs)
        output_shapes = tuple(shapes.shapes[tensor_refs[name]] for name in op.outputs)
        constraints.extend(semantics.shape_constraints(input_shapes, output_shapes, op_context))
    return constraints


def build_shape_constraints(
    stage: StageSpec,
    shapes: SymbolicShapeBook,
) -> list[z3.BoolRef]:
    """Build validity plus delegated relation and operator constraints."""

    constraints: list[z3.BoolRef] = []

    # 1. Every symbolic dimension is positive.
    for symbolic_shape in shapes.shapes.values():
        constraints.extend(dimension >= 1 for dimension in symbolic_shape)

    # 2. Known input relations.
    for index, relation in enumerate(stage.input_relations):
        constraints.extend(_relation_constraints(stage, relation, shapes, f"input_relations[{index}]"))

    # 3. Single-side operators.
    single_refs = {name: _single_ref(name) for name in stage.single.tensors}
    constraints.extend(_operator_constraints(stage.single, single_refs, shapes, "single"))

    # 4. Operators in each explicit distributed rank.
    for rank, program in stage.distributed.items():
        rank_refs = {name: _distributed_ref(rank, name) for name in program.tensors}
        constraints.extend(
            _operator_constraints(program, rank_refs, shapes, f"distributed.ranks[{rank}]")
        )

    # 5. Candidate output relation.
    constraints.extend(_relation_constraints(stage, stage.output_relation, shapes, "output_relation"))

    return constraints
