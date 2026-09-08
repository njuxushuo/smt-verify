"""Symbolic shape creation and Stage 2 shape constraints."""

from __future__ import annotations

import z3

from .shape_model import SymbolicShapeBook, TensorRef
from .stage_model import ProgramSpec, RelationSpec, StageSpec


class ShapeReductionError(ValueError):
    pass


class UnsupportedShapeSemanticsError(ShapeReductionError):
    pass


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


def _same_shape_constraints(
    left: tuple[z3.ArithRef, ...], right: tuple[z3.ArithRef, ...], context: str
) -> list[z3.BoolRef]:
    if len(left) != len(right):
        raise ShapeReductionError(f"{context}: tensors must have the same rank")
    return [left[index] == right[index] for index in range(len(left))]


def _relation_constraints(
    stage: StageSpec,
    relation: RelationSpec,
    shapes: SymbolicShapeBook,
    context: str,
) -> list[z3.BoolRef]:
    """Build only the symbolic shape equalities induced by one relation."""

    single_shape = shapes.shapes[_single_ref(relation.single_tensor)]
    constraints: list[z3.BoolRef] = []
    for rank, local_name in enumerate(relation.distributed_tensors):
        local_shape = shapes.shapes[_distributed_ref(rank, local_name)]
        if relation.type in {"replicate", "partial"}:
            constraints.extend(_same_shape_constraints(single_shape, local_shape, context))
        elif relation.type == "shard":
            if len(single_shape) != len(local_shape):
                raise ShapeReductionError(f"{context}: shard tensors must have the same rank")
            for index, (single_dim, local_dim) in enumerate(zip(single_shape, local_shape)):
                if index == relation.dim:
                    constraints.append(stage.world_size * local_dim == single_dim)
                else:
                    constraints.append(local_dim == single_dim)
        else:
            raise ShapeReductionError(f"{context}: unsupported relation {relation.type!r}")
    return constraints


def _operator_constraints(
    program: ProgramSpec,
    tensor_refs: dict[str, TensorRef],
    shapes: SymbolicShapeBook,
    context: str,
) -> list[z3.BoolRef]:
    """Build shape constraints for the Stage 2 operator subset."""

    constraints: list[z3.BoolRef] = []
    for index, op in enumerate(program.ops):
        op_context = f"{context}.ops[{index}]"
        if op.type not in {"matmul", "add", "mul"}:
            raise UnsupportedShapeSemanticsError(
                f"{op_context}: shape semantics for operator {op.type!r} are not supported"
            )
        if len(op.inputs) != 2 or len(op.outputs) != 1:
            raise ShapeReductionError(
                f"{op_context}: {op.type} requires exactly two inputs and one output"
            )
        left = shapes.shapes[tensor_refs[op.inputs[0]]]
        right = shapes.shapes[tensor_refs[op.inputs[1]]]
        output = shapes.shapes[tensor_refs[op.outputs[0]]]

        if op.type == "matmul":
            if len(left) != 2 or len(right) != 2 or len(output) != 2:
                raise ShapeReductionError(f"{op_context}: matmul currently supports only 2-D tensors")
            constraints.extend([left[1] == right[0], output[0] == left[0], output[1] == right[1]])
        else:
            constraints.extend(_same_shape_constraints(left, right, op_context))
            constraints.extend(_same_shape_constraints(left, output, op_context))
    return constraints


def _shard_divisibility_constraints(
    stage: StageSpec, relation: RelationSpec, shapes: SymbolicShapeBook
) -> list[z3.BoolRef]:
    if relation.type != "shard":
        return []
    single_shape = shapes.shapes[_single_ref(relation.single_tensor)]
    return [single_shape[relation.dim] % stage.world_size == 0]


def build_shape_constraints(
    stage: StageSpec,
    shapes: SymbolicShapeBook,
) -> list[z3.BoolRef]:
    """Build validity, relation, operator, and shard-divisibility constraints."""

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

    # 6. Explicit shard divisibility, after all shape equalities.
    for relation in (*stage.input_relations, stage.output_relation):
        constraints.extend(_shard_divisibility_constraints(stage, relation, shapes))

    return constraints
