"""Z3 Optimize-based concretization of relation-aware symbolic shapes."""

from __future__ import annotations

import z3

from .shape_constraints import build_shape_constraints, create_symbolic_shapes
from .shape_model import ReducedShapeResult, ShapeReductionError, TensorRef
from .stage_model import StageSpec


def _simplify_and_deduplicate(constraints: list[z3.BoolRef]) -> list[z3.BoolRef]:
    unique: dict[str, z3.BoolRef] = {}
    for constraint in constraints:
        simplified = z3.simplify(constraint)
        unique.setdefault(simplified.sexpr(), simplified)
    return list(unique.values())


def _concretize_shape(
    model: z3.ModelRef, dimensions: tuple[z3.ArithRef, ...]
) -> tuple[int, ...]:
    return tuple(model.eval(dimension, model_completion=True).as_long() for dimension in dimensions)


def reduce_shapes(stage: StageSpec) -> ReducedShapeResult:
    """Find minimum legal concrete shapes without modifying ``stage``."""

    shapes = create_symbolic_shapes(stage)
    constraints = _simplify_and_deduplicate(build_shape_constraints(stage, shapes))
    dimensions = [dimension for shape in shapes.shapes.values() for dimension in shape]

    optimizer = z3.Optimize()
    optimizer.add(*constraints)
    objective = z3.Sum(dimensions)
    optimizer.minimize(objective)
    result = optimizer.check()
    if result == z3.unsat:
        raise ShapeReductionError(f"shape constraints are unsatisfiable for stage {stage.name!r}")
    if result != z3.sat:
        raise ShapeReductionError(f"shape reduction returned unknown for stage {stage.name!r}")

    model = optimizer.model()
    single = {
        name: _concretize_shape(model, shapes.shapes[TensorRef("single", None, name)])
        for name in stage.single.tensors
    }
    distributed = {
        rank: {
            name: _concretize_shape(model, shapes.shapes[TensorRef("distributed", rank, name)])
            for name in program.tensors
        }
        for rank, program in stage.distributed.items()
    }
    objective_value = model.eval(objective, model_completion=True).as_long()
    return ReducedShapeResult(
        stage_name=stage.name,
        single=single,
        distributed=distributed,
        objective_value=objective_value,
    )
