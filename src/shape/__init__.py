"""Symbolic shape models, constraints, and reduction."""

from importlib import import_module

_EXPORTS = {
    "ReducedShapeResult": (".model", "ReducedShapeResult"),
    "ShapeReductionError": (".model", "ShapeReductionError"),
    "SymbolicShapeBook": (".model", "SymbolicShapeBook"),
    "TensorRef": (".model", "TensorRef"),
    "UnsupportedShapeSemanticsError": (".model", "UnsupportedShapeSemanticsError"),
    "build_shape_constraints": (".constraints", "build_shape_constraints"),
    "create_symbolic_shapes": (".constraints", "create_symbolic_shapes"),
    "reduce_shapes": (".reducer", "reduce_shapes"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
