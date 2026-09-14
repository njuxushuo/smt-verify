"""Symbolic tensors and Program/Stage execution."""

from importlib import import_module

_EXPORTS = {
    "SymbolicExecutionError": (".tensor", "SymbolicExecutionError"),
    "SymbolicProgramResult": (".tensor", "SymbolicProgramResult"),
    "SymbolicStageResult": (".tensor", "SymbolicStageResult"),
    "SymbolicTensor": (".tensor", "SymbolicTensor"),
    "create_symbolic_input_tensor": (".tensor", "create_symbolic_input_tensor"),
    "execute_program": (".executor", "execute_program"),
    "execute_stage": (".executor", "execute_stage"),
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
