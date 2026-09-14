"""Operator and distributed tensor relation semantics registries."""

from importlib import import_module

_EXPORTS = {
    "BroadcastShapeError": (".broadcast", "BroadcastShapeError"),
    "broadcast_index": (".broadcast", "broadcast_index"),
    "broadcast_shape_constraints": (".broadcast", "broadcast_shape_constraints"),
    "get_operator": (".operators", "get_operator"),
    "get_relation": (".relations", "get_relation"),
    "infer_broadcast_shape": (".broadcast", "infer_broadcast_shape"),
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
