"""Operator and distributed tensor relation semantics registries."""

from importlib import import_module

_EXPORTS = {
    "get_operator": (".operators", "get_operator"),
    "get_relation": (".relations", "get_relation"),
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
