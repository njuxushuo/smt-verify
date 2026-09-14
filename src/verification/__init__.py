"""Relation encoding and end-to-end Stage verification."""

from importlib import import_module

_EXPORTS = {
    "Counterexample": (".verifier", "Counterexample"),
    "EncodedRelations": (".relation_encoder", "EncodedRelations"),
    "VerificationError": (".verifier", "VerificationError"),
    "VerificationResult": (".verifier", "VerificationResult"),
    "VerificationStatus": (".verifier", "VerificationStatus"),
    "encode_stage_relations": (".relation_encoder", "encode_stage_relations"),
    "verify_stage": (".verifier", "verify_stage"),
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
