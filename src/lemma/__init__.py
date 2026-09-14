"""Certified lemma materialization and persistence."""

from importlib import import_module

_EXPORTS = {
    "CertificationResult": (".model", "CertificationResult"),
    "CertifiedLemma": (".model", "CertifiedLemma"),
    "LemmaMaterializationError": (".model", "LemmaMaterializationError"),
    "LemmaStoreError": (".store", "LemmaStoreError"),
    "build_certified_lemma": (".model", "build_certified_lemma"),
    "canonical_json_bytes": (".model", "canonical_json_bytes"),
    "certify_stage": (".model", "certify_stage"),
    "lemma_id_for_stage": (".model", "lemma_id_for_stage"),
    "lemma_to_dict": (".model", "lemma_to_dict"),
    "save_lemma": (".store", "save_lemma"),
    "semantic_identity_payload": (".model", "semantic_identity_payload"),
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
