"""Filesystem-only persistence for concrete certified lemmas."""

from __future__ import annotations

from pathlib import Path

from .lemma import CertifiedLemma, canonical_json_bytes, lemma_to_dict


class LemmaStoreError(ValueError):
    """Raised when filesystem persistence cannot preserve lemma identity safely."""


def save_lemma(
    lemma: CertifiedLemma,
    directory: str | Path,
) -> Path:
    """Write one canonical lemma JSON or validate an identical existing artifact."""

    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    path = target_directory / f"{lemma.lemma_id}.json"
    content = canonical_json_bytes(lemma_to_dict(lemma))

    if path.exists():
        if not path.is_file():
            raise LemmaStoreError(f"lemma path is not a regular file: {path}")
        if path.read_bytes() != content:
            raise LemmaStoreError(f"conflicting content already exists for lemma {lemma.lemma_id}")
        return path

    try:
        path.write_bytes(content)
    except OSError as exc:
        raise LemmaStoreError(f"cannot save lemma {lemma.lemma_id}: {exc}") from exc
    return path
