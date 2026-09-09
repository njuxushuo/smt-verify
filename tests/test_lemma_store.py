from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lemma import certify_stage, lemma_to_dict
from src.lemma_store import LemmaStoreError, save_lemma
from src.stage_loader import load_stage
from src.verifier import VerificationStatus


STANDARD_FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"


def _proved_lemma():
    result = certify_stage(load_stage(STANDARD_FIXTURE))
    assert result.verification.status is VerificationStatus.PROVED
    assert result.lemma is not None
    return result.lemma


def test_save_creates_directory_and_writes_parseable_canonical_json(tmp_path: Path) -> None:
    lemma = _proved_lemma()
    directory = tmp_path / "nested" / "lemmas"

    path = save_lemma(lemma, directory)

    assert directory.is_dir()
    assert path == directory / f"{lemma.lemma_id}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == lemma_to_dict(lemma)


def test_save_is_idempotent_for_identical_lemma(tmp_path: Path) -> None:
    lemma = _proved_lemma()

    first = save_lemma(lemma, tmp_path)
    second = save_lemma(lemma, tmp_path)

    assert first == second
    assert list(tmp_path.glob("*.json")) == [first]


def test_save_rejects_different_content_with_the_same_identity(tmp_path: Path) -> None:
    lemma = _proved_lemma()
    save_lemma(lemma, tmp_path)
    conflicting = replace(lemma, source_stage_name="different_source_name")
    assert conflicting.lemma_id == lemma.lemma_id

    with pytest.raises(LemmaStoreError, match="conflicting content"):
        save_lemma(conflicting, tmp_path)
