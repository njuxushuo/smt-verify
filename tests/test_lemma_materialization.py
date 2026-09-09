from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lemma import (
    LemmaMaterializationError,
    build_certified_lemma,
    certify_stage,
    lemma_id_for_stage,
)
from src.stage_loader import load_stage
from src.stage_model import RelationSpec
from src.verifier import VerificationStatus


STANDARD_FIXTURE = ROOT / "input" / "matmul_shard_to_partial.json"
WRONG_FIXTURE = ROOT / "input" / "matmul_shard_wrong_replicate.json"


def _proved_stage_and_verification():
    stage = load_stage(STANDARD_FIXTURE)
    result = certify_stage(stage)
    assert result.verification.status is VerificationStatus.PROVED
    assert result.lemma is not None
    return stage, result.verification, result.lemma


def test_proved_stage_materializes_complete_concrete_lemma() -> None:
    stage, verification, lemma = _proved_stage_and_verification()

    assert lemma.source_stage_name == stage.name
    assert lemma.world_size == stage.world_size
    assert lemma.input_relations == stage.input_relations
    assert lemma.single_ops == stage.single.ops
    assert lemma.distributed_ops == {rank: program.ops for rank, program in stage.distributed.items()}
    assert lemma.output_relation == stage.output_relation
    assert lemma.original_single_shapes == {
        name: tensor.shape for name, tensor in stage.single.tensors.items()
    }
    assert lemma.original_distributed_shapes == {
        rank: {name: tensor.shape for name, tensor in program.tensors.items()}
        for rank, program in stage.distributed.items()
    }
    assert lemma.reduced_shapes == verification.reduced_shapes


def test_certify_stage_keeps_disproved_counterexample_and_emits_no_lemma() -> None:
    result = certify_stage(load_stage(WRONG_FIXTURE))

    assert result.verification.status is VerificationStatus.DISPROVED
    assert result.verification.counterexample is not None
    assert result.lemma is None


@pytest.mark.parametrize("status", [VerificationStatus.DISPROVED, VerificationStatus.UNKNOWN])
def test_only_proved_results_can_materialize(status: VerificationStatus) -> None:
    stage, verification, _ = _proved_stage_and_verification()
    invalid = replace(
        verification,
        status=status,
        reason_unknown="test unknown" if status is VerificationStatus.UNKNOWN else None,
    )

    with pytest.raises(LemmaMaterializationError, match="PROVED"):
        build_certified_lemma(stage, invalid)


def test_materialization_rejects_mismatched_stage_name_and_inconsistent_proved_result() -> None:
    stage, verification, _ = _proved_stage_and_verification()

    with pytest.raises(LemmaMaterializationError, match="does not match"):
        build_certified_lemma(stage, replace(verification, stage_name="other_stage"))
    with pytest.raises(LemmaMaterializationError, match="counterexample"):
        build_certified_lemma(stage, replace(verification, counterexample=object()))  # type: ignore[arg-type]
    with pytest.raises(LemmaMaterializationError, match="unknown reason"):
        build_certified_lemma(stage, replace(verification, reason_unknown="inconsistent"))


def test_same_semantic_stage_has_stable_id_without_source_stage_name() -> None:
    stage, verification, lemma = _proved_stage_and_verification()
    renamed_stage = replace(stage, name="renamed_source")
    renamed_verification = replace(
        verification,
        stage_name=renamed_stage.name,
        reduced_shapes=replace(verification.reduced_shapes, stage_name=renamed_stage.name),
    )

    renamed_lemma = build_certified_lemma(renamed_stage, renamed_verification)
    assert renamed_lemma.lemma_id == lemma.lemma_id
    assert renamed_lemma.source_stage_name == "renamed_source"


def test_changed_output_relation_changes_semantic_identity_without_reverification() -> None:
    stage, _, _ = _proved_stage_and_verification()
    changed = replace(
        stage,
        output_relation=RelationSpec("C", ("C0", "C1"), "replicate"),
    )

    assert lemma_id_for_stage(changed) != lemma_id_for_stage(stage)
