"""Concrete certified Stage lemmas and Stage 6 materialization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .shape_model import ReducedShapeResult
from .stage_model import OpSpec, RelationSpec, StageSpec
from .verifier import VerificationResult, VerificationStatus, verify_stage


class LemmaMaterializationError(ValueError):
    """Raised when a verification result cannot safely become a lemma."""


@dataclass(frozen=True)
class CertifiedLemma:
    """A concrete semantic Stage artifact certified by one PROVED verification."""

    lemma_id: str
    source_stage_name: str
    world_size: int

    input_relations: tuple[RelationSpec, ...]
    single_ops: tuple[OpSpec, ...]
    distributed_ops: dict[int, tuple[OpSpec, ...]]
    output_relation: RelationSpec

    original_single_shapes: dict[str, tuple[int, ...]]
    original_distributed_shapes: dict[int, dict[str, tuple[int, ...]]]

    reduced_shapes: ReducedShapeResult


@dataclass(frozen=True)
class CertificationResult:
    """A Stage 5 result plus its PROVED-only concrete lemma, if any."""

    verification: VerificationResult
    lemma: CertifiedLemma | None


def _relation_to_dict(relation: RelationSpec) -> dict[str, object]:
    return {
        "single_tensor": relation.single_tensor,
        "distributed_tensors": list(relation.distributed_tensors),
        "type": relation.type,
        "dim": relation.dim,
        "reduce_op": relation.reduce_op,
    }


def _op_to_dict(op: OpSpec) -> dict[str, object]:
    return {
        "type": op.type,
        "inputs": list(op.inputs),
        "outputs": list(op.outputs),
    }


def _shape_map_to_dict(shapes: dict[str, tuple[int, ...]]) -> dict[str, list[int]]:
    return {name: list(shape) for name, shape in sorted(shapes.items())}


def _distributed_shapes_to_dict(
    shapes: dict[int, dict[str, tuple[int, ...]]],
) -> dict[str, dict[str, list[int]]]:
    return {
        str(rank): _shape_map_to_dict(rank_shapes)
        for rank, rank_shapes in sorted(shapes.items())
    }


def _distributed_ops_to_dict(
    distributed_ops: dict[int, tuple[OpSpec, ...]],
) -> dict[str, list[dict[str, object]]]:
    return {
        str(rank): [_op_to_dict(op) for op in ops]
        for rank, ops in sorted(distributed_ops.items())
    }


def _original_single_shapes(stage: StageSpec) -> dict[str, tuple[int, ...]]:
    return {name: tensor.shape for name, tensor in stage.single.tensors.items()}


def _original_distributed_shapes(stage: StageSpec) -> dict[int, dict[str, tuple[int, ...]]]:
    return {
        rank: {name: tensor.shape for name, tensor in program.tensors.items()}
        for rank, program in stage.distributed.items()
    }


def semantic_identity_payload(stage: StageSpec) -> dict[str, object]:
    """Return the concrete semantic content used for stable lemma identity."""

    return {
        "world_size": stage.world_size,
        "input_relations": [_relation_to_dict(relation) for relation in stage.input_relations],
        "single_ops": [_op_to_dict(op) for op in stage.single.ops],
        "distributed_ops": _distributed_ops_to_dict(
            {rank: program.ops for rank, program in stage.distributed.items()}
        ),
        "output_relation": _relation_to_dict(stage.output_relation),
        "original_single_shapes": _shape_map_to_dict(_original_single_shapes(stage)),
        "original_distributed_shapes": _distributed_shapes_to_dict(
            _original_distributed_shapes(stage)
        ),
    }


def canonical_json_bytes(payload: dict[str, object]) -> bytes:
    """Serialize JSON primitives deterministically for hashing and persistence."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def lemma_id_for_stage(stage: StageSpec) -> str:
    """Compute the stable concrete semantic identifier for one Stage."""

    digest = hashlib.sha256(canonical_json_bytes(semantic_identity_payload(stage))).hexdigest()
    return f"lemma_{digest[:16]}"


def _reduced_shapes_to_dict(reduced: ReducedShapeResult) -> dict[str, object]:
    return {
        "stage_name": reduced.stage_name,
        "single": _shape_map_to_dict(reduced.single),
        "distributed": _distributed_shapes_to_dict(reduced.distributed),
        "objective_value": reduced.objective_value,
    }


def lemma_to_dict(lemma: CertifiedLemma) -> dict[str, object]:
    """Return the complete JSON-safe structured representation of a lemma."""

    return {
        "lemma_id": lemma.lemma_id,
        "source_stage_name": lemma.source_stage_name,
        "world_size": lemma.world_size,
        "input_relations": [_relation_to_dict(relation) for relation in lemma.input_relations],
        "single_ops": [_op_to_dict(op) for op in lemma.single_ops],
        "distributed_ops": _distributed_ops_to_dict(lemma.distributed_ops),
        "output_relation": _relation_to_dict(lemma.output_relation),
        "original_single_shapes": _shape_map_to_dict(lemma.original_single_shapes),
        "original_distributed_shapes": _distributed_shapes_to_dict(
            lemma.original_distributed_shapes
        ),
        "reduced_shapes": _reduced_shapes_to_dict(lemma.reduced_shapes),
    }


def build_certified_lemma(
    stage: StageSpec,
    verification: VerificationResult,
) -> CertifiedLemma:
    """Materialize a concrete lemma only from a complete PROVED result."""

    if verification.status is not VerificationStatus.PROVED:
        raise LemmaMaterializationError("only a PROVED verification result can build a lemma")
    if verification.stage_name != stage.name:
        raise LemmaMaterializationError(
            f"verification stage {verification.stage_name!r} does not match {stage.name!r}"
        )
    if verification.counterexample is not None:
        raise LemmaMaterializationError("a PROVED verification result cannot contain a counterexample")
    if verification.reason_unknown is not None:
        raise LemmaMaterializationError("a PROVED verification result cannot contain an unknown reason")

    return CertifiedLemma(
        lemma_id=lemma_id_for_stage(stage),
        source_stage_name=stage.name,
        world_size=stage.world_size,
        input_relations=stage.input_relations,
        single_ops=stage.single.ops,
        distributed_ops={rank: program.ops for rank, program in stage.distributed.items()},
        output_relation=stage.output_relation,
        original_single_shapes=_original_single_shapes(stage),
        original_distributed_shapes=_original_distributed_shapes(stage),
        reduced_shapes=verification.reduced_shapes,
    )


def certify_stage(
    stage: StageSpec,
    *,
    timeout_ms: int | None = None,
) -> CertificationResult:
    """Reuse Stage 5 once and materialize exactly one lemma for PROVED only."""

    verification = verify_stage(stage, timeout_ms=timeout_ms)
    lemma = (
        build_certified_lemma(stage, verification)
        if verification.status is VerificationStatus.PROVED
        else None
    )
    return CertificationResult(verification=verification, lemma=lemma)
