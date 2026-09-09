"""Certify one existing Stage JSON and persist its PROVED concrete lemma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lemma import LemmaMaterializationError, certify_stage
from src.lemma_store import LemmaStoreError, save_lemma
from src.relation_encoder import RelationEncodingError
from src.shape_model import ShapeReductionError
from src.stage_loader import StageInputError, load_stage
from src.symbolic_tensor import SymbolicExecutionError
from src.verifier import VerificationError, VerificationStatus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_json", help="path to a validated Stage JSON input")
    parser.add_argument("--lemma-dir", default="lemmas", help="directory for lemma JSON artifacts")
    parser.add_argument("--timeout-ms", type=int, default=None, help="positive Z3 timeout in ms")
    arguments = parser.parse_args(argv)
    try:
        stage = load_stage(arguments.stage_json)
        result = certify_stage(stage, timeout_ms=arguments.timeout_ms)
        verification = result.verification
        print(f"Stage: {verification.stage_name}")
        print(f"Verification: {verification.status.value}")
        if verification.status is VerificationStatus.PROVED:
            assert result.lemma is not None
            target = Path(arguments.lemma_dir) / f"{result.lemma.lemma_id}.json"
            existed = target.is_file()
            path = save_lemma(result.lemma, arguments.lemma_dir)
            print(f"Lemma: {'EXISTS' if existed else 'CREATED'}")
            print(f"Lemma ID: {result.lemma.lemma_id}")
            print(f"Lemma file: {path}")
        else:
            print("Lemma: NOT CREATED")
            if verification.status is VerificationStatus.UNKNOWN:
                print(f"Reason: {verification.reason_unknown}")
    except (
        StageInputError,
        ShapeReductionError,
        SymbolicExecutionError,
        RelationEncodingError,
        VerificationError,
        LemmaMaterializationError,
        LemmaStoreError,
    ) as exc:
        print("Certification: FAIL")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
