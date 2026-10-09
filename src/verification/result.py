from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    PROVED = "PROVED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class VerificationResult:
    name: str
    status: Status
    scope: str = "ORIGINAL_SHAPES_EXACT_REAL"
    reason: str | None = None
    validated: object = None
    reduction: object = None
    inputs: object = None
    execution: object = None
    queries: list = field(default_factory=list)
    counterexample: dict | None = None
