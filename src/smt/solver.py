from dataclasses import dataclass
from time import perf_counter

import z3


@dataclass
class SolverOutcome:
    status: str
    query: str
    elapsed_seconds: float
    model: object = None
    reason: str | None = None


def check(assumptions, violation, timeout_ms):
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*assumptions, violation)
    query = solver.to_smt2()
    started = perf_counter()
    outcome = solver.check()
    elapsed = perf_counter() - started
    if outcome == z3.unsat:
        return SolverOutcome("UNSAT", query, elapsed)
    if outcome == z3.sat:
        return SolverOutcome("SAT", query, elapsed, model=solver.model())
    return SolverOutcome("UNKNOWN", query, elapsed, reason=solver.reason_unknown())
