# Optional shape reduction review

Pre-change checkpoint: `632e585` (the worktree was already clean).
The original-shape backend remains the default. Input JSON files are unchanged.

## Independent review and fixes

An explicitly requested subagent inheriting the parent model reviewed constraint
ownership, finite encoding, opaque sums, fallback, diagnostics and budgets.
It did not edit the repository. Review findings were discussed and corrected:

1. P2: certificate expansion could exceed the caller's contraction budget when
   contraction scales had gcd=1. Budget checks now use actual original-K/block-width
   counts, with a K=4/K=6 regression.
2. P2: the coordinator recognized reshape/expand by registered names. Parameter
   rewriting and contraction-axis metadata now belong to operators. Alias testing
   verifies that extension does not require coordinator changes. Generated invariant
   failures retain their plan and become explicit UNKNOWN/SHAPE_REDUCTION_ERROR,
   rather than misreporting invalid original input or being silently repaired.
3. P2: original collective deadlocks were discovered during candidate execution
   and misclassified as internal failures. Original scheduling now has a static
   dependency check before candidate construction. The full collective regression
   class runs in both modes.
4. P3: file/JSON parsing failure did not preserve the requested mode. It now reports
   SHAPE_REDUCTION_NOT_STARTED; missing-file and malformed-JSON tests cover this.

The reviewer found no erroneous PROVED from BlockContext or bypass of the reduced
proof/original certificate gate. Equal symbolic summands share atoms; unequal ones
remain unconstrained. Nested opaque sums fail closed. Reduced or abstract SAT is
not reported as an original counterexample.

Independent checks reported by the reviewer: 48 adversarial cases agreed with
the original route; all operators in the 11 fixtures could be registered under
aliases without fallback; 19 targeted checks passed; 400 random schedules agreed
with the runtime executor (163 executable, 237 deadlocked). These checks complement
the in-repository regression suite and do not constitute a general formal audit.

## Verification

Commands use `/data0/xs/miniconda3/envs/TrainVerify/bin/python` in this workspace.

- Unit/integration suite: 87 tests, including the original tests, both collective
  modes, constraint/relation checks, finite encoding, negative goals, fallback,
  certificate budgets, nested sums, diagnostics and report replay.
- All 11 inputs: PROVED in the default original-index route.
- All 11 inputs: PROVED in the optional route, with reduced UNSAT and original
  certificate UNSAT, without exact fallback.
- 11 new `demo.shape-reduced.txt` reports: 104 recorded queries replayed UNSAT.
- Ruff lint/format and `git diff --check` pass.

One in-memory FC2 probe changed K to 4096 while preserving TP4 mappings. Optional
mode took about 0.375s and original mode about 7.625s in single runs. Optional
mode reduced total declared tensor elements from 7,995,392 to 256, then certified
original shapes using width-512 sums (16 distinct atoms over two output groups).
These timings include construction and solving, are machine/run dependent, and
are not a benchmark guarantee. No JSON was changed for this probe.

## Limits retained deliberately

Candidate reduction uses integral divisor scales and conservative non-singleton
lower bounds; it is not an arbitrary segment optimizer. Shape constraints alone
are not a lifting theorem. The independent original-shape certificate supplies
the soundness obligation. Opaque nested sums and inconclusive certificates fall
back explicitly; unreduced exact verification can still hit resource limits.
Arithmetic remains exact Real with shared uninterpreted GELU, not floating point.
