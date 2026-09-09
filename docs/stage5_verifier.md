# Stage 5: Complete Stage Verifier MVP

Stage 5 verifies one already-defined Stage. It does not discover candidates, align graphs, schedule stages, or establish cross-Stage provenance.

## Verification query

After shape reduction, symbolic program execution, and relation encoding, the verifier asks exactly:

$$
R_{in} \land \neg R_{out}
$$

where an output relation with multiple constraints is represented as `Not(And(c0, c1, ...))`. There is deliberately no `Rin` satisfiability pre-check: input relations are a proof premise, not a property this local verifier must prove.

Stage 3 execution stores each produced tensor as the operator-generated Z3 expression. Consequently the query already contains `Sem(Gs)` and `Sem(Gd)` through expression substitution; no fresh output variables or separate semantic-equation block is needed.

`UNSAT` means `PROVED`, `SAT` means `DISPROVED` and returns an input-only counterexample, and Z3 `unknown` means `UNKNOWN`. A requested positive timeout configures Z3 and still maps to `UNKNOWN` when Z3 returns that result.

## Reduction soundness contract

For every original concrete tensor extent `d` and matching reduced extent `d'`, shape reduction explicitly adds:

$$
1 \le d' \le d
$$

For a MatMul contraction extent `K`, it also adds:

$$
K' \ge \min(2, K).
$$

Relations preserve each original single/local ratio with integer cross multiplication:

$$
local' \cdot original\_single = single' \cdot original\_local.
$$

Under the documented assumptions, this supports the TrainVerify-style lifting argument:

$$
Valid(S_{reduced}) \Rightarrow Valid(S_{original}).
$$

This is a scoped design argument, not a mechanically checked theorem. The supported fragment is MatMul, Add, and Mul; and Replicate, regular contiguous equal-size Shard(dim), and Partial(sum). It excludes broadcast, arbitrary layouts, uneven or interleaved shards, unsupported operators, communication value semantics, and whole-graph reasoning.
