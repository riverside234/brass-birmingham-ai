# B05: post-PPG search value and bounded continuation qualification

2026-11-07. Implemented; Linux/CUDA qualification is **Pending**. The completed
B04.5 r3 receipt is recorded in the migration plan. Production teacher modes
remain gated, search_v1 remains the default, and V205/DOMiNO/AZ_DB stay disabled.

R2 fixes the lab attempt ending `d6e77efd-221a8f6f`: all 70 CPU fixtures passed,
but restoration rejected differing saved/visible GPU counts before any update.
The B05-only launcher now defaults to the existing logical-device restore path
with the learner on `cuda:0`. It restores all overlapping saved RNG streams,
preserves initialized streams on additional devices and reports the counts.
`--strict-cuda-topology` retains the equal-count diagnostic gate. The checkpoint
library's strict default, training code and implementation identity are unchanged.
Regressions exercise one-to-six, six-to-one and equal-device restoration, exact
Adam/CPU RNG preservation, parent/child flag propagation and strict rejection.
Child exceptions are written into the exported qualification report, retaining
the initiating cause instead of only a generic subprocess failure.

## Training integration

`SearchValueIteration` connects collection admission, the ordinary PPG boundary,
and checkpoint/state commits. Its tensor-bound state stamp includes the full
companion Adam, eight-age buffer receipts, label probes, stage, lambda, blend,
trunk scale and gates. Resume rejects an old/new checkpoint-state mismatch,
missing head/companion state and incomplete Adam moments. Existing checkpoints
without a search head retain their parameter ordering and optimizer groups.
Buffer ages use collection generation G, while the committed checkpoint is
G+1; this preserves all eight generations needed by the gate.

The search head has one optimizer owner. Ordinary PPO and PPG exclude it.
Off/shadow construction leaves ordinary weights, gradients and RNG unchanged.
An existing head can be restored without enabling it. The post-PPG callback runs
before its prepared datasets close and uses the original frozen pre-PPG policy;
it does not reset the KL allowance. The guarded pass retains the stricter of
0.002 and the ordinary cap (0.003 for g1250). Current two-age caches, the
every-iteration PPG schedule, step-16 loss checks, step-32 KL checks, beta floor
0.125 and learning-rate factor 0.65 remain in place. Annealing defers search
training/progression. Missing ordinary probes defer trunk fitting.

Head-only exceptions restore the head, companion moments/counter and stage.
Trunk exceptions and stage-C guard rejection restore the entire post-ordinary-
PPG model, value Adam and search companion. Accepted trunk changes invalidate
the utility calibration. Gradients are cleared before the next ordinary pass.
The local-scoring companion receives no teacher objective and retains its
existing selection/rollback path.

Static producer/profile changes retire policy-label ages and beta together.
Expired teacher and value shards remain available to older full checkpoints;
future retention must inventory those dependencies before deleting them.
Unknown/partial/duplicate value publications fail closed. Policy and value
sealing share the remaining final-turn post-search allowance, with no second
allowance granted to either stage.

## Short qualification and its limits

Use [the B05 bundle](../distribution/rust-lab-b05-g1250/README.md). It binds the
same g1250 checkpoint/config/model as B04.5, includes the frozen B04.5 tables,
and pins the previously successful B03/B04 receipts. It collects **zero games**
and does not rerun B01–B04.5 qualifications or rebuild Rust.

The CPU suite covers off/shadow exactness, actual task-actor gradient sharing,
head-only and full rollback, all stage transitions using synthetic gate
receipts, three tiny search checkpoint cycles, missing/partial files, source/
config adoption retry, ordinary two-age restoration, PPG rejection/tails and
the current g1250 recipe. Synthetic stage evidence never qualifies a real
256-game strength gate.

The lab check restores g1250's full checkpoint, ordinary/local companion Adams,
RNG, gates, cache ownership and dependencies in an isolated fork. It compares
the two Python preparation paths on original B03 families without changing
their historical g570 lineage. It then runs small authored ordinary PPO/PPG
kernels with the actual g1250 model, optimizer states and recipe; only fixture
row selection is bounded. It reloads the resulting diagnostic checkpoint and
checks ordinary/value/local companion state and cache ages. A separate copy
commits/restores a stage-A head update on the frozen B04.5 value rows. An
authored CUDA teacher PPO update checks sharing against the actual ordinary
actor gradient and adds no optimizer step.

These are component update/commit tests, not new completed training games,
performance or strength evidence, or a normal production iteration. All
source-run dependencies are verified unchanged afterward. The existing B07
gate still owns production backend adoption and the next normally authorized
G→G+1 iteration, restart and rollback. B05 does not turn the g1250 shadow-only
null-readiness exception into production readiness.

The bundle records exact source pins and local test results. The source-review
registry preserves previous explicit config restrictions and adds only reviewed
routes to this implementation. No signature override or broad config exception
is introduced. Each lab attempt exports a new Git-visible results folder,
including failures; earlier successful qualifications remain reusable.

Local validation: 70 CPU regression tests completed in 130.4 seconds, with
69 passes and one platform-dependent symlink test skipped on Windows. No games
or lab training sessions were run locally. The actual g1250 CUDA update/restore
checks remain Pending until the lab command succeeds.
