# Features 2 post-implementation review

14 October 2026. Reviewed FE0–FE6 for rule correctness, feature meaning,
privacy, caching, artifact admission and live inference integration. Width 304,
the frozen 5,489/548 layout and the explicit 3M model cap are retained.

## Fixes

**Feature encoding did not receive live inference deadlines.** The encoders had
an optional deadline, but local, remote and personality inference did not pass
the request's absolute clock. Feature work could continue after the admitted
search time, even though its eventual prediction would be discarded. The same
clock and cancellation event now reach context construction, per-candidate
consequences, cached lookups and final tensor publication. A typed cooperative
stop returns `expired`/`cancelled` without aborting the actor or poisoning shared
inference. Other validation errors still propagate. No new timer or time allowance
is created. Validation and individual array operations remain bounded cooperative
work, rather than preemptible operations.

**A source repair made earlier Features-2 heuristic recordings unnecessarily
unconvertible.** The explicit raw converter previously required their encoder
source fingerprints to match before it could re-audit/re-encode them. It now
accepts known v1/v2 raw layouts with matching rules, wire, target and reference
contracts, checks source fingerprint syntax, and independently reproduces every
native state, legal list and transition. It regenerates tensors in a new directory
and retains the original contract in the new header. Normal model/data admission
has not been relaxed. Neural behavior, teacher labels, old tensors and checkpoints
cannot use this exception. The existing parallel conversion scheduler is preserved.

**Verification gaps were repaired.** Older R2/inference tests still assumed the
old model version, 2M ceiling, categorical-action offset and cache key. They now
assert the actual version-2 contract. Fabricated subchoices test categorical
encoding; native FE tests verify their dynamic effects in legal contexts. The VP
batch-parity check allows 0.00005 VP of FP32 rounding with the wider input matrix;
other head checks remain strict. Nine new regressions cover mid-encoding stops,
cache publication, clock forwarding, cancellation, malformed clocks/errors and
strict versus explicitly re-audited data admission.

Changed runtime files: `learning/encoding_budget.py`, `learning/features.py`,
`learning/batching.py`, `recovery/encoding.py`, `recovery/inference.py`,
`personality/search/inference.py`, `feature_engineering/context.py` and `data.py`.
Tests and the lab guide are updated. The active recipe JSON is byte-for-byte
unchanged: exploration, reward, PPO, opponents, teacher clocks/size/cadence,
distillation and `search_wall_fraction=null` are preserved. Existing production
runs and checkpoints were not modified.

## Feature quality and rule checks

All six groups remain connected to their intended model inputs. The review found
no further arithmetic mismatch in the checked financial, spending, income, VP,
sale-feasibility or development cases. This is bounded evidence, not a proof over
every reachable game state.

Two complete heuristic games were independently re-audited after conversion:
**691 positions and 27,866 legal candidates**, one training and one validation
game. State/action tensors, offsets and seat maps match the pre-review cache
**bit-for-bit**. Original game objects, choices, labels and split membership are
preserved. Incremental cash/income/VP and determined-part bounds were checked on
443 recorded non-automatic decisions with zero mismatches. Three additional
native games exercised Pass, Scout and liquidation as well as building and
payments; their checked immediate accounting also had zero mismatches.

Sale summaries equal deduplicated native legal sale targets in **248 main-action
positions**. Of 326 state measures, 324 were applicable somewhere and 194 varied
in these two games. Status counts were 165,560 exact, 59,052 inapplicable, 326
bounded and 328 unknown. Reachable and blocked higher-tier targets both occur.
These are diagnostic observations, not independent training samples or reasons
to remove fields that happened to remain constant. Debt, final-round and rare
merchant/development cases also have authored native regression coverage.

The features expose useful relationships without changing the objective:
financial exposure remains a signed fixed-position projection; sale potential
is separate from secured VP; individual beer feasibility is separate from joint
feasibility; development targets respect physical stack copies and blockers;
partial choices do not recharge earlier payments or assume later optional parts.
Policy candidates/IDs remain intact, and no extra production search or simulator
clone is introduced by this review.

## Validation and runtime

**188 tests passed in 95.595 seconds**, including all FE suites, legacy learning
consumers, replay/conversion, PPO behavior probabilities, teacher timing/quality,
shared transport, dynamic ONNX, inference and continuation. An initial expanded
run exposed the stale assertions above and sandbox-local temporary-directory
permissions; the final run used a separate workspace temporary directory.
`git diff --check` passes.

The separate CPU smoke completed in **257.828 seconds**: one real width-304 IL
epoch, completed-IL reload, fresh PPO handoff, two ordinary games/two PPO iterations,
Adam/RNG/progress restoration and four seat-rotated diagnostic evaluation games
with zero failed games. PPO actually used **138 and 32 distinct eligible actor
rows**, with three and one optimizer steps. The evaluation is one correlated
setup with a deliberately undertrained model; it provides no strength conclusion.

Four teacher-turn probes used the unchanged **12/12/16/24 second** phase clocks,
four workers and the existing quality/calibration gates:

| Phase | Searches attempted/completed | Rollouts attempted/completed | Visited/legal actions | Qualified | Insufficient / timeout | Elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early Canal | 1/1 | 16/8 | 4/387 | 1 | 0/0 | 12.063 |
| Late Canal | 1/1 | 17/10 | 4/101 | 1 | 0/0 | 12.172 |
| Early Rail | 1/1 | 16/8 | 4/198 | 1 | 0/0 | 16.016 |
| Late Rail | 1/1 | 57/50 | 4/4 | 1 | 0/0 | 24.047 |

All four reported zero search-deadline overrun; external elapsed time includes
completion/bookkeeping. The four qualified labels are diagnostics from partial
games: **0 distinct labels trained, 4 unused, 0 presentations, policy drift 0**.
Early-phase action coverage remains sparse. These probes neither measure a full
five-game collection nor demonstrate improved qualified-label yield.

On the unchanged 14-position/1,014-candidate corpus, seven local CPU passes gave
the following median/p95 encoding milliseconds, excluding construction/model work:

| Mode | Median | p95 |
| --- | ---: | ---: |
| Persistent misses, no request clock | 3.511 | 14.980 |
| Persistent misses, request clock enabled | 3.570 | 15.511 |
| Result-cache hits, request clock enabled | 0.650 | 1.007 |

These small sequential measurements do not isolate tiny timing differences from
noise. The larger Features-2 input cost remains; this fix primarily prevents
expired work from continuing. GPU throughput/memory and full collection throughput
remain unmeasured. No playing-strength or learning-quality gain is claimed.

## Evidence and use

Evidence is in `artifacts/feature-engineering/review-20261014/`: before snapshot,
initial/final test logs, native/replay accounting audits, quality audit, re-audited
v2 manifest/receipt, latency report and isolated native smoke. `review.json`
records the final source/compatibility identities and preservation checks.

Follow [the lab guide](../docs/FEATURE_ENGINEERING_LAB.md) for the explicit
reconversion command and fresh-IL workflow. Changed source pins invalidate old
encoded-cache receipts; use a new output directory. Exact model compatibility
also changes despite identical numeric tensors. Earlier neural checkpoints require
their pinned runtime; `--resume-reviewed-source` cannot override this learning
compatibility mismatch. The supported new-code path is fresh IL on the re-audited
manifest. No old checkpoint or run was rewritten, migrated or reset.
Assess eventual usefulness with held-out, matched training/ablation runs measuring
development-to-build-to-flip conversion, era-end waste, insolvency and official
outcomes. The bounded implementation checks do not replace that experiment.
