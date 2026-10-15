# HM0–HM3 implementation and bounded validation

**Archived:** Hand coverage was removed from the active pipeline on 15 October
2026 at the user's request. The measurements below describe the earlier
experiment. See [the six-group/five-block report](SIX_GROUP_FIVE_BLOCK_REPORT.md)
and [current lab guide](../docs/FEATURE_ENGINEERING_LAB.md).

14 October 2026. Implementation scope is complete. These checks establish feature
and pipeline behavior; they do not establish improved playing strength or training
stability. Commands for the lab are in [FEATURE_ENGINEERING_LAB.md](../docs/FEATURE_ENGINEERING_LAB.md).

## Implementation and compatibility

Hand matching reuses the completed six-group context and already-generated legal
Build choices. One per-observation bitset mapping deduplicates outcomes by
`(slot, industry)`. Physical hand counts distinguish a duplicate from its last
copy. Candidate removals are evaluated jointly, including Scout's three discards;
the selected completed Build is excluded from unused-opportunity loss. No extra
legal enumeration, simulator cloning or future search was added.

The 27 per-card support counts and wildcard-only count append 28 state facts.
Four action facts describe lost support, loss of last ordinary support while
wildcards remain, and ordinary/wildcard alternatives for a Build. Existing Measure
masks distinguish unavailable, unknown and zero. Coverage is unavailable outside
main-action phases. These are fixed-board card-removal facts: Scout's newly
acquired wildcards are not credited, and post-action legality is not predicted.
See [the field contract](HAND_MATCHING_CONTRACT.md).

The immutable schema, learning contract and model architecture are version 3.
State/action dimensions are **5,685 / 576**, with style[4]/intent[8] still last.
The old v1/v2 schemas and source-review registries remain unchanged. All old A–F
numeric components match exactly across the two audited full games.

Width **304** and four residual blocks remain. Parameter counts are **2,643,597**
base, **2,653,617** with utility and **2,719,450** with personality/goals. The state
encoder contains 2,473,344 parameters. Existing 3M/3.1M limits suffice.

The new contract flows through fresh IL, IL handoff, rollout actors, dynamic
inference/shared memory, replay caches, PPO, neural ISMCTS teachers, evaluation,
checkpoint resume and named export metadata. Replay sidecars are `.features-v3.*`.
Incompatible neural history, PPO behavior and teacher targets fail admission;
limited compatible history uses the existing fresh-anchor fallback. Rule-based v4
opponents remain supported. There is no old-weight/optimizer migration.

The config CLI now binds both initialization and rehearsal to the completed new
IL dataset. A focused regression caught the previous recipe's stale rehearsal
path, which would otherwise reject a new handoff. Rehearsal weight/duration stay
unchanged. Retention also needed a correction: pins and evaluation cleanup now
include unversioned, v2 and v3 sidecars. Pin/cleanup/restore regressions cover it;
retention quotas are unchanged.

## Three existing-feature semantic checks

| Check | Result and evidence |
| --- | --- |
| Affected player and scoring time | **Passed.** Native own/opponent last-iron consumption verifies the actual owner receives the flip/income change under every relative-seat rotation. Immediate VP remains separate from secured deferred industry VP. |
| Resources usable for the selected action | **Passed.** Equal coal totals with different connectivity change actual Build quotes. Sale tests distinguish accessible brewery beer and eligible merchant beer. Two individually feasible sales sharing one beer are not jointly feasible. Rail tests cover connected coal, own unconnected beer and inaccessible opponent beer. |
| Stack unlock versus practical realization | **Passed.** The same higher-tier stack target in an early rich and late constrained position has distinct encoded cash/projected cash, remaining actions and sale/beer feasibility. Existing FE2 regressions also cover incomplete development, era restrictions and blockers. No guaranteed future build/flip or profitability score is inferred. |

These native-transition and encoded-input checks are in `test_hm_semantics.py` and
the retained FE2/FE3 suites. They indicated no A–F semantic correction or new field.

## Data, tests and smoke

Two complete qualified heuristic games were explicitly re-audited and re-encoded:
**345 training + 346 validation positions**, **27,866 candidates**. Records,
labels, action IDs/order, setup/splits and original shard bytes are preserved.
The conversion used two processes and took 29.547 seconds. No new teacher games
were needed. Missing/incompatible raw inputs fail explicitly. This small sample
contains positive selected-action total-support loss in 108/248 main-action
observations, but no positive last-ordinary-support loss; authored native wildcard
cases and randomized algebra tests cover that less common branch.

Focused suites cover hand algebra, duplicate and joint removals, applicability,
stale caches, privacy, reference/prepared/rotated parity, conversion/admission,
shared transport, forced-action filtering, deadlines, checkpoints and retention.
All **216 focused tests passed in 87.148 seconds**; the receipt is in
`tests-final.log` and `review.json`.

The optimized six-game CPU smoke took **428.0 seconds**: fresh width-304 IL for
one epoch, IL handoff, two PPO iterations, save/resume of Adam/RNG/counters and
lineage, and four diagnostic evaluation games with zero failures. Actual tables
were two current-model, one historical and three rule-based v4. PPO used **211
and 206 distinct actor samples**, with no eligible actor samples unused; critic
usage was 256 each. All four current-model seats contributed eligible data.

After the retention-only source change, the final-source smoke repeated fresh IL,
two PPO iterations/save/resume, four evaluation games and teacher probes in
**226.375 seconds**. Its two ordinary tables were one current-model and one
historical; the unchanged v4/encoder path is covered by the preceding six-game
run. PPO used **135 and 35 distinct actor samples**, none unused; critic usage
was 192 and 64. Final source and compatibility identities are verified in the
receipt. Existing production checkpoints were not changed.

The final-source teacher probes used the inspected current recipe's unchanged
shared-turn clocks. Each row is one observed reserved turn, not a complete game:

| Phase | Budget s | Attempted/completed rollouts | Root actions visited/revisited/total | Qualified / insufficient coverage | Search / whole-turn elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early Canal | 12 | 16 / 8 | 4 / 2 / 387 | 1 / 0 | 11.891 / 12.094 |
| Late Canal | 12 | 16 / 8 | 4 / 2 / 160 | 1 / 0 | 11.859 / 12.375 |
| Early Rail | 16 | 16 / 8 | 4 / 2 / 68 | 1 / 0 | 15.890 / 16.062 |
| Late Rail | 24 | 56 / 48 | 4 / 4 / 4 | 1 / 0 | 24.000 / 24.172 |

Each phase completed one search; final probes had zero failures/timeouts and
zero reported deadline-guard overruns. Whole-turn time includes finalization.
**Four qualified labels were generated, zero trained, four unused; presentations
and teacher policy drift were zero.** These partial-turn labels remain diagnostic,
and ordinary games did not reach collection cadence. An earlier pre-optimization
probe rejected two labels at the existing overrun guard (0.250/0.359 seconds);
that receipt remains in `native-smoke/report.json`. Gates were not relaxed.

Official dynamic ONNX export is covered by the focused suite. Final trained
utility and initialized goal-branch exports passed ORT CPU parity at batch/candidate
shapes (1,1), (2,7), (4,31), (1,257), including legal masks and every named head.
Maximum finite absolute error was **3.53e-5**. This is functional export validation,
not a deployment release qualification.

## Incremental performance

The reproducible CPU comparison uses identical 15 permitted observations and
1,361 candidates, maximum 387; seven passes give 105 timings per mode, including
21 with at least 300 candidates. Cold includes encoder construction; persistent
misses reuse an encoder without a result cache; warm hits reuse complete results.
Python allocation peaks are measured separately. Hardware is Intel Family 6,
Model 170, Stepping 4, Windows x64, 14 logical CPUs. No CUDA, neural forward or
search work is included. Timings include feature-array construction.

Final sequential paired measurements (`quiet-before-benchmark.json`,
`quiet-after-benchmark.json`) are summarized below; times are milliseconds and
allocation peaks are decimal MB:

| Mode | Median before → after | p95 before → after | Peak Python MB before → after |
| --- | ---: | ---: | ---: |
| Cold/construction | 9.245 → 9.452 | 30.556 → 37.122 | 3.261 → 3.709 |
| Persistent miss | 4.836 → 6.905 | 23.140 → 33.772 | 3.716 → 4.173 |
| Warm complete-result hit | 0.764 → 0.730 | 1.380 → 1.919 | 6.195 → 6.777 |
| Large-candidate persistent miss | 21.274 → 32.188 | 31.227 → 38.466 | included above |

The persistent-miss median increased **42.8%** in this pair; Python allocation
peaks increased **9.4–13.7%** depending on mode. These are repeated calls on a
small desktop corpus, not independent statistical samples; earlier local pairs
also varied. In particular, the small warm-hit median decrease is not evidence
of a speed improvement. Peak allocation is not process RSS or model/GPU memory.

The immutable mapping overlay avoids copying all old facts for every hand-effect
combination, and separate tensor-block interning retains old card-alternative
reuse. A full-array comparison before/after that optimization is exact.

Encoder overhead is measurable. Wall-time search throughput was not compared
under matched model weights, so no regression percentage for completed search
work is established. Re-measure on lab hardware with production workers and GPU
inference before attributing training runtime or label yield to these features.

## Preserved settings and limits

The checked-in latest-model config changes only architecture version. Exploration
0.08, temperature 1, PPO/reward/utility settings, curriculum, latest-model four-seat
collection, disabled champion gate, v4 budgets, teacher five games/cadence 160/
eight distillation steps, label quality/reuse/freshness checks and null cumulative
wall fraction remain. Explicit current phase clocks stay 12/12/16/24; archived
run migrations are not applied to this latest-model recipe.

Not exercised end to end: CUDA/production concurrency, a full scheduled five-game
teacher collection with distillation, long training, statistical strength panels
and deployment qualification. Existing focused tests cover scheduling and
distillation admission/sample accounting. The smoke's evaluation is a functional
diagnostic only. Start fresh Features-3 IL after explicit raw conversion; resume
only new-schema checkpoints with the documented same-run command.

Evidence root: `artifacts/feature-engineering/hm0-hm3/`. Authoritative final files:
`review.json`, `tests-final.log`, `converted-final/conversion.json`,
`current-smoke/report.json`, `native-final/report.json`, `lab-config-final.json`,
`inspect-final.json` and `named-final-exports.json`. Earlier prototype artifacts
remain for traceability and are not interchangeable with final-source outputs.
