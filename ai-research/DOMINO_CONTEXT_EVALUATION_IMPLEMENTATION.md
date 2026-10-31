# DOMiNO context evaluation

Implemented 2026-10-31 (America/Chicago), following
[the context evaluation plan](DOMINO_CONTEXT_EVALUATION_PLAN.md).

The existing neutral PL slot remains 128 games every ten committed iterations.
The new report-only panel runs every 50 iterations after activation. It seals
16 setups before inspecting outcomes: 2 IL-reference, 2 v4-reference and 12
competitive setups, with an explicit proportional startup fallback. It reuses
all 64 neutral games and adds 384 games, changing only the candidate context.
No extra PL fit, training update, promotion, PFSP update or health evidence is
produced by this panel.

## Protocol and implementation

- `context_evaluation_config.py`: strict versioned configuration. Existing
  configurations without the block remain off. Fresh DOMiNO configuration
  generation enables report mode. The optional 100-iteration cadence is an
  explicit configuration change, never an automatic response to results.
- `context_evaluation_spec.py`: outcome-blind selection, parent-spec/model/runtime
  receipts, exact rotations and sampling seeds, and interleaved context jobs.
  Contexts share one registered candidate model identity. Neural opponents stay
  neutral; the existing frozen v4 policy remains search-free.
- `context_evaluation_collect.py`: validated candidate-only descriptors, official
  tie-aware outcomes, pass/action/industry denominators and insolvency accounting.
  Develop industry counts include the second tile chosen on continuation rows;
  these do not count as another main action.
  Task return uses the recorded combined reward and subtracts capped insolvency
  cost once. No intrinsic reward, teacher labels or training feature cache is
  generated. Neutral summaries are extracted while the main collector still
  holds each validated replay.
- `context_evaluation_analysis.py`: equal-context mixture within paired setup
  blocks; 2,000 stratified setup-bootstrap resamples; 1,000 matched label
  permutations; full and investment-plus-realization-removed distances without
  rescaling the remaining coordinates. Within-context variation, absent phases,
  rare events and insufficient support remain visible.
- `context_evaluation.py`: persisted activation/due schedule, sealed pending
  checkpoint, exact completed-job reuse, bounded collection retries, atomic data
  publication and owned replay cleanup. Python, NumPy and Torch CPU/CUDA RNGs are
  restored even on errors. The active training model is never passed to the
  evaluation collector. Incompatible pending source/runtime receipts fail
  admission rather than mixing executions or collecting replacement neutral games.
- `context_evaluation_plot.py`: deterministic selectable-text SVG from committed
  evidence and frozen history. The latest 12 panels remain separate comparisons.
  Both heatmaps share one numeric scale; the mixture uses a diamond. A render
  failure retains evidence and can retry without games or resampling.

The panel keeps at most one automatic pending specification. Collection and
renderer retries are each bounded at three failures. It records failed attempts
and retires a permanently incompatible or repeatedly failed panel.
It preserves compact reports and representative candidate traces; raw additional
replays are retained for the latest two complete panels. Required pending model
dependencies are pinned against ordinary retention; failed specs release these
pins. Evaluation-only source/config adoption
preserves the existing utility-calibration receipt as well as optimizer/RNG,
PPG replay, progress, DOMiNO evidence and controller baselines.

## Evidence and timing artifacts

Each panel writes under
`ratings/contexts/generation-NNNNNN/`:

- `spec.json`, `games.jsonl`, `summaries.jsonl`, `report.json`, `report.md`;
- compact per-game admission receipts and representative traces;
- `status.json` with attempts, failures and elapsed work;
- `progress.svg`, `visualization.json`, and a finalized iteration-time
  companion `cost.json` when the normal trainer interval is available;
- `progress-final.svg` and `visualization-final.json` after timing is finalized,
  preserving the first immutable SVG if the final redraw fails.

The immutable analysis report precedes rendering and the final iteration timing.
The timing companion supplies the same-interval normal-training denominator,
including ordinary PL work and excluding added diagnostic work. The renderer
records both report and timing receipts. Its own final redraw time is recorded
separately in the visualization receipt and cumulative accounting, avoiding a
self-referential render-time calculation. A hard process kill with an unknown
elapsed tail marks timing incomplete; missing overhead is not reported as zero.
After two complete panels, the 5% overhead budget is advisory and does not
change training or evaluation cadence.

`scripts/export-training-reports.py` exports the verified compact evidence,
Markdown, per-generation SVGs and the latest SVG alias. It verifies generation
identity and matching receipts and excludes model tensors and raw replay shards.

## g1340 reward, exploration and cache revision

`training_revision_g1340.py` is a separate, explicitly reviewed training change:

- Insolvency **event_cost=0.01, max_cost=0.075**. Learner, search and any existing
  teacher-phase reward identities stay consistent. Teacher mode is preserved.
- Ordinary exploration **epsilon=0.05**. Existing legal-family/context weighting,
  three T2 openings, decoupled PPO behavior verification and branch rules remain.
- Fixed entropy remains **0.10 in all four groups**, branching remains **1%** with
  **50% T2 + 50% q**, and snapshot weight remains fixed at 1.
- The current two-iteration PPG recipe is retained. Old-reward PPG caches are
  retired **after durable configuration adoption**, under the run lock. The first
  new iteration uses fresh rows; the second can use both new-reward iterations.
  Retries do not clear a newly committed, compatible buffer again.
- Weights, PPO/value/auxiliary Adam, beta, RNG, progress, population and checkpoint
  lineage are preserved. The existing reward-change safeguard invalidates the
  utility calibration and archives incompatible DOMiNO quality/centroid evidence.
  DOMiNO rebuilds support with task-only updates while keeping learned duals and
  their Adam state. Economic, KL-reference, GAE and snapshot baselines are not reset.

Fresh configuration generation also uses the new reward and epsilon. Archived
configs remain valid and unchanged; adoption requires the reviewed transition.
The context-only helper remains available for evaluation-only changes and does
not refresh PPG replay.

## Diagnostics and cleanup follow-up

The g1340 export is now available locally. Its last two reports measured
**11.32/11.62 s post-PPO diagnostics**, **12.43/25.69 s final diagnostics**, and
**32.38/35.66 s cleanup**. Each active-anchor probe took about 11.3–11.7 s;
the 2,048-row auxiliary-head diagnostic took about 12 s. The second final
diagnostic pass includes another anchor probe when retained PPG changes weights.

The performance revision preserves data eligibility, checkpoint dimensions,
optimizer/RNG/controller state, scoring targets and PPO/PPG acceptance rules.
The subsequent explicit policy-observation revision changes the critic inputs
and removes the anchor-distance gate as requested:

- **Auxiliary-head diagnostics:** every **five committed generations**, using
  at most **512 ordinary held-out learner rows** balanced by table/era. Retain
  ownership CE/positive-owner error counts, VP MAE and sample support. Drop
  confusion matrices, Brier scores, occupancy breakdowns and extra VP summaries.
  Skipped generations explicitly report `measured=false`, with no stale metrics
  or inference. The schedule follows the committed generation across restarts.
  Auxiliary heads still train on their existing schedule; this changes reporting.
- **AnchorKL and original-IL diagnostics:** disable their measurements,
  reference promotions and overall/group allowance calibration. Console reports
  `AnchorKL=off`; reports use a null measurement, never stale saved KL values.
  LR controller contract `ppo-categorical-kl-lr/1.3.0` uses fresh economic health
  plus the unchanged PPO categorical KL rules. Cuts, warmup, cooldown, multiplier,
  early stopping and empty/nonfinite-update rules remain. Archived anchor and
  calibration receipts are retained for lineage, but cannot gate LR increases.
  The existing economic baseline and rolling evidence are preserved and advanced
  from ordinary games. IL remains the fixed PL rating origin/reference opponent.
- **Seat fingerprints:** stop current, historical, bot and DOMiNO-context style
  fingerprint inference. The 28 fingerprint columns become zero; the five role
  columns and checkpoint-age column stay unchanged (34 dimensions total).
  An input hook at the critic descriptor encoder also neutralizes old PPG cache
  rows without rewriting their recorded descriptors or Monte Carlo targets.
  Model tensors and checkpoint shapes do not change. Hook installation is
  idempotent and repeated after each load of the new self-play configuration.
  DOMiNO strategy contexts, actor inputs and PFSP sampling remain unchanged.
- **Probe performance:** cache admitted input batches on CPU, bounded to 256 MiB;
  batch transfers and exact full-legal-distribution KL reductions replace per-row
  tensor copies. With AnchorKL disabled, run the fixed probe only on due, healthy
  historical-pool admission attempts and the associated eviction comparisons.
  Keep immutable probe positions and model-bound pool probability caches; never
  reuse learner probabilities after weights change. No per-iteration IL-data
  diagnostic reload or entropy-probe pass remains. Fixed entropy stays 0.10;
  preserve the saved entropy baseline and opening/branch staging.
- **Auxiliary performance:** transfer whole output batches and vectorize entity
  reductions. Restore model/head mode even after exceptions. Sampling, target
  masks and relative-seat mappings stay intact.
- **General cleanup:** every **five committed generations** by default. A small
  durable receipt prevents restart from resetting the cadence. Batch up to 64
  move/delete intents per journal write; retain fresh dependency scans, path
  checks, content hashes, quarantine rules and interruption recovery. Archive
  compact reports before retiring their raw evidence. A failed pass does not
  advance the schedule. Manual cleanup still runs immediately.
- **PPG cache expiry remains per commit**, including the requested old-reward
  refresh. Population selection/limits, active model pins, pending evaluations
  and per-iteration checkpoint commits remain unchanged.

Less frequent general cleanup amortizes repeated scans and journal work; it
does not remove the underlying file I/O. Up to four extra iterations of retired
replays/checkpoints can remain between passes. The exported run retired about
1.48–1.98 GB per pass, so allow roughly **6–8 GB extra temporary disk space**
(a rough estimate, not a hard bound; replay sizes and evaluation artifacts vary).
PPG's own memory/disk window does not grow. An optional strictly validated
`retention.every_iterations` integer (1–20) overrides the default; changing an
existing signed configuration still requires its normal reviewed transition.
No config change is needed for this five-iteration cleanup default. Retiring
fingerprints/AnchorKL is an explicit signed configuration change:
`policy_observations = {contract: selfplay-policy-observations/1.0.0,
seat_fingerprints: neutral, anchor_kl: off, fixed_probe: pool_admission_only}`.
Fresh config generation and the existing `configure-training-g1340.py` helper
include it. The documented g1340 recovery/config/train command is unchanged;
unreviewed configuration edits and incompatible pending collections still fail.

Isolated CPU measurements against Git `c8fa5a5`, two Torch threads, unchanged
small-model weights and authored inputs:

| Work | Before | After |
| --- | ---: | ---: |
| Fixed 2,048-row probe (2–3 legal actions/row; median of 3) | 0.380 s | 0.325 s |
| Auxiliary report, same 2,048 rows (median of 2; retained metrics match) | 2.010 s | 0.640 s |
| New default auxiliary report, 512 rows | 2.010 s every iteration previously | 0.163 s every 5 iterations |

The fixed-probe benchmark above is relevant to pool-admission work; it no longer
runs as an AnchorKL diagnostic. A separate bounded CPU training smoke accepted
one PPO step and two PPG steps; every observed descriptor forward received zeros
in the fingerprint columns, including rows carrying old nonzero descriptors.
The focused regressions cover strict config migration, economic evidence,
stale-anchor exclusion, LR rules, model/Adam/RNG preservation, neutral seat
rotation/context provenance, PFSP planner parity, due-only pool probes and fresh
branch staging without an entropy probe. These tests do not establish a playing
strength benefit or predict production CUDA timing.

Final follow-up validation: **32 continuation/controller/source-review tests**
and **29 diagnostics/retention/context-lifecycle tests** passed. Receipts are
`artifacts/policy-observations-continuation-tests.log` and
`artifacts/policy-observations-performance-tests.log`. The neutral-descriptor
smoke used 42 authored tables, one PPO step and two accepted PPG steps; all 146
observed descriptor forwards had zero fingerprint columns. It ran in an isolated
temporary directory and did not edit any existing training checkpoint. Current
source identity is `f2303e80d843bdce339df8e9ca59f37642d2a5addbc13f468df700b6e0b09d11`;
model/feature compatibility is unchanged. All 76,391 pre-follow-up source reviews
are retained in order, with 481 explicit reviews added.
| Cleanup of 100 temporary files | 1.702 s / 402 journal writes | 0.775 s / 10 writes |
| Separate 1,000-file cleanup fixture | 50.86 s / 4,002 writes | 24.93 s / 66 writes |

The reduced auxiliary check amortizes to **0.033 s per iteration** in this fixture.
The same selected 512 rows also match the old metric calculations within float
tolerance. The 1,000-file measurement preceded the reporting-frequency follow-up;
the retention code was unchanged between measurements. These are local Windows
CPU/filesystem measurements, not estimates of lab CUDA speedups or playing strength.
The 2–3-action probe is deliberately small and does not represent the production
candidate-count distribution. No long training/evaluation was launched.

Reproducible benchmark: `tests/Recovery/benchmark_diagnostics_cleanup.py` with
`--baseline-ref c8fa5a5`. Receipts are
`artifacts/diagnostics-cleanup-benchmark.json` and
`artifacts/diagnostics-cleanup-benchmark-final.json`.
New iteration reports expose `anchor_probe_seconds`,
`auxiliary_diagnostics_seconds`, `pool_admission_seconds`, and cleanup
plan/move/purge timings. The compact history retains these fields and cadence.

## Enable for the continuing run

Use the actual committed generation in `--expected-generation`. For the last
reported **g1340** checkpoint, stopped with Ctrl-C:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 1340 --apply && \
python scripts/configure-training-g1340.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 1340 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-domino-context-g1340.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-domino-context-g1340.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1340 --iterations 20
```

This requests **g1341–g1360**. First activation at g1340 makes the first context
panel due at **g1390**, then g1440, including across restarts. There is no new
context panel during these first 20 iterations. An already activated schedule is
preserved. If training has advanced,
use that current generation and a matching new output filename. Do not rewind,
repeat g1326 DOMiNO adoption or rerun IL. The requested PPG refresh is automatic
for this reward change; do not manually delete checkpoint or cache files.
The helper only writes the requested new config; continuation commits the
reviewed transition under the normal run lock.
The command also includes the diagnostics/cleanup follow-up above. The user has
not yet run the earlier g1340 command: retain this same recovery generation and
config filename. Auxiliary diagnostics and general cleanup first run at **g1345**,
then g1350, g1355 and g1360. No separate cleanup/reset command is needed.

To keep context evaluation off, retain the old config. To explicitly change
the interval later, use `configure-domino-context-evaluation.py` with
`--every-iterations 100`;
an already scheduled/sealed slot keeps its schedule, and subsequent slots use
the new interval.

Export after a completed context panel with the existing command:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic
```

The latest g1340 run/checkpoint is on the lab machine, not in this checkout.
The helper verifies its actual committed generation and checkpoint hash there;
no production checkpoint or run state was modified here.

## Validation and bounded measurements

**126 distinct focused regressions passed** across the main suite and targeted
reruns after fixes. The focused suite covers setup stratification, exact context/seat/RNG identity,
paired resampling, mixture covariance, descriptor ablation, corrupt/missing
neutral receipts, interrupted reuse, bounded failures, retention, compact export,
SVG identity, configuration migration and optimizer/RNG/cache preservation.
The existing main PL and report-export regressions are included.

The checks also caught and fixed an older resume-selection issue: an
inapplicable approved migration must not abort before another exact approved
path is considered. Unknown migrations and unmatched configurations still fail
closed. Calibration preservation is separately pinned to evaluation-only source
reviews; other source/reward changes keep their existing freshness checks.

Validation receipts: `artifacts/context-g1340-validation-summary.json` and its
listed logs. Current source identity is
`b514fd9ee9944d93bfd596968cba77c93a4c742118119c74610c5541dce6c29f`;
model/feature compatibility remains
`ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
All 75,952 previous source-review entries are retained in order.

An isolated real-game smoke completed **28 unique games** (one setup, four
rotations, seven contexts) in **272.39 s** on CPU. It used one untrained 1,511,464
parameter model, four actor processes, one host each and unchanged per-game
seeds/policies. After four additional-context games completed, an intentional
interrupt resumed with those four jobs cached and completed the remaining 20.
There were no additional PL fits or training updates. Model weights, RNG and
mode were verified unchanged. This tests execution, not strength.

The resumed additional-context call recorded **94.94 s collection + 77.91 s
strict replay extraction**, or **0.116 new games/s including extraction** for its
20 newly generated games. It loaded the candidate once; inference processed
2,743 positions in 2,183 batches (histogram: 1=1700, 2=415, 3=59, 4=9).
Reported main-process RSS was **662,040,576 bytes** at collection end. GPU peak
memory was unavailable on this CPU machine. Production telemetry additionally
records sampled RSS/disk maxima; GPU allocator peaks are explicitly labeled
process-lifetime counters, without resetting the trainer's profiling counters.

Four authored visual fixtures cover normal, identical-context, incomplete and
wide/asymmetric intervals with 12 history points. Final rendering took
**0.054–0.648 s**, including 1280-pixel PNG previews; SVGs were **3.4–116.1 KB**,
under the plan's 5-second/2-MB limits. Text remains selectable. The normal and
wide-interval views were inspected at 1280 pixels; the SVG preserves full-size
vector text. Fixtures are explicitly labeled synthetic.

The separate g1340 PPO/PPG fixture accepted one PPO step, verified epsilon 0.05
and rejected stale epsilon 0.08 behavior. Its retained PPG probe KL was 0.
No long training, 384-game production panel or CUDA performance campaign ran.
The additional default panel is **384 games every 50 iterations** (7.68 games
per ordinary iteration on average); its wall-time overhead is not inferred from
the reduced CPU smoke and must be assessed from actual completed panels.

## Interpretation

These are within-checkpoint candidate-context comparisons against the declared
neutral-context frozen opponents. They do not establish robustness to all
opponent styles, causal DOMiNO improvement, deployment qualification or seven
useful distinct strategies. The main PL series remains the neutral-policy
strength measure. A test-only untrained-model smoke cannot establish strength
or estimate the default-size production GPU panel overhead.
