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
