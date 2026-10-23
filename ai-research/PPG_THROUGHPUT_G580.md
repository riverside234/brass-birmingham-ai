# PPO/PPG throughput and fixed-budget continuation

The subsequent [PPG phase revision](PPG_PHASIC_G580.md) changes cadence to four
iterations, links the peak rate to PPO and detaches PPO critic representations.
The newer [g590 delayed-critic revision](DCPG_G590.md) supplies the current
continuation command. This report preserves the preceding throughput
implementation and its measurements; its caches and fixed guard settings remain active.

The requested revision keeps the model, MC targets, PPO objective, learner sampling,
eight-age uniform PPG buffer, whole-game holdouts and population unchanged. It adds
phase-local caches and an explicit `ppg-value-phase/8.0.0` recipe. The completed IL
folder remains `artifacts/step5/features4-v11-2-batch7-il`.

## Effective PPG settings

| Setting | New value |
| --- | --- |
| Schedule / buffer / heldout | Every 8 committed iterations / all 8 ages / all 8 ages |
| Epoch ceiling | Fixed 5; adaptive voting disabled (`epoch_controller: null`) |
| Heldout acceptance interval | 32 optimizer steps, including a short epoch tail |
| Complete raw-policy KL interval | 32 optimizer steps, at every acceptance boundary, plus final retained-state verification |
| Patience | 1 failed heldout interval; existing neutral-within-tolerance behavior retained |
| Heldout loss allowance | Mean paired per-game increase <= 1.5 standard errors of those paired differences |
| Policy KL caps | Overall and age 0 each remain 0.002 |
| Cloning / probes | Existing disjoint stratified samples and complete legal distributions unchanged |
| Value optimizer | Existing phase-start Adam reset; first-interval warmup now spans 32 steps, followed by existing decay to 0.3x at the fixed cap |

Five epochs is a ceiling, not a requirement to train five epochs. Keep-best and
patience can stop earlier. A policy failure rolls back the entire attempted
interval with matching value-Adam and companion-head/Adam state. Existing bounded
beta retries still require value improvement, replay their original cloning draws
and cannot waive either KL cap. Intermediate steps between checks are not measured;
only checked states can enter retention, and the final chosen state is checked again.
Zero standard error still permits no positive mean loss increase.

## Implementation

- `calibration_cache.py` caches the frozen context, outcome predictions, descriptors,
  mapping and remaining-cost inputs for **one** stock utility-head calibration.
  The same `predict_utility` function trains the same head, on the same samples,
  with the same three passes, minibatches, weights, clipping and Adam state.
  Feature encoding falls from roughly five dataset sweeps to one. Custom model
  subclasses keep the original path. The cache is destroyed before PPO changes
  the backbone. Teacher calibration retains its existing semantics.
- `ppg_policy_cache.py` copies admitted complete legal inputs into ragged contiguous
  tensors, uploads bounded chunks and captures the post-PPO raw probabilities once.
  Subsequent cloning/probes gather batches and old probabilities with tensor operations;
  no Python per-row probability upload is needed. Updated model outputs are always
  recomputed. Candidate counts, padding masks, duplicate sampled rows, order and RNG
  are preserved. Cache admission still checks selectors.
- Policy tensors prefer GPU storage when their estimated size fits both an 8 GiB
  cap and one third of currently free memory; otherwise they stay on CPU. Calibration
  features prefer GPU within 1 GiB and one eighth of free memory, otherwise CPU.
  Neither cache persists across phases or checkpoints. Logs report actual sizes,
  locations and build time; these are additional memory, not reductions to disk cache.
- `ppg_replay.py` gathers state/action/mapping and metadata columns in batches per
  mmap, restoring original order and duplicates. This change adds no prefetch workers
  or altered sampling. Recorded descriptors, targets and branch/holdout partitions
  are preserved.
- Auxiliary statistics accumulate detached scalar tensors on-device and are copied
  together when reporting. Optimizer losses, nonfinite checks, clipping, gradients
  and transactional retained counters stay active.
- Full/compact reports retain calibration-cache metadata, PPG cache metadata, guard
  counts, SE allowance, fixed cap, reference preparation, critic/cloning and guard
  timings. Component timing is host wall time, not a CUDA kernel profiler; asynchronous
  execution can overlap until synchronization.

## Continuation

The helper reads the actual committed config under the run lock, validates the last
checkpoint hash and changes only the value-phase recipe. It never edits run state,
checkpoints or caches. Adoption preserves the existing schedule and cached ages:
for the exported g580 state the next PPG phase remains **g584**, not g588. The old
epoch votes are archived in the migration report, and the fixed cap becomes 5.
Both Adams, beta, RNG, GAE/snapshot evidence, LR controller, population and ordinary
progress are retained. Existing source freshness checks still reject unfinished
old-source collection work.

After syncing the source and matching review registry, with committed generation 580:

```bash
cd /data/not_backed_up/yxu209/ai-training
python scripts/configure-ppg-throughput.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 580 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppg-throughput-g580.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-throughput-g580.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 580 --iterations 20
```

This continues g581–600. Do not repeat the old local-scoring adoption or
`--fresh-ppg-buffer`. If Ctrl+C left an unfinished iteration and startup reports it,
first use the existing checkpoint-bound recovery helper:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 580 --apply
```

Fresh configuration generation also uses the new recipe. Archived signed configs
remain valid with their original checks/caps until an explicit migration.

## Validation

Focused regressions cover cached-vs-uncached calibration weights/Adam/RNG, frozen
parameter preservation and failure cleanup; raw KL and cloning gradient parity;
legal masks/order/duplicate samples; vectorized mmap column parity; deferred
statistics; strict fixed-cap and 1.5-SE validation; whole-interval joint rollback;
short tails and final checks; g580 config generation and restart preservation of
committed caches, clock, optimizers and controller evidence.

The small smoke runs eight authored CPU iterations in a separate directory, including
PPO, the scheduled PPG phase, auxiliary heads, diagnostics, checkpoint publication,
and exact restart checks at iterations 4 and 8. It does not access production
checkpoints, run native evaluations or establish improved playing strength.

Run the bounded check locally with the existing training environment:

```bash
python scripts/smoke-ppg-throughput.py --output artifacts/ppg-throughput-isolated-check
```

Final focused validation: **37 passed, 1 skipped** (CUDA unavailable), covering
38 distinct cases across the throughput, paired-SE, value-batch, companion
rollback and targeted legacy GAE/value-phase regressions. An archived g540
source-admission test initially ran during registry regeneration; its isolated
rerun passed after the registry was finalized. The current g580 helper also passed
its narrow-transition, idempotence and unchanged-run checks.

An exploratory run of the three older `test_selfplay_major_learning` cases hit
their unchanged authored-fixture topology mismatch (`V11 link topology differs:
CanalBelperDerby`) before reaching the revised training paths. Those fixture
failures are separate from the focused results above. The new calibration failure
cleanup regression uses the current valid fixture and passes.

The final isolated eight-iteration smoke passed with 64 PPO steps and **35/35
retained PPG steps**. Exact restart checks passed at iterations 4 and 8, including
weights, PPO/value Adam, companion state, RNG, schedule and fixed epoch state.
The small fixture has seven steps per epoch, so it exercises five short epoch-tail
checks plus the final retained-state check; the focused regression separately
verifies a full 32-step interval and joint rollback. Forty-eight ordinary held-out
games supported acceptance. Overall KL was **0.0000051513**, and age-0 KL was
**0.0000071482**, both below 0.002. The selected mean game loss moved from 1.497210
to 1.447681 on this authored fixture; this is not playing-strength evidence.

The smoke's policy cache held 300 rows / 802 legal candidates in 4,820,224 bytes,
with 0.112 s construction time. Sampled phase RSS peaked at 479,072,256 bytes;
the process high-water mark was 483,414,016 bytes. These tiny-fixture measurements
do not estimate production eight-iteration memory usage.

The following paired CPU benchmarks had already completed before the user asked
to stop further performance testing. No additional performance runs were launched.
They used two PyTorch threads and the same inputs/weights for each comparison:

| Operation | Before | After | Time reduction | Measurement scope |
| --- | ---: | ---: | ---: | --- |
| Utility calibration | 0.6907 s | 0.1618 s | 76.6% | 192 rows, six head updates; includes cache construction; encoder calls 15 to 3 |
| Cloning forward/backward | 0.2051 s | 0.1910 s | 6.9% | 128 rows, complete legal distributions; one-time cache construction separately 0.0154 s |
| Critic input/metadata batch | 0.02470 s | 0.00797 s | 67.7% | 256 rows across two mmap ages; original order and duplicate rows retained |

These are warmed component medians, not a before/after full-run comparison and
not a prediction of GPU speedup. CUDA residency, transfer savings and GPU memory
were not measured. The complete smoke plus benchmarks took 187.9 s. The machine-
readable result is `artifacts/ppg-throughput-g580-final-smoke/report.json`.

Final implementation identity:
`4d7dbd64a757357973679d64b36eaf199724b659588c0023ea1d98322aa09283`.
Learning compatibility remains
`ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
The source registry preserves all 74,129 existing explicit approvals and their
definitions, appending only 13 explicit routes for this reviewed source. No
production run, checkpoint, optimizer or replay cache was modified by this work.
