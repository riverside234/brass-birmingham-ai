# g470: bounded throughput review and controller continuation

The exported run is committed at generation **470**, not 418. This revision adds ordered PPO preparation in **four CPU processes**, changes physical PPO batches to **512 rows / 131072 candidate cells**, permits snapshot critic weight reductions to **0.00**, and lowers entropy floors only on future qualified KL-reference promotions. PPG and teaching remain disabled. Existing training checkpoints were not edited and no long training or evaluation was launched.

## What the production log establishes

Source: `results/joint7-v11-2-grouped-lr-critic-g0470-20261022T011816523228Z/iteration-0469/report.json` and its recorded configuration.

- Ordinary collection: **159.70 seconds / 320 games = 2.004 games/s**. Inference handled **57616 positions in 30223 batches**, mean **1.906**, with **70.65% useful candidate cells** and **6629 padding splits**. Peak queued positions was 48; the batch cap of 64 was not reached.
- Queue wait summed to 1633.99 seconds, about **28.36 ms per position**. Forward calls took **85.83 seconds**, including packing, transfer and output handling; this is host elapsed time, not CUDA kernel occupancy.
- The GPU model cache already had **301918 hits over 301951 cumulative batches**, only 33 cold model uses. The 72-model bound is sufficient for the current 24-member history pool, learner, IL and retained evaluation references. Increasing that bound does not address this run's batching limit.
- The exported GPU counters show 0.767 GB live allocation and a 4.459 GB allocated / 6.858 GB reserved process peak. These are **process-wide peaks**, not isolated collection peaks. The lab used an RTX PRO 6000 Blackwell and 64 CPU threads.
- PPO took **73.89 seconds / 46 steps**. The major recorded pieces were head calibration 17.12 s, behavior/full policy scans 16.40 s, forward/backward 15.74 s, KL probes 14.04 s, initial preparation 4.62 s and critic preparation 3.62 s. Only about 0.13 s was optimizer stepping. The old CUDA path had two preparation threads; calibration also prepared batches serially.

This is a readiness/model-fragmentation and preparation problem, not evidence that a larger maximum batch would help. Per-model readiness and full-stream histograms were missing from old reports; those details cannot be reconstructed retrospectively.

## Implementation

`ppo_preparation.py` provides a persistent spawn pool, four workers by default, each restricted to one Torch thread. Eight workers and the prior inline/thread path (`0`) are also strictly validated options. The pool warms while collection runs. Tiny diagnostics below 2048 learner rows keep inline/thread preparation to avoid startup overhead.

Sampling, shuffles, RNG, optimizer steps, KL guards and gradient accumulation remain in the main process. Workers only collate CPU features/masks and exploration distributions. Results are consumed in their original order. The same path covers actor batches, critic batches, full behavior scans and ordinary head calibration; fixed-probe packed-input caching remains in place. Targets and recorded seat descriptors are untouched. Effective actor and critic batches remain 2048.

Iteration-local input arrays are staged once in a private mapped file under `artifacts/brass-ppo-*`; tasks pass small descriptors rather than entire datasets. Source seals are checked on reuse, and worker collation retains full selector admission. At most four output batches are in flight. Outputs are copied into owned buffers, optionally pinned for CUDA, and retired on consumption. An early-stopped scan cancels/joins unused prefetches and removes their outputs before the next scan; discarded preparation is reported separately. The context joins outstanding writers and deletes temporary files on normal return, failure or Ctrl+C. A hard OS kill can leave this temporary directory behind; it is unrelated to the checkpoint and never read on resume. The existing PPG-cache startup cleanup remains active.

Inference now reports batch-size histograms, per-model requests/batches/wait histograms and elapsed forward time, model loads/evictions, padding and cell-cap splits, and resident/registered model counts. Compact history retains the new counters. Per-collection counts and sums are deltas; per-model maximum wait is explicitly labeled as cumulative since service start. Forward durations use the high-resolution performance clock. Queue deadlines retain their existing clock.

Two optional collection controls were implemented and compared: `execution.padding_ratio` (default **2.0**) and `execution.model_cohorts` (default **false**). Cohorts only break ties after branch/long-work priority, using actual frozen model identities. They do not resample opponents, change seat rotations or mix weights. The benchmark rejected enabling them by default. The **1 ms flush, 64 batch cap, 131072 inference-cell limit, 24 processes × 2 lanes, 72-model cache and current CPU admission** remain unchanged. No extra game lanes or unmeasured CUDA-stream concurrency were enabled.

## Snapshot and entropy controls

The new `critic-rows/2.0.0` block changes only `rollout.reduced_snapshot_weight` from 0.5 to **0.0**. A reviewed resume retains the current saved weight, frozen R² reference and evidence. It does **not** forcibly rewrite `snapshots=1@0.50` to zero. Supported degradation can now reduce it to zero. Recovery proceeds **0 → 0.50 → 0.75 → 1.00** using the existing 20-commit window, ten qualifying iterations, whole-game minima, five new healthy commits, 0.02 degradation / 0.01 recovery tolerances, economic checks and KL safeguards.

At zero, snapshot rows are excluded from critic fitting so they cannot consume training slots or be reported as actual trained rows. Real learner critic work can qualify ordinary learner holdout evidence for recovery from zero. Above zero, actual retained snapshot training is still required. Branches and snapshot holdout rows never enter the rolling learner R². Reports include the minimum weight and explicitly identify zero-weight learner observations.

The new `entropy-floor-controller/4.0.0` uses, for each group:

```text
floor = original_baseline × max(group_minimum, initial_group_ratio − 0.02 × new_anchor_promotions)
initial_group_ratio = 0.95 for canal_action / rail_action
initial_group_ratio = 0.90 for canal_other / rail_other
group_minimum = 0.85 for canal_action / rail_action
group_minimum = 0.80 for canal_other / rail_other
```

There is **no iteration-based decay and no baseline reset**. The current saved baselines remain exactly: canal action **0.978153762348556**, canal other **0.37862535847684414**, rail action **1.3641296675490167**, rail other **0.4904388703717831**. Baseline samples, alpha values and controller windows are retained. The first future qualified promotion changes ratios to **0.93 / 0.88**, then **0.91 / 0.86**, **0.89 / 0.84**, **0.87 / 0.82**, and finally **0.85 / 0.80**. Each reduction is two percentage points of the original baseline, not a multiplicative 2% reduction of the current floor.

The signed config binds the existing four promotion receipts at g470 (references g260, g320, g340 and g440); none are counted retroactively. Fresh runs start with zero historical events. The controller counts only qualified promotion receipts from the LR gate's active policy-anchor mechanism, not rating slots, high scores or rejected candidates. Checkpoint counters and the signed history-prefix digest prevent duplicate/skipped reductions, including Ctrl+C after an anchor is published to `state.json` but before the next model checkpoint. The active IL/rated reference, PL origin, economic baseline, warmup, LR multiplier, GAE, population and lineage are unaffected. Epsilon stays 0.02, opening/fork temperature 2.0 and branching 0.5%.

## Bounded measurements

Local hardware: Windows, 14 logical CPUs, CPU-only Torch 2.8.0, two Torch compute threads. **CUDA is unavailable here. These results do not establish a lab GPU speedup.**

The native collection fixture uses eight identical seeded games per arm: four self-play, three history and one direct-v4 game, six in-flight lanes, four frozen current-size randomly initialized models, and two resulting branch continuations. This deliberately small mix is not the production 160/154/6 allocation. Models and actors were warmed; per-game seeds, model IDs, seat rotations and policies matched across arms.

| Collection setting | Wall seconds | Mean batch | Useful candidate cells | Host forward seconds |
| --- | ---: | ---: | ---: | ---: |
| Existing padding 2, 1 ms | 64.365 | 1.156 | 88.9% | 10.86 |
| Padding 3, 1 ms | 72.006 | 1.205 | 86.9% | 9.75 |
| Padding 4, 1 ms | 75.604 | 1.193 | 84.5% | 10.24 |
| Padding 3, 2 ms | 64.554 | 1.201 | 86.1% | 10.49 |
| Padding 3 + cohorts | 72.205 | 1.195 | 85.4% | 12.07 |

A reverse-order check isolated cohorts while retaining padding 2:

| Measurement | Existing order | Model cohorts |
| --- | ---: | ---: |
| Wall / games per second | **64.580 / 0.12388** | 75.573 / 0.10586 |
| Batch histogram: 1 / 2 / 3 positions | 860 / 167 / 12 | 924 / 141 / 8 |
| Mean queue wait | 14.34 ms | 13.51 ms |
| Useful candidate cells | 87.58% | 89.42% |
| Host inference forward / batch time | 10.51 / 11.11 s | 10.65 / 11.25 s |
| Mean process-tree CPU cores | 2.86 | 2.64 |
| Sampled process-tree RSS peak | 3.91 GB | 3.21 GB |
| In-phase model loads / evictions | 0 / 0 | 0 / 0 |

Both arms handled **1230 neural requests**: learner 1025, frozen opponents 68, 69 and 68. Complete model IDs, per-model histograms and wait bins are in the JSON reports. The queue bins place the 95th percentile at or below 32 ms. RSS includes shared pages and allocator history; it is not unique physical memory. The process-tree sampler was corrected to exclude CPU used by already-running workers before a timed phase. CPU totals from the first screen are therefore not used. Per-model forward durations from the early screen used a lower-resolution clock; use the aggregate high-resolution `inference.forward` values above. The shipping per-model timer is corrected.

Every native arm retained identical parent/branch moves and results. The reverse-order comparison's maximum behavior-probability difference was **2.93e-8**, within the 3e-6 tolerance. Wider batches alone did not improve collection wall time; no collection-throughput gain is claimed.

The PPO fixture uses the current **5928616-parameter** medium model, 2048 authored rows with 2–32 candidates, one bounded epoch and unchanged three-pass head calibration. It runs serial, four-process and eight-process modes with the same inputs, model, Adam and RNG. Process startup is warmed outside timing. All **nine** runs (three in each mode, forward then reverse order) produced **bit-identical model weights, Adam state, RNG, epoch metrics and KL decisions**.

| CPU preparation workers | First screen | Reverse-order repeats | Median of all three |
| --- | ---: | ---: | ---: |
| 0 | 12.549 s | 10.301 / 10.920 s | 10.920 s |
| **4 (default)** | 12.713 s | 10.369 / 10.580 s | 10.580 s |
| 8 | 10.823 s | 12.947 / 12.494 s | 12.494 s |

The four-worker median is about 3% lower, but order/warm-cache noise is larger than that difference. Treat this as correctness and feasibility evidence, not a proven throughput improvement. The reverse-order four-worker process tree peaked at **2.19 GB** versus 1.32 GB inline and 3.07 GB with eight workers. It used 1.47–1.52 average CPU cores, with collation spread across all four worker PIDs; this small CPU fixture is mostly model compute. Its 36 packed batches staged **92.7 MB** of inputs and wrote **376.9 MB** of total transient output, with a conservative **111.8 MB** in-flight output bound. Full production disk use scales with actual unique state/action inputs and is reported each iteration; this fixture is not a full-run capacity estimate.

Raw local results: `artifacts/g470-throughput/collection-screen5/report.json`, `collection-confirm/report.json`, `ppo-screen/report.json`, and `ppo-confirm/report.json`. Benchmarks use separate outputs and do not import or modify production training checkpoints.

An additional current-size PPO/checkpoint smoke completed in **7.26 seconds** (`artifacts/g470-throughput/ppo-resume-smoke/report.json`). It used 128 authored rows and verified that serializing/restoring the checkpoint produces a **bit-identical next PPO update**, including Adam, RNG and phase-controller state. Disabled PPG performed no steps and created no cache. This small smoke deliberately uses the inline preparation fallback; the separate 2048-row benchmark and process tests cover multiprocessing.

## Validation and lab commands

**205 focused checks passed; two CUDA-only checks were skipped.** The 207-case run is recorded in `artifacts/g470-focused-tests-passed.log`; its one test-fixture error used the old execution schema without `cpu_budget`. After switching that fixture to the current schema, the affected check passed in `artifacts/g470-metrics-tests.log`. Older resume fixtures were also updated to use their actual current-layout source and initialize the native sparse-label fixture; incompatible archived model/source combinations remain rejected. Coverage includes ordered process collation, masks/selectors and mutation rejection, early-stop cleanup, model routing and fairness, branch priority, PPO exact-state parity, strict config/source transitions, zero-weight recovery, preserved optimizer/controller state, qualified-promotion-only entropy reductions, exact clamps, pending-promotion restart continuity, and PPG-off GAE behavior. `git diff --check` passed with the repository's CRLF convention.

After syncing the revision, from the repository root on the lab machine:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 470 --apply && \
python scripts/joint7-throughput.py --expected-generation 470 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-throughput-g470.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This resumes at g471 and targets g490. Recovery archives only an uncommitted next iteration; the guard refuses a different committed generation. The helper reads the locked committed recipe, applies only the reviewed settings, writes the new config and leaves model/checkpoint files unchanged. The pre-generated config is bound to the supplied g470 export; conflicting local options are refused. Use a different `--output` if the committed recipe differs. On later clean continuations, repeat only the training command.

For a separate bounded GPU comparison, with an unused output directory:

```bash
PYTHONPATH=src:tests/Recovery:tests/R2:tests/FeatureEngineering \
python tests/Recovery/benchmark_collection_batching.py \
  --device cuda --games 32 --processes 24 --lanes 2 --cpu-budget 64 --threads 4 \
  --variants baseline padding3 wait2 cohorts2 \
  --output artifacts/g470-collection-gpu

PYTHONPATH=src:tests/Recovery:tests/R2:tests/FeatureEngineering \
python tests/Recovery/benchmark_ppo_preparation.py \
  --device cuda --rows 4096 --workers 0 4 8 --repeats 1 \
  --output artifacts/g470-ppo-gpu
```

These commands were **not** run on CUDA locally. They preserve matched conditions across arms and report GPU allocation peaks where available. No claim of improved playing strength follows from these implementation changes or small benchmarks.

Changed areas: `recovery/{ppo_preparation,updates,utility,league,config,resume,inference,inference_batches,execution,branch_scheduler,telemetry,history,exploration,critic_data,snapshot_r2,actor_value}.py`; the continuation helper/config and exact source-review receipts; focused tests and benchmarks; this report and the lab instructions. Concurrent rating-chart edits were preserved and do not alter saved PL fits or training behavior.
