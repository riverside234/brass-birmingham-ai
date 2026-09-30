# Recovery training and shared CPU/GPU execution

The revised defaults implement the [learning recovery plan](../ai-research/TRAINING_RECOVERY_PLAN.md) and [throughput plan](../ai-research/TRAINING_THROUGHPUT_PLAN.md). Use the existing **conda environment `ai-training`**. All lab paths below start at `/data/not_backed_up/yxu209/ai-training/`. The working private rules/v3 runtime is still required; this revision does not replace it or change the installed game.

This is a new training recipe, not a continuation with the old optimizer. Preserve the old run, reports and frozen checkpoints. Local checks establish wiring, legal four-player games, gradients and resume behavior. Playing strength and RTX throughput must be measured on the lab.

[Executed checks](TRAINING_RECOVERY_EXIT_CHECKS.md) record 88 passing focused/regression checks, ten complete four-player games and the final three-worker search smoke, including their source and timing limits.

## What changed

| Area | Revised default |
| --- | --- |
| Training config | `configs/step5/lab.json`, `step5-training/2.0.0` |
| Warm start | Audited current-v3 demonstrations, separate full-legal-choice imitation loss |
| BC update | Effective batch 1,024; memory microbatch at most 512; LR `1e-4`; max 10 epochs, patience 2 after at least 3 |
| PPO update | Effective batch 2,048; microbatch at most 512 / 131,072 padded candidate cells; two epochs; LR `3e-5`; post-update KL guard `0.01` |
| Ordinary games | Eight spawned Python processes, four private host lanes each: up to 32 live games |
| v3 / S2 games | Eight private process/planner lanes, one game per process |
| Inference | One coordinator service; immutable model cache; same-model batches up to 64, 2 ms partial flush |
| Evaluation | Shared actors/inference; development panel excludes held-out Pottery Flex; full panel retains it |
| R4 / R5 | Shared inference plus R4 permitted-input prior/value cache; optimized confirmed S3 distillation |

An **effective batch** controls optimizer-step frequency. A **microbatch** controls memory. The new PPO code steps after each effective batch, not once after accumulating the whole collection. For 77,095 eligible learner decisions, 2,048 gives up to 76 steps over two epochs, before KL stopping. Do not compare old and new raw actor losses as if their normalization were identical.

Five qualified v3 profiles participate evenly in new demonstration/curriculum tables. Original Hard is not a teacher. The Pottery Flex **opponent identity** stays held out; Pottery actions remain legal, receive balanced exploration, and have a learner focus context.

## Verify the updated checkout

Transfer the new authored code/configs to the lab first. Keep the already working environment and imported private runtime. There are no new mandatory Python dependencies for the recovery path.

```bash
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/check_qualification_reports.py
python /data/not_backed_up/yxu209/ai-training/scripts/verify-recovery.py
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda preflight
```

The small real-host recovery check is optional on the lab after the focused checks:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/verify-recovery.py --integration --device cuda
```

It collects two complete v3 demonstration games, audits their transitions, performs a short BC update, trains/resumes four curriculum games including mixed v3 opposition, and evaluates four rotated games. It is not a strength qualification. Results are in a fresh directory under `artifacts/recovery/`.

## 1. Collect and audit imitation data

Start with 1,000 new complete four-player v3 games. Generation uses the existing deterministic setup-family split: approximately 80% train, 10% validation and 10% test. It mixes all five admitted profiles and rotates seats. Eight independent legacy lanes preserve each frozen planner's clock.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-generate \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/recovery-v3 \
  --games 1000 --seed 17100000

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-audit \
  --replays '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/recovery-v3/*.jsonl.gz' \
  --output /data/not_backed_up/yxu209/ai-training/training/datasets/recovery-v3.json
```

Generation can resume with the same command/spec and verified completed shards. Audit restores the current rules host, reproduces every permitted observation and complete choice list, applies every selected action and checks the terminal result. It records hashes, labels, partitions and exclusions in an immutable manifest. Inspect its `partitions` and `excluded` fields.

To reuse older data, give `demo-audit` additional explicit **training** shard patterns, using a fresh manifest filename. Evaluation/qualification/diagnostic games, changed rules, old banks, held-out opponents, incomplete traces and missing reproducible setup seeds are rejected. Aggregate score reports cannot supply imitation labels. Duplicate semantic games are removed; setup families cannot cross partitions.

Keep test demonstrations untouched. Only increase to 2,000–5,000 games if development curves or coverage justify it. The sequential transition audit is intentionally strict and may itself take time; it is separate from game-generation throughput.

## 2. Warm-start a compatible checkpoint

Prefer the most recent completed checkpoint from the existing self-trained run. The filename below is an example: set `RECOVERY_INITIAL` to a file that exists. The fresh compatible control is `training/models/bootstrap/r2-fresh-medium.pt`.

```bash
RECOVERY_INITIAL=/data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0001.pt

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda imitate \
  --manifest /data/not_backed_up/yxu209/ai-training/training/datasets/recovery-v3.json \
  --initial "$RECOVERY_INITIAL" \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-bc \
  --epochs 10 --min-epochs 3 --patience 2 --learning-rate 0.0001
```

BC uses causal permitted inputs with neutral context, not the legacy planner's post-choice intent. Expert actions receive cross-entropy over **all** legal alternatives and four-seat value supervision. Their external behavior probability is never a student PPO likelihood. Training weights balance profile/era/choice family with a threefold cap; validation reports natural and balanced losses plus nonforced accuracy. Stop on held-out loss stagnation; do not add epochs just to imitate the teacher more exactly.

`best.pt` is the development-loss candidate; checkpoint state supports exact-spec resume. Before accepting it for main learning, run a fresh development comparison against the unchanged initialization:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-bc/best.pt \
  --references "$RECOVERY_INITIAL" \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-bc-development
```

The eight-setup development panel has four seat rotations and two lineups with one reference: 64 games, including mixed admitted-v3 opposition. Review VP, placement, productive builds/sales and industry coverage alongside imitation loss. This panel cannot promote a model and does not use Pottery Flex. A three-epoch control run needs a different run directory.

## 3. Run revised self-play

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --initial /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-bc/best.pt \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-main \
  --iterations 20

# Continue with the exact same configuration/source; omit initialization.
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-main \
  --iterations 20
```

Twenty iterations collect 6,400 ordinary games at 320 per iteration; this is a budget, not a strength guarantee. Review fresh development results early, then near curriculum boundaries. Collection outcomes describe the **incoming** model; the report identifies the saved **post-update** model separately.

| New completed ordinary games | Self-play tables | Historical neural tables | v3 tables |
| --- | ---: | ---: | ---: |
| 0–1,999 | 40% | 20% | 40% |
| 2,000–4,999 | 55% | 25% | 20% |
| 5,000–9,999 | 65% | 25% | 10% |
| 10,000 onward | 70% | 25% | 5% |

The scheduler smooths exact quotas over 20-table cycles and persists counters on resume. At a phase boundary inside a collection, tables follow their individual global indices. Historical/v3 tables contain one learner and three independent frozen opponents; 5% v3 **tables** means 3.75% v3 seats. Demonstrations/evaluation do not advance this counter.

To compare revised PPO without imitation, use a **different new run** and `train --from-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main`. It imports committed compatible weights/history and setup cursor, records ancestry, starts fresh optimizer moments, and starts the new curriculum count at zero. Do not run recipe 2.0 in the old directory. Old runs retain their original schema through `configs/step5/previous-lab.json`, subject to their recorded source signature.

Optional forgetting control is a separate BC rehearsal loss. In a new config copied from `lab.json`, set `rehearsal_manifest` to the audited manifest's absolute lab path; the default coefficient is 0.1, decaying to zero over 10,000 ordinary games. The default manifest is null, so rehearsal is off. Compare against no rehearsal. Never insert expert rows into the PPO ratio.

## 4. Evaluation, diagnostics and tuning

Repeat the development command with a completed recovery generation and fresh output directory. Use the unchanged parent and two distinct completed neural generations as references once available. Then evaluate shortlisted checkpoints with the default full panel, including the held-out Pottery Flex matchup:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-main/generation-0020.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-main/generation-0010.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-main/generation-0005.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-g20-full
```

The full 64-setup, four-rotation, four-lineup panel is 1,024 games with two references. Development selection and final test remain separate. Full promotion retains multiple trained-reference, clustered-confidence and Pottery coverage gates; BC checkpoints do not masquerade as independently trained PPO opponents. Also run `--focus Pottery` in a separate development output.

Inspect `iteration-*/report.json`: separate table-composition VP/wins/placement, action/phase frequencies, build/sale/loan/pass patterns, forced choices, cash/income, feasible sales, observed flips, merchant beer and bonuses; actor/value loss components, normalized advantage statistics, clip fraction, gradient norm, post-update KL and actual optimizer steps. Last-observed unflipped tiles are explicitly a **proxy**, not exact end-of-era waste or tile VP attribution. The report does not claim those missing per-tile endpoint labels.

Tune one dimension at a time in fresh recipe directories: LR `1e-4` versus `3e-5`, effective batch 1,024 versus 2,048, entropy 0.01 versus 0.005. Only increase epochs from two to three if KL stays stable and held-out play improves. If economic competence regresses, diagnose/roll back; increasing game count or GPU memory does not fix it. Keep optional Reptile/meta-RL outside this recovery comparison; its explicit historical config is `configs/step5/previous-lab.json`.

## 5. Measure the CPU/GPU pipeline

Reports contain actor host/encoding/policy timers, replay publication time, inference queue/pack/forward/transfer timings, actual/padded candidate cells, worker counts, effective update work, CPU affinity/thread settings and CUDA peak memory. Actor lanes persist between games; completed shards are published through bounded writers before a round commits. Models/choices/seat/revision/context are checked when routing replies. Ordinary collection, heavy v3 collection and learning run as separate phases; pure legacy generation naturally does not use the GPU.

For a short trace, copy the config to a **new filename** with `profile_steps: 4` and use a **new short run directory**. Profiling adds overhead; compare unprofiled throughput separately. Collection and PPO traces appear in `profile-collection/` and `iteration-*/profile-ppo/` respectively. Preserve FP32 initially. `update.precision: "bf16"` is a separate CUDA capability/parity experiment, not an automatic speed promise.

While collection and PPO run, sample the same intervals externally:

```bash
mkdir -p /data/not_backed_up/yxu209/ai-training/artifacts/performance/recovery
timeout 120s nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv -l 1 > /data/not_backed_up/yxu209/ai-training/artifacts/performance/recovery/gpu.csv
```

Benchmark 16/32/48 live games by changing `execution.processes` and `hosts_per_process` in separate configs, keeping one CPU tensor thread per actor and the same population/seeds. Supply enough games to occupy every slot. Try `max_candidate_cells` 262,144 or a longer batch wait only if measured work/sec improves without search latency regressions. Report cached games separately from fresh throughput. The goal is productive games/updates per hour, not filling 90 GB of VRAM.

## 6. S2 and the additional R4/R5 branch

Regular recovery learning uses shared-inference S2 teachers after 640 ordinary games, then every 1,600 games, with four new training-only teacher games at each due boundary. The counter persists across resume. Only fresh-confirmed, covered alternatives carry positive-confidence distillation; incomplete search cannot invent a teacher. This limited schedule does not turn the main run into S3 training.

The required additional R4/R5 implementation uses the same central inference service in `configs/searchlab/experiment.json` version 1.2.0. R4 combines/caches prior and value for identical **permitted** observations, context and complete choices. S3 distillation uses the optimized supervised updater in a separate run. A framework change alone does not increase GPU utilization.

```bash
# Three decisions only; exercises three spawned search workers and shared inference.
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-searchlab.py \
  --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/searchlab/experiment.json \
  --seconds 1 --workers 3

# Refresh the machine/device-specific gate before full comparison or teachers.
python /data/not_backed_up/yxu209/ai-training/scripts/verify-searchlab.py --integration --device cuda
```

See [the R4/R5 guide](R4_R5_SEARCH_LAB.md) for matched S2/R4/S3 comparisons and independent S3 distillation. Code changes invalidate old correctness records. Parallel search is screening evidence; promotion still requires uncontended one-worker equal-budget comparisons with at least 30 independent paired setups. Queue wait and inference count against search time, and overshoots remain visible. Preserve the parent checkpoint and main learning state throughout.
