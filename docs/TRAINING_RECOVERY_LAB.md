# Revised recovery training in conda

30 September 2026. This guide implements the [results 1.1 revision](../ai-research/TRAINING_REVISION_1_1_PLAN.md), building on the [learning recovery](../ai-research/TRAINING_RECOVERY_PLAN.md) and [throughput](../ai-research/TRAINING_THROUGHPUT_PLAN.md) plans. Keep your working **conda environment `ai-training`**, 64-CPU allocation and private rules runtime. All lab paths below start at `/data/not_backed_up/yxu209/ai-training/`.

**Use a new run directory.** Recipe `step5-training/2.1.0` changes actor weighting and population management. Initialize compatible weights with fresh optimizer moments; do not bypass an old source signature. Preserve `results/1.0`, `results/1.1` and the original three-second v3 bank. This update does not install anything into the laptop game. Strength, RTX throughput and one-second lab equivalence still require lab measurements. See [local exit checks](TRAINING_REVISION_1_1_EXIT_CHECKS.md).

## Defaults

| Area | Revised default |
| --- | --- |
| Collection | 320 games/round; 32 spawned ordinary actors, one private host each |
| Heavy pool | 8 concurrent games × 4 private search workers; global cap 32; ordinary/heavy pools overlap before PPO |
| Alternative | `lab-16x2.json`: 16 heavy games × 2 workers, separately calibrated |
| Lab v3 | One-second whole-turn target; 720 aggregate reconstruction tokens (3 × the parent's 240), divided across root partitions; mandatory incumbents can exceed the token target and are measured |
| Reference/laptop v3 | Original frozen 3-second profile and 240-token allowance |
| Inference | Same-model batches up to 64; 1 ms partial flush; 131,072 padded candidate cells |
| Replay | Up to 8 games / 256 MiB uncompressed estimate per shard; smaller packets preserve concurrency on short collections; hash-bound derived features |
| BC | Effective batch 1,024; memory microbatch 512; LR `1e-4`; up to 10 epochs, patience 2 after at least 3 |
| PPO | Nonforced actor batch 2,048; independent critic sample up to 2,048 from all learner states; microbatch 512; two epochs; LR `3e-5`; KL guard `0.01` |
| Population/storage | At most 8 recent + 4 anchor weight-only opponents; 2 full resume checkpoints; 4 replay rounds; BC best/last; 4 development shards; 2 profiler traces |
| Evaluation | 8-game monitor every 1,600 ordinary games; 16-game broader screen every 6,400; formal shortlist panel 480 games |
| Main teachers | S2 after 640 ordinary games, then every 1,600, subject to roughly 10% cumulative wall-time allowance; four training-only games |
| Additional R4/R5 | 24-game development screen; initially 8 independent S3 teacher games |

An effective batch controls optimizer steps; a microbatch controls memory. Forced single-choice states no longer dilute actor advantages, policy loss or actor sampling. They remain eligible for four-seat critic targets. The bounded critic sampler need not process every row in a round. Losses/step counts are not directly comparable with recipe 2.0. Rehearsal reports its actual active state; a null manifest means **off**. Reward remains terminal four-seat win share; shaping is an unenabled conditional ablation.

## 1. Transfer and check

The Windows checkout includes child-bank sources/profiles in `training/opponents/lab/current.json` and an authored `training/runtime/v3-lab/BrassLegacyAI.exe`. Its runtime/dependencies must also arrive in the updated private ZIP. On the Windows project machine, export with `scripts/step5.py --root <absolute-local-project> assets`, then transfer the ZIP outside Git. The child was built against the pinned research rules runtime; do not rebuild against an arbitrary newer Steam UI DLL or edit the parent's qualification reports.

```bash
conda activate ai-training
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py import-runtime \
  --bundle /data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime.zip
python /data/not_backed_up/yxu209/ai-training/scripts/check_qualification_reports.py
python /data/not_backed_up/yxu209/ai-training/scripts/verify-recovery.py
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda preflight
```

There are no new mandatory Python packages. Optional `verify-recovery.py --integration --device cuda` checks a small real-host collection/audit/BC/train/resume/evaluation workflow; it is not a strength tournament.

## 2. Calibrate the shorter teacher

The child is **not admitted until measured on the lab**. This command runs 25 complete four-player legality/economy games and early/late Canal/Rail position probes against the unchanged three-second parent on the same hardware. It writes immutable raw evidence and a source/hardware/budget-bound report. Predeclared margins require at least 80% aggregate reference reconstruction count, at most 1 heuristic VP loss per probe, no worse projected cash deficit, no debt decisions, and the probe's one-second elapsed target. Counts include duplicate mandatory incumbents; also inspect unique-plan coverage. Passing certifies engineering margins, not equal playing strength.

```bash
RECOVERY_CONFIG=/data/not_backed_up/yxu209/ai-training/configs/step5/lab.json
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --config "$RECOVERY_CONFIG" lab-calibrate --games 25
```

Check `training_eligible` in `/data/not_backed_up/yxu209/ai-training/artifacts/search-budgets/lab-v3-calibration.json`. Failure cannot admit the shorter teacher. For immediate training with the unchanged qualified control, explicitly use:

```bash
RECOVERY_CONFIG=/data/not_backed_up/yxu209/ai-training/configs/step5/lab-reference.json
```

For 16 × 2, select `/data/not_backed_up/yxu209/ai-training/configs/step5/lab-16x2.json` and run its separate calibration. Choose before collecting data or starting training. A changed clock, workers, token multiplier, source or CPU allocation requires new provenance/calibration. Existing calibration outputs are immutable; subsequent attempts need a new output/config path. Laptop play retains the original allowance.

## 3. Generate and audit demonstrations

Start with 1,000 games across all five training profiles. Original Hard is not a teacher. Pottery Flex's **opponent identity** remains held out; Pottery actions, exploration and learner contexts remain available.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --config "$RECOVERY_CONFIG" demo-generate \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/revision11-v3 --games 1000 --seed 17100000

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --config "$RECOVERY_CONFIG" demo-audit \
  --replays '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/revision11-v3/*.jsonl.gz' \
  --output /data/not_backed_up/yxu209/ai-training/training/datasets/revision11-v3.json
```

Generation resumes exact-spec verified shards. Eight independent audit processes reproduce observations, complete legal lists, selected actions and terminal results; the parent commits deterministic global deduplication and setup-family partitions, approximately 80/10/10. Inspect exclusions and profile/era/action coverage. Test demonstrations remain untouched. Existing **training** replays can be re-audited into a fresh manifest if identities, rules and seed provenance match. Diagnostic qualification/evaluation traces cannot become BC labels. Increase to 2,000–5,000 games only if coverage or held-out curves justify it.

Validation reuses exact checked content and unchanged history events. Mutations/new branches are checked independently. Feature receipts bind raw replay SHA, encoder/contract version, row context and array SHA; unsigned foreign caches are ignored. BC derives a separate neutral predecision context array from the pinned feature layout, with bitwise encoder parity, so it never learns the legacy teacher's post-choice intent as an input. Audited dataset manifests protect their raw sources and derived caches against cleanup.

## 4. Imitate and check play

Use an existing compatible committed self-trained checkpoint or the fresh medium control shown here. All legal choices remain in expert cross-entropy; external actions never become PPO likelihoods. Training balances teacher/era/action bins with a threefold cap. Validation reports natural/balanced loss, nonforced top-1, root-action/family accuracy and teacher/era/phase/action bins. Stop on held-out stagnation; compare 3- and 10-epoch caps in separate runs if overfitting appears.

```bash
RECOVERY_INITIAL=/data/not_backed_up/yxu209/ai-training/training/models/bootstrap/r2-fresh-medium.pt

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config "$RECOVERY_CONFIG" imitate \
  --manifest /data/not_backed_up/yxu209/ai-training/training/datasets/revision11-v3.json \
  --initial "$RECOVERY_INITIAL" --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-bc \
  --epochs 10 --min-epochs 3 --patience 2 --learning-rate 0.0001 --batch-size 1024

BC_INITIAL=$(python /data/not_backed_up/yxu209/ai-training/scripts/resolve_checkpoint.py \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-bc --select best)

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate \
  --candidate "$BC_INITIAL" --references "$RECOVERY_INITIAL" \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-bc-monitor
```

`resolve_checkpoint.py` returns the immutable best epoch, avoiding a mutable `best.pt` initialization. This monitor is **2 setups × 4 rotations × 1 recent-neural lineup = 8 games**, with no legacy/Pottery Flex matchup. It cannot promote. Retain an unchanged initialization/pre-BC control. Review VP, placement, sales, income recovery and industry coverage; imitation loss alone is insufficient.

## 5. Train and resume

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config "$RECOVERY_CONFIG" train \
  --initial "$BC_INITIAL" --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-main --iterations 20

# Same source/config; omit initialization on resume.
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config "$RECOVERY_CONFIG" train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-main --iterations 20
```

Twenty medium-preset iterations mean 6,400 ordinary games, not a strength guarantee. V3 **table** shares remain 40/20/10/5% at 0/2,000/5,000/10,000 new ordinary games; self-play and historical neural tables take the rest. Mature 5% v3 tables mean 3.75% v3 seats. BC, teachers and evaluations never advance this counter. The incoming policy stays frozen throughout overlapping ordinary/heavy collection; both pools finish before updating it.

Inspect load/preparation/optimizer timings, nonforced actor counts, actual critic positions, clip fraction and phase-balanced KL probes. Each epoch closes with a full nonforced KL check. Tune LR `1e-4` vs `3e-5`, entropy `0.01` vs `0.005`, and batch 1,024 vs 2,048 in separate runs. Do not increase epochs when KL/economics regress. Optional rehearsal needs a real `rehearsal_manifest`; its separate loss decays over 10,000 games and never enters the PPO ratio.

`--from-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main` imports committed weights, available opponents and setup cursor into a **new** recipe directory with fresh optimizer moments/new curriculum count. `recovery-v2-lab.json` preserves the 2.0 config for controls; it does not bypass recorded source signatures. `previous-lab.json` remains the independent historical Reptile/meta-RL recipe.

## 6. Smaller evaluation

The league persists completed panel slots, ordinary-game count, checkpoint hashes, seeds and reports. Resume finishes a due panel before collecting the next round and does not repeat completed slots. Default automatic panels are 8 games every 1,600 and 16 every 6,400 ordinary games. Broader panels wait for two distinct frozen references; evaluation traces never enter gradients.

Manual `/data/not_backed_up/yxu209/ai-training/configs/step5/broader-evaluation.json` uses two references and **2 × 4 × 2 = 16 games** (`mixed-neural`, `mixed-v3`). Select the matching admitted lab budget, or change its role to `reference` in a fresh config. `--lineups` explicitly selects names; `--setups` overrides setup count. A separate monitor output with `--focus Pottery` checks industry-conditioned play.

Resolve available weight-only opponents rather than assuming an old full generation survived retention:

```bash
MAIN_CANDIDATE=$(python /data/not_backed_up/yxu209/ai-training/scripts/resolve_checkpoint.py \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-main --select latest)
REF_OLD=$(python /data/not_backed_up/yxu209/ai-training/scripts/resolve_checkpoint.py \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-main --select oldest-opponent)
REF_RECENT=$(python /data/not_backed_up/yxu209/ai-training/scripts/resolve_checkpoint.py \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-main --select latest-opponent)

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda evaluate \
  --candidate "$MAIN_CANDIDATE" --references "$REF_RECENT" "$REF_OLD" \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-shortlist-full
```

Run the formal default only for shortlisted candidates: **30 setups × 4 rotations × 4 lineups = 480 games**. It retains the original v3 control and held-out Pottery Flex, two genuinely trained distinct references and all raw evidence. A BC/initialization anchor cannot satisfy that promotion gate; replace `REF_OLD` if it is still such an anchor. Reduced lineups cannot promote. Final `--partition test` stays separate and cannot authorize promotion.

## 7. Storage and restoration

Automatic pruning follows a durable checkpoint/state commit. It protects resume/population weights, BC best/last, downstream dataset/model manifests, unfinished jobs, formal evidence and explicit pins. Retired lineage remains metadata, not the sampling pool. BC aliases use hard links where possible; population copies omit optimizer state. Model registries and actor policy caches are bounded or evicted after jobs finish.

Inspect a dry run before the first production collection. Only a specific owned `artifacts/` run is accepted; linked/escaping targets and active-run cleanup are rejected.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-main

# Apply after inspecting paths/reasons.
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-main --apply

# Use the exact manifest printed by cleanup, before its grace expires.
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py restore \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/step5/revision11-main/.retention/cleanup-REPLACE.json
```

Quarantine lasts **24 hours**. Actual disk reclamation occurs after grace plus a later successful commit, or explicit `cleanup --apply --purge` after grace. Manifests record intended moves before moving, file hashes, bytes and exact restore paths; restore refuses overwrites. `pin --run <absolute-owned-run> --paths <absolute-file> ...` protects selected ablations. Pin while training is stopped before starting downstream experiments. Set `retention.enabled: false` in a new config/run to disable automatic cleanup. Dataset pins override the four-round limit.

Development summaries retain at most four sample shards and mark `raw_records_pruned`; repeating the exact command returns a cached summary, not fresh throughput. Formal raw evidence remains intact. No cleanup command targets game files or `results/1.1`.

## 8. Measure CPU/GPU work and run R4/R5

Compare fresh ordinary-only, v3-only and overlapping pools using equal seeds. Pure legacy generation/audit has no useful GPU work. Separate load/preparation from update time. Inference host timers are not CUDA kernel duration. Reports include first-call/steady mean/p95/maximum, candidate cells, cache use, affinity and memory. Optional `profile_steps: 4` in a fresh config/run produces bounded traces. FP32 remains the control; BF16 needs separate CUDA parity/throughput evidence.

```bash
mkdir -p /data/not_backed_up/yxu209/ai-training/artifacts/performance/revision11
timeout 120s nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv -l 1 > /data/not_backed_up/yxu209/ai-training/artifacts/performance/revision11/gpu.csv
```

Supply enough games for all actors. Compare 16/32 ordinary actors and 8 × 4 vs 16 × 2 heavy actors; additional concurrency can reduce timed per-turn work even when throughput improves. Optimize games/updates per hour, not filling 90 GB. Conditional microbatch/candidate-cell/queue-delay changes need measured throughput and latency.

Independent R4/R5 now uses `searchlab-experiment/1.3.0`: 8 concurrent games × 4 private search workers, a one-second lab target, 128 **aggregate** tree simulations, and a 24-game screen. S2 distributes fixed proposals/matched worlds; R4 merges compatible public-root visits only; S3 uses four-seat backups, private observation trees and fresh matched confirmation. One deadline covers both actions. Cold or blocked RPCs may overshoot; overshoots are visible and fail timing promotion. Timeout cannot invent positive-confidence teachers. If one second yields few confirmations, compare an explicitly identified three-second budget on the lab before using teachers.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-searchlab.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/searchlab/experiment.json --workers 3 --seconds 1
python /data/not_backed_up/yxu209/ai-training/scripts/verify-searchlab.py --integration --device cuda

python /data/not_backed_up/yxu209/ai-training/scripts/r4r5.py --device cuda compare \
  --checkpoint "$MAIN_CANDIDATE" --references "$REF_RECENT" "$REF_OLD" \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/searchlab/revision11-screen
python /data/not_backed_up/yxu209/ai-training/scripts/r4r5.py --device cuda distill-s3 \
  --checkpoint "$MAIN_CANDIDATE" \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/searchlab/revision11-s3-teachers
```

S3 collects eight train-partition games and updates a separate copy; main training remains S2/self-play. Promotion requires fresh correctness and **30+ paired setups, one uncontended concurrent game, equal wall time and equal within-turn CPU allowance** (`--workers 1 --setups 30 --search-workers 4`). One game worker does not mean one search replica. Laptop experiments use `configs/searchlab/laptop.json` (one game, one search worker, three seconds). See [R4/R5 details](R4_R5_SEARCH_LAB.md).
