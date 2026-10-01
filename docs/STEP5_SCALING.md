# Larger Step 5 experiments on the lab server

**Archived pre-2.2 instructions:** use [Fresh Masked ISMCTS training](TRAINING_RECOVERY_LAB.md) for current commands. This document records earlier recipes and checkpoint-import/search workflows; do not run those commands for the new random residual-model run.

**Superseded default, 30 September:** follow [TRAINING_RECOVERY_LAB.md](TRAINING_RECOVERY_LAB.md) for the new imitation/minibatch/curriculum and central-inference recipe. This document preserves the original scaling experiment; its flat training schema is now `configs/step5/previous-lab.json`. Recipe 2.0 intentionally starts fresh optimizer moments in a new run.

Updated 29 September 2026 for the reported 64 logical CPUs (`nproc`), approximately 90 GB of GPU memory and 755 GiB of system RAM. This scales the **current Step 5 self-training**, before the later personality/difficulty stage. Use the existing conda environment `ai-training`.

Evaluation has its own [larger preset and parallel scheduling guide](STEP5_EVALUATION_SCALING.md): 64 setups with four rotations, 32 neural simulators and eight independent legacy/search game workers. That follow-up source update also changes the trainer implementation signature; the guide explains continuing an already-started run with `--from-run`.

## What changed

The new default [lab.json](../configs/step5/lab.json) is a larger preset. The previous settings remain in [lab-small.json](../configs/step5/lab-small.json); [smoke.json](../configs/step5/smoke.json) remains a small correctness check.

| Setting | Previous main preset | New main preset | Meaning |
| --- | ---: | ---: | --- |
| `games_per_iteration` | 20 | **320** | Completed four-player ordinary games collected before each PPO update phase. This is the `num_games` equivalent. |
| `workers` | 4 | **32** | Concurrent ordinary game simulators. This is the `num_workers` equivalent, not PyTorch DataLoader workers. |
| `parallel_hosts` | Serial requests | **true** | Run independent rule processes' reset/observe/choices/apply requests concurrently. Each individual host still processes one command at a time. |
| `group_tables` | Consecutive table types | **true** | Fill worker batches across the iteration's table schedule, preserving its exact opponent quotas. |
| `batch_size` | 32 | **64** | Maximum positions per neural inference forward pass. Actual batch size is limited by ready games of the same policy and padded candidate capacity. |
| `minibatch_size` | Used `batch_size`: 32 | **512** | Maximum decision rows per training forward/backward pass, independent of inference batching. |
| `max_candidate_cells` | Inference cap 8,192; training unbounded | **131,072** | Maximum padded position-by-action cells per inference/training microbatch. Large legal-choice sets reduce actual batch size. |
| `search_games` | 2 | **8** | Additional S2 teacher self-play games when search is scheduled. |
| `search_workers` | 1 shared sequential search policy | **8** | Independent search-game workers, each owning its own policy copies, private contexts and simulator. |
| `distill_steps` | 4 | **8** | Gradient steps on confirmed teacher examples per scheduled distillation phase. |
| `threads` | 4 | **4** | PyTorch CPU intra-op threads, not the game simulator count. |
| `ppo_epochs` | 3 | **3** | Full-data PPO passes/optimizer steps under the existing accumulated-gradient algorithm. |
| `learning_rate` | 0.0001 | **0.0001** | Kept for an interpretable scaling comparison. |

Keep the medium trained backbone, four-player rules, legal masks, Pottery availability and independent seat objectives. Every 320 regular games contain **224 current self-play, 80 historical-neural and 16 qualified v3 tables** under the default mixture. The held-out Pottery legacy policy remains excluded. S2 teacher games are additional neural self-play, not legacy imitation. This does not turn Step 5 into S3 training.

## Why larger numbers alone were insufficient

Previously the coordinator waited for each rule process before querying the next. Multiple processes existed, but much of their work was serial. In addition, consecutive table types limited a batch to at most 14 self-play, five historical or one legacy game, regardless of a larger `workers` setting.

The new Step 5 collector runs independent host requests concurrently and leaves GPU inference grouped on the coordinator. Table bucketing removes that 14-game ceiling. The original collector is byte-pinned by the qualified v3 bank, so it remains intact; the new parallel adapter lives separately in [step5/collector.py](../src/brass_ai/step5/collector.py).

Ordinary collection has up to 32 live game hosts. A legacy batch can also have three frozen-opponent processes. The search phase follows PPO and uses up to eight independent search-game workers, each with a live-game host and a reconstruction/search host: roughly **16 rule processes**, not 32 plus 16 running together. Search policy/RNG/private-tree state is never shared between workers. Within each search worker, rollout decisions remain sequential; this is game-level parallelism, not a distributed MCTS service.

The learner remains frozen for an entire ordinary collection, and again for the entire teacher collection. Gradients begin only after their corresponding collection has finished. A worker failure aborts that collection; incomplete games do not acquire terminal targets.

Using only 800 MB of GPU memory is not itself an error for an approximately 1.9M-parameter model. Rules simulation, Python encoding, JSON transport and legal enumeration can limit GPU input. Aim for **more valid games and learner decisions per second with useful learning**, rather than filling 90 GB. GPU memory usage and GPU compute utilization are different measurements. This preset is a starting point for the reported hardware, not a measured server speedup.

## Continue the existing trained run with the larger configuration

Allow the current iteration to finish, then stop the trainer before updating its checkout. Transfer/pull these code and config changes into the lab project. Run migration reads the last **committed** checkpoint from `state.json`; it does not migrate partial collection from an interrupted iteration.

```bash
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/verify-step5.py

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/lab.json train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled --from-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main --iterations 2
```

`main-scaled` must be a new empty directory. Replace the source run path if your existing run has a different name. `--from-run` carries forward:

- Last committed model weights, optimizer moments and training RNG state.
- Historical neural opponents, by their immutable checkpoint references.
- Next unused setup seed and cumulative ordinary-game count.
- Meta-task exclusions and an origin record identifying the source run/checkpoint.

The new config supplies the learning rate and collection/update settings. Model/rules compatibility and source-checkpoint hashes are checked. New iteration numbering starts at zero, so the search schedule also restarts: by default it first runs at new zero-based iteration 10. The original run stays available; retain its older checkpoint files because the new history still references them.

After the first two scaled iterations, inspect their reports. Resume the new run with the same config and code, omitting `--from-run`:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/lab.json train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled --iterations 8
```

That reaches ten new iterations, or 3,200 additional ordinary games. An additional five iterations reaches 15 and includes the first search phase. Evaluate progress before making a long extension.

If there is no prior trained run, use `--initial` with the existing bootstrap checkpoint and a new directory instead. `--initial` imports weights with fresh optimizer/history/seed state; `--from-run` is preferred for scaling an already trained run. They are mutually exclusive.

An ordinary resume of the old run with the new config/code is deliberately rejected. Even selecting `lab-small.json` does not bypass the source signature. To compare the old workload under the new implementation, use a new directory with `--from-run` and `lab-small.json`.

## Measure the bottleneck by phase

Each iteration report now includes:

| Report field | Interpretation |
| --- | --- |
| `timings.collection_seconds` | Ordinary host collection, inference, replay writes and opponent setup. |
| `timings.ppo_seconds` | Preparation, exact behavior-probability verification and gradient passes. |
| `timings.search_and_distill_seconds` | Additional teacher games and distillation; near zero when not due. |
| `throughput.regular_games_per_second` | Ordinary completed games divided by collection time. Measure fresh iterations; reused cached games after a restart inflate this figure. |
| `throughput.learner_inference` | Learner forward-pass count, position count, maximum actual batch, padded candidate cells and total predict time. Historical/search policy activity is separate. |
| `actor.microbatches_per_epoch`, `actor.max_batch_positions` | Actual training batch behavior after candidate-memory splitting. |
| `actor.optimizer_steps`, `actor.learner_decisions` | Update/data volume; essential when comparing different games-per-iteration values. |
| `cuda_memory` | Allocated/reserved and peak allocated PyTorch GPU bytes for the iteration, when using CUDA. This is not total device use by all applications. |
| `search.collection_workers` | Actual independent teacher workers used when search runs. |

The console reports `concurrent_games` for each ordinary batch and announces the search phase. Whole-iteration elapsed time excludes final checkpoint/report serialization, so use wall-clock run time too for storage-heavy jobs.

Useful server observations during a fresh iteration:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 2
```

Check `top` during ordinary collection **and** during PPO/search. GPU use will fluctuate because collection and updates are separate phases. If ordinary collection dominates with one busy Python core and idle native hosts, Python feature construction/transport is still limiting throughput; larger GPU batches cannot create ready work. If training dominates and candidate cells repeatedly cap the batches, increasing that bound may help.

## Tuning after the first measured run

Change one bottleneck-related group at a time and fork into another new run. Keep the starting weights and benchmark workload comparable when measuring speed; a 320-game iteration will naturally take more time than a 20-game iteration even if games/second improves.

| Observed limitation | Next setting to try |
| --- | --- |
| Ordinary collection scales well at 32 hosts and CPU allocation remains available | Try `workers: 48`, then at most 64 for this allocation. Keep `games_per_iteration: 320` while measuring the concurrency effect. |
| GPU PPO passes are efficient and memory headroom is large | Try `minibatch_size: 1024` and `max_candidate_cells: 262144`. Candidate padding, not just position count, determines activation memory. |
| Ready inference batches approach their limit | Try `batch_size: 128`; actual batches still cannot exceed ready games for a given frozen policy. |
| GPU is mostly waiting on encoding/legal enumeration | Profile those phases; preserve the current model size and improve data supply before adding parameters. |
| Search dominates the cycle | Inspect confirmed/incomplete counts. Keep eight independent workers initially; more sampled worlds/depth are separate quality experiments, not necessary to scale ordinary PPO. |
| More hosts reduce throughput or responsiveness | Reduce `workers` to 16 and compare fresh collection rates. A logical-CPU count is not a speedup guarantee. |

Avoid raising `threads` to 64 along with many simulators; it controls CPU tensor parallelism and can add competing threads. The current preset leaves headroom for coordination and other server work.

`minibatch_size` is a **microbatch memory/throughput knob** in this implementation. Gradients accumulate across all learner rows in one epoch, followed by one optimizer step. It does not mean 512 rows per independent Adam update. Candidate splitting preserves row weights, all legal alternatives and optimizer-step count. A different stochastic minibatch PPO algorithm would be a separate learning change.

Increasing `games_per_iteration` changes the data collected per optimizer step. Therefore 313 new iterations (100,160 ordinary games) are **not learning-equivalent** to the previous 5,000 iterations of 20 games, even though the total number of games is similar. Likewise 5,000 iterations now means 1.6 million regular games. Budget using games, learner decisions, optimizer steps and validation improvement together.

Suggested scaled milestones are 2 iterations (640 games) for throughput, 10 (3,200) for early learning, 50 (16,000) for checkpoint comparisons, then 200 (64,000) if results improve. About 313–500 iterations is a substantial 100,160–160,000-game experiment; it is not a promised strength threshold. Additional historical games inherited by `--from-run` are recorded separately through the origin and cumulative count. Reserve evaluation compute and check opponents beyond weak stock/v3.

## Focused checks and limits

The new tests cover strict config fields, exact table quotas, filling 32-worker batches, padded candidate-memory bounds, equivalent accumulated PPO updates across microbatch sizes, and migration of weights/optimizer/history/setup cursor. R2 model/replay/export checks still apply.

An optional small real-engine check compares two identical setups under serial and parallel collection, then runs two independent short-search games:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/check-step5-scaling.py --device cuda --with-search
```

It writes evidence beneath `/data/not_backed_up/yxu209/ai-training/artifacts/step5/scaling-checks/`. The search smoke uses one-second budgets and accepts properly labeled incomplete searches; it verifies execution and isolation, not useful teacher strength. It does not rerun a qualification tournament. For results executed locally, see [scaling checks](STEP5_SCALING_CHECKS.md).

Windows CPU checks cannot establish Linux/Mono scaling or CUDA throughput. Measure the first fresh lab iterations and the first scheduled search phase before further expansion. The default model remains suitable for later compact laptop deployment; larger training throughput does not require a larger deployed network.
