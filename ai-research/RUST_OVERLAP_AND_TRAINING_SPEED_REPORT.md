# Overlapped collection, PPO/PPG speed and c_scale 1: review and implementation

Date: 2026-11-09. Evidence: `results/joint7-v11-2-g1290-trained-restart-g1292-20261109T202425416960Z`. Both iterations in that export are the first iteration of a fresh process (different PIDs), which matters below.

## 1. What the g1291 profile shows

| Phase | Seconds | Note |
| --- | ---: | --- |
| Teacher tables | 55.5 | 28 tables on 28 CPUs; each table takes 45–51 s |
| Ordinary pool | 42.4 | 64 lanes, 40 s live |
| Planning before the pool | 89 | Only on a fresh start; about 12 s when the process keeps running |
| Replay join | 12.1 | Mostly single-threaded sealing of 124k rows after the last game |
| Sampling | 15.8 | First iteration only (teacher label reload); 0.9 s otherwise |
| PPO | 47.5 | |
| PPG value phase | 24.8 | Search head is paused while eight return ages rebuild |
| Whole iteration | 290.6 | About 196 s without the fresh-start costs |

The 89 s of planning is `register_checkpoint` loading 24 pool checkpoints at about 3 s each. The CPU path reads those models from their ONNX bundles, so it never needed them loaded.

### Ordinary collection is limited by its pipeline, not by lanes

For the 320 ordinary games (sums over games, in CPU-seconds):

| Stage | Seconds |
| --- | ---: |
| Playing games | 861 |
| Publishing families | 489 |
| Reading shards | 478 |

Only 13.5 s of the 40 s collection is game play spread over 64 lanes. Lanes spend the rest waiting: `shared_write_seconds` sums to 1,235 lane-seconds (the 1 GiB transfer budget is full).

A discrete-event model of this pipeline, fed with the measured durations of all 320 games, reproduces the real run (39 s modelled, 40 s measured):

| Lanes | Ordinary wall (s) | |
| ---: | ---: | --- |
| 16 | 61 | |
| 24 | 43 | |
| 32 | 38 | |
| 40 | 39 | |
| 48 | 39 | |
| 64 | 39 | today |

**Decision: keep 64.** More lanes than about 32 add nothing for ordinary games alone, so neither 48 nor 32 is faster. They are also not better once tables overlap: the lanes that matter during overlap are those not held by tables, 64 − 28 = 36, which is the flat part of this curve, while 48 would leave only 20 at the start. The CPU count that matters is the machine's 64.

The readers are the slowest stage (16 readers at about 1.5 s per shard). Going to 24 removes that: the model gives 32.9 s instead of 39.0 s for ordinary games alone, and 54.8 s instead of 56.6 s with overlap. Going to 32 adds nothing further.

## 2. Overlap schedule (implemented)

The teacher tables and the ordinary games are independent frozen jobs, so they can share the CPUs. The schedule is list scheduling, longest work first:

1. Every teacher table starts at once on its own CPU, because tables are the long jobs (about 50 s against 2.7 s for an ordinary game).
2. Ordinary lanes run on the other CPUs, with the existing FIFO gate deciding between lanes, publishers and readers.
3. When a table ends, its CPU goes straight to a parked ordinary lane. This is `CorePlan` and `Lanes`, so both workloads keep every CPU busy to the end.

The training order is unchanged. Teacher admission still happens before ordinary replay is merged, the frozen jobs and their seeds are the same, and a failure in either side stops the other. Timed searches keep their dedicated run, so only fixed-count tables overlap.

Model result for the current workload, with the teacher tables taking 51 s: the combined phase takes about 55 s, against about 94 s run one after the other. The whole iteration should fall from about 196 s to about 120 s (see section 6). These are model estimates; the next export will show the real numbers.

## 3. Collection changes

| Change | Effect | Why it is safe |
| --- | --- | --- |
| Overlap above | about 40 s per iteration | Same frozen jobs, same admission order |
| Checkpoint hashes in parallel threads | Part of the roughly 12 s of planning; `timings.planning_seconds` will show how much | Every checkpoint is still fully hashed |
| CPU path defers loading pool checkpoints | about 80 s per fresh start | Workers read bundles; a missing bundle loads the checkpoint on demand |
| Replay shards sealed as they arrive | about 8 s of the join moved beside collection | The same seal runs once per row, on background threads |
| 24 replay readers on the CPU path | about 6 s in the model | Runtime setting only; readers still pass the CPU gate |

## 4. PPO and PPG

| Change | Cost it addresses | Effect on results |
| --- | --- | --- |
| Skip re-hashing sealed rows when staging batches | 20.8 s of PPO main-thread staging, mostly SHA-256 on every reuse | None; rows are immutable and bytes-backed, and unsealed inputs are still re-hashed |
| Compute exploration distributions only when used | Most of the 4.4 s `ppo.prepare`; never read with zero exploration | None |
| Measure actor-gradient norms on one step in four once beta is fixed | A second backward pass per actor microbatch (about 3 s) | Report only; the gradient applied is byte-identical, which a test checks |
| Write the PPG cache beside PPO (removed after g1294, see [G1294_REVIEW_AND_TRAINING_REVISION.md](G1294_REVIEW_AND_TRAINING_REVISION.md)) | 5.9 s at the start of the PPG phase | Same bytes, same receipts; the GIL made it no faster |

The search-head phase (`search_value`) is currently paused while the buffer rebuilds, so it costs 0.03 s. When it trains, the update is about 2 s and preparing its rows is under 1 s in a running process (up to 6.7 s on the first iteration after a restart). Neither is worth changing yet.

Not changed: microbatch size, precision, optimizer settings, KL guards and PPG acceptance. Those change numerics, and the costs they would save are smaller than the ones above.

Estimated PPO 47.5 s → about 36 s and PPG 24.8 s → about 19 s. These come from the measured components; the next export's `cpu_preparation.staging_seconds`, `ppo.*` and `value_work` will show actual values.

## 5. Gumbel c_scale 1

- Lab: the fixed-count teacher budget now carries `c_scale`, set by `--c-scale` (default 1). It reaches the search through the CPU worker overlay and is written on every label. The sealed job and its checks are unchanged, exactly like the budget change before it. Requires `prepare-rust-cpu-inference.py`.
- Label window: following the rule in `TEACHER_SEARCH_PHASE_REVISION_PLAN.md`, labels from before the change retire with the switch. The qualified fixed β stays: `c_scale` is not part of the buffer identity, so the weight is not reset, in line with the standing fixed-β rule. The window refills over the next three iterations.
- Local game: release `1.4.54-rust-local` sets `config.c2=1.0` in all 18 policy files of the three NN packages. Every other file is unchanged. See [NN_C_SCALE_GAME_UPDATE.md](NN_C_SCALE_GAME_UPDATE.md).

With raw utility Q and no normalization, c_scale 1 makes the improved policy much greedier than 0.1 (the Q term is ten times larger), so a label's top action gets a larger share of the target. Watch teacher gradient share and held-out KL in the first iterations.

## 6. Expected iteration after this change

| Phase | Before | After (estimate) |
| --- | ---: | ---: |
| Planning + teacher + ordinary | 110 | about 60 |
| Replay join | 12 | about 3 |
| PPO | 47.5 | about 36 |
| PPG | 24.8 | about 19 |
| Sampling, checkpoint and other | about 2 | about 2 |
| **Iteration** | **about 196** | **about 120** |

Fresh-start costs (about 90 s of planning, 15 s of label reloading) also drop by about 80 s.

## 7. New reporting

- `report.timings.planning_seconds`
- `report.exploration_metrics.branching.scheduling.overlap`: lanes parked for teacher tables
- `teacher-shadow/collection.json`: `overlapped_with_ordinary`, and `cpu_inference.budget.c_scale`

## 8. What was verified

On the laptop:

- Rust worker unit tests, including the c_scale budget parser (12 passed).
- CPU-inference tests with real compiled workers, including a real table at c_scale 1 whose labels recompute correctly (47 run, 2 expected skips).
- New tests for CPU hand-over, parked lanes, the overlapped begin and finish, a table failure stopping games, label retirement, sampled gradient norms keeping byte-identical gradients, sealed-row staging, shard sealing and deferred checkpoints.

**Not finished:** the wider Recovery and RustBackendLab suites were stopped before they completed. An earlier lab-suite run, made before the code identity was recorded, failed mostly on that missing identity. Two of its failures went through changed code and are fixed; the others either do not touch the changed files or are old fixture problems. The PPO, PPG and replay suites in tests/Recovery did not finish.

Also not verified: the overlap on real pinned CPUs and the throughput numbers, because the laptop has no CPU-affinity API. The new `overlap` qualification stage runs that on the lab.

## 9. Known limits

- A parked lane still starts on its held CPU, which costs a table about 1 s.
- Only count-budget tables overlap; a timed budget keeps its own window.
- The GPU path stays sequential.
