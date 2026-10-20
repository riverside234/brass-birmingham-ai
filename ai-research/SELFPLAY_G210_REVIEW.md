# Self-play g210 review and queued branch continuations

Reviewed 20 October 2026, from `results/joint7-v11-2-grouped-lr-critic-g0210-20261020T145156876449Z`. The committed state is **generation 210 / 67,200 ordinary games**. Report `iteration-0209` produced g210; compact history supplies the earlier iterations.

## Training findings

The reports do not show an economic collapse or a new PPO/PPG KL failure. The clear implementation issue is branch scheduling: the original worker finished its parent, then restored and played the branch on that same lane. This made dependent work unavailable to other idle workers and lengthened the final collection tail.

| Measure | g196–205, before branching | g206–210, branching active |
| --- | ---: | ---: |
| Mean collection wall time | 196.38 s | 227.66 s |
| Mean PPO time | 55.48 s | 63.43 s |
| Mean PPG time | 37.91 s | 47.41 s |
| Mean iteration time excluding scheduled evaluation | 312.18 s | 365.46 s |
| Mean categorical PPO KL | 0.00713 | 0.00738 |
| Mean accepted PPG steps | 31.0 | 47.2 |
| Mean branches | 0 | 65.2 |
| Mean branch continuation worker-seconds | 0 | 630.14 s |
| Mean fraction remaining at branch | — | 0.581 |

Collection rose **15.9%**, and non-evaluation iteration time rose **17.1%** across these windows. These are observational comparisons, not a paired experiment: game setups, hardware contention, accepted PPG work and training data change. Lambda also changed from 0.98 to **0.95** for g206, so performance or learning changes cannot be attributed to branching alone. g200 included both normal and temperature ratings; g210 included the normal rating. Those scheduled costs are excluded from the iteration comparison.

The ordinary pool has **24 processes × 2 host lanes = 48 game workers**. The historical branch work averages **630.14 / 48 = 13.13 seconds** of ideal evenly shared work, versus the observed 31.28-second collection increase. This is a useful scheduling reference, not a measurement of CPU utilization or a guarantee that all 18.15 seconds of difference are removable. Host/inference waits are included in worker-seconds, and the old reports lack per-branch enqueue/start/end timestamps.

At g210:

- Five-iteration all-learner health: **108.69 VP/seat**, **0.656% zero-VP seats**, **3.063% insolvent seats**, and **0.356 ordinary passes/seat**. Every configured economic check passes. IL sale/pass-probability checks remain absent; sales are reporting only.
- PPO exact categorical KL is **0.006695**, with **44/44 planned optimizer steps**, no KL early stop, and LR multiplier **1.5625** (effective PPO LR **9.375e-6** after warmup).
- Fixed IL KL is **0.6056**, below the current calibrated **0.7115** mean limit; the group limit is **1.4229**. The largest group, Canal actions, is **1.2118**. Both existing limits and the calibration state are retained.
- The four entropy coefficients remain **0.005**. Raw entropy is above every floor; no cap/excess-entropy alarms occur. Canal-action raw entropy is **0.9296**, against floor **0.8690**. There is no evidence here to reduce epsilon or change the floor.
- **64 branches** generated **5,942 actor-eligible** and **14,265 critic** continuation rows. All 5,942 eligible branch actor rows were actually used by PPO; PPO also used 7,886 distinct branch critic rows. Ordinary progress still counts 320 games. Parent/branch holdout families and exclusion of the fork move/prefix remain enforced.
- PPG accepted **32, 156, 48, 0, 0 steps** in g206–210. At g210, its first 16-step interval changed held-out loss from **1.18609 to 1.18731** (+0.103%) and was correctly restored. Attempted policy KL was only **0.0000601**, well below 0.002. Raising the KL cap would not fix this rejection. Retain the loss safeguard and assess accepted work over the rolling window.
- Ordinary held-out EV varies substantially by iteration; the g210 ordinary EV is **0.1996**, versus branch EV **0.4295**. These cover different positions and outcomes and are not directly comparable strength measures. The mixed-table rolling evidence remains supported: history EV **0.3684** over 140 games, v4 EV **0.4534** over 40 games, with matched self-play EV **0.3692**. No new automatic gate threshold is introduced.
- Exploration produced zero pass picks when a non-pass alternative existed. The 2,496 tempered decisions equal exactly two per ordinary neural seat-game. Both pass exploration picks were in pass-only positions.

## Strength evidence and recommendations

The latest common PL fit gives **g200 572.09** and **g210 571.34**. The paired fitted difference is **−0.75 points**, with setup-bootstrap 95% interval **[−65.44, +60.05]**. This does not establish an improvement or regression between those checkpoints. g210's v4 diagnostic panel has 81.25% learner win share over 32 games / eight setups, with interval [62.5%, 96.9%]. The IL panel is 96.875%, with interval [90.625%, 100%].

At g200, temperature 0.7 beats 1.0 by a point estimate of 7.92, but the interval **[−34.92, +61.95]** spans zero. Keep the existing evaluation/deployment temperature and human panel review.

The immediate improvement is scheduling and measurement. Keep the 0.005 branch rate, 50/45/5 table mixture, current reward, GAE, PPG, LR controller and entropy recipe. Review the next ordinary training logs for the new tail metrics, actual branch sample usage, ordinary held-out EV and the common rating series before increasing branch frequency or tuning value acceptance. A sustained pattern of zero accepted PPG work would justify a separate value-update experiment; the current five-iteration window still contains 236 accepted steps.

## Implemented scheduling

1. At capture, export the trusted native fork checkpoint and immediately enqueue its continuation. The parent keeps playing. The checkpoint never enters policy inputs or the replay/IL dataset.
2. Dispatch one parent/continuation per free ordinary lane, without prefetching a sequential packet of games. Ready continuations have priority; earlier-era/earlier-round continuations run first as an estimate of remaining work. Branch-producing parents start early; other parents prefer the previous collection's measured duration by table type. Stable indices break ties. These estimates affect scheduling only, with no game RNG consumption.
3. Preserve the worker cap and frozen model/policy identities. Import the exact native checkpoint on the selected lane, preserving request identity/revision, all hands/public state and the deck-only reshuffle semantics.
4. Join parent and child asynchronously in the parent's process. A worker or writer never waits for an unfinished sibling. Only compressed child JSON crosses the completion channel; the parent's encoded arrays stay local. Publish a replay family only after both finish and validation succeeds.
5. Move replay-reader submission off the scheduling thread. Bounded reader backpressure cannot delay a fork notification or prevent dispatch to a free lane. Reader failures abort shared admission before executor shutdown.
6. Reuse complete cached families on restart. Unpublished families are recollected from their original deterministic setup/sampling/fork seeds; a partial parent is never admitted to PPO. No standalone child can bypass the existing parent/provenance checks.

Full and compact history retain `exploration_metrics.branching.scheduling`, including actual workers, branch worker-seconds and worker-seconds/worker, queue wait, branches started before their parents finish, parent overlap, work utilization, branch tail after the last parent, and publication/reader-submission tail. Actor phase and Python-CPU summaries now include remote branch work exactly once. Cached work is explicitly excluded from live worker timing.

The timing scope differs slightly from the older `branching.seconds`: the new worker metric also includes final continuation bookkeeping. Neither sum is CPU busy time. Overall `timings.collection_seconds` also includes collection planning/startup outside the dispatch interval. Compare consistent scopes.

## Validation and limits

The isolated native benchmark used the current **5,928,616-parameter v11.2 utility model**, randomly initialized, eight fixed self-play setups, four warmed worker lanes, and the production **0.005** fork rate. Two branches were selected. No training update was run.

| Collection arm | Wall time |
| --- | ---: |
| Same parent games, branches disabled | 46.546 s |
| Original serial branch execution | 58.297 s |
| Queued branches | 47.843 s |

Queued collection was **17.9% faster than serial**. Both branches started before their parent finished; branch work after the last parent finished was **0 seconds**. Publication tail was **1.625 seconds**. The new timing record gives **25.671 branch worker-seconds / 4 lanes = 6.418 seconds**, with 83.6% measured work-interval lane utilization. Mean queue wait was 10.85 seconds because lanes were occupied when the forks arrived; queued does not mean an extra worker is created.

Parent and branch actions, results, policy metadata and contexts matched the serial run; maximum behavior-probability differences were below **3e-8**. A separate cross-host native check reproduced all **295 continuation decisions** and terminal state after the same deck shuffle, while preserving all four permitted seat observations. Full cached-family reuse ran zero additional parent/branch games.

**93 focused regressions passed**, including queue priority/worker limits, nonblocking parent-child joins, duplicate/foreign work rejection, reader backpressure/failure handling, actor cache identity and partial publication, native restore/rollback, behavior probabilities/openings, branch holdout and training exclusion, compact timing retention, g210 reviewed continuation, optimizer/RNG preservation, and existing PPG/GAE evidence-window/disk-buffer checks.

The final fresh-process recovery smoke retained seven complete families and omitted one branched family. It recollected exactly **one parent plus one branch**, preserved all seven existing replay files byte-for-byte, and reproduced the missing family's choices/results with probability differences below **1.5e-8**. Real streamed ingestion prepared **10 games / 2,947 rows**, while ordinary health counted only **eight** parent games. With two live lanes, the new branch waited **0.016 seconds**, overlapped its parent for **12.015 seconds**, and left a **0.203-second** branch work tail. Its receipt is `native-restart2/report.json`.

The measured +2.8% queued collection cost versus this small no-branch arm is **not** a production overhead estimate. The benchmark is too small, uses CPU inference/random weights/all-self-play tables, and the no-branch arm shared the machine with focused checks. Scheduling also changes batch utilization. The target remains **+6–8% non-evaluation iteration time** on the real 48-lane run; it is unverified until matched lab logs include collection, ingestion, PPO and PPG costs. The supplied g206–210 averages do not meet it.

Receipts and focused-test logs are under `artifacts/selfplay-g210-review/`; the A/B receipt is `native-benchmark2/report.json`. Completed source approvals are retained, and the learning/feature compatibility digest is unchanged. No live checkpoint or result export was edited, and no long training/evaluation campaign was launched.

## Continue

The existing config exactly matches the g210 exported signature. After syncing this revision, continue from the same run:

```bash
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-exploration.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

No config regeneration, fresh start or IL retraining is needed. Weights, both optimizers, training RNG, population, checkpoint lineage and saved exploration origin **185** / baseline arming **197** are preserved. Existing pending-old-source safeguards still apply. The local verification uses the exported g210 state and isolated checkpoints/collections; the remote full training checkpoint is not available locally.
