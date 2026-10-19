# PFSP history sampling and slower rehearsal decay

Date: 2026-10-19. Implements the supplied same-day PFSP plan. This is an experimental sampling change, not demonstrated improvement in playing strength.

## Implemented settings

- `pfsp-pool/1.0.0`: every completed history-table learner/snapshot pair contributes 1/0.5/0 after official VP/income/cash tiebreaks. Two learner seats against the same snapshot in both opponent seats produce four comparisons. Self-play, v4 and rating games do not update the estimator.
- All members, including IL, receive one float64 update per committed iteration: `S=d*S+outcomes`, `C=d*C+comparisons`, `d=0.5**(1/5)`. New members use S=5, C=10. No importance weighting or q scores in this contract.
- Conditional scored-member probability is `0.8*(1-S/C)^2/sum(weights)+0.2/N`. All-zero weights fall back to uniform probabilities. IL is excluded from these weights.
- IL receives `1/(k+1)` of snapshot seats, where k includes IL. IL alone gets 100%. Default optional cap is null; the configured alternative cap works without changing the IL-only fallback. Both seats remain independent draws using the existing seeded planner.
- Capacity is 16 scored members + IL below 100,000 committed ordinary games, then 23 + IL. No additional admissions occur at the capacity boundary; the existing five-iteration admission cadence still applies.
- Admission still needs five healthy iterations and minimum probe `KL(member || candidate) >= 0.02`. Eviction protects IL, the two newest members and members younger than 20 iterations. Eligible members need C>=15. Among those within 0.05 of the highest eligible outrank rate, evict the most redundant by nearest-other-member probe KL. No eligible victim means skip admission.
- An outrank rate below 0.4 with C>=20 for three consecutive iterations raises a **review-only** alarm. It does not alter shared health, gates, mixture, rehearsal or rating decisions.

50/45/5 tables, six learner-seat pairs, exploration, PPO/reward/LR/IL-KL settings, GAE/PPG, staged snapshot-weight recovery, official v4 policies, disabled search/teaching and held-out PL ratings retain their settings. No model or feature changes; completed IL remains compatible.

## Rehearsal

The coefficient remains `max(floor, 0.1 * max(0, 1-games/10000))`; only the default interval changes from 30 to 60 committed iterations. The first drop still follows the first 60 iterations.

| Next training iteration, one-based | Floor |
| --- | ---: |
| 1–60 | 0.03 |
| 61–120 | 0.02 |
| 121–180 | 0.01 |
| 181 onward | 0 |

Continuation uses the existing committed counters, without a new rehearsal clock. Explicit old interval-30 configs keep their old behavior until the reviewed config switch.

## Continuation and provenance

The source keeps the legacy quality-score contract for explicitly unchanged old configs. Fresh config generation adopts PFSP and the interval-60 default. The new `scripts/configure-pfsp.py` helper copies an existing config to a new filename, changing only these recognized settings. It refuses overwrite and does not modify IL, checkpoints or run state.

The switch is allowed only at a committed generation divisible by five. Existing members get C=10 and S=10 times their logged five-iteration moving outrank rate, or the 0.5 prior when no comparisons exist. Membership, checkpoint/probe/fingerprint identities, admission ages and the health window are retained. q is never treated as an outrank estimate.

The atomic source transition saves a backup and a reproducible old/new pool-digest receipt. A restart immediately after the switch can still load the unchanged original generation checkpoint and reconstruct/verify the migrated pool. Subsequent checkpoints bind the PFSP state normally. No pending old collection is relabeled or silently discarded. Oversized legacy populations and non-boundary switches fail explicitly instead of dropping members or resetting progress. Both optimizer states, RNG, fixed validation probe, recovery/gate state and counters follow their existing restore paths.

The g85 export contains IL, g60, g75 and g85. A read-only migration preview gives:

| Member | Seed outrank estimate | Absolute snapshot-seat probability |
| --- | ---: | ---: |
| IL | 0.816949 | 20.000% |
| g60 | 0.536932 | 23.905% |
| g75 | 0.476040 | 29.110% |
| g85, no comparisons yet | 0.500000 | 26.985% |

IL is initially still at 20% because k=4. One further admission changes its share to 1/6; at k=17 it is 1/18, and at k=24 it is 1/25. This preview did not alter the exported results or a training run.

## Reporting

The existing `quality_pool` state/report key remains, with the new contract distinguishing PFSP. Per-member reports contain age, actual seats/comparisons, raw outrank rate, S/C, decayed estimate, weight, used probability, and alarm streak. `planning` holds the completed collection's probabilities; `next_distribution` and `summary` hold the post-admission plan, including k, IL share, effective members, largest share, share to admissions younger than 20 iterations, and the p-weighted mean outrank rate. IL has a separate share/seats/comparisons/estimate summary. Compact history and the existing report exporter preserve these fields. Console adds PFSP IL share, effective count and alarm count; gate diagnostics list review warnings.

## Validation

**213 tests passed:** 193 training/sampling regressions in 241.226 seconds, plus 20 export/source-review regressions in 6.108 seconds. Results are recorded in [the validation receipt](../artifacts/pfsp-revision-20261019/validation.json), [training test log](../artifacts/pfsp-revision-20261019/regressions-final.log) and [report test log](../artifacts/pfsp-revision-20261019/report-regressions.log). Tests cover strict contracts, hand-computed decay, official ties, no-game decay, all four comparisons, floor/normalization/all-zero fallback, independent seat draws, capacity boundaries, eviction protections/redundancy/count support, admission prior, alarms, frozen planning, JSON restart, atomic migration, corruption rejection, configuration review, rehearsal boundaries, retention, existing GAE/PPG/LR/IL-KL behavior and snapshot-weight recovery.

An existing rehearsal worker-death test could occasionally execute its injected `os._exit(93)` in the test parent when the sampled window contained one distinct shard. The first broad run stopped there. The test now explicitly supplies a process executor, ensuring the injected failure occurs in a child; production rehearsal code is unchanged. Initial fixture-only assertion failures were corrected before the final run.

The [isolated compute smoke](../artifacts/pfsp-revision-20261019/compute-smoke/report.json) passed in **16.24 seconds**, using authored mixed-table rows and the 5,928,616-parameter medium v11.2 model on CPU. It completed two bounded PPO/PPG iterations, verified bitwise matching next-update weights, PPO Adam, value Adam and gate/recovery state after restoration, kept the previous checkpoint unchanged and deleted expired value-cache data. It did not collect native games or evaluate strength. Timing overlapped regression work and is not a controlled throughput comparison.

The [synthetic benchmark](../artifacts/pfsp-revision-20261019/simulation/report.json) used 16 seeds at each pool size, 144 history tables per iteration, correlated four-seat Plackett–Luce outcomes, 100 stationary iterations then 40 forgetting iterations. It used the implementation's estimator and eviction decision. Pool identities were kept fixed for precision measurement; eviction choices were inspected on copies. The plan's referenced `pfsp_sim.py` was not available for that benchmark, so it remains an independent diagnostic, not a reproduction of its exact simulation.

**Follow-up:** the user subsequently supplied the simulation. It is archived unchanged at [simulations/pfsp_sim_original.py](simulations/pfsp_sim_original.py), with a maintained runner at [scripts/pfsp_sim.py](../scripts/pfsp_sim.py). See [rerun instructions and cross-reference findings](simulations/PFSP_SIMULATION.md). Rerun it when history share, collection volume or pool size changes. Direct reference tests match trainer arithmetic and sampling; simulator response-reporting and IL-only-generator edge cases were fixed in the maintained copy. The production trainer and continuation source identity are unchanged by this follow-up.

| Scored members | Median iterations to half of new share | Median to 90% | Forgotten evictions while below median | 2×RMSE, hardest/easiest estimate | Estimator time/iteration |
| --- | ---: | ---: | ---: | --- | ---: |
| 16 | 5 | 9 | 0 | 0.0362 / 0.0528 | 0.72 ms |
| 23 | 4 | 9 | 0 | 0.0403 / 0.0612 | 0.94 ms |

The response checks (median <=8/16 iterations) passed. The hardest-member error was somewhat above the plan's illustrative figures; those figures are not guarantees. No half-life, floor or acceptance threshold was relaxed to match the simulation. Real-game correlation and forgetting response still need review in ordinary logs. No playing-strength inference follows from these results.

## Runtime implications

Sampling/estimation adds no model forward pass and measured under 1 ms per simulated update. Existing admission probe inference remains; eviction can perform more cached pairwise KL comparisons. At 16 scored models, expected scored snapshot seat-games rise from 230.4 to 272 per 320-game iteration, potentially increasing eligible critic/PPG work once its gate is active. Total games and learner-seat counts are unchanged. Rehearsal lasts 60 iterations longer than before; the existing 16-worker bounded loader and cleanup behavior are preserved. End-to-end training latency was not benchmarked.

## Use in the existing run

At g85 or another committed multiple of five:

```bash
python scripts/configure-pfsp.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic.json \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-pfsp.json

python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic-pfsp.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 20
```

Use the actual existing config/run/IL paths if different. Between boundaries, keep the old config until reaching the next multiple of five. After switching, repeat only the train command with the PFSP config. No fresh run or new IL is required. The normal shared-search build/preflight instructions remain in [the lab guide](../docs/MLP_V11_LAB.md).

## Changed files

- Sampling and configuration: `pfsp_pool.py` (new), `quality_pool.py`, `quality_config.py`, `ordinary.py`, `config.py`, `configs/step5/joint7-latest.json`.
- Continuation, reporting and schedule: `resume.py`, `league.py`, `review_metrics.py`, `rehearsal.py`, source-resume and IL-handoff review receipts; `scripts/configure-pfsp.py` (new).
- Validation: new `test_pfsp_pool.py` and `benchmark_pfsp.py`; existing quality/default/rehearsal/smoke tests adjusted for the new defaults and the worker-death fixture fix.
- Documentation: lab guide, report-export guide and this report. Existing generated run configs, exported results, completed IL and production checkpoints are unchanged.
