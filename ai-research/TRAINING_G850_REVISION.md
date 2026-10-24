# Training revision from g850

Date: 2026-10-24. Incoming evidence:
`results/joint7-v11-2-grouped-lr-critic-g0850-20261024T181807288965Z`.
This is an explicit reviewed continuation and the new fresh-config recipe.
Existing run files, checkpoints, exported reports and completed IL were not modified.

## Effective changes

| Setting | New value |
|---|---|
| Opening temperature | 1.5 for each neural seat's first two action decisions, including fresh counters after a fork |
| Entropy coefficient | Fixed 0.005 for all four groups; adaptive controller off |
| Epsilon mixture | 0.05; existing context/family/industry-weighted distribution unchanged |
| Branch probability | 0.010 per eligible decision; existing one-branch/game and eligibility limits retained |
| Insolvency | 0.02/event, capped at 0.075 per seat/game |
| PPG contract | `ppg-value-phase/15.0.0` |
| PPG cadence / training data | Every iteration; current plus previous committed iteration, uniformly weighted rows |
| PPG loss acceptance | Full held-out set from both available iterations; strictly non-increasing pooled row-mean loss |
| PPG epoch cap | 2, fixed; shuffle each epoch |
| PPG beta | Preserve saved beta/trip window; floor 0.5, maximum 32 |
| PPG LR peak | 0.65 × this iteration's last actually applied PPO LR |
| PPG LR schedule | Cosine warmup from 0.1× peak at step 1 to peak at step 16, then cosine decay to 0.3× peak at the full two-epoch cap |
| Loss / KL intervals | Both every 16 attempted steps, plus short epoch tails and retained-state verification |
| PPG hard KL cap | Unchanged 0.002 |
| PL bootstrap | Same 200 draws, full-fit warm starts, four CPU processes by default; configurable 4–8 |

The fourth insolvency event charges only the remaining 0.015; further events
add no penalty. Ordinary and search reward specifications agree. Frozen opponent
checkpoints keep their own reward identities.

The decoupled raw-T1 PPO surrogate is retained for the new T1.5 opening rows.
Its detached importance denominator remains the actual temperature-plus-epsilon
behavior probability. Opening objective and branch contracts are versioned;
old T2/epsilon-zero collection data cannot masquerade as new behavior. The
excluded fork move's alternative-selection distribution is unchanged.

Entropy baseline, saved controller evidence and rollout age are preserved.
The operative coefficient is 0.005 regardless of a previously saved adaptive
alpha; reports expose `effective_alpha`. Anchor promotions no longer lower the
inactive entropy floors. Existing branch-readiness rules remain in force.

## PPG acceptance, state and cache handling

Both ages' whole-game/branch-family heldouts are excluded from fresh critic
fitting and pooled for every loss check. There is no SE tolerance. Cloning and
the raw-T1 complete-distribution KL probe still use current learner data only.

`acceptance_heldout_before/after`, `acceptance_heldout_rows`,
`acceptance_generations` and `acceptance_support` report the pooled test.
`heldout_before/after`, descriptor diagnostics and GAE/snapshot evidence remain
current-only, preventing overlapping validation windows from reusing older
iterations as new gate observations. These fields also survive compact history.

The cosine schedule uses attempted step number and the actual row-derived full
budget; early stopping never compresses the curve. At g850's 260,547 train rows,
the same-sized collection would have 64 steps/epoch and a 128-step ceiling.
Tiny fixtures ending before step 16 remain in warmup. Value-Adam moments persist
across phases, and rejected intervals restore matching core/auxiliary weights
and Adam state. PPO's DCPG preservation objective remains unchanged.

The saved g850 beta is 1.0 and remains 1.0 on adoption. A clean phase with accepted
steps and final KL strictly below 0.001 halves beta, now down to 0.5. A new KL trip
with at least three trips in the latest five phases doubles it up to 32.
No accepted steps means no low-KL decrease. There are no within-phase beta retries.

Changed reward targets invalidate the old PPG critic cache. Only owned caches
are retired, after the reviewed signature is committed under the run lock.
Unknown files/owners still block deletion. This preserves all model weights,
PPO/value/auxiliary Adam moments, RNG, generation/game counters, population,
checkpoint lineage, economic baseline, LR controller and GAE/snapshot evidence.
g851 uses one new age; g852 onward uses two. Restarting before the first new
checkpoint can repeat cleanup safely; normal later resumes do not reset beta,
trip history or cache origin. Expired new caches are deleted after commit.

## PL implementation and comparison

Execution settings are in a separate strict `rating_compute` block. They do not
rewrite the statistical rating contract, saved slots, priors, IL zero point,
seat correction, reference selection or promotion rule. All original rating
history remains available to the production fitter.

The main process generates the exact original seeded setup multiplicities.
Each setup receives one weight across every historical slot and seat rotation.
Spawned workers receive immutable encoded data once and consume draws in fixed
order. Their BLAS/OpenMP thread counts are set to one before library import.
Configured workers cannot exceed the existing CPU-reader budget, and actual
parallelism respects CPU affinity. Worker failure aborts rather than accepting
an incomplete bootstrap. A failed warm-start convergence check retries the
original cold fit; its existing convergence safeguards still apply.

Local benchmark, same available 9,728 main games, 87 agents, 32 setup clusters
and 200 refits for each implementation:

| Path | Wall time |
|---|---:|
| Existing cold serial | 188.70 s |
| Warm serial | 163.41 s |
| Warm four-process | 54.54 s |

Four-process warm refitting was **3.46× faster** than cold serial locally.
Point ratings were identical. Warm serial/parallel bootstrap samples matched
exactly. Relative to cold starts, maximum differences were 0.000808 in sampled
theta, 0.0462 PL points in marginal interval endpoints and 0.1326 PL points in
paired interval endpoints. None of the 3,741 pairwise stronger/weaker decisions
changed. These small numerical differences arise from the unchanged optimizer
stopping tolerances with a different initialization; samples are not relabeled
as bit-identical to cold starts.

The incremental export lacks raw g760–840 slot records locally. This comparison
uses the full available archive plus g850, so it does not reproduce the complete
saved g850 fit. Unit regressions separately exercise actual initial-reference
and later-reference promotion selection on paired samples. Historical files
and production promotion criteria are unchanged. Local timing is not a lab
throughput guarantee; short correctness tests overlapped parts of the serial
measurements, so the speedup should be remeasured on the training server.

## Validation

- **29 focused tests passed** in 51.24 s: config/continuation, stale behavior,
  post-fork counters, cosine endpoints, beta bounds, two-age loss acceptance,
  current-only gate support, KL/loss rollback, reward-cache retirement,
  deterministic resampling, serial/parallel parity, convergence failure,
  paired confidence intervals, promotion selection and preserved source reviews.
- Three isolated authored CPU PPO/PPG iterations passed in **22.02 s**. Accepted
  PPG steps were **2, 4, 4**; beta was **4 → 2 → 1 → 0.5** in this fresh fixture.
  Pooled heldout rows were **55, 109, 108** versus current-only gate rows
  **55, 54, 54**. Restart after iteration two reproduced model/optimizer/RNG,
  beta/trips and gate state exactly. Source checkpoint checksum was unchanged.
- The smoke's 121 policy probes and short phases do not exercise production
  CUDA throughput or a complete 16-step warmup. Focused regressions cover the
  16/32-step transaction boundaries and the complete cosine curve separately.
- No long training or new evaluation games were launched. These checks do not
  demonstrate better playing strength or training stability.

Evidence: `artifacts/g850-two-holdout-validation.log`,
`artifacts/g850-two-holdout-smoke/report.json`,
`artifacts/g850-pl-benchmark.json`. The refit comparison is reproducible with
`tests/Recovery/benchmark_pl_bootstrap.py`.

More frequent KL checks and twice as much heldout support increase validation
work. Two rather than three epochs lowers the maximum update budget, but phases
can still stop after the first interval. More branches can increase collection
cost. No net training-speed claim follows from the PL-only timing.

## Continue g851–870

Sync this revision on the Linux training machine, then run from the repository
root. The helper validates generation/source/checkpoint hash and writes only a
new config. The trainer performs adoption; do not edit `state.json` manually.

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 850 --apply && \
python scripts/configure-training-g850.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 850 --rating-workers 4 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-training-g850.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-training-g850.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 850 --iterations 20
```

`--rating-workers` accepts 4–8. Start with the measured four-worker setting.
After completing these iterations, reuse the generated config with the new
committed generation; do not repeat an adoption to reset beta or replay.

## Implementation files

- Recipe/config/resume: `training_revision_g850.py`, `config.py`, `resume.py`,
  `utility.py`, the source-review registry, and `scripts/configure-training-g850.py`.
- Behavior: `exploration.py`, `opening_objective.py`, `branching.py`.
- PPG/reporting: `ppg_g210.py`, `value_phase.py`, `review_metrics.py`.
- Rating execution: `plackett_luce.py`, `ratings.py`.
- Regressions/benchmark/smoke: `tests/Recovery/test_training_g850.py`,
  `test_pl_parallel.py`, `benchmark_pl_bootstrap.py`, `smoke_training_g850.py`.

Production modules above are under `src/brass_ai/recovery/`. Current lab entry
points are updated in `docs/TRAINING_RECOVERY_LAB.md` and `docs/MLP_V11_LAB.md`.
