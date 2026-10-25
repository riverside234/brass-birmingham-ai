# g950 review and PPG continuation

Date: 2026-10-25. Reviewed export:
`results/joint7-v11-2-grouped-lr-critic-g0950-20261025T053358889475Z`.
Committed generation is **950**; the corresponding report directory is `iteration-0949`.

## Implemented settings

The new `ppg-value-phase/16.0.0` recipe changes only the PPG schedule and beta floor.

| Setting | Effective value |
| --- | --- |
| Peak PPG learning rate | 0.65 × this iteration's last actually applied PPO LR |
| Warmup | Linear, 0.1 × peak at step 1 → peak at step 16 |
| Decay | Cosine, peak → **0.001 × peak** at the full two-epoch budget |
| Budget | `2 × ceil(actual training rows / 4096)` attempted steps |
| Beta floor | **1**, with existing upper bound 32 |
| Current-run hotfix | Set saved beta **0.5 → 1 once**, when adopting v16 from v15 |
| Beta feedback | Existing 3-of-5 new KL-trip escalation and accepted low-KL decrease; decreases cannot cross 1 |
| Frequency and buffer | PPG every iteration; current plus previous committed critic data, uniformly shuffled each epoch |
| Loss acceptance | All held-out games and their branch families from both ages; check every 16 steps and short epoch tails |
| Policy guard | Existing exact raw-policy KL cap 0.002; check every 16 steps, epoch tails and unchecked retained states |

The endpoint follows the final request to use a very small positive number. At g950's actual
PPO LR of `7.8125e-6`, the PPG peak would be `5.078125e-6`, starting at
`5.078125e-7` and ending at `5.078125e-9` if both epochs complete. Positive-LR
checkpoint validation remains strict. Tiny test phases shorter than 16 steps
do not compress warmup. Loss/KL rejection does not rescale the schedule to the
shortened phase; it restores the matching retained model and Adam state.

The one-time reset is part of the reviewed configuration adoption, before the
next PPO/PPG iteration. Its receipt is saved in the actor/value checkpoint and
reported at startup and in `value_phase.schedule_migration`. A retry before the
first new checkpoint reapplies the same deterministic transition. Once a v16
checkpoint exists, restart restores its adaptive beta without resetting it to 1.
The beta trip window is preserved. Fresh runs retain the existing initial beta 4
with the new floor 1; the requested current-run hotfix starts at 1.

The conversion preserves weights, PPO/value Adam moments, RNG, utility targets,
two-age cache receipts, ownership heads, GAE/snapshot evidence, population,
checkpoint lineage, ordinary progress, rating history and completed IL. There
is no reward or behavior change and no fresh PPG cache reset. g951 can retain
g950's eligible cached data. Whole-game gate evidence remains current-only, so
previous held-out games are not counted twice as new GAE/snapshot evidence.

## Review findings

The review covers **100 committed generations, g851–950**, using compact history
and the two full reports. No retained PPG KL above 0.002 was found. From g852,
every report names the correct current/previous acceptance generations. Ordinary
composition stayed at 160 self-play / 154 history / 6 v4 tables per collection.
The g950 full report records temperature 1.5 openings, epsilon 0.05, the expected
v4 bank, and all five training v4 profiles. There is no new evidence here of a
probability/provenance, table-mix, holdout-age or restart-state bug.

The material concern is **low PPG acceptance**, not a disabled value phase:

| Measurement | g851–900 | g901–950 | g931–950 |
| --- | ---: | ---: | ---: |
| Iterations | 50 | 50 | 20 |
| Phases retaining zero steps | 29 | 31 | 13 |
| Retained PPG steps, total | 746 | 803 | 256 |
| Phases stopping on loss increase | 47 | 45 | 18 |
| Phases stopping on KL cap | 3 | 4 | 2 |
| Maximum retained PPG KL | 0.001977 | 0.001989 | 0.001939 |
| Median post-PPO exact KL | 0.00726 | 0.00737 | 0.00736 |
| PPO early stops | 0 | 0 | 0 |
| Self-play VP per seat | 122.60 | 124.18 | 124.65 |
| Self-play zero-VP seat fraction | 0.313% | 0.309% | 0.344% |
| Self-play insolvent seat fraction | 1.772% | 1.709% | 1.781% |
| Median current held-out learner EV | 0.412 | 0.382 | 0.373 |

The EV figures are medians of existing per-iteration diagnostics, not pooled R²
or a common-model re-evaluation. The decline deserves monitoring but does not
establish a training regression or identify a faulty implementation.

At g950, PPG retained 48 of 64 attempted steps. The fourth interval increased
pooled held-out loss and correctly rolled back. Its retained KL was 0.001288.
Acceptance covered **16,259 rows / 47 game identities across two ages** (including
branch families); current-only learner gate diagnostics had 18 ordinary games,
split 8 self-play / 8 history / 2 v4. This loss rejection is expected under the
existing strict non-increase rule. The change does not weaken that rule or claim
to fix the 60/100 zero-step rate. Near-zero late decay cannot directly solve
phases rejected in the first interval; linear warmup is not uniformly smaller
than the previous cosine warmup.

Economic health passed at g950. The active KL reference remains g890; the
post-PPO LR decision was `hold`, multiplier 1.302083, actual LR `7.8125e-6`.
The calibration raised overall/group budgets to 0.1728915 / 0.3457830 at g950.
History/v4 mixed-table EV evidence was supported and healthy at g950; two earlier
v4-window warnings in g901–950 had cleared. No new thresholds or automatic
rollback were added.

Within the **same saved g950 PL fit**, g950 has 723.90 PL points. Its paired
difference from g850 is +67.28, 95% interval **[7.06, 124.59]**. Its difference
from the active g890 reference is −4.46, interval **[−48.81, 35.93]**; keeping
that reference is appropriate. This is evidence from the existing evaluation
model, not proof that these new PPG changes improve playing strength. The saved
fit contains 12,160 main games and 200 bootstrap refits; no evaluation games or
bootstrap campaign were run for this revision.

Median g931–950 wall times were collection **191.0 s**, PPO **66.6 s**, PPG
**22.1 s**, total **321.1 s**. The schedule change adds negligible computation
per step; actual runtime depends on acceptance/early stopping. It retains the
two-epoch ceiling and the existing validation/probe sizes.

## Validation and files

**24 focused tests passed** in 69.6 seconds:

```bash
python -m unittest test_training_g950 test_training_g850 test_ppg_g210 \
  test_source_reviews.SourceReviewTests.test_repository_registry_is_canonical_and_keeps_pinned_history -v
```

Tests cover strict versioned options, unchanged unrelated configuration, linear
warmup and positive cosine endpoint, beta floor/up/down behavior, a one-time
adoption preserving cache bytes/Adam/RNG/gates, retry and subsequent restart,
full two-epoch completion and a positive final-LR restart, next-phase LR reset
from actual PPO, loss/KL cadence, model/Adam rollback, and read-only generation-
bound config generation. Legacy v14/v15 configurations retain their behavior.

The isolated authored CPU smoke passed in **21.4 seconds**:

```bash
python tests/Recovery/smoke_training_g950.py --output artifacts/g950-smoke
```

It runs three small PPO/PPG iterations with branches, snapshot rows, ownership
heads and four preparation workers, then repeats iteration 3 from the saved
iteration-2 checkpoint. Retained PPG steps were **2, 4, 4**; fresh-fixture beta
was **2, 1, 1** after each phase. Model/auxiliary weights, both training optimizers,
ownership optimizer, RNG, beta/trips and gates matched exactly after restart.
It checked pooled two-age acceptance and current-only gate evidence, unchanged
incoming checkpoint bytes and commit-only cache expiry. The small smoke does
not estimate server performance or playing strength; the focused endpoint test
separately exercises phases longer than 16 steps.

Implementation files:

- `src/brass_ai/recovery/ppg_g210.py`: v16 schedule, beta adoption and receipt.
- `src/brass_ai/recovery/value_phase.py`: dispatch the new schedule through the existing guarded optimizer loop.
- `src/brass_ai/recovery/config.py`: fresh-config defaults.
- `src/brass_ai/recovery/resume.py`, `league.py`: narrow reviewed transition and startup reporting.
- `configs/learning/selfplay-resume-sources.json`: pinned source approvals, preserving prior approvals and restrictions.
- `scripts/configure-training-g950.py`: exact-generation, checkpoint-hash-checked config helper; no run mutation.
- `tests/Recovery/test_training_g950.py`, `smoke_training_g950.py`, and the fresh-default expectation in `test_training_g850.py`.
- `docs/TRAINING_RECOVERY_LAB.md`, `docs/MLP_V11_LAB.md`: current instructions.

Detailed local artifacts: `artifacts/g950-review-summary.json`,
`artifacts/g950-validation.log`, `artifacts/g950-smoke/report.json`.
Unrelated Rust backend changes were left intact. Production training checkpoints
were not modified, and no long training was launched.

## Continue g951–970

After syncing these changes to the lab checkout, run from the repository root:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 950 --apply && \
python scripts/configure-training-g950.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 950 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-training-g950.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-training-g950.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 950 --iterations 20
```

The recovery command handles an interrupted uncommitted iteration without
rewinding a newer committed generation. The config helper verifies the full
g950 checkpoint and writes only the new JSON config. Training adopts the beta
hotfix under its existing run lock; **do not edit state.json or checkpoint tensors
manually, and do not use a fresh-buffer flag**. The new recipe and beta floor are
also selected by future fresh-config generation; completed IL filenames stay unchanged.
