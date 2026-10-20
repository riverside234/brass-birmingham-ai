# V4 share gate and critic holdout revision

Date: 2026-10-20.

**Superseded allocation:** this report records the original gate v1 implementation.
The user clarified that the ten released v4 tables belong to **history**.
Use the [gate v2 correction and generation 213 continuation](V4_HISTORY_G213_REVISION.md):
**160 self-play / 154 history / 6 v4**, with **308 PFSP opponent seats / 616 comparisons**.
The old v1 recipe remains readable; new configs and the helper use v2.
The historical behavior and validation below describe v1 only.

Fresh recipes and the explicit continuation helper now enable a one-way
ordinary-table share gate. PPG remains disabled.

## Signal and allocation

Use the existing scheduled rating **v4-panel**, which measures one learner
against three fixed, search-free v4 opponents, with four seat rotations. The
current recipe supplies 32 games / 8 setups every ten iterations. This is a
win-share statistic using official ties, not the sum of two training learners'
wins, PFSP pairwise outrank rate, or a deployment qualification.

The latest completed panel must have win share **strictly greater than 0.95**.
At the current panel size, 31 outright wins out of 32 would qualify. Missing
panels, incomplete support, a changed receipt or exactly 0.95 cannot qualify.
This is the requested point-estimate threshold, not a 95% confidence-bound
test. No extra evaluations are scheduled.

| Ordinary games per collection | Self-play | History | V4 |
| --- | ---: | ---: | ---: |
| Before qualification | 160 | 144 | 16 |
| After qualification | 170 | 144 | 6 |

The exact reduced percentages are **53.125% / 45% / 1.875%**. The change moves
ten tables from v4 to self-play. Ordinary games remain 320; eligible learner
seat trajectories increase from 960 to 980 before forced-action filtering.
Self-play still has four learners; history/v4 still have two learners and two
frozen opponents.

The gate is checked after pending scheduled ratings finish and before a new
ordinary collection is planned. Its activation progress and triggering panel
receipts are saved in `state.json` before collecting any games and copied into
the next checkpoint. Resuming a partial collection reproduces the same plan.
On adoption into an existing run, only its latest completed panel is eligible,
not an older peak. Once activated, the reduction persists; later v4 weakness
remains a review warning and does not automatically reverse the gate.

The exported g210 v4 panel is 0.8125, so it would **retain 160/144/16** until a
later completed panel qualifies. Live lab checkpoints were not changed.

## PFSP and seat coverage

The scheduler preserves every historical table's absolute game index, setup,
learner pair, random seed, opponent draws and recorded probabilities. Thus the
history pool still receives **144 tables, 288 opponent seats and 576 pairwise
comparisons** each iteration. PFSP's decayed S/C estimator, five-iteration
half-life, reciprocal IL share, hardness weights, uniform floor, admission and
capacity rules need no rescaling.

V4 seat-pair and profile cycling use the **actual v4-table ordinal**, including
the reduced portion after activation. Six v4 games cover all six learner pairs
once. Their twelve opponent seats cover all five admitted bot variants, with
two or three appearances each; two distinct variants remain on each table.
All v4 search remains disabled. Rating panels and their reference are unchanged.

The existing LR controller detects the changed actual table/learner counts and
resets its measurement window while retaining its multiplier and Adam state.

## EV holdout

The versioned `ppg-critic-holdout/3.0.0` contract selects **8 self-play, 8 history
and 2 v4 parent games per 320 ordinary games** in either allocation. Selection
uses an outcome-independent hash within each table type. Branches follow their
parent's selection and do not increase the population used to set quotas.
For other collection sizes the quotas scale with total parent-game count,
retain the existing minimum-two rule when feasible, and always leave at least
one game for training in each represented type.

Every seat and branch in a held-out family stays excluded from all critic fits
and utility calibration. Their learner rows remain eligible for actor losses,
so this is an online critic diagnostic, not an untouched playing-strength test.
At the reduced v4 allocation, two games provide holdout evidence and four games
remain available for critic training.

Old v2 receipts still reproduce their original 8/7/2 selection and hash. A
reviewed v2-to-v3 continuation starts fresh EV/stage/recovery evidence windows.
Attained GAE lambdas, mixed-table activation, snapshot weight/baseline, model
weights, both optimizers, RNG and training progress remain intact. Existing
economic health, minimum-game support, twenty-observation GAE and PPO KL guards
remain active. A PPG-enabled g210 checkpoint can adopt the previous PPG-off
transition and this holdout change in the same reviewed continuation.

## Commands

Generate from the actual committed run, then continue for twenty additional
iterations:

```bash
python scripts/joint7-v4-share.py && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-v4-share.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The helper writes only a config. It refuses to overwrite a different existing
file; use `--output configs/step5/<new-name>.json` if the run has other reviewed
settings. Fresh `feature-engineering.py configure` commands inherit the gate
and quota holdout. No new IL or dataset is required.

Iteration reports, compact history and exported reports retain `v4_share_gate`
including trigger evidence, actual table counts/fractions and historical
exposure. `actor.critic_split` and `value_phase.split` retain actual holdout
counts. Console lines show `v4=16/320[base]` or `v4=6/320[reduced]`.

## Validation and runtime

Focused regressions and an isolated PPO-only checkpoint/resume smoke are
recorded under `artifacts/v4-share-gate`. These exercise the real planning,
PFSP and holdout paths without running an evaluation campaign or modifying
existing training checkpoints.

- **184 distinct focused regressions passed**, including reruns after fixes.
  The final 33-test run passed in 18.01 seconds. Coverage includes threshold
  boundaries, missing/changed panel evidence, latest-panel selection,
  persistence, quota counts at irregular offsets, PFSP draw/probability parity,
  all six v4 seat pairs and all five bot variants, branch-family holdout
  exclusion, legacy receipts, composed PPG-disable/holdout migration, GAE
  support, LR composition changes and exact source/config review.
- The isolated smoke used the current **5,928,616-parameter utility model**,
  CPU/two threads and 128 authored Features-4 rows from eight game identities.
  PPO took **1.227 seconds**; the entire smoke took **5.052 seconds**.
  PPG accepted **zero** steps, created no cache, and its diagnostics left
  weights, both optimizers and RNG unchanged. The next PPO update after a
  saved checkpoint matched the uninterrupted path bitwise. These are bounded
  compute timings, not full game-collection throughput.
- The continuation helper passed against the exported g210 state. Only
  `value_phase` and `v4_share_gate` differ from that export's config. All
  **3,806 earlier source approvals** remain unchanged, and the model/feature
  compatibility digest is unchanged.

The aggregate receipt is
[validation.json](../artifacts/v4-share-gate/validation.json); the final smoke
is [report.json](../artifacts/v4-share-gate/ppo-smoke-final/report.json).
Earlier regression logs contain the initial failures; the aggregate checks
that every affected test passed its subsequent run.

The gate reads existing panel receipts and adds no inference. After activation,
20 additional learner seat trajectories are collected per iteration, about
2.1% more than before. Four more critic seat trajectories are held out in the
additional history game (when all snapshot rows are eligible). Wall time still
depends on game length, branching and batching; no production speedup or
playing-strength improvement is claimed.
