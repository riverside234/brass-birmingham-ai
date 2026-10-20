# Snapshot critic weight: rolling R² and frozen stage references

20 October 2026. This supersedes the EV weight-recovery rule in
[SNAPSHOT_WEIGHT_RECOVERY.md](SNAPSHOT_WEIGHT_RECOVERY.md). It changes only snapshot-critic
weight control and its migration/reporting. The v4-to-history correction from
[V4_HISTORY_G213_REVISION.md](V4_HISTORY_G213_REVISION.md) remains intact.

## Evidence and decisions

`snapshot-weight-r2/2.0.0` keeps at most 20 **committed** iteration entries.
A supported decision needs at least 10 qualifying iterations, ordinary whole-game
holdout support (at least two games per represented required table type and eight
total games), finite nonconstant targets, and actual retained PPO or accepted PPG
snapshot training. Production holdouts remain 8 self-play / 8 history / 2 v4.
No critic fit sees those game families. Only their ordinary **learner** seat
moments enter this controller; branch continuations and snapshot seats do not.

Target central moments are merged with the parallel variance formula. Error
squares are reconstructed as `error_m2 + rows * error_mean**2` and summed.
The result is `R² = 1 - pooled_MSE / pooled_target_variance`, including bias.
It is neither centered-error EV nor an average of iteration scores. Reports give
R², learner MSE, signed prediction bias (`prediction - target`), target variance,
rows, game counts by table and contributing generations for the current iteration,
rolling estimate and frozen reference.

The initial reference is calibrated once from the latest ten qualifying
iterations. No weight change happens before this reference exists. Once the
rolling R² is at least the frozen reference, **five newly committed consecutive
healthy, qualifying iterations** permit 0.5 → 0.75, then 0.75 → 1.0. The reference
is frozen from the pre-increase rolling evidence; the next weight stage starts
with an empty evidence window and streak. Therefore a subsequent stage needs its
own ten qualifying iterations before it can make a weight decision.

A supported fresh observation below `reference - 0.02` reduces 0.75 or 1.0 to
0.5. A reduction clears stage evidence but **retains** the reference. Poor results
cannot ratchet it down. Changing the effective training/measurement regime also
clears the window and streak while retaining an established reference. The regime
includes PPO/PPG mode, holdout selection, actual table mix, GAE lambdas, reward and
exploration stage. Changing source alone does not discard compatible evidence.

Overlapping windows are correlated diagnostics. The checkpoint generation check
admits each new iteration once; replay/restart does not increment a streak.
Unhealthy observations with valid exposure remain in the performance estimate,
so poor results cannot disappear just because they fail health. Economic health,
warmup/retained PPO work, PPO early-stop and KL checks, and the PPG KL safeguard
remain required for increases. Actual PPO snapshot work can qualify even when an
enabled PPG phase accepts zero steps. GAE continues to use its existing EV windows
and health/work gates. Descriptor-based snapshot admission is unchanged.

## One-time joint7 baseline hotfix

The g213 export confirms the saved EV reference is `0.44701269779353614`,
snapshot weight is 0.5, and the current regime began at g211 (PPO only and
8/8/2 holdouts). Only **g211, g212 and g213** qualify under that regime.
Earlier reports cannot supply the requested ten-iteration baseline.

| Available evidence | Value |
|---|---:|
| Qualifying iterations | 3 (211–213), 7 still needed |
| Ordinary held-out games | 54 = 24 self-play + 24 history + 6 v4 |
| Ordinary learner rows | 14,917 |
| Provisional pooled R² | 0.3624826418501088 |
| Learner MSE | 0.15825827904097786 |
| Prediction bias | -0.0014482716411157864 |
| Target variance | 0.24824152161166513 |
| Valid ten-iteration baseline | **Not yet available** |

The provisional R² is **not installed as a baseline**. The helper can seed the
three available iterations and arm the once-only calculation. It freezes the
latest ten qualifying current-stage iterations when enough evidence exists,
earliest g220 if the next seven qualify without a regime change. The weight stays
0.5 meanwhile. Five later healthy qualifying commits are still needed for the
first increase. The old EV is archived for audit and never treated as R².

The helper checks the run lock, reviewed source, real committed checkpoint,
report/model lineage and pooled game support. It saves a state backup and an
exact checkpoint-bound override in `state.json`; it does not rewrite checkpoint
bytes. The next ordinary checkpoint contains the new controller state. Retries
are idempotent. Applying with insufficient evidence requires the explicit
`--defer-until-ready` option; otherwise it refuses to install a baseline.

Read-only preview of the received bundle:

```bash
python scripts/joint7-snapshot-r2.py \
  --reports results/joint7-v11-2-grouped-lr-critic-g0213-20261020T173708767483Z
```

For the stopped lab run at generation 213, after updating the code:

```bash
python scripts/recover-step5-iteration.py \
  --root "$PWD" --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 213 --apply && \
python scripts/joint7-snapshot-r2.py --apply --defer-until-ready && \
python scripts/joint7-v4-share.py && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-v4-history.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The first command archives any unfinished Ctrl+C collection without deletion;
it is a no-op if there is none and refuses a different committed generation.
The v4 helper preserves the previously requested corrected recipe. No new
snapshot-controller config block is needed. If ten eligible iterations already
exist when applying the hotfix, it computes and installs their baseline
immediately; `--defer-until-ready` is unnecessary in that case. Do not rerun an
iter-160 restore or restart training.

Subsequent continuation uses only the final train command. Twenty additional
iterations from 213 end at 233. The completed IL path, model/optimizer tensors,
RNG, progress, population, PFSP, GAE evidence, reward, LR/IL-KL controls, teacher
settings and checkpoint lineage are preserved. Fresh runs automatically use the
R² controller without this run-specific hotfix.

## Implementation and validation

- `snapshot_r2.py`: deterministic pooled statistics, bounded stage evidence,
  frozen references, strict committed state validation.
- `critic_data.py`, `actor_value.py`, `league.py`: controller-only migration,
  integration, report/console fields, exact state overlay on resume.
- `snapshot_r2_hotfix.py`, `scripts/joint7-snapshot-r2.py`: read-only preview,
  lineage-bound one-time application or deferred calibration.
- Source-review and IL-handoff registries: append the unchanged-architecture
  continuation routes without replacing earlier reviews.
- Focused tests cover bias, unequal target means/counts, zero variance,
  branch/snapshot exclusion, whole-game support, stage boundaries, 0.02 tolerance,
  health and actual work, restart/duplicate-commit rejection, reference preservation,
  g213 evidence insufficiency, complete/incomplete hotfixes, atomic backups and
  unchanged checkpoint/model/Adam/RNG/GAE state.

**139 focused checks passed** (138 in the final regression suite and one added
production-path health-guard regression). This includes the previous v4/PFSP
correction, GAE/PPG compatibility, source review and interrupted-run recovery tests.
Authored old checkpoints in legacy tests now explicitly use the archived EV
schema; new checkpoints enforce the R² stage contract. All 5,091 previous source
reviews remain unchanged, with 646 continuation routes appended.

The isolated CPU smoke used the current **5,928,616-parameter utility model**,
128 authored Features-4 rows, two CPU threads and 1–4 legal candidates. PPO took
**1.329 seconds**, and the full smoke took **5.807 seconds**. PPG accepted zero
steps because it remains disabled. The next PPO update after checkpoint/resume
matched bitwise for model weights, both optimizers, phase state and RNG; the
original checkpoint bytes remained unchanged. This smoke does not supply
production whole-game support for weight changes; dedicated synthetic stage and
real exported-moment tests exercise those decisions.

Logs, the baseline preview and machine-readable validation are under
`artifacts/snapshot-r2-revision/`. Production source identity:
`efc26563d6d0daa236d81a4d428a38b36624d7e72391c67bb84236a2515676b2`.
The actual lab hotfix has **not** been applied from this local workspace; use the
commands above on the stopped lab run.

No long training, native game campaign or strength evaluation ran. The change
uses existing held-out moments; it adds no model forward passes or game collection.
The window stores at most twenty iterations of small moment/game-ID records.
Training stability or playing-strength improvement is unproven.
