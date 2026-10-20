# PPO-only self-play continuation

2026-10-20. PPG is disabled in the fresh template/generator and in
`configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only.json`. The continuation
config differs from the exported g210 recipe only in `value_phase.enabled=false`.
No live checkpoints or exported logs were edited.

## Training and gates

PPO still updates both policy and critic. The existing MC critic targets, value
loss weight, head calibration, rehearsal, behavior verification, exploration,
branching, opponent mix, ratings and LR/IL-KL controllers are unchanged.
Read-only diagnostics cannot certify a pre-PPO utility-calibration receipt:
that stale receipt is cleared, retaining the existing critic-train-only head
calibration safeguard for the next iteration.

The previous flag alone was insufficient: configuration validation required PPG,
the disabled path discarded the critic holdout, and GAE/snapshot adoption waited
for PPG optimizer steps. Those dependencies are now separated:

| Mechanism | PPG disabled |
| --- | --- |
| Extra value fitting, clones, value Adam updates | Zero attempted or accepted steps |
| Two-iteration PPG cache | No load, rebuild, stage or use |
| Whole-game critic holdout | Same stratified selection; excluded from PPO critic and head calibration |
| EV and descriptor diagnostics | Read-only current post-PPO inference with true/unknown descriptors |
| First GAE transition | Existing minimum 10 critic-update iterations, EV threshold/streak, or 30-iteration fallback; counts actual PPO critic work |
| GAE 0.98 → 0.95 | Same 20 healthy observations, 16 critic steps over ≥3 iterations, economic/PPO health and common-fit rating evidence |
| History/v4 GAE extension | Same bounded EV window, ≥8 distinct games, ≥3 paired iterations, ≥3 critic-update iterations and 10-iteration health streak |
| Snapshot critic admission | Actual PPO critic work plus held-out descriptor gain and existing health |
| Snapshot weight reduction/recovery | Same EV baseline and actual retained snapshot exposure; PPO snapshot rows qualify |
| Entropy, LR, IL-KL and pool controls | Existing health/KL inputs; no artificial PPG failure when disabled |

Only retained PPO steps with a nonempty critic batch and positive `value_weight`
count. Actor-only steps, head calibration, rejected PPG attempts, available but
unused rows and old cached rows do not count. The existing field names
`value_iterations` and `accepted_steps` remain in gate state for compatibility;
reports explicitly identify their evidence source as `ppo` or `ppg`. PPG
`optimizer_steps` remains zero.

GAE support does not establish improved critic quality by itself. Held-out EV,
economic health, PPO KL and rating checks still control adoption. Already
attained lambdas remain attained; no automatic rollback or new thresholds were
added.

## Checkpoint and cache handling

The exact v7 → v8 actor/value checkpoint migration changes only this requested
mode. It retains model weights, both Adam states, PPO warmup/LR controller, RNG,
committed iteration/game counts, population, checkpoint lineage, active lambdas,
table activation/origins, snapshot weight/baseline and exploration stages.
Inactive PPG Adam, clone beta and trip history remain unchanged in checkpoints.

New PPO-based evidence starts at the actual committed progress. GAE qualifying
work counters, health/EV windows and unfinished snapshot recovery streaks restart;
old PPG evidence is not relabeled. The old count and buffer receipt remain as
transition provenance. This does not restart training or lower an attained gate.
Repeating startup against the old checkpoint is deterministic; subsequent v8
checkpoints restore the new windows exactly.

The disabled path does not require an old derived PPG cache to exist. After a
successful PPO-only checkpoint/state commit, cleanup validates ownership and
every file name/path before unlinking only known derived arrays/receipts. It
never recursively deletes a directory. Source replays, model checkpoints and
ordinary/rehearsal caches are unaffected. Unknown artifacts fail closed.

Source/config changes still require `--resume-reviewed-source`. Pending
incompatible ordinary collections are rejected through the existing safeguard.
Feature/model compatibility and completed IL identity are unchanged.

## Continue the current run

From the repository directory on the lab machine:

```bash
python scripts/joint7-ppo-only.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only.json

python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The helper reads the actual committed recipe, validates the exact reviewed
transition, and writes only a config. It is idempotent for identical output and
rejects a conflicting file; use a new output name if the run's unrelated recipe
has since changed. Do not use `--fresh` for continuation. No new IL is needed.

For future fresh training, the normal `feature-engineering.py configure` command
now sets `value_phase.enabled=false`, including when its input template had PPG
enabled. Previously generated enabled configs remain explicit historical recipes.

## Reporting and runtime

Console: `value_steps=0/0 value_stop=disabled` and
`PPO-critic-window=...steps/...iters`. Reports/compact history retain current
post-PPO held-out metrics and `critic_training.source=ppo`, separate from the
zero PPG-step counts. The old `value_phase_seconds` timing now includes only
diagnostic inference in this mode, so it need not be zero. The exporter already
retains these nested fields.

Expected savings are the removed PPG optimization, policy cloning, per-step KL
checks, rollback copies and cache I/O. Holdout/descriptor inference remains.
No production speedup or playing-strength improvement is claimed.

## Validation

**126 focused tests passed.** Coverage includes current-g210 source/config
admission, exact mode migration and restart, missing PPG cache tolerance,
post-commit cache cleanup, whole-game holdout exclusion, actual PPO critic-step
counts, snapshot activation/recovery, both GAE transitions, history/v4 support,
warmup/empty/NaN/early-stop/economic-health rejection, and the existing optional
PPG, disk-buffer, exploration and queued-branch regressions. Corrupt checkpoint
streaks and evidence predating the mode transition are rejected.

The isolated CPU smoke used the current **5,928,616-parameter utility model**,
128 authored Features-4 rows from eight game identities, and 32 holdout rows.
It completed one PPO step with critic work and **zero PPG steps**. Holdout
diagnostics changed no weights, optimizer state or RNG. Saving/loading and
performing the next PPO update reproduced weights, both Adam states, gate state
and RNG **bit for bit**. No PPG cache was created.

The smoke took **4.27 seconds**, including **0.92 seconds** for its first PPO
update and **0.035 seconds** for the read-only holdout diagnostics. These are
small authored-input CPU checks, not native gameplay, production throughput or
strength measurements. No long training or evaluation campaign was run; the
remote full training checkpoint was not available locally.

Validation receipts/logs: `artifacts/ppo-only-revision/validation.json`,
`focused-final.log`, `integration-final.log`, `continuation-final.log`, and
`smoke-verified/report.json`. The config helper was exercised against the
exported g210 state and produced the checked-in continuation config with only
the enabled flag changed. All 3,170 earlier source approvals were preserved;
the learning/feature compatibility digest is unchanged.

## Changed files

- Defaults and continuation: `configs/step5/joint7-latest.json`, new
  `joint7-v11-2-grouped-lr-critic-ppo-only.json`, `scripts/joint7-ppo-only.py`,
  `recovery/config.py`, `resume.py`, and the two source-review registries.
- Execution and evidence: `recovery/actor_value.py`, `value_phase.py`,
  `value_buffer.py`, `updates.py`, `gae.py`, `gae_health.py`, `ev_evidence.py`,
  `critic_data.py`, `league.py`, and `review_metrics.py`.
- Validation: new `test_ppo_only.py` and `smoke_ppo_only.py`; default/holdout
  expectations updated in `test_gae_value_phase.py` and `test_ppg_refinement.py`.
- Instructions: `docs/MLP_V11_LAB.md` and this report. Existing branch-scheduling
  changes remain intact.
