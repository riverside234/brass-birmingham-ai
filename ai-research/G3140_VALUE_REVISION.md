# g3140 value-phase revision: build steps 1, 5, 6, 7 and 8

- **Date:** 2026-11-17. **Run:** `joint7-v11-2-g1190-restart-3`, committed at g3140.
- **Design:** [TRAINING_IMPROVEMENT_PLAN_G2910.md](TRAINING_IMPROVEMENT_PLAN_G2910.md), sections 4.4–4.10 and the build order in section 8.
- **One change from the plan:** the search-value head is **not** paused (your decision). It keeps training at Stage C as now.
- **Tested:** focused local tests only (`tests/Recovery/test_training_g3140.py`). Nothing has run on the lab.

## What changes without any flag

These two parts of step 1 change no number in training and need no setting. They apply from the first iteration run with the new code.

| Change | Where | What to look for in the report |
| --- | --- | --- |
| Teacher label rows are staged on the GPU once per iteration; each PPO step gathers its 96 rows there instead of packing them on the host. The tensors are identical. A label whose width differs from its encoded choices keeps the old path. | `teacher_term.StagedRows` | `teacher_phase.term.staged_rows` (bytes, staging seconds). If host packing is what makes 61% of iterations slow (a hypothesis), `term_seconds` loses its 20–37 s iterations |
| The final cleanup purge (fresh reference scan, rehash, delete) runs beside the next iteration. It is joined before the next state commit, the next cleanup and the release of the run lock. A crash mid-purge resumes from its journal, as before. | `retention.BackgroundPurge` | `cleanup.purge = "background"`; the next report's `previous_cleanup_purge`; `cleanup_seconds` about 35 s lower |

## Items adopted with `--value-revision`

Add `--value-revision ITEMS` to your usual `scripts/continue-rust-g1190-restart-2.py` command, with `--expected-generation` set to the committed generation. Items are comma-separated, persist in the recipe, and only move forward: the signature transition refuses any step back.

| Item | Plan step | Effect |
| --- | --- | --- |
| `time` | 1 | The two full-buffer train scans of the value phase run every tenth generation. In between, the utility receipt is decided by the acceptance held-out games. |
| `validation` | 5 | The early stop validates on a quarter of the new games (56 self-play, 24 against the pool). In the next iteration they train like any other game. The permanent 5% hold-out goes. The check itself is unchanged: four main terms, real outcomes, weights .25 and .1, any increase rejects the interval. |
| `heads` | 6 | Fresh era heads replace the 2.0.0 local-scoring heads at this generation: each slot and link reads its own 16-wide trunk slice; Canal, Rail and Canal→Rail-end outputs; VP target divided by 5 and summed over seats, with the owner term's weight; weight .05. Head steps are kept when an interval is rejected (until `ownership-stop`). The trunk gradient ramps in through the existing rule (two fitted iterations head-only, then five ramp iterations). |
| `targets` | 7 | Win and utility heads train on half the outcome and half the λ=.95 return built from the critic outputs PPO freezes before its update. Each replay age stores its returns (`generation-N.bootstrap.npy`); an age staged before adoption uses its outcomes. The check stays on real outcomes. |
| `weight=0.15`, `weight=0.5` | 8 | Local-scoring weight for both eras, owner and VP. Needs `heads`. |
| `vp` | 8 | Total-VP weight .1 → .5 in training only. |
| `rail-end` | 8 | Canal rows also predict the Rail-end map, at half the Canal weight. Needs `heads`. |
| `ownership-stop` | (g3318) | The era heads get their own early stop: every 16 steps their held-out owner CE on the validation games decides whether they keep the steps (an increase keeps 0 and ends their fitting); after the value check stops, they fit alone with the trunk frozen. Needs `heads`. See [G3318_OWNERSHIP_CE_STOP.md](G3318_OWNERSHIP_CE_STOP.md). |

The value phase's cap (.003), beta floor (.25), schedule and strict early stop are unchanged.

### Suggested order

1. **g3140:** `--value-revision time,validation`. With a sharper early stop, the share of kept intervals tells sooner whether the later items help.
2. **About ten iterations later:** `--value-revision heads`. Wait until `local_scoring_auxiliary.warmup.after_commit.stage` reads `full` (about seven iterations).
3. Then `--value-revision targets`.
4. Then `--value-revision weight=0.15`. If kept intervals and the PPO baseline's explained variance on new games hold for about twenty iterations, `--value-revision weight=0.5,vp`.
5. Last, `--value-revision rail-end`.

Several items can go in one command; each is still adopted at that one committed generation.

## Where to read the effect

| Item | Report field |
| --- | --- |
| `time` | `value_phase.train_metrics_scheduled`, `train_metrics_measured`; `value_phase.timings.train_before_seconds` absent on unscheduled iterations |
| `validation` | `value_phase.split.heldout_games` (80), `acceptance_support`, `acceptance_heldout_rows`; first-interval keep rate in `interval_reports` |
| `heads` | `local_scoring_auxiliary.replacement` (once), `by_era` (training sums by era and open/settled), `heldout` every fifth generation: per era, kind and state the head's owner cross-entropy beside `reference_owner_ce` (in-sample copy-the-board rule), wrong-seat rate, VP error on the scoring seat beside always-zero |
| `targets` | `value_phase.buffer.bootstrap` (mean shift of the targets), `bootstrap_ages`, `training_targets` |
| weights | `local_scoring_auxiliary.weights`, `value_phase.training_targets.loss_weights` |

What to watch, from the plan's section 9: kept value steps falling toward zero for many iterations, or the PPO baseline's explained variance on new games (`actor.baseline_explained_variance`) falling, means the last item hurts; step its weight back by restarting from the checkpoint before it.

## Code

| File | Change |
| --- | --- |
| `recovery/training_revision_g3140.py` | Items, validation, forward-only projection, filename tag, the `previous` peel used by recipe recognition |
| `learning/local_scoring_heads_v2.py`, `recovery/local_scoring_v3.py` | Era heads and their companion (a subclass the 2.0.0 `Companion` builds for a 3.0.0 config) |
| `recovery/local_scoring_auxiliary.py`, `value_preservation.py`, `retention.py`, `local_scoring_adoption.py` | Dispatch, preservation weights by output, head-file hash by contract |
| `recovery/actor_value.py`, `ppg_replay.py`, `ppg_g210.py` | Rolling validation split, replay receipts under either fraction, bootstrap sidecar |
| `recovery/critic_bootstrap.py`, `updates.py`, `value_phase.py` | λ-returns; frozen outputs handed to the value phase; training-only targets and weights; scan schedule |
| `recovery/teacher_term.py`, `retention.py`, `league.py` | GPU-staged label rows; background purge |
| `rust_backend/lab_shadow_config.py`, `scripts/continue-rust-*.py`, `recovery/config.py` | Reviewed transition, `--value-revision`, config validation |
