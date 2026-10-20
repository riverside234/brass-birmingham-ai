# Smaller PPG updates and stratified critic holdout

This implements the follow-up to the [g110 review](SELFPLAY_G110_REVIEW.md).
Continue from **generation 110 / 35,200 ordinary games**, with the same run and
completed v11.2 IL. No restart, IL regeneration or model architecture change.

## Effective settings

| Setting | Previous | Revised |
| --- | --- | --- |
| Separate value Adam LR | 6e-6 | **3e-6** |
| PPO base LR / controller | 6e-6 / existing controller | Unchanged |
| PPG max epochs / batch size | 3 / 4,096 | Unchanged |
| Value-policy KL cap | 0.002 | Unchanged |
| Held-out-loss acceptance | Must not increase | Unchanged |
| Holdout | Independent game-ID hash threshold | Deterministic hash ranking within each table type |
| Nominal holdout fraction | 5% | 5%, with minimum two games per type when feasible |
| Ordinary game mixture | 50/45/5 | Unchanged |

The signed block is `ppg-value-phase/3.0.0`, with `learning_rate=3e-6` and
`holdout={"contract":"ppg-critic-holdout/2.0.0","minimum_games_per_table":2}`.
Its LR is independent of PPO warmup/multiplier and must not exceed PPO base LR.
Older contracts retain their equality rule. The exact reviewed migration halves
the preceding value LR and changes only this block, preserving other settings.

For n games of each type, select `min(n-1, max(2, floor(0.05*n + 0.5)))`, or zero
for n=0/1. A singleton trains only; n=2 gives one holdout and reports limited
support. Ranking uses a versioned hash of table type and recorded game UUID,
never scores, rewards, EV or predictions. Row/worker order does not matter, and
selection consumes no Python/PyTorch training RNG.

| Type | Collected | Critic holdout | Critic training |
| --- | ---: | ---: | ---: |
| Self-play | 160 | **8** | 152 |
| History | 144 | **7** | 137 |
| v4 | 16 | **2** | 14 |
| Total | 320 | **17** | 303 |

Every eligible seat and forced/nonforced row of a held-out game is excluded
from PPO critic fitting, utility calibration and current/previous PPG critic
caches. Actor and policy-cloning/probe eligibility are unchanged. This is a
critic holdout, not an untouched policy evaluation. Rare types are deliberately
overrepresented in diagnostics; aggregate loss/EV remain row-weighted. Compare
per-table EV as well as aggregate EV across this selection change. Two v4 games
per collection improve support but do not guarantee low-variance estimates.

## Continuation and cache behavior

`actor-value-phases/6.0.0` preserves weights, both Adam moments/steps, RNG, PPO
warmup/LR-controller state, counters, population, lineage, cloning beta and trip
history. Only the value optimizer group's LR changes.

Start selection-dependent evidence at the committed migration generation: clear
the mixed-table EV window, GAE EV/healthy/table streaks and snapshot recovery
streak. Preserve active lambda, qualified tables and their origins, cumulative
accepted-value count, snapshot weight, fixed phase-one EV baseline, activation
progress and fingerprints. No gate is promoted, rolled back or re-baselined.
The migration is reported and does not repeat after the next checkpoint.

The first new-split iteration excludes the previous hash-selected cache. If the
reward coefficient changes simultaneously, this is the same single current-only
iteration, not two separate skips. Old files survive until the next successful
commit; then cleanup removes them and normal two-iteration reuse resumes.

`ppg-critic-disk-buffer/3.0.0` receipts bind the full collection's game/type map
and held-out membership. Missing caches rebuild from admitted sealed replays
using that selection and original learner/snapshot order, reproducing the
checksum. Tampered selection, held-out cache members, changed game/type identity
and reuse of a preceding collection's held-out games are rejected. Cache tensor
layout and directory ownership are unchanged.

The prior one-time IL-KL limit **0.2288818359375**, group limit
**0.457763671875**, insolvency cost **0.0075** and cap **0.075** remain in the
prepared continuation. Repeating the helper cannot multiply the limit again.

## Continue from generation 110

Run in the updated lab checkout:

```bash
python scripts/joint7-g110-hotfix.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-g110-ppg.json \
  --refine-ppg --apply && \
python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic-g110-ppg.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 110 --iterations 20
```

This works whether the preceding one-time g110 hotfix was applied or not, while
training is still at 110. An identical output config is accepted; different
contents are never overwritten. Omit `--apply` to preview and validate. The
helper checks the actual checkpoint, frozen opponents and source/config approval
before writing. The lab checkpoint is not available in this checkout; no lab
files were changed here. The committed g110 rating is reused.

Later use the same config with the actual expected generation, or omit that
optional assertion. Fresh config generation also adopts the revised defaults;
explicit old configs keep their old LR/split until migrated.

## Validation and limits

**161 distinct focused tests passed** across the suites and targeted reruns.
The final smoke completed in **17.54 seconds**. Hash selection alone averaged
**0.93 ms** for 320 game IDs across 100 synthetic collections; this excludes
row partitioning, tensor preparation and model work.

Focused tests cover quotas, deterministic selection, small strata, leakage,
separate LR, exact g110 admission, unauthorized-edit rejection, both optimizers,
evidence migration, receipt-bound rebuild, tampering, cache cleanup and hotfix
idempotence. Existing GAE, critic, snapshot recovery, PFSP, IL-KL/LR, buffering
and report regressions were also run. Logs and the final validation receipt are
under `artifacts/ppg-refinement-20261019/`.

The bounded smoke uses the **5,928,616-parameter v11.2 model**, authored CPU rows
and generation-110 checkpoint metadata. Both g111 and g112 accept one value
step at 3e-6, with policy KL around **1.6e-6 / 1.2e-6** and lower held-out loss.
One branch continues in memory; another restores g111. Next weights, both
optimizers, gates and RNG match exactly; the old checkpoint stays unchanged.
The first iteration uses current rows only, the next reuses previous rows.
See `tests/Recovery/smoke_ppg_refinement.py` and
`artifacts/ppg-refinement-20261019/smoke-verified/report.json`.

There are no extra games or configured optimizer epochs. Selection adds bounded
game metadata, not tensor copies. If smaller steps allow more epochs past the
guards, actual PPG time can increase toward the unchanged cap. Production GPU
timing, sustained acceptance rates and playing-strength improvement are unproven.

## Changed files

- `value_phase.py`, new `critic_holdout.py`, `config.py`: versioned defaults,
  independent value LR and whole-game stratification.
- `actor_value.py`, `value_buffer.py`: optimizer/evidence migration, leakage
  checks, deterministic cache rebuilding and transactional cleanup.
- `resume.py`, source-review registries, `g110_hotfix.py` and its CLI: explicit
  continuation and optional `--refine-ppg`.
- `league.py`: value LR and actual v4 holdout counts on the console. Existing
  full/compact reports retain split, migration and buffer details.
- Latest template, separate g110-PPG config, tests, smoke and lab instructions.
  The prior g110-only config is unchanged.
