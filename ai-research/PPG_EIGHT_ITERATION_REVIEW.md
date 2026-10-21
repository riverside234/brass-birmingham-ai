# Eight-iteration PPG continuation

2026-10-21. Final requested recipe: **every eight committed iterations, uniform age weights, ordinary held-out games from all eight iterations, and an adaptive epoch cap**. This supersedes the initially proposed four-iteration cadence, recency decay, latest-four-only acceptance set, and two-consecutive-phase cap rule. The final controller uses three of the last four phases. Value Adam resets at phase start, with warmup then decay within the phase.

The exported run is committed at **g418 / 133,760 ordinary games**. This change prepares a reviewed continuation; it does not edit the existing training checkpoints or start production training. Model architecture, IL identity, PPO optimizer and LR controller, reward, exploration, opponent sampling, teacher settings, rating settings and economic safeguards are unchanged.

## Recipe and implementation

| Setting | Effective value |
| --- | --- |
| Auxiliary contract | `ppg-value-phase/5.0.0` |
| Enablement | Enabled in newly generated configs and the explicit continuation config |
| Buffer | Eight sealed committed iterations, ages 0–7 |
| Schedule | Every eight committed iterations, persisted in the checkpoint |
| Age weights | All 1; no recency decay |
| Epoch | One shuffled pass over every eligible critic training row, at most 4,096 rows per step; initial cap 2, adaptive bounds 1–6 |
| Value LR | Base 3e-6; first interval warms 0.1× → 1×, then linear decay to 0.3× at the planned cap; PPO LR multiplier does not apply |
| Value Adam | Reset moments and step counters at the beginning of each scheduled phase |
| Acceptance | All eight ordinary learner holdout sets, normally 144 games: 64 self-play, 64 history, 16 v4 |
| Validation interval / patience | 32 optimizer steps, also at epoch ends / two checks without a new best |
| Loss comparison | Equal-per-game paired differences against the best retained checkpoint; reject if mean increase exceeds one standard error |
| Policy clones | Up to 2,048 per age, 16,384 total |
| Independent safeguard probes | 2,048 age 0 plus 512 per older age, 5,632 total |
| KL caps | Overall and age-0 exact categorical KL must each be at most 0.002 after every attempted step |
| Cloning beta | Preserve g418's saved 8; double and retry improving but KL-unsafe intervals, bounded at the user-approved 32 |

The schedule starts at the committed adoption boundary, without reconstructing old data. For this run it collects g419–g426 before the first auxiliary phase at **g426**; the next phases are **g434, g442, ...**. A restart midway through the cycle restores the same origin, next phase, cache receipts, Adam moments and RNG state. Fresh runs start their first phase at g8.

Uniform age weights mean every eligible critic row has the same age multiplier. The existing snapshot critic weight still applies once to snapshot losses. A complete planned epoch includes every training row exactly once; repeated epochs and bounded beta retries are counted separately. Rejected intervals, unresolved KL stops and patience can reduce retained usage. Reports separate attempted presentations, retained presentations and distinct retained rows by age/role, including branches.

The epoch controller raises the cap by one when the best retained checkpoint falls in the final quarter of the planned budget in three of four phases, and lowers it by one when the best falls in the first epoch in three of four. The initial unchanged state is position zero, so consistently unsuccessful value training can lower the cap. The first decision needs four scheduled observations. Cap changes reset the window to avoid repeatedly voting with the same old budget's evidence. Window, cap and schedule persist together. KL-limited phases remain in the window without an up/down vote: they request more cloning pressure rather than being treated as evidence about overfitting or the value-training budget. The best position excludes retry work.

Critic caches preserve recorded Monte Carlo targets, original relative seat descriptors and descriptor generation. Holdout selection remains the existing 8/8/2 quota per 320 games. Whole parent/branch families stay in the same partition. Held-out families enter neither auxiliary critic fitting nor cloning/probe selection. Acceptance and loss-by-age diagnostics use ordinary learner holdout rows only, excluding branch and snapshot outcomes. Ordinary PPO actor treatment of these games remains unchanged.

Cloning and probe sets are disjoint, deterministic and stratified by age, table type, and Canal/Rail × action/other, with minimum represented-stratum quotas. Their weights correct quota sampling to the eligible decision population without recency decay or a second loss correction. Only non-forced learner decisions qualify, including eligible branch continuations; frozen opponents, fork moves and copied prefixes do not. The frozen reference is captured immediately before the auxiliary phase from the post-PPO model in eval mode, at raw temperature 1 without exploration mixing. Every selected state's entire legal-action distribution is used. Probes report age, table and decision-group KL/support separately.

The best validation checkpoint is retained only while both KL caps hold. Model weights and **matching value-Adam moments, step counters and rate** are restored together; the PPO optimizer remains unchanged. A tolerance-safe but worse final interval cannot displace the best checkpoint. Utility calibration is rebound to the retained weights and original current training targets; no stale calibration receipt is reused.

When a tentative step exceeds either KL cap, its paired held-out value loss is diagnostic only. If that loss improves on the safe best, restore the safe best weights/Adam, double beta and retry the same interval using the same cloning draws. For this run beta 8 allows retries at 16 and 32. At the bound, or without held-out improvement, stop safely. No unsafe weights are accepted, no KL allowance is relaxed, and retries do not extend the logical epoch budget. Attempted and retained work, unsafe diagnostics, and beta changes remain explicit. Any attempted cap violation continues to fail that iteration's existing KL health safeguard.

At each scheduled phase, discard stale **value** Adam moments before capturing its initial rollback state. The preserved incoming model weights are unchanged by this reset. Apply 3e-7 at planned step 1, reach 3e-6 at step 32, then decline linearly to 9e-7 at the planned cap (approximately step 506 initially). Retrying a step reuses that step's rate. The retained checkpoint saves matching moments and rate and resumes exactly; only the next scheduled phase performs another deliberate reset. The separate PPO Adam, PPO warmup and LR-controller state are never reset. This is an experimental optimization schedule, not proof that stale moments caused the earlier behavior.

GAE and snapshot controllers continue to use actual retained PPO critic work on ordinary iterations. Auxiliary KL violations also fail health. Gate diagnostics use only the current iteration's ordinary holdouts, so the eight cached sets are not recounted as eight new observations. Existing GAE evidence and attained lambdas survive adoption; snapshot R² starts evidence for the changed training regime while preserving the frozen reference and current weight.

## Disk ownership and continuation

`.ppg-eight-cache` contains mmap critic arrays, complete legal-action inputs for selected policy states, and immutable JSON receipts. Checkpoints bind the exact receipt list and array hashes. Only minibatches are transferred to the training device. Small row metadata remains resident; mmap does not guarantee that all mapped pages stay out of RSS or the OS page cache.

Expired files are deleted only **after checkpoint and state commit**. Staging can temporarily hold a ninth iteration. A disk-space check rejects a collection cache write before attempting to discard required old data. Missing or corrupt committed arrays fail explicitly. No cached trajectory is admitted to a new PPO update.

Back up or move this directory with the latest full checkpoint and run state. An older full checkpoint can require its matching expired cache backup. Existing ordinary replay retention stays at two iterations and cannot reconstruct the eight-age cache. The normal report exporter includes the full and compact PPG reports, but intentionally omits these bulk training arrays.

Run from the Linux repository after syncing the changes:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 418 --apply && \
python scripts/joint7-ppg-eight.py --expected-generation 418 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-eight.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The recovery command archives any uncommitted collection after Ctrl-C; its generation guard prevents accidentally applying g418 instructions to a later checkpoint. The config helper verifies the exact reviewed transition and changes only `value_phase`. It refuses conflicting output and does not modify a checkpoint. This continuation targets g438, with auxiliary phases at g426 and g434. On later clean continuations use only the training command, with the same config. Retain the existing IL folder name. Do not rerun older entropy or restart-160 hotfixes.

Source review is from `d0860bb58611f1645d27d6a3df2e5c308d567854808315d8d434b104895ad151` to `bd1a6bd5522fc0e54a18309afa9582cfcfd9f64c4b0622274e3081ae290fdfe8`. Learning/model compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. All 34,396 pre-existing expanded source-review records remain unchanged. Existing reviewed routes are carried forward, with a separate exact PPG-config migration; unknown source/config edits and incompatible pending collections remain rejected.

## Validation and measured scope

The focused suite completed **123 tests: 122 passed, one CUDA-only check skipped** on this CPU-only machine (174.05 s). It covers the new schedule/weights/eight-age holdouts, quota correction, complete legal distributions, branch-family separation, original targets/descriptors, pre/post-commit cache expiry, corruption rejection, paired acceptance, adaptive-cap boundaries/restart, beta retries/bounds, value-Adam resets, warmup/decay, KL rollback with matching Adam, exact g418 configuration migration, existing GAE/PPG behavior, PPO probability guards, source reviews and interrupted-iteration recovery. The broader run caught a legacy manually constructed controller lacking the new rolling-mode flag; the compatibility fallback is fixed and the final suite passes.

```bash
python -B -m unittest \
  test_ppg_eight test_ppg_window test_ppg_refinement test_ppo_only \
  test_gae_value_phase test_value_buffer test_value_batch_performance \
  test_categorical_kl_guard test_selfplay_input_cache_resume \
  test_source_reviews test_interrupted_iteration -v
```

These test modules require the repository's test directories on `PYTHONPATH`, including `src`, `tests/Recovery` and `tests/R2`, and the normal installed runtime dependencies. Local log: `artifacts/ppg-eight-review/focused-tests-final.log`.

The isolated smoke used the current **5,928,616-parameter v11.2 utility model on CPU**, with authored 18-game fixtures and reduced row/sample limits. It filled eight caches, performed ordinary PPO between auxiliary phases, and compared uninterrupted g426 with an exact restart from g425. Weights, both optimizer states, RNG, beta, epoch-controller evidence and schedule matched exactly; incoming checkpoint hashes were unchanged. It exercised all eight holdout ages (six ordinary held-out fixture games per age, 48 total), not 144 full-size production games. The fixture uses 128-row steps and a two-step validation/warmup interval; the production recipe's 4,096-row/32-step configuration is checked separately. The real smoke did not trip KL; injected-violation regressions exercise beta 8 → 16 → 32, refusal at the bound, and matching safe Adam rollback.

| Smoke measurement | Result |
| --- | ---: |
| Total verification wall time | 87.24 s |
| First auxiliary phase wall time | 19.48 s |
| Planned / attempted / retained value steps | 14 / 14 / 12 |
| Distinct rows per epoch / distinct retained critic rows | 867 / 867 |
| Planned / retained critic presentations across two epochs | 1,734 / 1,478 |
| Retained distinct learner / snapshot rows | 771 / 96 |
| Retained distinct branch rows | 3 |
| Selected cloning / independent probe states | 320 / 124 |
| Retained cloning presentations / distinct states | 288 / 192 |
| Final overall / age-0 KL | 0.00000983 / 0.00000773 |
| Beta before → after | 8 → 8; no retry needed |
| Phase LR first → peak → last | 3e-7 → 3e-6 → 9e-7 |
| Epoch cap before → next | 2 → 2; first of four evidence phases |
| Mean equal-per-game held-out loss, before → retained | 1.359727 → 1.328652 |
| Measured eight-cache peak for one smoke run | 21,313,830 bytes |
| Sampled RSS peak during the first auxiliary phase | 836,710,400 bytes |
| OS lifetime RSS high-water observed in that phase | 849,633,280 bytes |
| OS lifetime RSS high-water observed in the repeated restart phase | 1,046,876,160 bytes |

The first two-step interval increased held-out loss by 0.017226 versus SE 0.000843 and was rejected; later intervals improved the retained checkpoint. The best state is at the end of epoch two, supplying one late-best observation without prematurely raising the cap. These results verify rollback and acceptance mechanics. Authored-fixture loss reduction is not evidence of playing-strength improvement, production GPU speed, or full-scale memory use. The cloned restart uses another isolated cache directory; the cache figure above is per run, not total smoke artifact storage. The report also records the repeated restart phase's RSS and OS lifetime high-water separately. The suite and smoke overlapped on the same host, so these wall times are verification timings, not an isolated throughput benchmark.

Reproduce with a new isolated destination:

```bash
python -B tests/Recovery/smoke_ppg_eight.py \
  --output artifacts/ppg-eight-smoke-new
```

Local evidence: `artifacts/ppg-eight-review/smoke-final/report.json`. No production checkpoints, long training, or rating campaign were run.

An isolated config-helper check copied only the exported state into a temporary owned test run. It verified the g418 → g426 first-phase schedule, idempotent output, wrong-generation rejection, exact agreement with the checked-in continuation config, and unchanged state hash.

## g418 observations and runtime implications

The export reports PPO times of **71.21 s at g417 and 70.07 s at g418**, with 46 retained PPO steps in each. PPG is disabled in those reports, so they cannot show auxiliary benefit or overhead. The existing v4 EV warning remains: pooled v4 learner EV is about **0.162**, versus self-play **0.369**, across the 20-iteration evidence window. The anchor budget/latch also blocks LR increases. This implementation does not weaken either safeguard or claim PPG has resolved those conditions.

At g418's 129,522 eligible critic training rows per iteration, eight uniform ages give about **1,036,176 row presentations / 253 steps per epoch**. The initial two-epoch cap is therefore about **506 steps**, or at most **63.25 planned steps per ordinary iteration** averaged across eight iterations, before retry overhead. The earlier single-epoch comparison of about 31 steps per iteration no longer describes this two-epoch cap. Six epochs would permit roughly 1,518 steps; increases require the controller's evidence and do not occur immediately. Actual retained work can be much smaller because of KL or validation stops.

At 32-step validation intervals, 253 steps yield eight checks per epoch (16 for the initial two-epoch cap). The all-eight acceptance set has approximately **40,480 ordinary learner rows** at g418's 5,060 per iteration; the approximately 57,704 total critic held-out rows also include snapshot seats, which are excluded from acceptance. The acceptance set still contains 144 complete ordinary games. A beta retry may require another unsafe-state loss check, and every attempted step checks the complete policy safeguard probe. No production wall-time reduction is claimed from these arithmetic counts.

The current row layout is 14,503 bytes. Using g418's 136,735 total cached critic rows gives **15.86 GB** for eight critic arrays. If each age stores 4,096 selected policy states at the observed mean 86.10 legal actions, complete 327-float action vectors add approximately **3.69 GB**. This projects **19.56 GB steady cache data**, with roughly **22.00 GB** during staging of a ninth iteration, plus metadata and any retry-overwrite overlap. These are decimal-GB projections from report counts, not measurements of full production caches; represented policy strata and action counts can differ.

Reports now measure actual critic/action bytes, pre-commit disk peak, phase RSS, OS process-lifetime RSS high-water, CUDA allocator high-water when available, and phase seconds. Inspect the first full g426 phase before making throughput or memory claims. Losses are reported for every age, table type and era so that stale-data behavior is visible; no automatic decay or buffer shortening is introduced.

## Files changed

- `src/brass_ai/recovery/ppg_config.py`: strict recipe, exact migration, durable cadence, uniform age weights and per-phase rate schedule.
- `src/brass_ai/recovery/ppg_epochs.py`: persisted three-of-four adaptive epoch cap and best-position evidence.
- `src/brass_ai/recovery/ppg_replay.py`: owned mmap critic/policy storage, provenance, receipts and post-commit expiry.
- `src/brass_ai/recovery/ppg_sampling.py`: disjoint stratified policy samples and inclusion-weight correction.
- `src/brass_ai/recovery/ppg_auxiliary.py`: uniform epochs, paired eight-age holdouts, raw-policy cloning/KL, bounded beta retries, value-Adam reset/warmup/decay, best weights/Adam retention and resource/sample reports.
- `src/brass_ai/recovery/actor_value.py`, `value_phase.py`, `config.py`, `resume.py`, `league.py`: lifecycle, gate evidence, fresh defaults, reviewed continuation and console reporting.
- `scripts/joint7-ppg-eight.py`, `configs/step5/joint7-v11-2-grouped-lr-critic-ppg-eight.json`: exact continuation helper and config.
- `configs/learning/selfplay-resume-sources.json`, `imitation-handoff-sources.json`: source compatibility receipts without model reinterpretation.
- `tests/Recovery/test_ppg_eight.py`, `smoke_ppg_eight.py`: focused regressions and isolated restart smoke.
- `tests/Recovery/test_gae_value_phase.py`, `test_ppg_refinement.py`, `test_ppo_only.py`: fresh-default expectations and consistent archived test fixtures; explicit legacy recipes remain testable.
- `docs/MLP_V11_LAB.md` and this report: current recipe, continuation command, validation and limitations.
