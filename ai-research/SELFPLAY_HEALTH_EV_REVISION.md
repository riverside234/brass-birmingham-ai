# Self-play health and mixed-table EV revision

Date: 2026-10-19. Follows the [generation-55 review](SELFPLAY_G55_REVIEW.md), with two changes explicitly requested by the user.

## Removed IL sale-probability check

`il_kl_probe.py` no longer calculates IL/current sale mass, tests sale probability, or emits sale-check results. Shared health derives its behavior result exclusively from the remaining pass check; it cannot reuse an archived aggregate flag that included sale failures. New measurement/health reports, gate diagnostics and console lines contain no sale-probability fields. This reaches the calibrated KL limit, PPO LR gate, history-pool admission, GAE and snapshot-critic adoption through their existing shared-health plumbing.

The fixed probe, its metadata/tensor hashes and source checks are unchanged, so the completed IL and existing validation cache remain reusable. Overall/grouped IL KL still measure the full legal categorical policy. Observed sales per seat remain an economy metric with the existing threshold. Pass, VP, zero-VP and insolvency checks, LR bounds, calibrated limits/recovery latch and economic baselines are preserved. Removing this guard does not immediately clear a previously tightened distance limit or its latch; their existing update rules continue.

## Bounded mixed-table EV evidence

New `table-gae-ev-gate/2.0.0` defaults:

| Setting | Value |
| --- | --- |
| Maximum window age | 20 committed iterations |
| Minimum distinct held-out games | 8 for each mixed type and its self-play comparison |
| Minimum contributing iterations | 3 |
| Minimum healthy iterations with accepted PPG steps in window | 3 |
| Consecutive current health requirement | 10 iterations |
| Relative EV requirement | Mixed learner EV ≥ matched self-play learner EV − 0.05 |
| Activation freshness | New paired mixed/self-play sample in the current iteration |

At 16 v4 games per ordinary iteration and a 5% hash holdout, nominal support over 20 iterations is about 16 held-out v4 games before health exclusions. Eight games is an initial minimum, not a claim of statistical certainty. History and v4 qualify independently. Snapshot rows cannot substitute for learner evidence.

Each held-out learner group supplies row count, distinct game IDs, target mean/centered sum of squares, and residual mean/centered sum of squares. Parallel-variance merging gives `EV = 1 − Var(prediction − target) / Var(target)` over pooled rows; arithmetic averaging of iteration EVs would be incorrect. The self-play reference uses exactly the contributing generations for that mixed type. Repeated game IDs are rejected. Missing or degenerate evidence cannot pass. Health failures reset the ten-iteration health streak and are excluded from pooled evidence. Old entries expire by generation, including iterations with no v4 holdout.

The evidence describes **online predictions at each iteration's retained post-value weights**. It does not re-evaluate older states with the current model, and correlated decisions are not independent samples. Whole-game counts prevent many decisions from one game being mistaken for broad support. No extra inference pass, tensor cache, replay collection or evaluation campaign is added.

A restored held-out-loss rejection can still provide valid online predictions even with zero accepted steps that iteration. The gate requires accepted value work across the bounded window instead of on every sparse sample. The PPG KL guard, accepted-weight metrics, current PPO/economy/pass health, and existing self-play lambda-stage and snapshot-adoption requirements remain active. A missing v4 sample preserves unexpired evidence; activation still requires a fresh sample and enough support. Enabled tables keep their activation origins and follow later staged lambda changes without automatic rollback.

## Continuation and reporting

The current run's old table-gate block upgrades through an explicitly reviewed source/config transition. `actor-value-phases/4.0.0` adds the persistent EV window. At migration, start the window at the current committed generation and clear only the superseded mixed-table streaks. Do not synthesize old moments/game counts from scalar EV reports. Preserve model weights, both optimizers, RNG, global lambda state, already-enabled tables/origins, critic/fingerprint state, clone beta/trips, population, progress and checkpoint lineage. A restart immediately after source-transition commit performs the same migration from the unchanged old checkpoint.

Full reports and compact history retain window bounds, contributing generations, distinct games/rows, paired EVs, recent value-update support, health streak, freshness and reasons. Console `GAE-mixed` adds game/iteration support; `pass_fail` replaces the removed sale field. Existing exported reports remain historical evidence.

Continue with the same IL and run directories. An old local config is normalized through the reviewed migration; the checked-in examples already contain the new table-gate block. No config regeneration or IL training is needed:

```bash
python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 20
```

The source-review allowlist remains exact; unrelated config changes and pending old collections are rejected. The official search-free v4 bank and all preceding fixes remain in place. See [lab instructions](../docs/MLP_V11_LAB.md).

## Validation

**132 focused tests passed in 69.8 seconds** ([log](../artifacts/selfplay-health-ev-20261019/final-regressions.log)). Coverage includes sale removal with preserved pass/economy checks and probe-cache hashes; pooled-moment parity; distinct-game support versus correlated rows; sparse/absent/expired v4 samples; paired comparison generations; health/value-update gates; exact mid-window resume; restart immediately after the source/config transition; preservation of weights, Adam/RNG, beta, critic state and enabled table origins; PPO/PPG holdout and buffer exclusions; quality/LR gates; search-free v4 and five-profile scheduling.

The [isolated two-iteration compute smoke](../artifacts/selfplay-health-ev-20261019/compute-smoke/report.json) passed in **7.62 seconds**, using authored mixed-table Features-4 rows, the current 5,928,616-parameter v11.2 model and two CPU threads. Each iteration completed one PPO step. PPG accepted 2/2 steps in the first iteration; the second attempted two steps and correctly restored them after a held-out-loss increase. Both produced learner EV moments for self-play/history/v4 from retained weights. Warmup and insufficient support kept the mixed gates closed. Exact resumed next-update weights, both optimizers and gate states matched; the old disk cache was deleted after commit and the earlier checkpoint stayed unchanged. This is a compute/continuation smoke, not a native-game strength evaluation.

The [bookkeeping benchmark](../artifacts/selfplay-health-ev-20261019/benchmark.json) used 3,072 authored held-out rows per iteration, 8 self-play / 7 history / 1 v4 games, 20 warmup iterations and 200 timed updates. Median additional statistics plus window handling was **1.595 ms** (p95 **1.946 ms**); the full 20-iteration window serialized to **17,234 JSON bytes** for those game IDs. No additional model forward call or tensor cache is required. This measures bookkeeping, not end-to-end training throughput; its cost is expected to be small relative to the existing collection/update passes.

Changed runtime files: `il_kl_probe.py`, `il_kl_limit.py`, new `ev_evidence.py`, `gae.py`, `actor_value.py`, `value_phase.py`, `league.py`, `review_metrics.py`, `resume.py`, and the config-generation comment in `config.py`. Both current Step 5 example configs, exact source/handoff review receipts, focused tests, lab instructions and the historical review follow-up note are updated. No architecture, features, PPO reward/settings, teacher settings, opponent mix, frozen v4 bank or retention policy changed.

Final implementation: `997ce5bc26b8416d694f42ddac196feac8b365ee40bc412c6321ec85ae8885a5`. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. All preceding exact source approvals are retained; five explicit transitions add this revision and its table-gate migration. No intermediate development source is approved.

This change does not establish better training stability or playing strength. Production checkpoints, the completed IL and the exported 55-iteration run are not modified.
