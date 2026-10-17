# V11.1 fresh self-play recipe and PPO LR feedback

Implemented October 16, 2026. This changes self-play training, not the v11.1 model or Features-4 encoding. Completed IL from the exact preceding v11.1 source remains admissible through a pinned handoff receipt. Existing IL and self-play runs were not modified.

## October 17: calibrated IL-KL limits and rehearsal floor

The current fresh-run default is now controller **1.2** with `auto-il-kl-limit/1.0.0`. The October 16 sections below remain implementation history. Current configuration, fresh-start/continuation commands and exact health thresholds are in [the lab instructions](../docs/MLP_V11_LAB.md#ppo-learning-rate-controller-and-calibrated-il-kl-limits). The config/run name remains `joint7-v11-1-grouped-lr`; completed IL remains `features4-v11-1-batch7-il`.

Rehearsal uses `max(floor, 0.1 * max(0, 1 - committed_games / 10000))`: floor 0.03 for training iterations 1–60, 0.02 for 61–90, 0.01 for 91–120, then zero. Loading, application and release all use the same committed-progress function. Explicit old configs retain their original schedule.

The new fixed probe selects up to 2,048 distinct nonforced IL-validation positions with seed 710193 and balanced era/phase strata. Cached IL probabilities, source/feature identities and a deterministic probe hash are bound to the run. Raw temperature-1 policies replace the exploration-mixed ordinary probe for this IL-distance gate. The four groups now mean Canal/Rail × action/other. For numerical stability, the same 1e-6 floor and renormalization are applied to both distributions for KL, preserving exact identity; sale/pass constraints use the unmodified raw probabilities. No validation positions enter optimizer batches. A full 2,048-position probe and at least 64 positions in each group are required for increases.

The initial overall/group limits are **0.15 / 0.30**. The group threshold is always twice the current overall limit. A frozen ten-iteration post-warmup health baseline and five-iteration rolling all-learner metrics control immediate tightening and at-most-ten-iteration growth, capped at 25% and 1.0 overall. Budget violations latch increases until mean/group distances fall strictly below 80% of their respective limits with healthy evidence. Protective PPO LR cuts remain independent. Run state and generation checkpoints carry identical calibrated state; resume verifies that state and probe/data receipts. Legacy 1.0/1.1 controller semantics and completed IL compatibility remain explicit.

New complete-game diagnostics preserve zero-VP seat counts, insolvency event counts and distinct affected seats for every composition before replay IPC compaction. Compact history, recent state and exports retain those counts, the rehearsal coefficient and calibrated-limit evidence. Probe manifests and generation-zero reports are exported; dense validation tensors stay local.

The archived selfplay-v2 evidence can reproduce a **VP-only first failure at 11,520 games** when assuming warmup ended at generation 3. The archive lacks per-composition zero-VP/insolvency counts, non-pass-alternative pass counts, fixed-probe behavior and warmup-completion evidence. Consequently the full historical health check is **not verified**; missing values are never replaced by zero. Audit: `artifacts/auto-il-kl-20261017/historical-health-audit.json`.

Files changed for this follow-up:

- `recovery/il_kl_probe.py`, `il_kl_limit.py`, `rehearsal.py`: fixed validation evidence, deterministic limit transitions and committed rehearsal schedule.
- `recovery/config.py`, `lr_controller.py`, `league.py`: strict opt-in contract/default generation, LR-increase gate and checkpoint integration.
- `recovery/diagnostics.py`, `history.py`, `retention.py`: complete-game health counts, compact evidence and explicit probe/cache retention pins.
- `configs/step5/joint7-latest.json`, `configs/learning/imitation-handoff-sources.json`: fresh defaults and exact pre-change completed-IL admission.
- `scripts/export-training-reports.py`, its regression suite, `test_auto_il_kl.py`, `test_ppo_lr_controller.py`, `smoke_fixed_selfplay.py`: report selection, new/legacy contracts, rules, cache integrity and native integration checks.
- This report, `docs/MLP_V11_LAB.md`, `docs/EXPORT_TRAINING_REPORTS.md`: settings, evidence and launch/export instructions.

The learning/model compatibility digest is unchanged at `b3b944dbc492bc1483c41bdb5a9d566f13e2b2a8b7988dbeec8fcb9ebb372674`. Config generation was exercised against completed v11.1 IL. It retains the 60/30/10 table mixture, four/two/two learner seats, population 16, teacher disabled, exploration 0.08, PPO base LR 6e-6, 100-step warmup, insolvency costs 0.005/0.075, dormant 9/12/12/12-second teacher budgets and `search_wall_fraction=null`. No existing training state or IL weights were modified. Audit: `artifacts/auto-il-kl-20261017/final-config-audit.json`.

**67 focused tests passed** in 28.293 seconds. Coverage includes the floor boundaries; seeded probe selection/rebuild/source rejection; zero identity KL; raw sale/pass checks; warmup/baseline timing; linked overall/group thresholds; immediate tightening, growth cap/ceiling and latch recovery; missing evidence; protective cuts and shadow mode; serialized mid-window restart; probe retention; compact/export evidence; and legacy LR/source-review behavior. Log: `artifacts/auto-il-kl-20261017/focused-tests-final.log`.

The final native smoke used a separate output directory and the existing tiny completed diagnostic IL: **three real games, one of one planned PPO steps**, two self-play tables plus one history table, ten learner trajectories, 343 eligible actor decisions and 421 forced decisions. Total invocation time was **224.375 seconds**; the iteration itself took 124.672 seconds (77.641 collection, 12.609 PPO including probe/controller work, 32.828 retention cleanup). Teacher search and evaluation stayed disabled. Rehearsal ran with effective weight 0.1 and floor 0.03. Initial model weights matched IL, source IL files were unchanged, and Adam/RNG/controller/calibrated-state restoration plus repeated cached-probe measurement passed.

Initial IL-to-IL KL was exactly **0**. Post-PPO fixed-probe KL was **3.01015e-7 nats**. This diagnostic validation set contains only **144** eligible positions: Canal action/other **60/5**, Rail action/other **64/15**. Measuring them took **0.359 seconds on CPU with two learner threads**. This is not a 2,048-position timing measurement. No passing-check failures occurred among 124 eligible positions; the smoke has no positions eligible for the sale check, which is exercised by the focused regression instead. Insufficient probe support and warmup correctly prevented an increase; overall/group limits stayed at 0.15/0.30.

The smoke retained learner-seat counts separately: self-play had 7 zero-VP seats and 8 affected seats / 58 insolvency events across 8 seat-games; history had 1 zero-VP seat and 2 affected seats / 14 events across 2 learner seat-games. These are pipeline checks from a tiny diagnostic IL model, not an assessment of the user's completed IL or evidence of improved strength. The first-generation compact archive is not yet present; compact-field preservation is covered by regression tests.

The real exporter copied **six source JSON files**, including both fixed-probe reports, with exact source bytes/checksums. Dense probe/preparation data were excluded. The missing first-generation archive was explicitly recorded and there were no export warnings. Export evidence is retained under `artifacts/auto-il-kl-20261017/report-bundle/`.

The probe's one-shard validation preparation took 17.716 seconds and 24.4 MB of prepared disk storage in this smoke; production startup depends on the full validation partition and source-hash scan. Later iterations reuse cached tensors/probabilities, while the positive rehearsal floor extends rehearsal compute and residency beyond 10,000 games through training iteration 120. No long training or strength evaluation was launched. Final source identity: `6869d1f9bcf0b97937892b4b856a829e3d3a60fd71191ac50be7dbcbf9bfe4a9`. Combined evidence: `artifacts/auto-il-kl-20261017/validation.json`; smoke: `artifacts/auto-il-kl-20261017/smoke-final/report.json`.

## October 16 recipe (before the calibrated-limit follow-up)

| Setting | New default |
| --- | --- |
| Recipe identity | `latest-model-selfplay/2.0.0` |
| Teacher collection / distillation | `teacher_enabled=false` |
| History population | 16: 4 recent + 12 anchor/reservoir |
| Self-play tables | 60%; four learner seats |
| History tables | 30%; two learner seats + two frozen historical policies |
| v4 tables | 10%; two learner seats + two searched v4 bots |
| v4 scheduling | Fixed from game zero; no boost window or later search-mode transition |
| PPO base LR | 6e-6 |
| PPO controller | `ppo-categorical-kl-lr/1.1.0`, mode `apply` |
| PPO multiplier bounds | 0.5–3, giving 3e-6–1.8e-5 after warmup |
| IL-anchor increase gate | **0.15 nats overall; 0.30 per era/round group**, at least 64 nonforced positions per group |

Twenty-table cycles enforce the quotas. A normal 320-game iteration has **192 self-play / 96 history / 32 v4 tables** and **768 / 192 / 64 learner seat trajectories**, totaling **1,024**. A short smoke does not necessarily complete a full quota cycle. Existing deterministic seat rotation is retained, and all actual learner seats contribute eligible PPO rows. Frozen opponents do not contribute PPO rows.

Historical policies retain exploration 0.05; all ordinary learner seats retain 0.08 and temperature 1.0. Existing family/industry exploration, forced-action filtering, stale-behavior rejection, rewards, PPO loss weights, rehearsal, evaluations, utility calibration and disabled champion gate are unchanged. The history selection preserves its initial anchor and priority reservoir; the 16-model population bound is separate from the existing full resume-checkpoint and evidence pins.

Disabling teaching returns before search cost estimation, calibration admission, seed allocation, game collection or optimizer work. Teacher auxiliary value learning is also inactive without a teacher collection. The saved 9/12/12/12-second budgets, five-game collection size, eight workers per teacher turn, cadence, eight distillation steps and `search_wall_fraction=null` are unchanged and dormant. Ordinary head-only utility calibration still runs when required.

## LR controller

The exact categorical KL is **KL(old behavior || post-PPO behavior)** on the existing proportional era/phase nonforced probe, limited to **2,048 rows**. No extra random sampling was introduced. The existing sampled-action early-stop guard remains unchanged and is separately labeled `KLsample` in the console; controller input is `KLcat`.

- Raise by 1.25 after a full three-valid-iteration median below 0.0045, with no early stops, no cooldown and passing health checks. Clear the window after raising so another raise needs three new valid iterations.
- Cut immediately by 1.5 if the current early stop occurs before half of planned optimizer steps, two iterations in the window early-stop, or current exact KL exceeds 0.015. Cuts remain active during cooldown and do not depend on health checks.
- A cut starts three valid iterations of cooldown. Clamp the multiplier to 0.5–3. Empty/nonfinite updates neither advance the KL window nor consume cooldown. An iteration starting before the 100-update warmup is complete holds its LR decision.
- Health gates for increases: exact overall anchor KL at most **0.15**, each sufficiently sampled era/round group mean at most **0.30**, with all four groups supported; current self-play VP per seat at least 90% of the value five iterations earlier; fewer than 5% of ordinary learner seats finishing at zero VP. Missing anchor/history/VP evidence blocks increases. The first five VP observations supply the baseline; they are retained during warmup.

Anchor distance is **KL(fixed IL behavior || post-PPO behavior)** on the same observations, legal actions, contexts and exploration mixture. The anchor is loaded from the checksum-verified original checkpoint, is frozen, and is not replaced by newer history models. It adds one bounded anchor forward pass to each measured probe batch. This is an LR increase gate, not an anchor loss or automatic rollback controller.

The 0.15-nat budget is an experimental setting, not an empirically established optimum. Available historical logs measure update KL, not fixed-IL distance, so the quoted 119-iteration sensitivity replay cannot establish this anchor threshold or forecast this new recipe's strength. The [PPO reference](https://spinningup.openai.com/en/latest/algorithms/ppo.html) describes successive-policy KL guarding; it does not prescribe a lifetime IL-anchor budget.

The optimizer parameter group stores `brass_ppo_lr_controller`: multiplier, KL window, cooldown, five prior VP observations, config and collection signature. Resume restores these with Adam moments and RNG. The multiplier is independent of group `lr`, so the existing base-LR reset cannot erase it. Changes in collection volume/seat mixture or relevant update settings reset only the KL window; multiplier and cooldown survive. Random decision-length variation does not reset the streak.

The strict config block supports `off`, `shadow`, `apply` and is included in the existing run signature. Unknown same-run config/source transitions continue to fail closed. In shadow mode hypothetical decisions are saved but the applied multiplier is 1. Only explicitly marked PPO optimizer steps use the multiplier; IL and teacher steps keep their prior warmup rate, and calibration keeps its existing temporary 1e-3 rate.

Full and compact reports contain exact KL, full-window median (null before three measurements), early stop, completed/planned steps and fraction, multiplier before/after, decision/reason, effective LR, next PPO LR, cooldown, anchor identity/distance and health evidence. Console output includes multiplier and mode/decision.

## Grouped anchor gate follow-up

Controller 1.1 adds only the first requested refinement: overall anchor KL **≤0.15** and all four group means **≤0.30** before an LR increase. It reuses `teacher_schedule.PHASES` and `phase_for`: Canal rounds 1–4 / 5–8, Rail rounds 1–6 / 7–8. The current proportional 2,048-row nonforced probe, behavior mixture and legal masks are unchanged. Per-position anchor KL is reduced into groups after the existing forward passes; no additional inference, sampling or RNG use is introduced. Overall KL remains weighted by probe positions rather than averaging group means equally.

The first native smoke exposed a necessary integration detail: ordinary replay ingress previously retained only `era` and action `phase`, dropping `round`. It now also retains the verified round integer across reader IPC so the real PPO probe can classify early/late positions. Feature tensors and replay files are unchanged. A reader-compaction-to-probe regression and explicit four-group coverage assertions in the native smoke prevent this from silently disabling all increases.

Each group requires **64 positions** by default. This configurable initial evidence floor is not a confidence guarantee. An underfilled group is marked `insufficient_samples`; missing or nonfinite evidence is `unavailable`. These conditions and unmapped era/round positions block only the current increase, without triggering a cut or persisting a lock. Group mean violations are reported separately. Original protective cuts, warmup, three-iteration windows, cooldown, shadow/off behavior, PPO-only scope and other health checks remain unchanged. The proposed 0.12 hysteresis is not implemented.

Group means/counts appear in the full categorical probe and, with status and thresholds, in `actor.lr_controller.health.anchor_groups` in full and compact history. Config fields `anchor_group_kl_budget` and `anchor_group_min_positions` are strictly validated and signature-bound. Explicit 1.0 configs continue to use their original overall-only gate. No existing training directory is edited or automatically upgraded; a committed run requires the existing reviewed-transition mechanism. Fresh configuration generation uses 1.1. The model and completed-IL compatibility are unchanged.

Follow-up files: `lr_controller.py`, `review_metrics.py`, `ingress.py`, `configs/step5/joint7-latest.json`, `tests/Recovery/test_anchor_group_gate.py`, the existing controller test fixtures and smoke assertions, and these lab/report instructions. The earlier unrelated recipe changes are preserved.

**Follow-up regressions: 37 passed in 22.364 seconds.** The command was `python -m unittest test_anchor_group_gate test_ppo_lr_controller test_v11_learning_rate test_source_reviews.SourceReviewTests -v` with the repository test/dependency paths. New coverage includes inclusive 0.15/0.30 thresholds, a low overall mean hiding late-Rail drift, uneven group sizes, legal masks and the 0.08 mixture, the 64-position boundary, missing/nonfinite/unmapped evidence, unchanged protective cuts, immediate eligibility recovery without hysteresis, legacy-contract semantics, reader IPC compaction, compact history, and no additional forwards or RNG use. Existing tests also verify exact optimizer/checkpoint restart and unaffected IL/teacher/calibration rates. Log: `artifacts/anchor-group-gate-20261016/final-tests.log`.

One additional old regression, `test_joint7_review.ReviewTests.test_categorical_kl_is_full_distribution_on_fixed_probe_and_rng_neutral`, is blocked before reaching the probe by its legacy `fixtures.game()` topology (`V11 link topology differs: CanalBelperDerby`). The new direct-probe regressions exercise the current implementation without that obsolete fixture. No topology or admission rule was weakened. Details: `artifacts/anchor-group-gate-20261016/related-tests.log`.

**Final native smoke passed:** three real CPU games in **250.765 seconds**, from existing completed v11.1 diagnostic IL, under `artifacts/anchor-group-gate-20261016/smoke-final`. It collected 343 eligible probe positions across ten learner seats and completed one PPO step. All positions were classified; unassigned count was **0**. Per-group evidence was:

| Anchor group | Probe positions | Mean anchor KL (nats) | Status |
| --- | ---: | ---: | --- |
| Early Canal | 79 | 1.7393e-8 | ok |
| Late Canal | 106 | 3.7169e-8 | ok |
| Early Rail | 128 | 1.4441e-8 | ok |
| Late Rail | 30 | -8.9057e-9 | insufficient_samples |

The tiny negative value is float32 roundoff near zero, retained without clamping individual terms. The position-weighted overall distance was **2.0103e-8**. Late Rail correctly failed the 64-position evidence floor rather than being reported as a KL-budget violation. The LR decision remained `hold/warmup`; actual increase/cut decisions were checked by deterministic regressions, not inferred from this near-untrained diagnostic model. Teacher collection remained disabled. Adam/controller/RNG resume was exact, and source IL files were unchanged. The smoke's total time includes game collection and retention; it is not a comparative performance benchmark or evidence of playing-strength improvement. Source identity: `e4fb9b106be898d2c6522fa9ab945f1360de755bd7e58e910e784ba58cb9c302`.

## Validation

The 40-test and three-game results below describe the original 1.0 implementation; grouped-gate follow-up validation is recorded separately.

**40 focused regressions passed in 18.742 seconds.** Coverage includes decision sequences, strict config validation, both clamps, cooldown, warmup/NaN gating, the two-of-three rule, the 0.15 boundary, shadow isolation, exact Adam/checkpoint restart mid-streak, real PPO probability reconstruction and planned-step reporting, unchanged real teacher/calibration updates, source guards, fixed quotas/rotation, eight-versus-two learner attribution, 16-model retention, and v11.1 model/export compatibility.

Command:

```bash
PYTHONPATH=src:tests/Recovery:tests/R2:tests/FeatureEngineering:tests/Personality \
  python -m unittest test_ppo_lr_controller test_v11_learning_rate \
  test_v11_selector_resume test_selfplay_major_population \
  test_source_reviews.SourceReviewTests test_v11_1 -v
```

An earlier broader selection ran 36 tests: 34 passed and two inherited `CompactResumeTests` errored because they assume an obsolete source transition is approved against whichever implementation is current. The registry also lacks that transition against the pre-change v11.1 source. Its protections were retained; no obsolete architecture/source approval was added to make those fixtures pass. This is not a full-repository green claim. Existing ONNX exporter/tracer and nanobind shutdown diagnostics appeared during passing export tests.

The **final three-game native smoke passed in 99.922 seconds**, with the 0.15 default, existing completed v11.1 diagnostic IL and a new isolated output directory:

| Measurement | Observed |
| --- | ---: |
| Tables | 2 self-play, 1 history |
| Learner trajectories | 8 + 2 = 10 |
| Learner decisions | 764 |
| Actor-eligible / forced critic-only | 343 / 421 |
| PPO steps done / planned | 1 / 1 |
| Exact post-PPO KL / IL-anchor KL | 2.3783e-8 / 2.3783e-8 |
| Controller | hold: warmup, multiplier 1 |
| Actual PPO LR | 6e-7 |
| Teacher collections / optimizer steps | 0 / 0 |
| Collection / PPO wall time | 50.297 s / 7.469 s |
| Fresh handoff | IL weights exact; empty initial PPO Adam |
| Resume | Adam, RNG, controller, progress and lineage exact |
| Existing IL files | Hashes unchanged |

The smoke used a minimally trained diagnostic IL model, not a production-strength policy. No v4 search or evaluation campaign ran; v4 two-learner assignment and fixed scheduling were tested in the planner. The controller's increases/cuts were exercised in deterministic tests; the smoke correctly remained in warmup. The first generation has no archived compact history yet; compact field retention has a focused regression. No playing-strength or long-run stability improvement is demonstrated.

The configuration CLI also successfully bound the preceding v11.1 IL to a new config. All model/feature compatibility fields remain byte-identical. The only top-level template config changes are `teacher_enabled`, `population`, `curriculum`, `legacy_boost` removal, `legacy_opponents`, the new recipe identity and `ppo_lr_controller`.

Evidence: [verification](../artifacts/selfplay-lr-controller-20261016/verification.json), [test log](../artifacts/selfplay-lr-controller-20261016/final-tests.log), [final smoke](../artifacts/selfplay-lr-controller-20261016/smoke-final/report.json), [bound config](../artifacts/selfplay-lr-controller-20261016/cli-bound-config.json).

## Runtime implications and launch

Teacher search/distillation cost is removed. History storage is bounded to one quarter of the former population size. The new recipe produces more learner trajectories than the old early curriculum (1,024 versus 464 at 320 games), so PPO preparation and training can increase even as teacher/v4 search work decreases. Anchor measurement adds a frozen model of roughly 24 MB of float32 parameters plus a forward pass on at most 2,048 probe rows. No representative full-iteration speedup is claimed.

Use the commands in [the updated lab instructions](../docs/MLP_V11_LAB.md#fresh-self-play-from-completed-v111-il), substituting your actual completed IL path. Generate `configs/step5/joint7-v11-1-grouped-lr.json`, start a **new** `artifacts/step5/joint7-v11-1-grouped-lr` run, and omit `--fresh` for subsequent matching-source continuation. Keep the completed IL directory `artifacts/step5/features4-v11-1-batch7-il` unchanged; no new IL is needed for this self-play revision. `python scripts/export-training-reports.py` exports this self-play run by default; see [report export instructions](../docs/EXPORT_TRAINING_REPORTS.md).

## Changed files

- `src/brass_ai/recovery/lr_controller.py`: validated deterministic controller, state, health and reporting.
- `config.py`, `ordinary.py`, `teacher_collection.py`: new recipe, 16-model default, two-learner opponent tables, teacher switch.
- `learning_rate.py`, `updates.py`, `review_metrics.py`: PPO-only scaling, planned-step accounting and frozen-anchor measurement.
- `league.py`, `resume.py`, `history.py`: initialization/resume, commit timing, console and compact reporting.
- `configs/step5/joint7-latest.json`, `configs/learning/imitation-handoff-sources.json`: active defaults and exact completed-v11.1 IL admission.
- `tests/Recovery/test_ppo_lr_controller.py`, `smoke_fixed_selfplay.py`, `test_v11_learning_rate.py`: focused validation and warmup fixture integration.
- `docs/MLP_V11_LAB.md` and this report: effective settings, evidence and launch/continuation instructions.
