# Qualified policy references, competitive ratings and epsilon 0.02

Implemented 20 October 2026 for the current v11.2 PPO-only run and newly generated configs. Existing model architecture, PPO losses, ordinary table quotas, PFSP statistics, teacher/PPG/search settings and original IL handoff are preserved. This is an experimental training adjustment; no improved stability or playing strength is established here.

## Current-run evidence and activation

The supplied g270 export records **270 committed iterations / 86,400 ordinary games**. Both panels show 100% learner first-place win share across 32 games/eight setups. IL qualification uses the **combined** first-place share of its two learner seats against two IL seats, with the inclusive **≥90%** threshold. The one-learner/three-v4 panel supplies the separate v4 observation. Saved retirement receipts remain valid after panels stop running and after full slot reports move into the verified archive.

Within the latest completed g270 PL fit, **g260** has the highest available-report point estimate: **622.34093 points**, paired 95% difference interval against IL **[570.87019, 689.99513]**. g270 is 603.69384 points. Therefore g260 is the expected first reference, **subject to the lab helper verifying that its checkpoint and the attested bootstrap are still present and unchanged**. Missing candidates are skipped; changed files fail verification. The local report export contains no checkpoint/probability/bootstrap arrays, so actual **KL(g260 || current learner)** has not been measured locally and no live training state has been changed. The helper performs that measurement when applied on the lab host.

The export was also used for an isolated calibration-state check: resetting with an authored over-budget measurement preserved its actual economic baseline, rolling evidence, probe receipt and committed generation and blocked LR increases. This checks migration semantics, not its real new-reference KL.

## Policy reference and calibration

`policy_anchor` is a strictly validated, signed configuration block. Initial selection orders available rated checkpoints by their point estimate in the **same current fit**, then requires a paired difference interval with lower endpoint strictly above zero versus original IL. The selected model stays frozen. Later decisions consider only generations newer than the active reference; a higher point estimate alone cannot promote them.

For each proposed promotion, subtract candidate and reference theta **within every shared saved setup-bootstrap draw**, then take the 2.5%/97.5% quantiles and existing 173.7 point scale. Verify that this reproduces the fit's saved paired-difference report. Never subtract marginal interval endpoints or scores from separate fits. Shared resampling retains dependence between estimates, consistent with the pairing principle in [SciPy's bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). This uses the existing percentile procedure and 200 setup-cluster refits; it does not replace the rating estimator or claim guaranteed nominal coverage.

On promotion:

- Measure raw-temperature-1 **KL(reference || learner)** on the existing fixed **2,048 positions** and original four era/decision-phase groups. Preserve the numerical floor and legal-action masking.
- Start at **0.15 nats overall / 0.30 per group**. Clear only the prior reference's KL calibration window, last-healthy distance, update clock and budget latch; immediately recompute them against the new measurement and existing economic health.
- Preserve the ten-iteration calibration cadence, 1.25 growth cap, maximum allowances, tightening and 0.8 recovery rules. An over-budget learner remains eligible for ordinary PPO; LR increases are blocked.
- Preserve the independent economic baseline and rolling VP, zero-VP, aggregate-pass and insolvency evidence. No IL sale/pass-probability checks are restored.
- Preserve original IL identity, PL zero point, IL opponent, fixed probe positions/arrays, population, model/Adam/RNG, warmup/LR multiplier, progress, lineage and GAE/snapshot-R² state.

Reference probabilities live in a separate immutable cache under `policy-anchor/probabilities/`, bound to model identity and the existing probe hash. Original IL arrays are untouched. Active-reference and original-IL KL share one learner forward pass per probe batch. Retention pins the active checkpoint, cache manifest/array and promotion fit/spec/bootstrap. Old events remain diagnostic history; they do not supply the new calibration window.

The initial hotfix and later post-rating promotions use a checkpoint-bound, reproducible pending receipt in `state.json`. Restarting before the next PPO commit reuses it; the next checkpoint incorporates the reference and new calibration. The original committed checkpoint is never rewritten. The helper verifies the saved checkpoint and restores optimizer configuration before making the transition.

Full/compact reports retain reference generation/identity, paired promotion evidence, KL to the active reference, original IL KL, overall/group allowances and the anchor/economic LR permission. The console uses **`AnchorKL[gN]`**, `OrigILKL`, `group_limit` and `LRraise`; a promotion also prints the paired interval and immediate budget result. `LRraise=1` means these checks permit an increase, not that warmup, PPO health, median/cooldown and other LR conditions necessarily choose one. The historical `il_kl_limit` JSON key stays compatible.

## Competitive main rating coverage

A separate persistent gate opens when **both panel win shares are strictly greater than 80%**. A verified retired 90% receipt qualifies; exactly 80% does not. At g270 both gates qualify immediately without another evaluation campaign.

| Setups | Main lineup | Games |
| ---: | --- | ---: |
| 4 | Candidate + IL + 2 rated checkpoints | 16 |
| 4 | Candidate + v4 + 2 rated checkpoints | 16 |
| 24 | Candidate + 3 rated checkpoints | 96 |
| **32** | Four candidate-seat rotations each | **128** |

Use the previous completed PL fit, estimating candidate strength from its latest rated learner. The eight-model shortlist includes the previous rated checkpoint and active KL reference, then fills by rating proximity. Selection deduplicates frozen **model identities**, balances appearance counts deterministically and keeps all four rotations together. With eight distinct eligible models, the 88 checkpoint setup slots give each 11 appearances. The eight reference setups are spread every four positions and shifted between slots. Explicit startup/limited-population fallbacks use references and report any unavoidable repeated seats.

When sufficient distinct checkpoints exist, checkpoint comparisons rise from **256 to 352 per rating slot**, or **37.5%**, without adding games. Keep all historical outcomes, IL theta fixed at zero, agent/seat priors, seat correction, official tie handling and paired setup bootstrap. Direct IL/v4 edges preserve comparison paths to the earlier scale; numerical anchoring alone does not remove uncertainty from weak connections, as discussed in the [Plackett–Luce overview](https://hturner.github.io/PlackettLuce/articles/Overview.html). Improved precision must be measured from subsequent real slots.

Existing independent 90% panel retirement, v4 training-share reduction to six tables, post-qualification IL PFSP membership, disabled temperature comparison and manual deployment review remain unchanged.

## Epsilon and continuation

The current shared ordinary neural policy uses **epsilon 0.02** in the new config, including learner and frozen historical seats. Preserve q v3, legal family/industry weighting, opening temperatures, raw entropy control and branching settings. Evaluation stays at epsilon zero and v4 remains search-free.

Actual sampled behavior probabilities and policy/provenance identities carry 0.02; collection and PPO reconstruct the same mixture. Old 0.03 metadata is rejected. No existing trajectories are relabeled. The existing LR data-signature rule clears its short decision window after this collection-setting change, while retaining multiplier, cooldown and health; reference promotion alone does not reset that window. Existing signed old configs retain their previous epsilon until explicitly migrated.

With training stopped, from the lab repository:

```bash
python scripts/joint7-qualified-anchor.py --apply && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-qualified-anchor.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The helper applies to the current committed generation; optional `--expected-generation 270` requires g270. It verifies and writes `policy-anchor/hotfix-report.json`, including the actual new KL and LR permission. Omit `--apply` for a preview. Later continuation uses the train command alone. Completed IL keeps its existing path; no new IL or fresh PPO run is required. New fresh configs generated through the lab command also include these settings, with reference/coverage adoption waiting for their panel gates.

An unfinished collection must be recovered under the existing freshness rules first. If still committed at g270:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 270 --apply
```

This archives the unfinished next collection; it does not rewind progress. Use the actual committed generation if later. The helper refuses conflicting config output rather than overwriting it. No silent checkpoint, optimizer or population migration is permitted.

## Validation and runtime

**222 focused regressions passed in 161.55 seconds** ([captured output](../artifacts/qualified-anchor-review/final-tests.log)). The suite covers reference selection/pairing and missing/tampered evidence, archived panel qualification, budget resets/overrun and independent health, pending-receipt restart, 80%/90% boundaries, exact balanced 128-game scheduling and limited pools, zero-exploration evaluation, active cache/fit retention and export, epsilon correctness, existing PPO/PPG-off/GAE/R²/PFSP rules and exact reviewed continuation.

```bash
PYTHONPATH=src:tests/Recovery:tests/R2:tests/Reports:tests/FeatureEngineering:tests/Personality \
python -m unittest -v test_policy_anchor test_auto_il_kl test_anchor_group_gate \
  test_ppo_lr_controller test_panel_gates test_v4_share test_snapshot_r2 \
  test_pfsp_pool test_quality_pool test_strength_rating test_rating_retention \
  test_ppo_only test_exploration_revision test_source_reviews \
  test_interrupted_iteration test_export_training_reports
```

The isolated [smoke report](../artifacts/qualified-anchor-review/smoke-2/report.json) passed in **23.57 seconds**, on CPU with two threads and the **5,928,616-parameter current utility model**. It used eight authored game identities/128 training rows per update and a 2,048-row authored probe (512/group). It exercised real checkpoint loading, reference cache generation/reuse, dual KL measurement, repeated hotfix application, overlaid calibration verification and PPO updates. Checkpoint bytes, independent health, warmup/multiplier and model/Adam/RNG were preserved; the next resumed PPO update matched weights and Adam bitwise. It rejected epsilon-0.03 cached behavior. The IL-lineage discovery and rating evidence were authored fixtures, with no native game collection or real strength inference.

Reproduce the bounded smoke with a new output directory:

```bash
PYTHONPATH=src:tests/Recovery:tests/R2:tests/FeatureEngineering:tests/Personality \
python tests/Recovery/smoke_policy_anchor.py --output artifacts/qualified-anchor-smoke-new
```

Rating game count is unchanged. More neural and fewer rule-bot seats may change collection latency, so equal game count is not a measured equal wall time. Promotion incurs one frozen-reference probe pass/cache write plus the immediate learner measurement; ordinary dual KL adds probability arithmetic but no second learner forward pass. Cache size scales with legal candidates on the fixed probe. No long training or expensive evaluation campaign was launched.

## Changed files and source review

- New reference/coverage modules: `src/brass_ai/recovery/policy_anchor.py`, `rating_coverage.py`; hotfix entry point: `scripts/joint7-qualified-anchor.py`.
- Integration: `panel_gates.py`, `ratings.py`, `il_kl_probe.py`, `il_kl_limit.py`, `league.py`, `retention.py`, `history.py`, `resume.py`, `config.py`, `exploration.py`.
- Configs: `configs/step5/joint7-latest.json` and new `joint7-v11-2-grouped-lr-critic-qualified-anchor.json`. Earlier signed g270 config remains available.
- Reports/docs: `scripts/export-training-reports.py`, `docs/MLP_V11_LAB.md`, `docs/EXPORT_TRAINING_REPORTS.md`, this report.
- Regressions/smoke: `tests/Recovery/test_policy_anchor.py`, `smoke_policy_anchor.py`, updated fresh/legacy config expectations in `test_auto_il_kl.py`, `test_ppo_lr_controller.py`, `test_exploration_revision.py`, `test_quality_pool.py`; active-anchor export test in `tests/Reports/test_export_training_reports.py`.
- Reviewed source registries: `configs/learning/selfplay-resume-sources.json`, `imitation-handoff-sources.json`. All **6,723** pre-existing resume approvals are preserved; new paths remain exact source/config migrations, with unchanged-config continuations also allowed. Learning compatibility stays `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.

Validated implementation identity: `174712e17d5ab459d030aaab046acacfe3575c30223826da73fae33114661284`. Earlier g270 fixes and the separately updated rating renderer/exported chart were preserved.
