# Self-play correctness review — 11 October 2026

Scope: active Step 5 PPO, shared reward, Masked ISMCTS NN teachers, frozen incumbent/historical opponents, evaluation, checkpoint continuation and retention. Changes are limited to training infrastructure. The installed game AI and personality algorithms are unchanged. Pre-existing machine-transfer work was preserved.

## Confirmed issues and repairs

1. **Stale utility readiness after distillation.** Policy-only distillation changes the shared state encoder even though official-outcome and utility-head tensors are frozen. The previous numerical-fit receipt remained ready, so another due teacher could use a changed leaf evaluator without recalibration. Actual distillation now invalidates that receipt. The league performs the existing bounded three-pass, utility-head-only calibration using the same iteration's fresh ordinary-game returns; no searched-game outcome is a value target. Readiness is checked before every collection. Failed/missing calibration defers the next allocation before it spends seeds or counters. Successful fitting preserves the configured two-collection maximum. Calibration receipt/time are recorded per collection, and time is charged to the search allowance.

2. **Interrupted evaluations blocked otherwise valid source continuation.** A committed checkpoint could pass the source transition, then fail because the pending monitor/broader/champion specification still named the previous implementation. A verified source-transition chain now selects a deterministic fresh evaluation namespace. Old files remain intact and are not mixed with new-source evidence. Same-source configuration mismatches still fail. Restarting reuses the exact new namespace; completed committed screens are not rescheduled. Retention recognizes the suffixed directories as the same panel family.

3. **Accepted-best verification happened too late during an upgrade.** A champion outside the current historical pool was not checked in resume preflight. Its hash/path are now verified before committing a changed run signature, alongside the latest learner, live pool and evaluation anchor.

Old headed checkpoints may contain the stale utility receipt described above. Reviewed resume replaces only the in-memory readiness receipt, keeping checkpoint bytes, all model weights and optimizer state intact. Fresh ordinary rollouts recalibrate before teaching. This also works after a crash between committing the source transition and producing the first new checkpoint.

No additional confirmed defect was found in official tiebreaks, stable-seat utility labels, undiscounted causal/capped insolvency costs, on-policy learner attribution, the 20,000/10,000-game opponent window or historical-model sampling. No reward, search budget, worker count, PPO hyperparameter or model-layout changes were made.

## Continuation contract

- Previous source: `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`.
- Reviewed source: `f9f94171ead6383c012d402aadf8781f51074a1f4cb5e6575cfb1487deb4e5f0`.
- Learning compatibility remains `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`.
- Checkpoint format/tensor layout and generated-data formats remain unchanged. Resume preserves Adam moments/steps, CPU/CUDA RNG metadata, latest learner, accepted incumbent, historical population, game/seed counters, v4-window origin and teacher counters.
- `configs/learning/selfplay-resume-sources.json` pins the exact preceding-source transition with unchanged configuration. Previously approved older source/config transitions are retained to the new source; no arbitrary configuration or objective migration is added.
- `configs/learning/imitation-handoff-sources.json` explicitly retains completed IL from the preceding source. Its original completion, dataset, lineage and weight hashes are still required.

Use the [lab guide continuation command](../docs/TRAINING_RECOVERY_LAB.md#self-play-correctness-review-and-continuation-11-october), preserving your original arguments and adding `--resume-reviewed-source`. No retraining from IL is required for a compatible saved run. Existing source guards continue to reject an uncommitted ordinary iteration or uncommitted next checkpoint; recover that work with the prior source before updating. The local checks cannot establish whether the lab's particular run directory has unfinished files.

## Verification and limits

**Results:** all **86 runnable checks** from the targeted 87-test selection passed after the affected checks were rerun; one CUDA-only check was skipped because this laptop has no CUDA device. Thirteen new regression methods cover readiness and reviewed evaluation/checkpoint recovery. Two historical test fixtures were corrected to reconstruct the actual approved configurations instead of assuming the old search contract still matched today's preset. An initial retention run hit a Windows file-access error during parallel testing; its serial rerun passed. The additional personality source-boundary check also passed. Source receipt, guide links and `git diff --check` passed.

Focused checks cover modern and previously approved checkpoint continuation, exact weights/Adam/RNG preservation, repeated transition recovery, stale-teacher rejection, real distillation followed by head-only recalibration, calibration-failure deferral, intact old evaluation evidence, champion caching and retention, reward/insolvency accounting, reserved teacher budgets and replay attribution.

The native continuation smoke completed **four four-player ordinary games across two iterations in 131.2 seconds**, upgrading through the actual review gate between iterations. It verified unchanged previous-checkpoint/incumbent bytes, preserved opponent-window counters and fresh utility calibration. It uses a small old-source-format fixture, not the trained model still on the lab. Native artifact/report: `artifacts/recovery/selfplay-review-20261011/report.json`. No long training, full champion tournament or production-model qualification was run.

Calibration remains an in-sample numerical check. It does not prove a strong or well-generalized critic. CUDA/Linux utilization, recalibration overhead at lab batch sizes and playing-strength changes need lab measurements. The fit uses existing fresh ordinary data without generating extra games; memory holds those compact rows until that iteration's teacher work ends.
