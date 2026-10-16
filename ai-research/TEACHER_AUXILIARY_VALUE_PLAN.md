# Teacher Outcome Auxiliary Value Plan

Status: implemented as `teacher-outcome-value/1.0.0`; enabled by default for fresh self-play at the user's subsequent request. Diagnostics remain opt-in. See [implementation, validation and lab instructions](../docs/TEACHER_AUXILIARY_VALUE_REPORT.md). The representative full-iteration diagnostic overhead target remains unverified.

Implement a **configurable auxiliary value update using existing completed teacher games**.

1. **Preserve collection settings.** Keep five teacher games per collection, existing search budgets, cadence, eight distillation steps, and policy-label quality checks.

2. **Build outcome targets.** Reuse four-seat win/rank/VP targets and utility returns with future insolvency costs and remaining penalty budgets. Identify these as outcomes of the recorded mixed search/direct policy.

3. **Add bounded value learning.** Reuse existing heads and encoded inputs. Add a separately configurable auxiliary loss with bounded samples per game. Validate value samples independently from policy-label eligibility.

4. **Keep training semantics consistent.** Teacher trajectories stay outside PPO actor updates. For enabled runs, define explicit ordinary/teacher weighting for the shared critic and matching utility calibration. Preserve freshness and readiness safeguards.

5. **Record and preserve state.** Version configuration/provenance, preserve continuation state, and add the bounded teacher-value diagnostics below to collection reports and compact training history. Enable auxiliary training in fresh self-play defaults; preserve explicit opt-out and omitted-block semantics in existing signed configurations. Keep diagnostics disabled by default.

6. **Validate in isolation.** Test seat mapping, cost-to-go accounting, stale-data rejection, calibration, and resume behavior. Run a bounded comparison against policy-only distillation using identical teacher data; report overhead without claiming improved playing strength.

## Low-overhead report requirements

- **Labels and usage:** Report target source/reward identity, distinct completed games, valid value positions, distinct positions trained, unused positions, and sample presentations. Keep policy-label and value-label eligibility/counts separate. Store compact totals and four-phase summaries, not per-position vectors or replay contents.
- **Bounded probe:** Use at most **128 distinct positions per collection**, configurable downward. Select deterministically across available games and phases before encoding; do not consume training RNG. Reuse the same positions and targets for before/after comparisons, and record actual probe size, game count, phase coverage, and sampling scope. Correlated positions from one game are not independent games.
- **Errors and drift:** On that probe, report win-share Brier error, rank/VP MAE, and utility MAE/MSE before and after the auxiliary update. Report per-head prediction-change summaries and reuse the existing policy-KL diagnostic. Distinguish post-update results from post-calibration results; reuse bounded calibration outputs where available and mark unavailable stages explicitly. These are training-data diagnostics, not held-out strength evidence.
- **Reuse computation:** Construct outcome targets once per completed game. Reuse encoded inputs, existing model outputs, and fit/calibration passes wherever their checkpoint stage matches. Any necessary extra inference must stay within the bounded probe and use batching. Add no games, searches, optimizer steps, full-dataset diagnostic passes, or per-row GPU synchronization. Diagnostics must not change gradients, optimizer state, RNG, or sample selection for training.
- **Measure overhead:** Record diagnostic elapsed time and extra forward-pass/position counts separately from auxiliary training and calibration cost. Validate telemetry enabled versus disabled on identical saved data/checkpoints, using warmed, alternating repeated measurements on the same hardware. The acceptance target is **no more than 1% added median full training-iteration wall time attributable to these diagnostics**; report incremental seconds as well. A component benchmark alone cannot certify that iteration-level target. If it is exceeded, reduce the probe or improve reuse before enabling by default. Until measured on representative training, label the overhead target as unverified.
