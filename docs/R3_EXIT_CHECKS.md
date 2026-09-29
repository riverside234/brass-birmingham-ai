# R3 H0–H5 exit checks

Historical H5 v2 qualification record. Current v3 completion/cash behavior, research admission and game installation are recorded in the [v3 follow-up](LOCAL_AI_V3_CONVERSION.md). The v2 results below are not attributed to v3.

27 September 2026. Four-player CPU legacy bank, rules patch `steam-brass-patch-1.1.0`, R1 contract `0.2.0`. H3 tactical profiles, required H4 reply search and H5 freeze/integration are implemented. The [guide](R3_TACTICAL_BANK.md) contains usage and lab commands. [Historical H0–H2 results](R3_H0_H2_EXIT_CHECKS.md) remain separate.

**Implementation, correctness and integration checks pass. The 156-game study is complete, with version-specific admission below. Full R3 playing-strength/deployment qualification remains open wherever its gates are not met.** This is not a deployed or calibrated Hard release. The original H5 work left the installed game unchanged; later deployment work is recorded separately above.

## Frozen identity

- Bank: `bank-804c85204da65824cb9b`.
- Worker SHA-256: `962f3dca57bee47dfb4705427f8c290c86920ae081c14947e5d54d65de668e7b`.
- Manifest: [original sources, profiles, contracts, compiler and 112 runtime pins](../training/opponents/bank/bank-804c85204da65824cb9b/manifest.json).
- Recompiling the frozen original sources produced an identical worker binary. Game binaries remain private and ignored.
- Evaluation: [qualification-d9db1fbda13fcfcb173d](../training/opponents/evaluations/qualification-d9db1fbda13fcfcb173d/spec.json), declared before its matches began. Profiles/settings and evaluation seeds were not tuned against its results.

## Executed correctness and integration checks

| Area | Result |
| --- | --- |
| Frozen-bank suite | 30 tests pass: the 17 H1/H2 regressions, 9 H3/H4 tests and 4 initial bank/admission tests. A supplemental failure-accounting test also passes; the admission/statistics file now has 5 passing tests. |
| H3 intent | Conditional entry, retained feasible targets, cross-seat/profile/future-revision rejection, lost-card recovery, missing-beer brewery contingency and abandonment when it is unaffordable. |
| H3 style selection | All specialists differ from Balanced in at least one identical permitted position; every selected full turn remains within its credible score bound. This is conditional diversity, not equal difficulty or proven entertainment value. |
| H4 privacy | Root decisions and masked rollout choices pass hidden-world invariance checks for all four seats. Policies receive permitted views and semantic legal choices, never authoritative hidden assignments. |
| H4 scheduler | Executed zero- and six-opponent-turn fixtures, Canal-to-Rail transition and final Rail terminal settlement. Current-turn plans stay fixed across sampled worlds. |
| H4 sampling/fallback | Matched pilot batches, disjoint fresh confirmation samples, four-seat estimates and complete S1 fallback when reply work is exhausted. Partial batches do not become confirmed estimates. |
| .NET and Unity bridge | 25 assertions pass in each runtime, including async incumbents, Play Now, stale/duplicate requests, cancellation, failed workers, one confirmation, intent save/load/undo/corruption and S2 deadline fallback. |
| Full application lifecycle | Both .NET and Unity Mono complete 64 turns / 124 actions, 32 Rail turns, 15 save/load cycles and 7 debt episodes. Public final-score projection agrees for all four seats. Human metadata is scripted, not a human playtest. |
| Unity routing | Actual woven `PlayAITurn` and `MonoBehaviour.Update` with a mock view router: thinking remains asynchronous, duplicate turn requests and menu cancellation are safe, and intent is persisted. |
| R2 integration | One complete frozen-bank game, 359 decisions, all four seat trajectories, external-policy provenance, permitted style/intent features and terminal targets. Two real value-learning updates change a small model's weights; legacy trajectories are correctly rejected by the on-policy learner. |
| Mixed neural/legacy collection | A second complete game records 368 decisions: 89 from a fresh small neural model and 279 from three frozen legacy profiles. The header and every seat trajectory retain exact model/bank identities. Mixed replay is rejected before an on-policy optimizer update. This checks integration, not neural playing strength. |
| R2 regression | All 24 current scaffold tests pass: the original 23 plus rejection of evaluation-only opponent replay before a training update or checkpoint write. |
| Reproducibility/admission | Source/runtime tampering, path escape, changed qualifications, retargeted identities and overlapping training/evaluation pools are rejected. Setup-clustered interval and failed-match accounting tests pass. |
| Final admitted pool | All six actually admitted profiles load through their required purpose and commit a legal opening turn without planning mutating the source state. Fourteen cross-pool/unadmitted uses are rejected. |
| Installed rules/R1 | All 111 installed managed dependencies and the existing R1 host match their pinned hashes. No game binary was changed. |

The machine-readable [correctness record](../training/opponents/evaluations/qualification-d9db1fbda13fcfcb173d/checks.json) pins the evidence files and relevant checking sources. The later [mixed-collection check](../training/opponents/evaluations/qualification-d9db1fbda13fcfcb173d/mixed-integration.json) separately pins its shard, collector and verifier. A [supplemental R2 regression](../training/opponents/evaluations/qualification-d9db1fbda13fcfcb173d/r2-regression-final.json) pins the evaluation-replay guard and its 24 passing tests. The frozen suite was rerun through `FrozenOpponent`, not merely against mutable development files.

The style diagnostic uses the first ordinary-action position from each of 16 era/rounds in a complete game, selected chronologically before comparing the profiles. At identical fixed work, Rail Tempo differs from Balanced in 4 positions, Cotton in 3, Goods/Beer in 4, Supplier in 3 and Pottery Flex in 1. Shared openings are common. Pottery's observed distinction is particularly modest; these counts do not establish six equally strong human-like personalities. Fixed-work tests isolate policy differences from clock noise; they are not wall-time strength measurements.

## Tournament and admission

<!-- FINAL_TOURNAMENT_RESULTS -->

All **156 prescribed games** finished processing; **0 failed games**. Failures are not credited as wins. Bank source/configuration was frozen throughout the study.

Admitted training profiles: `balanced-v2`, `rail-tempo-v2`, `cotton-v2`, `goods-beer-v2`, `supplier-v2`. Held-out evaluation profiles: `pottery-flex-v2`.

The [complete qualification record](../training/opponents/evaluations/qualification-d9db1fbda13fcfcb173d/report.json) and [archived v2 pools](../training/opponents/archive/pool-v2.json) preserve the exact identities and evidence hashes.

| Configuration | Guarded-stock games | First-place share | Setup-bootstrap 95% interval | Mean rank | Mean VP | All-table turn p95 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| balanced-v2 | 16 | 87.5% | 75.0%–100.0% | 1.12 | 133.1 | 3.60 s |
| rail-tempo-v2 | 16 | 75.0% | 56.2%–93.8% | 1.38 | 130.0 | 3.58 s |
| cotton-v2 | 16 | 81.2% | 62.5%–100.0% | 1.38 | 133.4 | 3.58 s |
| goods-beer-v2 | 16 | 75.0% | 56.2%–93.8% | 1.25 | 134.1 | 3.53 s |
| supplier-v2 | 16 | 87.5% | 62.5%–100.0% | 1.25 | 137.8 | 3.56 s |
| pottery-flex-v2 | 16 | 81.2% | 62.5%–100.0% | 1.19 | 133.2 | 3.61 s |
| balanced-s2-v2 | 16 | 56.2% | 37.5%–75.0% | 1.56 | 133.3 | 10.44 s |
| balanced-s1-10s-v2 | 16 | 100.0% | 100.0%–100.0% | 1.00 | 148.1 | 10.34 s |

These are exploratory empirical intervals over only four setup clusters per configuration. A narrow or degenerate bootstrap interval is not certainty about unseen games. Identical and mixed tables are included in failure and latency reporting; the win/rank/VP columns above are against three guarded stock opponents.

**H4 equal-time gate:** 16 paired seat/setup results, both at 10 seconds per turn. S2 minus S1 first-place share is -43.75 percentage points; grouped interval -62.50 to -25.00 percentage points. The prespecified benefit check does not pass; S2 is implemented, but it is not promoted as a stronger replacement for S1.

Across all S2 tables, 545 decisions completed fresh confirmation and 95 kept the S1 incumbent because a fresh matched batch did not finish. There were 0 rollout failures. These counts cover guarded-stock and identical-profile S2 matches.

Specialist S2 configurations were evaluated only in their declared identical tables and lack the full individual admission matrix. They remain diagnostic versions. The 10-second S1 control also lacks the required mixed/identical-table coverage and is not admitted. No profile passes the general improved-Hard or live-deployment gate in this study.

<!-- END_FINAL_TOURNAMENT_RESULTS -->

The niche gate is deliberately separate from a general improved-Hard claim: all declared table categories, zero observed failures and a mean rank better than 2.5 against guarded stock. General improved-Hard additionally requires at least 30 setup groups and a lower win-share uncertainty bound above 25%. The present four-group study cannot satisfy that broader release gate. Pottery Flex is the held-out opponent family. S2 benefit is assessed at equal total turn allowance and is not assumed from implementation.

The [final admission check](../training/opponents/evaluations/qualification-d9db1fbda13fcfcb173d/admission-check.json) exercises the real pools and pins their contents. The S1 bank can supply the next model-training experiments. H4 promotion remains open: use a new version and new setup groups to test proposal-budget allocation, opponent-model accuracy and horizon evaluation separately. This study does not identify which of those causes the S2 regression. Include stronger frozen S1 opponents in that next comparison; retain the present results and profiles unchanged.

## Reproduction and limits

```powershell
./scripts/build-legacy.ps1
./scripts/build-legacy-overlay.ps1
./scripts/check-legacy-bridge.ps1 -Overlay
./scripts/check-legacy-bridge.ps1 -Unity -Overlay
# Wait for the isolated Unity process and inspect its newly written result.
./scripts/check-legacy-overlay-ui.ps1
python scripts/check-frozen-bank.py --root . --bank bank-804c85204da65824cb9b --output artifacts/legacy-ai/new-platform-check.json
python scripts/verify-r3.py --root . --bank bank-804c85204da65824cb9b --unit-report artifacts/legacy-ai/new-platform-check.json --output artifacts/legacy-ai/new-correctness.json
python scripts/check-legacy-admission.py --root . --output artifacts/legacy-ai/new-admission-check.json
```

The aggregate check additionally requires the separately executed replay/style diagnostics described in the guide and the R2 regression report. Use new output paths so evidence is not overwritten. The matching private game dependencies/compiler and Python dependencies must already be available.

Tests ran on Windows .NET Framework and Unity 2019.1.7f1 Mono. The portable `check-frozen-bank.py` runner supports lab checks without a Unity installation, but Linux/Mono on `/data/not_backed_up/yxu209/ai-training/` has not been tested here. Match workers ran concurrently; reported headless ordinary-turn timing includes capture, search and commit, and excludes process IPC, separate debt decisions and rendered presentation. It is not a foreground 135U latency certificate.

Rendered human play, actual UI controls/animations, enjoyment, equal-difficulty personalities, new-Easy-above-original-Hard, lab throughput and neural playing strength remain separate gates. H4 is implemented and tested; any unmet equal-time benefit gate is reported rather than treating it as optional. The overlay remains uninstalled. No commit or public upload was performed by this delivery.
