# search_v1 default restored — 1.4.42

Installed and verified on 2026-10-29 (local date). The user's requested **old `search_v1`** is again the game default. Part 2 of the [migration plan](RUST_AI_BACKEND_MIGRATION_PLAN.md#part-2--lab-server-migrate-training-after-local-acceptance) now selects search_v1 for explicitly enabled training teachers and other neural-search consumers. Search_v2 adaptive focus and V205 are not selected.

## Installed change

The release overlays 1.4.41 with one scoped managed-source change: `NativePackage` admits explicit `search_v1`. All four installed NN policies (`policy.json`, `policy-k8.json`, `policy-t15.json`, `policy-k8-t15.json`) select that method, and the native package hashes are refreshed. The existing Rust worker already implements search_v1; its executable, source archive, shared game/search core, model and ONNX Runtime are byte-identical to 1.4.41. This restores the original whole-turn/session search path without adaptive-focus switching or V205 statistics reuse.

The NN K8/K16 and temperature 1.0/1.5 controls, their saved preferences, per-seat 4s/8s/10s/12s allowances, two one-thread search sessions, completed-turn saving, cancellation fixes and actual-state action validation remain in place. K16 and temperature 1.0 remain the defaults. The game and patch assemblies are unchanged; only `BrassLegacyAI.dll`, the four policy files and their package manifest are installed. All 146 existing saved-game/preference files remain byte-identical.

## Short verification

| Check | Result |
| --- | --- |
| Candidate, 4s/K16/T1.0, complete two-action turn | Passed; 754 Unity checks; 3604.65 ms delivery |
| Candidate, 4s/K8/T1.5, cancel after action one | Passed; 750 checks; action two prevented |
| Candidate, 4s/K8/T1.5, load after action one | Passed; 751 checks; unsaved action discarded and action two prevented |
| Isolated install, rollback, interrupted-install recovery | Passed; exact prior runtime/receipt restored; sentinel save unchanged |
| Installed game, 4s/K16/T1.0, complete two-action turn | Passed; 3760.25 ms delivery; one planning request; completed-turn save/load passed |
| Installed policies and package integrity | All four policy methods are search_v1, all budget mappings correct, all package hashes verified |

Warmup also exercises the alternate temperature under the scheduler lease, including retiring the prior worker. Candidate tests cover both K variants and both temperatures. Every production search reply uses two sessions with one thread each and has no focus/composite revision, as required by search_v1. These are bounded smoke checks, not a new strength or throughput comparison; no full games or long sessions were run.

Evidence:

- [Deployment verification](../artifacts/search-v1-default/deployment.json)
- [Candidate/build identities](../artifacts/search-v1-default/candidate/build-manifest.json)
- [Candidate qualification](../artifacts/search-v1-default/candidate/qualification.json)
- [Installed-player verification](../artifacts/search-v1-default/installed-k16/verification.json)
- [Upgrade/recovery verification](../artifacts/search-v1-default/upgrade-test.json)

## Lab plan and existing bundles

**B01 and B02 lab qualification remain Pending.** The plan now requires explicit search_v1 teacher admission, existing completed-round/whole-turn evidence, conserved budgets, exact source/model identity and rejection of search_v2/V205. Ordinary policy-only sampling and production teacher enablement are unchanged. The local temporary g890 readiness exception still does not qualify a lab teacher.

Existing B01/B02 bundles were pinned to the earlier cache-only profile and are retained as historical development artifacts. Before qualifying the revised default, refresh their profile/admission checks, local-evidence/input pins and transfer identities as listed in B1/B01. Their old receipts must not be relabeled search_v1. This task updates the game and migration plan; it does not deploy or qualify the lab server.

## Reproduction and recovery

Source tooling: `scripts/build-search-v1-default.py`, `scripts/qualify-search-v1-default.py`, `scripts/install-search-v1-default.ps1`, `scripts/smoke-installed-search-v1.ps1`. Builders preserve existing evidence rather than overwriting it; build from the exact preceding release in a fresh artifact directory when reproducing.

The installation receipt records the exact prior-release backup as `searchV1DefaultRecovery`. With the game closed, restore 1.4.41 using:

```powershell
./scripts/install-search-v1-default.ps1 -Action Rollback
```

Use `-Action Recover` only for an interrupted installation with its matching pending record. These commands restore the recorded prior runtime, policy files and receipt; they do not rewrite saved games.
