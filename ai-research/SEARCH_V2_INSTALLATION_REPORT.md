# Cache-only search_v2 installation

Installed **2026-10-28**, deployment revision **1.4.39-rust-local**, following the user's explicit installation request.

The installed policy is **`search_v2_focus_only`**. Adaptive second-action refinement and exact NN-cache reuse are enabled; **V205 statistical subtree reuse is disabled**. The temporary g890 model and ONNX Runtime CPU remain in use, with two independent sessions and one compute thread each.

The installed files match the selected [cache-only candidate](../artifacts/search-v2/20261028/v205-cache-candidate/build-manifest.json). Only `BrassLegacyAI.dll` and five native package files changed: the executable, policy, package manifest, corresponding source archive and README. Game/UI patch assemblies, model/export, difficulty settings and root K mapping were preserved: 4s/K8 and 8s/10s/12s/K16. The package README describes the original study build; this report records its subsequent deployment.

## Post-installation smoke

The test launched the **actual game executable in the Steam installation**, using the installed managed/native files. It used the existing self-test entry point, a temporary test save under the game's artifacts directory and in-memory test AI preferences. Prior test DLL/output/image artifacts were restored afterward.

- One short ordinary two-action NN turn at **4s/K8**, following a separate warmup.
- **745 checks passed**; actual cache-only focus occurred and both independent owners used effective K8.
- One native request; action one committed at **3,722.96 ms**, action two at **3,849.85 ms**.
- Both actual actions validated and committed. No individual-action/search save occurred; AI End Turn saved exactly once, and loading verified the checkpoint including the next-turn boundary.
- The live reply identified `search_v2_focus_only`; installed policy and binary pins confirm V205 remains disabled.
- All 216 installed managed/native file pins matched before and after the smoke.

This is a bounded installation smoke, not a new performance study or a guarantee about latency in every position. The previous [cache-only evaluation](SEARCH_V2_EVALUATION_REPORT.md) and [V205 decision](SEARCH_V205_EVALUATION_REPORT.md) remain the comparison evidence.

Evidence: [installed verification](../artifacts/search-v2/20261028/installed-smoke/verification.json), [turn result](../artifacts/search-v2/20261028/installed-smoke/result.json), [Unity checks](../artifacts/search-v2/20261028/installed-smoke/unity.json), and [current installation receipt](../artifacts/patch-installation.json).

## Recovery

The scoped [installer](../scripts/install-search-v2.ps1) verified the selected candidate/evidence hashes, preserved an exact rollback copy and published the native directory and managed replacement with an interruption recovery journal. An install followed by exact rollback passed in a disposable directory before the real installation: [receipt](../artifacts/search-v2/20261028/deployment-check/result.json).

The real rollback copy is under `artifacts/patch-recovery/search-v2-a4a21b372fe743bab7edb65fc12dfcb3/`. With the game closed, `pwsh -NoProfile -File scripts/install-search-v2.ps1 -Action Rollback` restores the prior runtime and receipt. `-Action Recover` handles an interrupted installation with a pending journal. Neither path changes saved games.

The [installed smoke runner](../scripts/smoke-installed-search-v2.ps1) can repeat the same short check with a fresh evidence directory. No lab deployment or final-model promotion was performed.
