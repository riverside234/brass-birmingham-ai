# All-budget K16 game configuration

Installed 2026-10-28 over local release 1.4.39: requested Gumbel **K=16 at 4, 8, 10 and 12 seconds**. Both independent single-thread sessions use the whole-turn allowance mapping. Effective K can still decrease when legal choices or remaining trajectory quota require it.

Only the installed `BrassNative/alpha-g890/policy.json` and its `build.json` integrity manifest changed. The package source setting in `configs/search/local-gumbel-k.json` was updated too. The binary's legacy base `config.k=8` remains an admission-format constant; the engine replaces it with the per-allowance K16 value before every search. No binary rebuild is required.

Cache-only `search_v2_focus_only`, V205 disabled, ONNX Runtime CPU and the current test model remain selected. Four bounded native-process checks confirmed requested K16 at every allowance. The actual installed Unity player then passed **745 checks** at **4s/K16**, completing two actions from one focused request in **3,843.42 ms**. The smoke also verified completed-turn save/load using an isolated test save. This is a short functional smoke, not a new performance or strength study.

All 213 native package files and unchanged managed assemblies were verified. See [installation verification](../artifacts/search-k16-config/installed-verification.json), [candidate checks](../artifacts/search-k16-config/verification.json), and [installed-player smoke](../artifacts/search-v2/20261028/installed-k16-smoke/verification.json).

With the game closed, exact configuration rollback is available through:

```powershell
& ./artifacts/search-k16-config/install.ps1 -Action Rollback
```

Use this rollback before reverting the older search-v2 binary installation. `-Action Recover` handles an interrupted configuration install. Historical comparison receipts and the B01 lab transfer bundle retain their original settings.
