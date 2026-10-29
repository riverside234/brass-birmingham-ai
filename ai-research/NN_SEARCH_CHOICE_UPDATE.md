# NN search K selection — local release 1.4.40

The AI selection panel now offers **NN search: K = 8 / K = 16** underneath Difficulty on the NN / Alpha Test page. The controls use the existing light theme, gold selection state, check marks and keyboard navigation. The selected K applies to every NN seat at 4, 8, 10 and 12 seconds; seats can still use different time budgets.

Apply commits the draft and saves the setup preference. Cancel leaves the previous setting intact. K16 remains the default. New NN games bind K in their assignment integrity hash, so continuing a game uses its saved K independently of current menu preferences. Older saves and older setup preferences without a K field use K16; reading an older save does not rewrite it.

The two choices use integrity-checked `policy.json` (K16) and `policy-k8.json` (K8). The scheduler retires a previous worker before selecting a different K. The existing Rust binary, source archive, g890 test model, ONNX Runtime CPU inference and two independent single-thread sessions are preserved. Cache-only `search_v2_focus_only` remains selected and V205 stays disabled. Effective K can still be reduced by legal-action count or remaining simulation quota.

Short qualification covered both K choices at every difficulty in the UI, Apply/Cancel, whole-table consistency, invalid K rejection, preference restoration in a fresh Unity process, signed save compatibility, native readiness mappings, and actual 4-second searches with worker replacement and completed-turn save/load. UI layout was checked at 1920×1080 and 1920×1200 with the largest text size. The candidate passed 868 UI/preference checks, 793 restart checks, and 750 checks for each live K smoke. The live turn completed in 3,788.85 ms at K8 and 3,659.32 ms at K16. These are functional smokes, not a performance or playing-strength comparison.

Installation interruption/recovery and exact rollback were tested on an isolated copy with a saved-game sentinel. Only `BrassLegacyAI.dll`, the native `build.json` integrity manifest and the additional K8 policy are published. All other managed/native files are verified against the installed baseline.

The actual installed game subsequently passed **750 checks per K choice**, completing both actions in **3,733.33 ms at K8** and **3,645.56 ms at K16**. All 214 native package files and three managed assemblies were verified after both smokes. The game and native workers are closed. See the [final installed verification](../artifacts/nn-search-choice/installed-verification.json) and [immutable deployment receipt](../artifacts/nn-search-choice/deployment.json).

- [Qualification](../artifacts/nn-search-choice/candidate/qualification.json)
- [Upgrade/recovery/rollback check](../artifacts/nn-search-choice/upgrade-test.json)
- [Installed K8 smoke](../artifacts/nn-search-choice/installed-k8/verification.json)
- [Installed K16 smoke](../artifacts/nn-search-choice/installed-k16/verification.json)
- [Large-text panel preview](../artifacts/nn-search-choice/candidate/evidence/images/setup-nn-1080-largest.png)

With the game closed, restore the exact previous managed runtime, policy manifest and receipt using:

```powershell
& ./scripts/install-nn-search-choice.ps1 -Action Rollback
```

Use `-Action Recover` if installation was interrupted. Rollback does not change saves; K8 games should be resumed with the selector release installed. The separately pinned B01 lab bundle and its 1.4.39 shared native source baseline are unchanged.
