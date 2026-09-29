# Six-strategy AI and city locator — combined patch 1.3.0

**Release status: installed and verified on 27 September 2026.**

Historical release: superseded by [installed patch 1.3.1](LOCAL_AI_1.3.1_RELEASE.md), which protects AI turns from input, removes Play now, increases the search allowance, adds Random and brightens the city highlights. The behavior and evidence below describe 1.3.0.

## What changes in the game

The local custom-game screen offers **Balanced, Rail Tempo, Cotton, Goods/Beer, Supplier, and Pottery Flex** using the existing arrow buttons. These replace Easy/Medium/Hard for new custom games. Hovering a strategy shows its explanation. Human players, player counts, introductory mode, portraits, and colors remain available.

Each game's chosen strategies and profile parameters are saved with that game. Changing the next-game settings cannot change a running match. Continue displays its saved strategies; older games retain their original AI and display **Original AI**. The shipped profiles use a soft three-second allowance for a complete turn, with **Play now** becoming available after a complete legal plan is ready.

City-card previews use rounded **warm ivory `#F2E8D5` outlines at 55% opacity** for spaces without a green build highlight, including occupied spaces. The original green highlight is preserved. The outlines are hollow and do not intercept clicks.

All previous B01–B16, R01–R03, save-recovery, human-liquidation, and city-selection repairs remain included. **Online play remains disabled.**

## Integration and additional repairs

- A separate deployment fork in `src/BrassLocalAI` preserves the original research runtime and frozen opponent bank. All six S1 v2 profiles are embedded in `BrassLegacyAI.dll`; no external AI program, profile-editing step, or network service is required to play.
- Two-, three-, and four-player normal/introductory games carry explicit configuration through observation, reconstruction, card accounting, evaluation, and final scoring. Color identifiers are separate from seat positions.
- Strategy assignments and per-seat intent share the existing recoverable metadata/events save transaction. Intent is bound to profile, revision and event-history prefix, so it cannot carry across an abandoned undo branch.
- New-game creation preserves earlier save directories. Continue chooses the latest completed creation. Malformed or incompatible saved games show an error instead of deleting their files.
- Search runs asynchronously on detached information. Each completed action is validated before a single main-thread commit. Cancellation and generation/revision checks cover pending startup, search, loading, history navigation and menu return.
- Refreshing a resume row no longer calls its absent nickname-editing callback.
- **Brewery liquidation save repair:** a full mixed-game reload test found that `IndustrySellingEnded` could serialize a brewery's runtime `Func<int>` callback and fail to deserialize it. The weaver marks only `Brewery._numberOfBeersFunc` with `JsonIgnore`. Saved brewery summary data remains intact; the normal game setup still constructs each live brewery's production callback. Regression tests cover both new JSON and legacy JSON containing the unusable callback.

## Verification

Final results, exact binary hashes, and installation status are recorded in `artifacts/patch-build/validation.json`, `local-ai-qualification.json`, `packaging-results.json`, and `installation-verification.json`.

The installed game passed **47 startup/menu checks and all ten Unity regression suites**. All three installed DLL hashes match the qualified release. Packaging passed **78 assertions**, including interrupted installation and recovery; the original backup and retention of previous patch helper calls were verified. The game is closed and ready to launch normally through Steam.

The strategy reliability matrix completed **252 games**: 216 mixed-strategy games covering all six profiles in every active seat, two seeds, all three player counts, and normal/introductory mode; plus 36 tables where every bot used the same strategy. These deterministic tests use a one-node work limit while still constructing complete legal turns. They are not a comparison of playing strength.

The native Unity test instantiates the shipped custom-game and AI-turn prefabs and runs 36 strategy turns with the shipped search allowances. It checks selectors, tooltips, defaults, color changes, removal/re-addition, read-only resume labels, preservation of corrupt saves, pending-creation handling, repeated Play now input, native button raycasts, cancellation, action confirmations, and action/intent reload. Six additional mixed games passed **237 turns, 456 actions, 231 save/reloads, and 10 debt settlements**, using a scripted human seat through the full supported game length. The local smoke suite passed **968 assertions**, including every legal color/seat permutation and hidden-hand/deck invariance; the mixed-game/coordinator suite passed **1,397 assertions**.

The managed rules suite includes 90 completed original-AI games, independent era/player scoring comparisons, and targeted rules, replay, Scout, liquidation, compatibility and failure-recovery tests. The ten existing Unity suites are rerun against the combined patch. Locator verification uses the actual board prefab plus 18 day/night renders at zoom 0.028, 0.064 and 0.100, with native pointer checks and rounded-mesh checks.

Long-running strategy simulations ran from private runtime copies. Qualification verifies unchanged Host/Policy source files, embedded profiles, and game-method IL; it explicitly lists presentation-only differences and the final brewery serialization annotation. Native, JSON, mixed-game reload, and installed-player tests use the final candidate. Original test evidence is retained instead of being relabeled as testing different binary hashes.

These are automated fixtures, including scripted human decisions, not a physical mouse playthrough of every interaction or every monitor/DPI setting. The menu test records one unrelated localization component warning about writing its Google-update timestamp to the sandbox account's preferences; there are no unexpected native test errors. No real user game save is opened, migrated, or deleted by validation.

## Installation and recovery

The package consists of `Assembly-CSharp.dll`, `BrassPatch.dll`, and `BrassLegacyAI.dll` in `BrassBirminghamUnity_Data/Managed`. The installer requires matching validation hashes, a verified original backup, and a closed game. It backs up the previously installed release, publishes dependencies before the engine assembly, verifies all three results, and retains a recoverable transaction record after an interrupted operation.

`artifacts/patch-releases/1.2.0` retains the previous installed pair and record. The original stock assembly remains under `artifacts/patch-baseline`. Do not use the historical R3 `restore-legacy-overlay.ps1` on this combined release; it now rejects installed releases from 1.2.0 onward.

To reinstall this validated candidate with the game closed:

```powershell
pwsh -NoProfile -File scripts/install-patch.ps1
```

Stock restoration refuses pending recovery journals and games carrying the new AI state. Those games require a compatible combined runtime to retain their chosen strategies. Recovery scripts do not delete saves.

## Reproduce the checks

```powershell
pwsh -NoProfile -File scripts/build-patch.ps1
pwsh -NoProfile -File tests/BrassPatch/validate-patch.ps1
pwsh -NoProfile -File tests/BrassLocalAI/run-runtime.ps1
pwsh -NoProfile -File tests/BrassLocalAI/run-live.ps1
pwsh -NoProfile -File tests/BrassPatch/CityLocator/run-unity.ps1 -Render
pwsh -NoProfile -File tests/BrassLocalAI/run-unity.ps1
pwsh -NoProfile -File tests/BrassLocalAI/run-unity.ps1 -ExistingSuites
pwsh -NoProfile -File tests/BrassLocalAI/qualify-release.ps1
pwsh -NoProfile -File tests/BrassPatch/packaging-regression.ps1
pwsh -NoProfile -File scripts/install-patch.ps1
pwsh -NoProfile -File scripts/test-installed-local-ai.ps1 -Smoke
pwsh -NoProfile -File scripts/test-unity-patch.ps1
pwsh -NoProfile -File tests/BrassPatch/CityLocator/verify-release.ps1
```

The full strategy matrices are generated by `LocalRuntimeTests.exe <output.json> <2|3|4> 1`, with an additional `identical` argument for uniform tables. The qualification script requires their retained, fingerprinted runtime copies; changed policy or rules code requires rerunning the corresponding matrices.
