# Adaptive AI and gameplay review — patch 1.3.2

**Status: installed and verified on 28 September 2026.** All three installed DLL hashes match the qualified candidate; online play remains disabled and there is no pending installation transaction.

## Random, persistent intent and fallback

Random chooses an **adaptive profile** at game creation. That profile supplies preferences, rather than an instruction to repeat one industry or plan throughout the match. Its current intent is reconsidered at each decision: feasible targets can persist, missing prerequisites can trigger a different plan, and stalled or poor opportunities can fall back to balanced choices. For example, Cotton can switch its immediate focus to building a brewery when that can make an existing cotton investment sellable.

This follows the distinction in [H3's tactical guide](R3_TACTICAL_BANK.md#h3-six-conditional-styles) and [the exit report](R3_EXIT_CHECKS.md). The original statement that Random “keeps its strategy” referred to the selected profile, but did not explain adaptive plans clearly. The menu tooltip now says: **“Chooses one of six styles for this game. Plans adapt, with balanced fallback when needed.”** Random does not reroll the profile on every turn. The existing H5 tactical implementation remains unchanged.

The deployed-DLL audit passed 62 behavioral assertions before the repair. It covered feasible target retention, lost-card/funding recovery, Cotton's brewery contingency, recovery after four stalled decisions, bounded specialist preferences, complete legal plans, and detached per-seat intent in memory. These checks are also included in final-candidate validation.

## Confirmed gameplay bugs repaired

| Finding | Reproduction and consequence | Repair |
| --- | --- | --- |
| Human city-click building stayed disabled after AI | 1.3.1 set `BoardUseCase._allowBuildingFromBoard` false when scheduling an AI and never restored it on the local-human path. With a cached board, the next human turn opened city details instead of entering the build flow. The new native regression fails against the archived 1.3.1 DLLs. | Remove the persistent flag change. The guarded city/action/card entry points still reject input on a strategy AI's turn. A real mixed AI-to-human transition now verifies that direct building remains available. |
| Historical human replay accepted action input | The previous guard checked whether the historical actor was human, but not whether the displayed game was a replay. Replaying the human's own just-completed action rewinds to a human state with remaining actions; city and action callbacks could enter action routes there. | Reject those callbacks while `BrassGameHistory` is moved back in history. Native tests drive the actual replay use case and check blocked input throughout historical events, then restored building access on return to the live game. Original-game/tutorial routing is preserved. |
| Final introductory AI compared incompatible score units | A final introductory pass could stop in mandatory liquidation, leaving its candidate scored in heuristic VP. A loan could finish settlement and be scored as terminal win share. In the reproduced case, the AI chose the pass's **70.96 heuristic points** over the loan's **1.0 win share**. Actual settlement made the pass lose **78–82**, while the loan won **86–82**. | Complete forced final settlement inside each isolated candidate before comparing terminal outcomes. All completed final candidates use official win share/rank. Earlier ordinary turns still stop at the S1 action boundary. |

The liquidation loop is bounded by the number of occupied industry spaces, since each step removes a tile. It uses public board values, introduces no opponent ordinary turns, and never changes the live game. Its primary liquidation valuation matches the live AI's existing valuation; semantic slot IDs break ties without using hidden setup identifiers. This is a public liquidation model: human choices and the live bot's different tie ordering can produce different outcomes when several choices exist. The exact win/loss regression uses a unique liquidation choice, so that limitation does not explain the original failure.

The final-introductory scoring defect was introduced by the deployment's support for introductory games. The frozen four-player H5 worker did not implement that introductory projection; this is not evidence that its original tournament contained the same defect.

## Scope and preserved behavior

The repair changes only four deployed C# files: `Unity/LocalMenu.cs`, `Unity/LocalTurnBridge.cs`, `Unity/TurnInputGuard.cs` and `Policy/Planner.cs`. The installed rules/helper assemblies are byte-identical to 1.3.1. Existing rules fixes, brighter city highlights, disabled Play now, ten-second search allowance, 1,024-node ceiling and local-only configuration are retained.

No save-game bug investigation or save-code repair is included. Dedicated save/reload suites are excluded from this release's new validation. Full mixed-game tests use application games and event histories held in memory. Existing native UI fixtures still exercise ordinary game lifecycle calls through a fixture repository; no real user save is opened.

All six embedded S1 v2 profiles still match H5's frozen `bank-804c85204da65824cb9b` exactly. The frozen research sources, profiles, contracts and runtime are unchanged. The game deployment has its own adaptations and package version; H5's strength/admission results are not new results for this repaired deployment or its larger live budget.

## Verification

All checks below passed. The release gate requires final-DLL fingerprints on the following evidence in `artifacts/patch-build`:

- `local-ai-runtime-0.json`: 1,022 S1 complete-turn, legal-choice, hidden-information and player-mapping assertions, with save tests disabled.
- `local-ai-tactical.json`: 62 adaptive-intent and bounded-preference assertions.
- `local-ai-search-gameplay.json`: 1,441 assertions covering final/nonfinal boundaries, all six profiles, 2/3/4 players, introductory/full games, one/two actions, several debtors, VP-only debt and unchanged live state.
- `local-ai-gameplay-live.json`: six complete mixed games through live capture, validation and commit; 2,042 assertions, 237 turns, 456 actions and ten debt episodes, all in memory.
- `local-ai-native.json`: 34,569 assertions over 36 native turns across all profiles/configurations, including repeated input during thinking, AI-to-human city building, historical replay protection and cancellation.
- `local-ai-runtime-{2,3,4}.json` and `local-ai-identical-{2,3,4}.json`: 252 freshly executed complete strategy games on the repaired binary. These use deterministic one-node work for reliability coverage; they do not measure playing strength or ten-second search performance.
- `local-ai-h5.json` and `local-ai-h5-installed.json`: candidate and installed profile provenance, including 39 research source files and 112 frozen runtime files.
- `packaging-results.json`, `local-ai-installed.json` and `installation-verification.json`: 79 upgrade/recovery assertions, 57 installed menu/startup assertions, and installed-file/hook checks.

Test runners clear stale reports, pin candidate fingerprints before execution, and reject failed child processes before publishing success. The qualification gate verifies unique scenario identities, including player counts, styles, seeds and active seats; row totals alone cannot qualify a release. An isolated regression passed 25 cases using the actual runner scripts with mock children, including stale success, missing output, nonzero exits, compiler failure and changed DLLs (`artifacts/re-c0806b2a/runner-evidence-result.json`).

The previous managed rules, city-render and ten-suite Unity results are retained specifically for the **byte-identical engine/helper DLLs**. They are not described as newly executed 1.3.2 checks. The scoped qualifier rejects a change to either binary and rejects source changes outside the four reviewed AI/UI files. This preserves previous rules verification without reopening save-game work.

Baseline reproductions and the full review are retained under `artifacts/ai-review-1.3.2/`; the prior installed package, source and evidence are archived in `artifacts/patch-releases/1.3.1`.

## Installation record

Installed at `2026-09-28T07:42:39Z`. The installer retained the original stock assembly and a verified copy of the previous three-file package at `artifacts/patch-recovery/c3451fb2c3d84a4c9780e4753313d8e1`. The installation record is `artifacts/patch-installation.json`.

| Installed file | SHA-256 |
| --- | --- |
| `Assembly-CSharp.dll` | `0FCF229D5373A389AE855FC00F7F93628E5731160B0D90DADA26A05B3322CDCD` |
| `BrassPatch.dll` | `61C79AF0B4DAF4697FE26C7238D65CF00F4E7E0FA65D56FAA11C972D56BF22DF` |
| `BrassLegacyAI.dll` | `0A611374A49A40C07BABE8CB67B8BF1CDC2A68E5ED33BFD8663D0F6EB6087FF8` |

Installed verification confirms all 116 prior hooks, six merchant-bonus rewrites, 92 methods bearing earlier rules-helper calls, and the online-disabled route. The first isolated packaging attempt encountered a Windows file-replacement error in its temporary fixture; a fresh run passed all 79 checks. The actual game installation completed successfully on its first attempt. No user save was opened, migrated or deleted.

## Reproduce this scoped qualification

```powershell
pwsh -NoProfile -File scripts/build-patch.ps1
pwsh -NoProfile -File tests/BrassLocalAI/run-runtime.ps1 -GameplayOnly
pwsh -NoProfile -File tests/BrassLocalAI/run-tactical.ps1
pwsh -NoProfile -File tests/BrassLocalAI/run-search-gameplay.ps1
pwsh -NoProfile -File tests/BrassLocalAI/run-gameplay-live.ps1
# These use separate runtimes and may run concurrently.
pwsh -NoProfile -File tests/BrassLocalAI/run-gameplay-matrices.ps1 -Players 2
pwsh -NoProfile -File tests/BrassLocalAI/run-gameplay-matrices.ps1 -Players 3
pwsh -NoProfile -File tests/BrassLocalAI/run-gameplay-matrices.ps1 -Players 4
pwsh -NoProfile -File tests/BrassLocalAI/run-unity.ps1
pwsh -NoProfile -File tests/BrassLocalAI/qualify-release.ps1
pwsh -NoProfile -File tests/BrassPatch/packaging-regression.ps1
pwsh -NoProfile -File scripts/install-patch.ps1
pwsh -NoProfile -File scripts/test-installed-local-ai.ps1 -Smoke
pwsh -NoProfile -File tests/BrassLocalAI/verify-h5-provenance.ps1 -Installed
pwsh -NoProfile -File tests/BrassPatch/CityLocator/verify-release.ps1
```

Automated replay/pointer-route fixtures do not certify every physical mouse sequence or display/DPI setup. No additional non-save defect was established in the reviewed Random setup, dispatch/cancellation, live commit, semantic action generation and S1 boundary paths.


