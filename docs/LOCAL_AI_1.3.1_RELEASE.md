# AI input and city highlight repair — patch 1.3.1

**Historical release: installed and verified on 28 September 2026.** See the [1.3.2 follow-up](LOCAL_AI_1.3.2_RELEASE.md) for corrections to direct human building, historical replay input and final AI settlement.

## Changes

- **Random AI option:** each bot can choose Random in the custom-game selector. At game creation it independently draws one of the six S1 profiles for the match. Its current plan and intent adapt, including fallback when needed. The next-game preference stays Random; Continue displays the actual chosen strategy. Selection uses separate unbiased randomness and does not consume the board/deck RNG.
- **Clicks cannot take over a strategy AI's turn.** City clicks, card selection, the seven action buttons, undo and end-turn are accepted only for the current human player in games using the new AI. The board's direct-build mode is also disabled when an AI starts. Original games and tutorials retain their existing input routing.
- **Play now is disabled and removed from the panel.** Its previous callback is inert, so even a stale reference cannot select the first available plan early. The panel stays at “Thinking…” until search completes and then displays “Playing…”.
- **More search capacity:** the deployment runtime grants at least **10,000 ms and a 1,024-node ceiling** per complete turn, up from 3,000 ms and 240 nodes. Search may finish earlier when its candidates or work limit are exhausted. Preparation of the live board no longer consumes this allowance. The limit is soft: completing a legal action can exceed it slightly. This increases search opportunity; it is not a measured playing-strength claim.
- **Skip to me only affects replay.** It cannot replace or rewind a live game while an AI is thinking or awaiting dispatch. Once the AI has finished, normal replay skipping remains available. Hiding an AI panel alone no longer abandons its worker; changing games, moving through history, or exiting still cancels stale work.
- **Brighter city highlights:** warm ivory `#FFF0C7`, a 100%-opacity border replacing the previous 55% treatment, a thicker 4%-of-slot stroke, a feathered outer halo and an 18% interior wash. Rounded corners remain. The original green build highlights and build eligibility are unchanged; the extra graphics cannot intercept clicks.

All prior rules fixes and the six strategy choices remain included. Online play remains disabled. Saved profile identities and strategy preferences stay intact; the larger execution allowance applies to existing strategy games too. No additional save-bug investigation or migration is part of this repair.

## Diagnosis and scope

The stock game computed AI turns synchronously. The new worker runs across frames, exposing human UI routes that previously had no opportunity to execute during planning. `BoardUseCase.StartBuildActionFromBoard` checks remaining actions but does not check whether the current actor is human. The 1.3.0 bridge also treated an invisible AI panel or a changed revision as grounds to stop its work.

A native Unity regression against the installed 1.3.0 binaries reproduced a city click entering the human build route on an AI's turn. The failed baseline is retained at `artifacts/patch-releases/1.3.0/input-regression-reproduced.json`. The repair guards production use-case entry points, including delayed callbacks, instead of relying solely on a hidden button or panel.

The policy, evaluator, strategy profiles, belief reconstruction and board-game rules are unchanged. The live turn allowance and UI routing are corrected. There are 116 method hooks (104 retained plus 12 input guards), six merchant-bonus rewrites and the existing brewery JSON annotation.

## S1 research cross-check

The implementation was checked against [R3's complete-turn specification](R3_LEGACY_BANK.md#h2-complete-turns), [the tactical profile description](R3_TACTICAL_BANK.md#h3-six-conditional-styles), the original `src/BrassLegacyAI/Planner.cs`, and `tests/legacy-ai/test_legacy.py`.

The deployed catalog still requires `mode: S1`. The planner builds a complete incumbent, retains setup families in the first-action beam, generates the second action from the actual first-action afterstate, and compares complete-turn scores. The opening Canal turn has one action. Ordinary future draws and opponents' decisions are outside the S1 horizon. Scout's specified wildcards remain usable in the second action. Each style uses its own intent and bounded preference; private opponent hands and the live RNG are excluded.

The research includes an explicit `balanced-s1-10s-v2` control, confirming that a ten-second allowance is compatible with S1. Its research node ceiling is 240; this deployment deliberately grants 1,024 nodes so the additional time can explore more of the existing beam. First-action width 28, beam width 10, second-action width 14, tactical weights and credible-score windows retain their original profile values. Both action commits and any replan share one running turn clock; it is never reset for action two. Capture is outside this deployment's search allowance and is not represented as a research whole-turn benchmark.

The frozen research bank and its reported strength/admission results are unchanged. They must not be presented as results for this different live budget. New deployed-runtime regressions exercise enabling loan/build and develop/build sequences, complete legal alternatives, widening beyond the greedy incumbent, fixed-work repeatability and absence of live-state mutation.

### H5 version verification

**The game uses the six S1 v2 strategies included in H5's `bank-804c85204da65824cb9b`.** H5 is the immutable-bank and evidence-gate milestone; S1 and S2 are the search algorithms inside that bank. It does not define an additional “H5” search mode. The [H5 exit record](R3_EXIT_CHECKS.md#tournament-and-admission) admits Balanced, Rail Tempo, Cotton, Goods/Beer and Supplier to training and Pottery Flex to held-out evaluation. All six are suitable identities for this local-game selector; Pottery Flex's separation concerns research data use. S2 did not pass the equal-time benefit gate and was not promoted over S1.

`tests/BrassLocalAI/verify-h5-provenance.ps1` checks the manifest against the original correctness record, all 39 frozen and active research source files, all 112 frozen runtime files, each pool's qualification-report hash and all six profiles embedded in the deployed DLL. **Each embedded profile is an exact byte match to its H5 `*-v2.json` file, with `mode: S1` and persistent tactical intent enabled.** Release qualification now requires this check. The separate `*-s2-v2` files and older v1 profiles are not selected.

Source comparison also confirmed that `Tactics.cs` differs only in looking players up by their actual color/seat instead of assuming array position. The deployed planner retains the H5 S1 proposal, full-turn completion, beam and bounded style-selection logic. Its reviewed differences handle 2/3/4 players, reject S2 profiles, use the same seat lookup and settle the actual final game state for scoring. The evaluator similarly supports variable round limits, active seats and introductory games. These adaptations were introduced for the 1.3.0 deployment and remain unchanged in 1.3.1.

This is a versioned game deployment derived from H5's S1 v2 code, rather than the byte-identical four-player research worker. Its 10-second/1,024-node execution allowance is an explicit deployment override. H5's original qualified S1 settings remain 3 seconds/240 nodes, and their strength results do not transfer automatically to the changed allowance.

## Verification and installation records

The final three-DLL candidate passed the following checks. Its installed hashes match the tested candidate, all 104 previous AI/rules/UI hooks remain present, and online play remains disabled.

| Check | Result |
| --- | --- |
| H5 provenance | Six exact embedded S1 v2 profiles; 39 research source files and 112 frozen runtime files verified; repeated against the installed AI DLL |
| Deployed S1/runtime contracts | 1,030 assertions, including loan → build and develop → build enabling sequences, complete alternatives and unchanged live state |
| Native strategy turns | 36/36 passed across six strategies, three player counts and normal/introductory modes; repeated input attempts, no Play now, full action completion and cancellation checked |
| Full live mixed games | 6/6 completed; 1,397 assertions |
| Previous managed rules/AI suites | Passed, including 90 complete original-AI games |
| Existing Unity suites | All ten passed on the candidate and again on the installed files; no native errors |
| City visuals | 18 day/night and zoom captures; mixed green/ivory and zoomed-out frames visually inspected |
| Upgrade/recovery | 79 assertions passed upgrading a fixture from 1.3.0, including partial publication and locked-file recovery |
| Installed menu/startup | 57 assertions passed, including Random resolution and game creation |

The previously recorded 252 strategy simulation games remain supporting evidence for unchanged policy/domain code. They were not rerun or relabeled as tests of the new live budget. Windows sandbox runs encountered file-replacement errors in packaging fixtures; the complete fixture suite and actual atomic installation passed with normal host access. No installer safety checks were bypassed.

Current evidence is recorded under `artifacts/patch-build` with the tested DLL hashes:

- `local-ai-native.json`: native strategy turns for every strategy, player count and normal/introductory configuration; input guards before dispatch and during search, repeated input, absent Play now, panel hiding, cancellation and exactly one confirmation per action.
- `local-ai-runtime-0.json`: legal-turn, configuration, hidden-information and state-contract smoke tests.
- `local-ai-h5.json` and `local-ai-h5-installed.json`: frozen-bank/profile provenance for the candidate and installed AI DLL respectively.
- `local-ai-live.json`: six complete mixed games through the live action/intent path.
- `validation.json` and `unity-isolated-result.json`: existing managed and native rules/UI regressions.
- `city-render-result.json`: 18 day/night and zoom comparisons using the shipped board prefab.
- `local-ai-qualification.json`: exact source/IL checks distinguish retained 252-game simulation evidence from final-build native tests. Historical simulations use a deterministic one-node work limit; they do not measure the new thinking allowance or AI strength.
- `packaging-results.json` and `installation-verification.json`: upgrade/recovery and installed-file verification.

Native tests use real Unity prefabs with a fixture router and scripted input routes. They do not certify every physical mouse sequence or display/DPI setting. Real user saves are not changed by these tests. Previous 1.3.0 binaries and affected sources are retained in `artifacts/patch-releases/1.3.0`; the installer also retains a verified backup of the replaced package.

The installed record is `artifacts/patch-installation.json`; the replaced 1.3.0 package is also backed up in `artifacts/patch-recovery/6fb1ca7820454f798d0a6f6b2ac02014`. The three installed SHA-256 values are:

```text
Assembly-CSharp.dll  0FCF229D5373A389AE855FC00F7F93628E5731160B0D90DADA26A05B3322CDCD
BrassPatch.dll       61C79AF0B4DAF4697FE26C7238D65CF00F4E7E0FA65D56FAA11C972D56BF22DF
BrassLegacyAI.dll    05C2686BB4C89BAF53A9EF857C5DC31A2081E22E37F7B4BD80B48EDF3D4F97D0
```

Build and qualification commands remain in [the 1.3.0 release notes](LOCAL_AI_1.3_RELEASE.md). The updated packaging test defaults to upgrading from 1.3.0. `tests/BrassLocalAI/verify-h5-provenance.ps1 -Installed` repeats the H5 comparison without starting the game. The game is ready to launch normally through Steam.

