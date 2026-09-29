# Brass: Birmingham — reference cross-check and additional fix plan

Reviewed 27 September 2026 against local patch 1.0.0, using [npow/brass-birmingham](https://github.com/npow/brass-birmingham) as an independent implementation. **R01–R03 were delivered in patch 1.1.0 and are retained in the installed combined patch 1.2.0**. The current release adds city-card locator outlines and retains all previous B01–B16 fixes and the save-integrity repair. The original review evidence and repair rationale below are retained for traceability.

## Completed implementation and installation

| Finding | Delivered repair | Verification |
| --- | --- | --- |
| R01 | Per-attempt game/player/action ownership; stale callbacks have no UI/domain effects; confirmation uses the captured flow before presentation; owned animations stop on cancel/destruction and reset on cached-view reuse | 125 managed Scout assertions include cancellation, restart, Undo, navigation, reentrant presentation, actual replay/hand binding and filesystem reload; 128 native assertions cover nine timing/input-handler scenarios |
| R02 | Canonical game-owned card validation, distinct exchange checks, preserved start-time wild restriction, immediate native input lock, ignored excess/repeated input | Invalid domain commands leave their preceding state unchanged; normal 2/3/4-player exchanges conserve cards and reload correctly; same-update repeated card-handler call is rejected |
| R03 | Early era/round check and null guard, preserving the original applicable opening heuristic | 3,490 assertions; 296 executable public AI decisions; all three legal exhaustion histories and event reconstructions now return legal moves |

Implementation: [ScoutFixes.cs](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Scout/ScoutFixes.cs>), [ScoutViewFixes.cs](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Scout/ScoutViewFixes.cs>), and [AiFixes.cs](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/AI/AiFixes.cs>). The shared original-method invoker now handles private patched methods declared on base classes. Event schemas and replay handlers are unchanged.

**Combined validation passed:** all previous regression suites, 90/90 full original-AI games (9,210 actions), three prior-patch save upgrades, and 28 packaging/recovery checks. All nine suites passed inside the installed Unity player, including the unchanged 19-check liquidation popup suite. The installed build retains all 67 prior method patches, adds 12, and preserves the six merchant-bonus rewrites. Both installed hashes and the online-disabled method were verified. Exact fingerprints, commands, rollback instructions and limitations are in [LOCAL_PATCH.md](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/LOCAL_PATCH.md>).

Current regression commands are `tests/BrassPatch/validate-patch.ps1`, `tests/BrassPatch/packaging-regression.ps1` and `scripts/test-unity-patch.ps1`. Historical failure probes below remain pinned to the reviewed 1.0.0 hashes and deliberately reject the upgraded installation. Neither upgrade testing nor installation opened or migrated real user saves. Existing corrupt histories are not silently rewritten. Full gameplay-prefab/physical-input coverage and natural AI-trigger frequency retain the limitations described below; those limits do not prevent the new guards from rejecting the demonstrated invalid operations.

## Original review scope and reference versions

| Input | Exact version |
| --- | --- |
| Reference repository | [`fdeb63c6aa92cc4ba44a721482316e0b8db10e06`](https://github.com/npow/brass-birmingham/tree/fdeb63c6aa92cc4ba44a721482316e0b8db10e06), committed 2 September 2026 |
| Reviewed 1.0.0 `Assembly-CSharp.dll` SHA-256 | `34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A` |
| Reviewed 1.0.0 `BrassPatch.dll` SHA-256 | `4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D` |
| Official rulebook SHA-256 | `BB627A11C769957FBB5F26A210E3957AE27A2BAD054BA9CB3B4FADC6C2BA73E5` |

The rule authority remains [Roxley's official rulebook](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281), retained as [official-rulebook.pdf](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/official-rulebook.pdf>). Page numbers below are printed pages. The reference is useful for identifying differences and missing boundaries; it is not an oracle. This review reproduced errors in the reference itself.

The original Steam comparison included decompiled original code **and** 1.0.0 replacement methods/manifests under `src/BrassPatch`. Original source alone would incorrectly report already-fixed defects. Scope is local human/hotseat/original-AI play, consistent with the offline patch. The review itself changed no production binary or real user save; the subsequent authorized implementation and installation are recorded above. Executable probes use artificial games and save directories.

## Original findings and repair rationale

Three additional repair candidates remain after excluding already-fixed defects and errors in the reference. The Scout use-case failures reproduce against production code, and a separate isolated Unity test verifies the shipped animation's late-callback behavior. That native test uses a minimal in-memory hierarchy; serialized gameplay prefabs, physical mouse input and full interactive cancellation were not exercised. The AI finding reproduces through the current public decision entry point, including after a legal scripted history and event reconstruction.

| ID | Priority | Finding | Evidence status |
| --- | --- | --- | --- |
| R01 | P1 proposed | Cancelled Scout completion can mutate the restored game or a subsequent action | Production use-case/persistence reproduction; shipped Unity animation delivers callback after target destruction; full interactive path unverified |
| R02 | P1 conditional candidate | Scout can exchange the same physical card twice | Production use-case/domain reproduction; native card handler accepts repeated calls before tween update; physical-input trigger unverified |
| R03 | P1 conditional | Hard AI crashes on an exhausted brewery stack in the Canal Era | Public AI decision crash after legal scripted actions in 2/3/4-player games; same failure after event replay |

These proposed priorities reflect blocked progress, corrupt turn/card accounting, or materially illegal actions when triggered. They do not imply that a trigger is common. R02 remains a conditional input-hardening candidate until an ordinary-input trigger is established; R03 has not been observed in a game where the original Hard AI itself selected the entire preceding history. The R03 brewery defect is separate from the prior manufacturer's-stack finding B14.

## R01 — Scout cancellation does not invalidate delayed completion

**Expected behavior:** Cancelling an unfinished action restores its starting state. A callback belonging to that cancelled action must not discard cards, grant wild cards, consume an action, or alter a later action. The Scout action's three-card exchange and normal per-turn action allowance are specified on printed pp. 6 and 11.

**Relevant code:** [ScoutActionUseCase.cs:42](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.Actions.ScoutAction/ScoutActionUseCase.cs:42>) queues `OnCompleteAction` through `SetupCardInSlot`. `CancelAction` at line 83 resets counters and hides the view, but does not invalidate that callback or clear its references to the old action/player. `OnCompleteAction` at line 66 uses the old Scout references and then calls [BaseActionUseCase.ConfirmAction](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.Actions.BaseAction/BaseActionUseCase.cs:24>), which operates on `_gameHistory.CurrentGame`—potentially the restored game or a new action. [ScoutActionView.cs:71](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/Frameworks.View.Game/ScoutActionView.cs:71>) creates the animation sequence and appends the completion callback after a one-second final pause.

**Controlled-callback reproductions:** Start Scout, discard its action card, select the two trade cards, cancel through the production cancel use case, then deliver the previously queued completion callback. The restored game consumes an action without discarding a card. In a separate case, complete a Pass after cancellation and then deliver the old callback: confirmation attempts another action decrement, throws, and leaves a revision gap. The patched filesystem writer correctly rejects the next save with `Incoming event revisions are not contiguous.` The earlier good save remains intact; this is a live-state/save-progress failure, not a reproduction of the already-fixed file truncation bug.

**Native timing evidence:** In Unity 2019.1.7f1 with the shipped DOTween and unchanged `ScoutActionView.AnimateCardToSlot`, completion fires at approximately 2.93 seconds. Destroying the view/card targets at 0.50 seconds or 2.50 seconds still delivers the final callback once. These destruction points represent cancellation plus the 0.30-second hide period. The late-pause case emits no warnings; earlier destruction produces missing-target warnings that DOTween swallows without cancelling the outer completion. `ScoutActionPresenter.SetupCardInSlot` disables hand interaction after the second selection, but does not clear the cancel delegate; the action-progress view retains its Cancel callback and an unguarded Escape handler. Escape delivery itself is unverified because the top hand view can consume the key. The callback lifetime is a demonstrated defect, while full prefab routing and actual button hit-testing remain a release-test requirement. Detailed evidence and limitations are in [phases-findings.md](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/reference-review/phases-findings.md>).

**Reference comparison:** The reference's [Scout selection/commit branch](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/uiManager.js#L990-L1025) commits through the current pending action. Its simpler synchronous design does not justify replacing the Steam presentation system, but it highlights the missing lifetime check around Steam's asynchronous completion.

**Proposed repair:**

1. Give each Scout interaction a generation/token tied to the exact application-game instance, player, and Scout action. Capture it in the completion closure. Cancellation, leaving the view, successful completion, and starting another Scout invalidate the previous token.
2. Before hiding views or changing domain state, require the token to remain current, the same game/action/player to be active, and the selection to remain valid. Mark a valid completion consumed before invoking code that can reenter the UI.
3. Store and cancel the view's owned sequences on cancellation/destruction, while restoring card parents/layout. Animation cleanup complements the domain guard; it must not be the only protection.
4. Ensure confirmation cannot accidentally confirm a different action. Preserve the existing event schema and normal Scout replay/undo behavior.
5. Keep the save writer's contiguous-revision rejection. Do not suppress it or renumber a damaged history to make the symptom disappear. Broadly changing aggregate exception handling requires a separate atomicity design; decrementing the revision after a failed handler alone can leave other partially changed state.

**Acceptance tests:** Cancel before either selection, between selections, during the second animation, and during its final pause; deliver the old callback before/after a Pass, a new Scout, game navigation, and save/load. Every invalidated callback must leave revision, event list, hand, discard pile, wild decks, action count, and current action unchanged. A normal valid Scout commits once. Verify contiguous events and save/reload equivalence after every cutoff, then exercise the shipped view with real tween updates.

## R02 — Scout selection lacks distinct-card validation

**Expected behavior:** A Scout costs three distinct cards: the action card plus two further cards from the player's current hand; it grants one wild location and one wild industry card. Printed p. 11 also prohibits Scouting while a wild card is already held.

**Relevant code:** `ScoutActionUseCase.SelectCard` at line 42 increments a counter and appends cards without checking whether the selected card is already queued. The domain [ScoutAction.Scout](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/ScoutAction.cs:22>) checks the number of submitted entries, but not distinct identities/current ownership; its event handler at line 35 adds both submitted references to the discard pile. The native view delays disabling the first selected card until a tween callback. Repeated input before that callback can therefore submit the same card twice.

**Controlled reproduction:** Supply the same card as both trade-card selections. After completion, the player has eight cards instead of the legal seven before the end-turn refill; the discard pile contains the same card ID twice. This demonstrates an actual card-accounting defect. Native input timing and ordinary click likelihood must be reported separately from the deterministic use-case result.

**Native boundary check:** Calling the shipped `BaseCardView.OnMouseClicked` twice on the same card before the first tween update accepts both calls and completes both slot animations. The first call does not immediately lock the selected card. This verifies the missing synchronous guard, but it does not establish that a normal mouse can generate both accepted inputs through the shipped event routing in one update. Do not describe this as a confirmed ordinary double-click exploit. Repair the invariant with R01, and keep normal-input reproduction as an explicit acceptance target.

**Reference comparison:** The reference explicitly ignores a card index already in `pendingData.scoutCards` at [uiManager.js:1001–1008](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/uiManager.js#L1001-L1008). Adapt the invariant, not its positional-index representation: Steam should validate actual game-owned card identities.

**Proposed repair:**

1. Reject duplicate, stale, non-owned, already-discarded, or excess selections in the use case before incrementing its counter or starting an animation. Resolve selections to canonical cards in the captured game's current hand; ordinal IDs can repeat across separate games, so ID equality alone is insufficient.
2. Disable/reserve a selected card immediately. Ignore repeated input after the selection becomes complete.
3. Validate the complete exchange in the domain before emitting `WildCardsTaken` or mutating either trade card/wild supply: two distinct additional cards, both still owned, neither the already-used action card, and both wild supplies available. Preserve the no-wild-card check at action start, before discarding its action card, and revalidate the remaining conditions when the guarded R01 completion executes. The existing action-card discard precedes this exchange validation.
4. Treat R01 and R02 as separate regression contracts even if implemented in one Scout patch. R01 lifetime checks alone do not make two duplicate selections legal.

**Acceptance tests:** Same card twice; two input events in one update; stale card after cancellation; a fourth selection; a card from another player/game, including a foreign card with the same ordinal ID; unavailable wild supply; an existing wild card, including an attempt to discard it as the action card; legal selection in either order. Invalid attempts must leave the immediately preceding domain state unchanged. Legal cases must conserve physical card identities and produce exactly three distinct discards and two wild acquisitions, followed by one action decrement. Verify end-turn refill, era hand exhaustion, undo, and event-based reload.

## R03 — Another exhausted-stack Hard-AI crash

**Cause:** [BuildBreweryForCanalEra.cs:43](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.AI.RuleBasedSystem.Rules/BuildBreweryForCanalEra.cs:43>) reads `firstBuildingOfType.IsForCanalEra` without checking whether the brewery stack is empty. The `CurrentRoundNumber <= 2` condition appears **after** that dereference. Thus a rule intended only for the opening can crash in a later Canal round before it gets a chance to return false. The current patch fixes B14's manufacturer rule, but does not patch this brewery rule.

**Rule authority:** Developing one or two lowest-level tiles and taking loans with their three-income-level reduction are on printed p. 11; round income and handling an uncovered shortfall are on p. 6. Era restrictions apply to building tiles, so developing Rail-only Brewery IV during Canal is allowed. Nothing requires a player to retain a brewery tile. A lack of a build candidate must not prevent an otherwise legal turn/Pass.

**Legal-history reproduction, seed `20260928`:** Other players Pass. The target takes a Loan in round 1 and two Loans in round 2. In rounds 3–4, it develops all seven brewery tiles in batches `2,2,2,1`. In rounds 5–6 it develops all seven coal tiles in the same batch pattern. All iron is bought through the normal market/general-supply path; every action's affordability, development permission, card discard and action count are checked. Round-six income settlement leaves £0 and income −9. On that player's Canal round-seven turn, the public `HardAiBehavior.GetNextAIMove` throws `NullReferenceException` in `BuildBreweryForCanalEra.IsApplicable`.

| Players | Legal actions through the trigger | Trigger | Event-reconstructed trigger |
| --- | --- | --- | --- |
| 2 | 24 | Crash | Same crash |
| 3 | 37 | Crash | Same crash |
| 4 | 50 | Crash | Same crash |

No direct cash, mat, board, income or round mutation is used in that trace. It is a scripted legal sequence, **not** evidence that the original AI policy naturally chooses those preceding moves. A smaller arranged fixture confirms that an empty brewery stack alone suffices when higher-priority actions are unavailable; it does not require empty coal or other stacks. Controls leaving one brewery, changing to Rail, or using Easy/Medium all complete. The broader 1,152-case synthetic empty-stack sweep produced six exceptions, all this same cause; impossible combinations in that stress sweep are not counted as legal reachability evidence.

**Policy reachability limit:** The original AI's normal Develop rules only select Brewery I, so they cannot select the scripted all-seven-development history unchanged. Its Gloucester bonus fallback can select any next brewery, including Brewery IV, after earlier breweries are gone. This is a source-traced path to an exhausted Canal brewery stack, not an executed natural-policy trajectory. The crash also requires no owned unflipped brewery and no applicable higher-priority action. An [independent review of the legal history and AI policy](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/reference-review/actions-r03-review.md>) records the arithmetic, rule checks and limitations.

**Reference comparison:** The reference checks for an absent next tile before reading its properties in [gameLogic.js:79–83](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L79-L83), and its AI selects from those validated candidates. It does not contain the same Steam heuristic, so this is a boundary-design comparison rather than a line-for-line port.

**Proposed repair:** Add an early era/round check and an explicit null guard before any brewery property access. Preserve the heuristic's priority and selection behavior for valid opening states; return “not applicable” when the stack is empty or the opening window has passed. Implement through the existing version-gated weaver/helper architecture. Do not catch and ignore arbitrary AI exceptions or force a Pass over a still-legal higher-priority action.

**Acceptance tests:** Exhausted/one-remaining brewery in Canal rounds 1, 2 and 3+, Rail controls, owned unflipped brewery present/absent, all player counts and difficulties. Repeat the legal trace and event replay above. Assert the public decision returns an executable legal move and consumes exactly one card/action when executed. Retain B14 controls and rerun the existing full-game batch after the change.

## Differences that must not become Steam fixes

| Reference discrepancy | Evidence and decision |
| --- | --- |
| Four-player merchant mix adds an extra universal buyer instead of a blank | Official p. 4's full setup illustration shows three blanks and one universal buyer; current Steam matches. Reference has two blanks and two universal buyers, adding a beer. Do not import it. The separate three-player difference is recorded without claiming unreadable printed pips prove every subset assignment. |
| Single rail can use a farther coal mine | Reference `executeNetwork` queries coal before placement and its endpoint merger sorts free/cost without global distance. An executed Birmingham–Dudley fixture consumes Redditch coal while Dudley coal remains. Current Steam selects Dudley and rejects Redditch. B03 is already fixed. |
| Two merchant beers can make a two-beer sale appear legal | Reference offers Manufacturer V with two merchant barrels and no brewery, then reports failed sale after consuming one Oxford beer and gaining two income spaces. Current patch's B04 permits at most one merchant barrel per industry and rejects that funding arrangement. Do not port the reference's Sell eligibility/partial commit logic. |

Sources: [merchant mix](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameData.js#L393-L407), [rail selector](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L631-L649), [Sell eligibility and execution](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L760-L887). Detailed [component findings](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/reference-review/components-findings.md>) and [action findings](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/reference-review/actions-findings.md>) retain the individual comparisons and limits.

## Repair sequence and release gates

1. **Preserve and promote the new reproductions.** Turn each failure into a regression expectation for the proposed corrected behavior. Keep distinct tests for native presentation timing, domain invariants, save continuity, and AI legal-state reachability. Preserve the current supported binary hashes as the baseline.
2. **Repair Scout lifetime and selection together.** Implement R01's generation/ownership/one-shot completion and R02's use-case/domain checks, then repair owned-animation cleanup. Test cancellation while the final animation pause is active and ensure a new action cannot be affected.
3. **Apply the small R03 guard independently.** Keep it separately reviewable and retain its legal-history and difficulty controls. It can be delivered first if the Scout UI work takes longer, but do not mark Scout resolved because AI tests pass.
4. **Validate save compatibility.** New guards should prevent bad new events without changing valid legacy schemas or versioned IDs. Use copies of artificial histories, not real user saves. Already-corrupt duplicate-card/revision-gap histories need a separate repair analysis; do not silently “repair” them by deleting or renumbering events.
5. **Run targeted and existing release gates.** Execute new Scout/AI cases, prior B01–B16 and save-failure tests, existing resource matrices, and the 90 complete AI games across 2/3/4 players/difficulties. Run the actual Unity callback/input tests for the exact candidate. Check both installed candidate hashes, atomic installation recovery, and restore/reinstall.
6. **Publish the repair record.** Update the patch guide with resolved R IDs, evidence, exact binaries, remaining limitations and rollback instructions. Retain local-only mode. This plan does not authorize an online compatibility claim or a new save migration.

## Historical review evidence and reproduction

The following commands produced the original failure evidence before the upgrade. They require the reviewed 1.0.0 installation fingerprints; use the current regression commands above to verify 1.1.0. PowerShell probes run in separate processes from the game directory. Windows PowerShell is required for the application/persistence Scout harness; PowerShell 7 is used for the engine probes.

```powershell
pwsh -NoProfile -File rules-audit/reference-review/components-probe.ps1
pwsh -NoProfile -File rules-audit/reference-review/actions-probe.ps1
pwsh -NoProfile -File rules-audit/reference-review/sell-ai-probes.ps1
pwsh -NoProfile -File rules-audit/reference-review/ai-brewery-repro.ps1
pwsh -NoProfile -File rules-audit/reference-review/ai-brewery-legal-trace.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File rules-audit/reference-review/phases-probe.ps1
```

With the game closed, `powershell -NoProfile -ExecutionPolicy Bypass -File rules-audit/reference-review/phases-native-run.ps1` compiles and runs the isolated native timing probe through the installed opt-in self-test entry point. It temporarily substitutes an **artifact test DLL**, then restores it and the prior self-test result/log in `finally`. Normal game initialization and real save loading are skipped. [phases-native-results.json](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/artifacts/reference-review/phases-native-results.json>) records the four timing cases; [phases-native-run.json](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/artifacts/reference-review/phases-native-run.json>) confirms that both installed binaries, `validation.json`, and all three original test artifacts remain unchanged/restored. The native runner requires the existing compiler and self-test artifacts from this local patch workspace.

The `components-compare.cjs`, `actions-reference-probe.cjs` and `sell-reference-probe.cjs` scripts run with Node against the pinned checkout under `artifacts/reference-npow-brass`. They produce reference-comparison JSON beside themselves. The external repository and generated binaries/data remain local artifacts; authored reports/probes are retained separately.

**New coverage:** 1,825 matching component/data comparisons; 57 passing Build/Develop/Network assertions; 1,152 synthetic AI decision/action cases; 15 reduced AI cases; three legal AI-trigger histories and their event reconstructions; five controlled Scout use-case/persistence cases; four native Scout animation/input-handler cases; executed external-reference counterexamples. These are checks of different kinds, with overlaps, not thousands of independent rules or additional completed full games. The earlier 90-game validation is existing baseline evidence and is not counted as newly rerun here.

**Still unresolved:** the prior introductory-scoring interpretation is not resolved by this reference. Full interactive Scout routing and a physical-input trigger for R02 remain unverified; AI legal-state reachability does not establish natural policy frequency. Passing data comparisons do not independently certify every physical-component value. General-purpose cloning/AI-training integration and online compatibility remain outside this review.
