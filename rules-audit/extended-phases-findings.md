# Extended phases, persistence, replay and scoring audit

Date: 2026-09-27. Original installed Assembly-CSharp.dll SHA-256: `600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB`.

These findings extend the first audit; they do not modify the game or its saves. The probes use the original `BaseRepository`, `BrassBirminghamFactory`, `BrassGameHistory`, `ReplayUseCase`, and rules engine with `InMemoryGameStorage` and `InMemorySnapshotStorage`. A small presenter stub pauses animation callbacks deterministically, allowing the production replay logic to be tested without Unity. UI reachability is traced in source; live animation timing was not tested in Unity.

## New confirmed findings

### EP01 — P1: Skipping a Gloucester bonus replay can prevent ending the next turn

**Observed result:** A complete ordinary AI turn has already finished its Gloucester development bonus. While replaying that turn for the next player, skip after `MerchantBonusReceived(Develop)` has been applied but before `MerchantBonusDevelopFinished` has been applied. The original game resumes, but `DevelopMerchantBonus.Instance` still points to the historical replay object with `MerchantDevelopmentNeeded=true`. The next player may complete actions, but `GameFlow.EndTurn()` throws `InvalidOperationException: You can't end turn before finishing develop merchant bonus`.

**Runtime reproduction:** `extended-phases-probes.ps1`, two players, seed20260927. The original Hard AI generates the first five turns; Gloucester's bonus occurs at revision59. The replay is paused at the subsequent `IndustryFlipped` event, after applying the bonus event. Before skip, the original game's own bonus flag is false and the replay's static flag is true. After `ReplayUseCase.SkipReplay()`, history points to the original latest game, its own flag remains false, and the static flag remains true. Performing legal Pass actions then ending the turn raises the exception. No board, money, card, income, or scoring fields were directly altered to create this trace; only the viewer's last-seen revision is set to the previous turn boundary.

**Controls:** On the same saved game, replaying all events allows EndTurn. Skipping before the bonus event also allows EndTurn. Both controls leave the static flag false. This isolates the failure to the replay cutoff, not an unfinished actual bonus or malformed board fixture.

**Production path / exact code:**

- `BrassApplication.UseCases.Game.PlayHumanTurn/PlayHumanTurnUseCase.cs:24-35` starts unseen-event replay and exposes the skip-to-turn view.
- `Frameworks.View.Game.Replay/ReplayView.cs:79-82` shows the skip button; lines153-155 route its click to the controller.
- `InterfaceAdapters.Game.Replay/ReplayController.cs:24-26` calls the tested `SkipReplay()`.
- `BoardGameRules.Entities.Actions/DevelopMerchantBonus.cs:17-24` defines a static `Instance` and reassigns it whenever a game is constructed.
- `BoardGameRules.Entities.Board/Board.cs:413` and `:419` mutate the static instance from bonus events.
- `BrassApplication.UseCases.Game.Replay/ReplayUseCase.cs:126-132`, `:274-276` restore the latest game when skipping, without applying skipped events to the replay object or rebinding its singleton.
- `BrassApplication.GameHistory/BrassGameHistory.cs:82-85`, `:34-39` restore `CurrentGame` but do not repair that global reference.
- `BoardGameRules.Entities.TurnPhases/TurnFlow.cs:89-91` checks the static flag when ending a turn.

**Rule connection:** The official rulebook printed p10 grants Gloucester's optional development benefit to the player consuming that merchant's beer; pp6-7 require normal turn progression. A historical animation must not create a fresh mandatory bonus in another player's live turn. This is primarily a game-state isolation defect, not a disputed board-game rule.

**Repair plan:** Remove the process-wide singleton from rules state. Store the pending bonus on the owning game, player/action, or injected game-scoped service; board event handlers, SellActionUseCase, AI helpers, and TurnFlow must access that same owning instance. Replay must be unable to alter live-game flags. Avoid treating a simple singleton rebind on `MoveToLatestGame` as a complete fix: while a historical game applies its events, handlers currently target whichever game was most recently constructed. If a minimal compatibility patch is necessary, isolate/rebind every historical application and restoration as a temporary measure, with tests covering multiple loaded games.

**Regression tests:** Full replay versus skip at each event boundary around `MerchantBonusReceived`, `IndustryFlipped`, `IndustryDeveloped`, and `MerchantBonusDevelopFinished`; skip in a turn containing several sales; human/network opponent replays; cancellation/undo after a pending bonus; loading a second game during a replay. In each case compare the live game to an uninterrupted control and verify EndTurn. Verify the legitimate current-action development decision remains available. Also ensure failed EndTurn does not increment revision without an event: `AggregateRoot.ApplyEvent` currently increments revision before dispatch, so this exception can leave a temporary revision gap.

### EP02 — P2: Reloading a finished introductory game loses its scoring breakdown and crashes its results screen

**Observed result:** A legal introductory game completed through 20 two-player Pass turns has a two-player `IntroductoryPoints` dictionary and displays its summary successfully. After `BaseRepository.Save()` and `Load()`, total VP remains4 for each player, but `GameProgressInformation.IntroductoryPoints` is null. Calling the production `FinishedGameReviewUseCase.ShowGameSummaryTable()` with a real no-op output presenter throws `NullReferenceException`. The same presenter succeeds on the original game.

**Reachability:** EndTurn saves the completed game before review (`BrassApplication.UseCases.Game.PlayGame/PlayGameUseCase.cs:102`). A completed local game remains until `FinishedGameReviewUseCase.FinishGameReview()` deletes it. Closing/reopening before finishing review therefore reloads this state; the game-over branch at `PlayGameUseCase.cs:39-52` returns to review. Network finished-game review uses the same table path.

**Root cause / code:** `BoardGameRules.Entities.GamePhases/IntroductoryGameScoring.cs:36-49` fills the dictionary directly in the command method; its event handlers at lines93-103 are empty. `Frameworks.Persistence.Repository/BaseRepository.cs:75-92` reconstructs by applying persisted events, so that direct assignment is never repeated. `BrassApplication.UseCases.Game.FinishedGameReview/FinishedGameReviewUseCase.cs:54-59` dereferences the missing dictionary; `ReplayCompleted()` reaches the method at line77.

**Rule connection:** Printed p12 describes the introductory money, income, and industry scoring components. Saving and reloading must preserve those components and must not prevent viewing the result.

**Repair plan:** Make the introductory breakdown derived from its existing persisted component-scoring events, or add a versioned summary event whose handler sets it. For existing saves, replay the existing `IntroductoryGameAdditionalMoneyVictoryPointsChanged`, `...Income...`, and `...Industry...` events into the breakdown. Do not reconstruct money bonuses from final cash because subsequent current code can change cash after the bonus calculation. Add a defensive results-view fallback for legacy incomplete data rather than throwing. Keep total VP unchanged during reconstruction.

**Regression tests:** Complete introductory games at2/3/4 players, with positive and negative income, cash bonus capped/uncapped, and high-level industries. Save and reload before review, during review, and at the terminal revision. Compare breakdown, total VP, ranks, and visible table to uninterrupted play. Verify repeated loads do not add VP twice.

### EP03 — P2: Resuming in the Rail Era incorrectly includes Canal merchant VP in the Rail results breakdown

**Observed result:** Merchant VP totals are correct, but their per-era accounting is not reconstructed correctly. In a normal two-player Hard-AI Canal Era (seed20260927), Blue earns4 merchant VP. At the era transition, the original game clears its current-era bonus counter to0. A reload reconstructs it as4. After finishing the Rail Era with only legal Pass actions, both branches have Blue totalVP53, but the uninterrupted branch reports RailBonus0/CanalVP46, while the resumed branch reports RailBonus4/CanalVP42. Thus this does not duplicate actual VP or change the winner; it misattributes earned points in end-of-era and final results.

**Root cause / code:** `BoardGameRules.Entities.GamePhases/BaseEra.cs:60-65` copies and clears `VPFromBonusesInCurrentEra` only in `ScoreBonusVP()`, outside event application. `BoardGameRules.Entities.GamePhases/CanalEra.cs:63-72` handles `CanalEraEnded` without clearing/restoring the bonus counters. `RailEra.cs:48-50` similarly does nothing at era end. Loading applies the original merchant events, including their `Board.cs:404` counter increments, but never performs the corresponding reset. Later scoring picks up the stale count. `FinishedGameReviewUseCase.cs:66-68` derives the Canal result by subtracting Rail links, industries and bonus fields from total VP, so stale Rail bonus also lowers the displayed Canal result.

**Repair plan:** Move counter finalization/reset into replayed era-end events; use their recorded bonus dictionaries to restore historical summaries. Ensure reset happens once for original execution and once during reconstruction, without awarding/removing actual VP. Consider replacing the mutable era counter with an event-derived per-era ledger so a reconstruction cannot omit reset state.

**Regression tests:** Canal-only merchant VP, Rail-only VP, both-era VP, no merchant VP; save/reload immediately before and after the era transition and midway through Rail; then compare end-of-era and game-end breakdown with an uninterrupted control. Include repeated load, undo during Rail, and replay skip.

## Verified invariants and scope

- `extended-phases-roundtrip.ps1` ran original Hard-AI full games at2/3/4 players with seed20260927, plus3-player seed270906 and4-player seed270904, saving and reloading after every completed turn:40+54+64+54+64=276 comparisons.
- Compared revision, era/round/turn/current player, remaining actions, money, marker positions, total VP, spending/order, hand/discard/deck identities and order, mat tile identities, built tiles/flip/resource state, link owners, merchants/beer, market capacities, removed tiles, PRNG advancement, rankings, and final score records.
- Only the current-era merchant VP counter differed in those comparisons:21 two-player states and28 three-player states for seed20260927, and33 four-player states for seed270904, starting at the Canal/Rail transition. Four-player seed20260927 and three-player seed270906 had no differences. Player fields excluding the bonus counter were independently compared and matched in every case. A separate branch test demonstrated EP03's eventual visible result, rather than merely reporting a private-field mismatch.
- A partial Build cancelled after card discard and an original-AI first action followed by Undo each restored the compared game state without differences.
- The two targeted debt samples entered forced liquidation3 and9 times respectively. All12 completed debt phases reloaded with matching cash, total VP, industries/resources and other compared game state (apart from the independent era-bonus counter defect in the four-player run). Source review shows current debt settlement runs synchronously before the save boundary; absent debt dictionaries during historical replay therefore do not by themselves demonstrate a present save/resume bug. Debt state should become event-reconstructible when implementing the interactive liquidation repair already planned in B09.
- The copy constructor that aliases a snapshot payload is not reached in this audit: the installed repository saves event streams, and no normal producer of snapshots was found. It is not promoted as a confirmed ordinary-game defect.
- Introductory unflipped-industry extra scoring and final-Canal income ordering remain rule-interpretation follow-ups from the first audit. No new definitive rulebook evidence was found to justify upgrading them.

## Evidence files

- `extended-phases-probes.ps1` and `extended-phases-results.json`: EP01 runtime failure plus two controls; EP02 original-versus-reloaded summary check.
- `extended-phases-roundtrip.ps1` and `extended-phases-roundtrip-results.json`:276 save/load comparisons including12 debt phases, undo/cancel checks, and EP03 uninterrupted-versus-resumed final scoring branches.

All paths to source above are relative to `ai-research/decompiled/`. Official rulebook: [Roxley English PDF](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281), as linked from [the publisher's Brass: Birmingham page](https://roxley.com/products/brass-birmingham).
