# Brass: Birmingham — rules audit and bug-fix plan

> **Implementation update — 27 September 2026:** Installed combined patch **1.2.0** retains B01–B16, the P1 save-integrity repair and confirmed follow-up repairs, and adds R01–R03 from the [reference cross-check](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/REFERENCE_CROSSCHECK_BUG_FIX_PLAN.md>). Targeted regressions, 90 complete original-AI games, prior-patch save compatibility, installation recovery fixtures and all ten in-player Unity suites passed (including the city-card locator UI improvement). [LOCAL_PATCH.md](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/LOCAL_PATCH.md>) records current fixes, fingerprints and validation. Source, tests, reversible installation and restoration scripts are provided; the original assembly and previous patch are backed up. Online play is disabled at the user's request. The introductory scoring interpretation remains unresolved and unchanged. The original stock-code findings and checklist below remain historical reproduction evidence.

Reviewed and expanded 27 September 2026. Scope: the original Steam version, including ordinary two-, three-, and four-player games, introductory results, original AI decisions, replay, cancellation, undo, and event-based save reconstruction. The findings, unchecked release checklist, and coverage tables below preserve the **pre-patch audit**. Use the implementation update above and linked patch guide for current status and validation.

## Findings at a glance

**The original code contains reproducible rules and game-state defects.** The expanded review retained B01–B11 and added **five findings, for 16 total**. The additional high-priority cases were **B12, a replay skip that blocks a later turn; B14, a potential Hard-AI crash in a crafted but rules-permitted stock state; and B15, rare seed-generated identity collisions with a demonstrated wrong-tile flip in an arranged board state**. B13 and B16 affect reloaded results. Evidence strength and reachability are stated separately below. The audit itself did not modify game binaries or user saves; the subsequent implementation is documented separately.

| ID | Priority | Finding | Evidence |
| --- | --- | --- | --- |
| B01 | P1 | Final Rail round incorrectly starts debt settlement | DLL reproduction |
| B02 | P1 | Legal two-coal builds are rejected when nearer coal would run out | DLL reproduction, all player counts |
| B03 | P1 | Rail construction can consume coal from a farther mine | DLL reproduction; human/AI callers traced |
| B04 | P1 | Original AI combines two merchants' beer for one industry sale | Original AI helper reproduction |
| B05 | P1 | Endgame tiebreak uses income-marker space instead of income level | DLL reproduction |
| B06 | P2 | Complete ties receive different ranks | DLL reproduction |
| B07 | P2 | Human loan availability rejects a legal loan at income −7 | DLL reproduction; human caller traced |
| B08 | P2 | Legal overbuilds in dual-icon spaces are blocked | DLL reproduction, all player counts |
| B09 | P2 | Forced liquidation chooses industries for human players | Source-confirmed rules deviation |
| B10 | P2 | Gloucester bonus may trap a human with no development target | Partial DLL reproduction + UI source; live UI verification pending |
| B11 | P2 | Merchant VP animation displays 3/4 as the total score | Source-confirmed presentation defect |
| B12 | P1 | Skipping a Gloucester bonus replay can block the next End Turn | Naturally played original-AI trace + production replay reproduction and controls |
| B13 | P2 | Reloading a completed introductory game crashes its results table | Legal completed game + production repository/summary reproduction |
| B14 | P1 potential | Hard AI dereferences an exhausted manufacturer stack in the first Rail round | Public AI decision reproduction in crafted 3/4-player states; natural policy reachability unproven |
| B15 | P1, rare/conditional | Duplicate industry IDs can flip the wrong industry | Unmodified seeded setup collisions + arranged original-resource-consumption reproduction |
| B16 | P2 | Reload in Rail attributes Canal merchant VP to the Rail breakdown | Naturally played Canal game + save/load and legal continuation; total VP unchanged |

P1 means a game-progress blocker, materially illegal action, incorrect outcome, or similarly serious state-corruption risk. P2 means an important boundary, player-choice, or presentation repair. **Severity does not imply frequency or proven natural AI reachability.** B14's public decision failure is real, but its trigger was arranged; B15's collisions arise naturally at setup, while its gameplay consequence was isolated in a fixture. None is an observed loss of a user's saved game. The new passing matrices do not resolve any listed defect.

## Basis, sources, and limits

The production code is the managed Unity assembly [Assembly-CSharp.dll](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll>), not an upstream source repository. Readable C# already existed in [ai-research/decompiled](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled>). The assembly fingerprint matches the earlier review:

```text
SHA-256 600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB
```

Rule authority: Roxley's [product page](https://roxley.com/products/brass-birmingham), which links its English **2018.11.20 rulebook**, and the [publisher-hosted PDF](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281). The downloaded document identifies itself as v2018.11. A [local reference copy](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/official-rulebook.pdf>), extracted text, and rendered pages are retained. PDF SHA-256: `BB627A11C769957FBB5F26A210E3957AE27A2BAD054BA9CB3B4FADC6C2BA73E5`.

**Page references below are printed page numbers.** The PDF has seven sheets: sheet 3 contains printed pages 4–5; sheet 4 contains 6–7; sheet 5 contains 8–9; sheet 6 contains 10–11; sheet 7 is page 12. The relevant spreads were visually checked.

Probes load the unmodified DLL into separate PowerShell processes. The expanded review combines arranged boundary fixtures, naturally played original-AI games, and production application/repository methods. Rules and AI matrices use PowerShell 7.6.5 / .NET 10.0.11; application-flow/persistence harnesses use Windows PowerShell 5.1 / .NET Framework with stub presenters. Each finding identifies which kind of evidence supports it. The repository tests use original event serialization/reconstruction with isolated in-memory storage; they do not read or alter the user's saves or prove filesystem crash recovery. Production replay logic was exercised with controlled callbacks, not live Unity animation. No running Unity UI, online server, controller interaction, or mobile build was tested. Runtime parity with Unity Mono remains a release requirement. This review does not establish the developer's maintenance status.

## Detailed findings and repairs

### B01 — Final-round debt settlement changes scores before final scoring

**Rule reference:** p. 6, End of Round / Take Income: the final game round omits income settlement.

**Code:** [GameFlow.cs:115](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.GamePhases/GameFlow.cs:115>), `BeginEndRound`, sets the debt flag whenever cash plus income is negative. `EndRound` resolves that debt at lines 126–132 before its final-Rail branch at lines 149–152. [SellIndustryFlow.cs:92](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.SellIndustry/SellIndustryFlow.cs:92>) then deducts VP for an uncovered shortfall.

**Reproduction and impact:** Last Rail round, £0 cash, −1 income, 10 VP, no industries. Finish the last turn and round. The installed code starts a debt phase and ends at **9 VP**, although the expected result is **10 VP**. With industries, this path can instead force their removal before scoring, affecting industry and link points. The latter consequence follows from the source; the retained fixture demonstrates the VP deduction directly.

**Repair:** Introduce one `ShouldSettleIncomeThisRound` decision and use it before debt preparation and income payment. In the normal game, skip both on the final Rail round; retain final Canal income. Avoid creating debt events at all when settlement is not due. Review the introductory variant separately before changing its final-Canal behavior, because its “play the entire Canal Era as normal” wording needs an explicit interpretation.

**Acceptance tests:** Repeat the fixture with no assets and with a flipped industry plus adjacent link; test positive and negative income, penultimate Rail round, final Canal round, and 2/3/4 players. Assert events, cash, industries, scores, and exactly one round completion.

### B02 — Build affordability counts only the currently nearest coal

**Rule reference:** p. 8, Consuming Coal: depletion permits moving on to the next closest mine.

**Code:** [BuildAction.cs:621](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/BuildAction.cs:621>), `CostOfCoalForGivenIndustry`, totals mines that are legal **right now**. It mistakes the rest of a multi-cube requirement for a market purchase. The boolean/reason predicates at lines 421–429 and 464–480 use this estimate; [BuildActionUseCase.cs:47](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.Actions.BuildAction/BuildActionUseCase.cs:47>) exposes them to humans.

**Reproduction:** Next pottery is III; player has £22 and a Coventry card. Coventry and rail-connected Nuneaton each have an opponent's mine with one coal. No merchant is connected. The build predicate reports `NO_COAL`. Direct Build followed by Coventry coal and then Nuneaton coal succeeds, leaves £0, and flips both mines. Reproduced for all player counts.

**Repair:** Separate current source selection from whole-action affordability. For an industry build, count all connected available board coal, then price only the shortfall and require merchant access only for that shortfall. Keep the actual delivery sequence constrained by nearest remaining coal. Share this estimate among UI, reason text, and AI candidates. Source also indicates possible overpricing when a merchant is connected; retain that as a regression case.

**Acceptance tests:** Distances 0/1 and 1/2; tied nearest mines; all coal in one mine; disconnected farther coal; one cube plus market supply; empty market; exactly tile-cost cash. Cover Manufacturer III and Pottery III/V, with one-coal controls.

### B03 — Rail coal selection compares endpoints separately

**Rule reference:** pp. 8 and 11, Consuming Coal / Network: use the nearest coal after placing the rail.

**Code:** [ConsumingCoal.cs:60](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Consumables/ConsumingCoal.cs:60>) accepts a source valid for `Any` endpoint. [NetworkAction.cs:81](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/NetworkAction.cs:81>) supplies those endpoints; [NetworkActionUseCase.cs:167](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.Actions.NetworkAction/NetworkActionUseCase.cs:167>) and [NetworkActionAIHelper.cs:27](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.AI.AIHelpers/NetworkActionAIHelper.cs:27>) use the same union of candidates.

**Reproduction:** A new Birmingham–Dudley rail, an existing Birmingham–Redditch rail, and stocked mines in Dudley and Redditch. Dudley's distance to the new rail is 0; Redditch's is 1. Both nevertheless become selectable because they tie relative to Birmingham alone. Supplying Redditch coal succeeds while Dudley coal remains untouched.

**Repair:** Evaluate a rail as one target: compute each mine's minimum distance to all of its non-null endpoints, then select the global minimum across mines. Reuse that query in validation, UI, AI, and feasibility. Include the third farm endpoint on the southwest link. Preserve singleton targets for industries and reevaluate after depletion.

**Acceptance tests:** This fixture permits Dudley only; equal distances at opposite ends allow a choice; depleted sources disappear; connected board coal prevents market use. Cover three-endpoint links and each rail of a double action separately.

### B04 — The AI sells one industry using two merchants' barrels

**Rule reference:** pp. 8 and 10, Consuming Beer / Sell: merchant beer belongs to the merchant selected for that sale.

**Code:** [SellAction.cs:142](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/SellAction.cs:142>) sums all matching merchants. [SellActionAIHelper.cs:19](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.AI.AIHelpers/SellActionAIHelper.cs:19>) searches for a merchant afresh for each barrel; lines 115–119 repeat this during a sale. Similar feasibility logic exists at `SellActionAIHelper:98` and [BoardHelper.cs:498](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.AI.AIHelpers/BoardHelper.cs:498>). The domain supply method does not remember the merchant already used.

**Reproduction:** Manufacturer V in Birmingham, a Birmingham–Oxford link, Oxford's goods and wildcard merchants each holding beer, and no brewery. The human initial preflight correctly returns `NO_BEER`, but the eligible-slot query returns true. The original AI helper consumes **both merchant beers**, flips the industry, and awards **six income spaces**: two for the industry and four from two Oxford bonuses.

**Repair:** Add a per-industry sale context with an explicit chosen merchant. Before selection, eligibility must find at least one single merchant whose permitted beer plus brewery supply meets demand. After selection, enforce that merchant's identity in the engine. Update AI and UI to use the same query and reset the context between industries in a multi-sale action. Fixing only the human menu will not fix the shipped AI.

**Acceptance tests:** Two merchants/no brewery rejects a two-beer sale; one merchant plus a valid brewery accepts; two breweries accepts; zero-/one-beer industries still work. Multiple industries may use different merchants, with each bonus applied once. Test continuation after a different legal sale as well as the initial action menu.

### B05 — The first tiebreak compares the wrong income quantity

**Rule reference:** pp. 6–7, income definition / Winning the Game: compare income, then remaining cash.

**Code:** [EndOfGame.cs:38](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.GamePhases/EndOfGame.cs:38>) uses `IncomeMarkerPosition` in a weighted score.

**Reproduction:** Both players have 100 VP and income £1. Blue has marker 12 and £0; Yellow has marker 11 and £10. The installed code ranks **Blue** first; **Yellow** should win the cash tiebreak.

**Repair:** Replace weighted arithmetic with a descending lexicographic comparison of `(VP, IncomeLevel, Money)`. Keep stable presentation order separate from competitive rank. This also removes implicit numeric-weight limits. Pair this change with B06 and audit consumers of the ranking metadata.

**Acceptance tests:** Same income/different spaces and cash; different income; different VP; positive and negative income; large cash balances; every seat permutation for 2/3/4 players. Assert the winner independently of original player order.

### B06 — Complete ties are not represented as draws

**Rule reference:** p. 7, Winning the Game: a remaining tie is a draw.

**Code:** [EndOfGame.cs:39](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.GamePhases/EndOfGame.cs:39>) stores a list and gives every player its distinct `IndexOf` rank. [FinishedGameReviewUseCase.cs:48](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.FinishedGameReview/FinishedGameReviewUseCase.cs:48>) forwards those ranks; [UnityAnalytics.cs:34](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/Frameworks.Analytics/UnityAnalytics.cs:34>) takes the first player as sole winner.

**Reproduction:** Two players at 100 VP, marker 12, and £10 receive ranks 0 and 1. The sort's apparent random fraction evaluates to zero through integer division; changing that fraction would still not implement a draw.

**Repair:** Group equal B05 comparison keys and assign shared ranks. Represent co-winners explicitly; adapt final-screen and result consumers. Preserve a stable display order without treating it as a victory. Do not consume randomness for tie resolution. Version any changed result schema.

**Acceptance tests:** Two-, three-, and four-way first-place ties, tied lower ranks, different spaces within one income level, save/load, and the results screen. Each tied winner must receive the same outcome.

### B07 — A legal loan at income −7 is blocked in the menu

**Rule reference:** p. 11, Loan: a loan is prohibited only if its three-level reduction goes below −10.

**Code:** [LoanAction.cs:36](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/LoanAction.cs:36>) rejects `IncomeLevel <= -7`, whereas execution at line 25 rejects only `<= -8`. [PlayerActionsUseCase.cs:93](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.PlayerActions/PlayerActionsUseCase.cs:93>) uses the former gate.

**Reproduction:** At −7, the menu predicate returns `INCOME_TOO_LOW`. The original execution method accepts the same state, grants £30, and sets income to −10. Controls at −8 and −6 behave as expected.

**Repair:** Use one eligibility predicate for UI and execution: `IncomeLevel - 3 >= -10`. Check AI callers for duplicated boundaries. Preserve the action's card-discard and confirmation lifecycle.

**Acceptance tests:** Income −10 through −6, multi-space positive income bands, exactly £30 gained, and correct target marker. Rejected actions must not change cash, cards, actions, income, or event revision.

### B08 — An occupied dual-icon space is incorrectly subject to vacant-slot priority

**Rule reference:** pp. 9–10, Build placement / Overbuilding: the vacant-space preference must not redirect a valid overbuild.

**Code:** [BuildAction.cs:263](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/BuildAction.cs:263>) and [BuildAction.cs:603](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/BuildAction.cs:603>) duplicate single-icon preference checks without first distinguishing occupied targets. Execution rejects the move at lines 77–80 as well as hiding it in availability queries.

**Reproduction:** Rail Era; own Coal II in Cannock's Coal/Manufacturer space, its single Coal space empty, next coal tile III, Cannock card and sufficient funds. The overbuild-specific predicate permits the upgrade, but full validation returns `FIRST_BUILD_IN_SINGLE_SLOT`; execution throws. This can arise after another player's Coal I is removed from the single space between eras. Reproduced for 2/3/4 players.

**Repair:** Centralize placement validation with separate empty-space and occupied-space branches. Apply single-icon priority only to the former. Retain all overbuild restrictions, costs, cards, era checks, and opponent-resource conditions. Update both duplicated helpers and execution, not only highlights.

**Acceptance tests:** Cannock Coal II→III and Coalbrookdale Iron II→III in dual spaces; wrong type/level rejection; opponent overbuild restrictions; removal of old stock; retained previously earned income; no future scoring of the replaced tile. A new industry must still prefer a vacant single-icon space.

### B09 — Humans cannot choose which industries to liquidate

**Rule reference:** p. 6, Take Income: the indebted player chooses their own industries for liquidation.

**Code:** [PlayGameUseCase.cs:96](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.PlayGame/PlayGameUseCase.cs:96>) starts liquidation and proceeds to end the round. [SellIndustryFlowUseCase.cs:30](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.SellIndustryFlow/SellIndustryFlowUseCase.cs:30>) unconditionally executes an AI rule. [SellIndustryBecauseOfDebt.cs:37](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.AI.RuleBasedSystem.Rules/SellIndustryBecauseOfDebt.cs:37>) loops over every indebted player, without distinguishing humans.

**Impact and confidence:** The source establishes an automatic choice even where a human has alternatives. This may be a deliberate digital simplification, but it differs from the cited tabletop choice; no documented variant was established. A live human UI reproduction was not performed.

**Repair:** Make settlement a resumable phase. Humans choose eligible owned industries with proceeds and remaining debt visible; AI retains a selector. Stop when the shortfall is covered, keep surplus, and resume the round once. Add ownership and phase validation to `SellIndustryFlow.SellIndustry` before exposing interactive input. Its current caller filters ownership, but the domain method does not enforce it itself.

**Acceptance tests:** Two legal liquidation alternatives; multiple indebted humans; mixed human/AI game; opponent tile rejected; surplus retained; no further sale after covering debt; no-assets shortfall; save/reload during selection. Test synchronous and delayed completion to prevent premature `EndRound`.

### B10 — No-target Gloucester development can leave an unresolved modal

**Rule reference:** p. 10, Sell / Merchant Beer Bonuses. The listed sale requirements do not require a remaining development target.

**Code:** [SellActionUseCase.cs:130](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.Actions.SellAction/SellActionUseCase.cs:130>) enters the bonus UI whenever the flag is pending. [SellActionPresenter.cs:130](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/InterfaceAdapters.Game.Actions.SellAction/SellActionPresenter.cs:130>) produces no selectable tile if none is eligible. [IndustriesDetailsView.cs:103](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/Frameworks.View.Game/IndustriesDetailsView.cs:103>) hides the back button; line 230 blocks Escape in merchant mode. [TurnFlow.cs:89](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.TurnPhases/TurnFlow.cs:89>) refuses to finish with the pending flag.

**Evidence:** In a no-target fixture, the original engine reports `NOTHING_MORE_TO_DEVELOP`, yet the bonus remains pending and `EndTurn` throws. The UI call chain suggests a lockup; it has not been observed in a running game. The AI already has a no-target `FinishDevelop` branch at [SellActionAIHelper.cs:160](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.AI.AIHelpers/SellActionAIHelper.cs:160>).

**Repair:** Check legal targets before showing the bonus modal. When none exist, emit the existing finish event and continue the sale. Keep normal selection mandatory when targets exist unless a separate authoritative clarification allows skipping. Share resolution between human and AI paths.

**Acceptance tests:** Empty mat; only prohibited pottery remains; one legal target; Gloucester plus brewery beer; another sale afterward; replay/undo around the bonus. A Unity UI test must demonstrate that the modal closes and the turn finishes before calling this fixed.

### B11 — Merchant VP replay animation replaces the total with 3 or 4

**Rule reference:** p. 10, Merchant Beer Bonuses. Bonus VP must be added to the existing total.

**Code:** [ReplayView.cs:346](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/Frameworks.View.Game.Replay/ReplayView.cs:346>) evaluates an expression equivalent to `(currentVP + regionEnum == Shrewsbury) ? 4 : 3`. It passes that result as the new total. [PlayerView.cs:63](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/PlayerView.cs:63>) and lines 120/127 write that text into the score label. Fresh decompilation from the inspected DLL independently confirmed the expression in [ReplayView.verified.cs:346](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/ReplayView.verified.cs:346>).

**Impact:** With 20 VP, Shrewsbury should display 24; this expression displays 3. The core [Board.cs:398](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Board/Board.cs:398>) correctly adds the bonus, so this is a presentation defect, not evidence that the final stored score loses those points. Duration until refresh and live visual behavior remain untested.

**Repair:** Compute the bonus separately and display `currentVP + bonus`. Prefer a shared bonus-value function used by the rules and presentation. Review capped income display in the same animation code, without assuming it has the same root cause.

**Acceptance tests:** Starting VP 0, 1, 20, and above 100 at Shrewsbury and Nottingham; normal replay and skip; reload/refresh. The displayed and authoritative totals must agree after every completed event.

### B12 — Replay skip leaks a historical Gloucester bonus into the live turn

**Priority/evidence:** P1, confirmed production replay failure on a naturally played game. This is the most clearly demonstrated new gameplay blocker.

**Rule reference:** p. 10, merchant development benefit; pp. 6–7, turn progression. A completed historical bonus cannot become a new mandatory choice for a later player.

**Code:** [DevelopMerchantBonus.cs:17](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/DevelopMerchantBonus.cs:17>) stores a process-wide `Instance`, replaced whenever a game is constructed. Board bonus handlers mutate that static object. [ReplayUseCase.cs:126](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.Replay/ReplayUseCase.cs:126>) skips pending events and eventually restores the latest game through `MoveToLatestGame` at line 276 without restoring the static reference. [TurnFlow.cs:89](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.TurnPhases/TurnFlow.cs:89>) checks that reference when ending a turn. The human replay skip button reaches this tested method through `ReplayController.SkipReplay`.

**Reproduction:** Two-player Hard AI, seed 20260927, first five turns played by the original AI. Gloucester bonus occurs at revision 59. Replay the unseen turn and skip after `MerchantBonusReceived` but before `MerchantBonusDevelopFinished`. Production history resumes the latest game, whose own pending flag is false; the global flag remains true on the historical object. After legal Pass actions, End Turn throws `You can't end turn before finishing develop merchant bonus`. Completing the same replay, or skipping before the bonus, succeeds. Presenter callbacks were paused deterministically; actual Unity animation timing remains untested.

**Repair:** Remove global ownership of the bonus state. Event handlers, TurnFlow, human Sell flow, and AI helpers must use their owning game's instance. Replaying another game must never mutate live-game flags. Rebinding only on return from replay is insufficient: historical event application itself currently targets whichever game was last constructed. If a temporary interception patch is needed, scope every historical application and restoration, and test multiple loaded games. Also prevent a rejected End Turn from advancing revision without recording a corresponding event.

**Acceptance tests:** Full replay versus skip at every event boundary around bonus receipt, sale flip, development and finish; several sales; undo/cancel; loading a second game; legitimate pending human bonus. Compare live state and revision with the uninterrupted control, then complete the next turn.

### B13 — Finished introductory results fail after reload

**Priority/evidence:** P2, confirmed supported-results-flow crash; accumulated VP remains correct.

**Rule reference:** p. 12, introductory scoring components. Results must remain available after reconstruction.

**Code:** [IntroductoryGameScoring.cs:36](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.GamePhases/IntroductoryGameScoring.cs:36>) creates `IntroductoryPoints` directly during command execution; the component-score event handlers at lines 93–103 do nothing. Repository loading applies events, so the dictionary is absent. [FinishedGameReviewUseCase.cs:54](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.FinishedGameReview/FinishedGameReviewUseCase.cs:54>) dereferences it. The game is saved at completion and can be reopened before review removes the local entry.

**Reproduction:** Finish a two-player introductory game with 20 legal Pass turns. The original summary succeeds. Save/load using the original repository and in-memory stores; both players retain 4 VP, but `IntroductoryPoints` is null. The original summary method now throws `NullReferenceException`, using the same working no-op presenter.

**Repair:** Reconstruct component totals from existing persisted introductory money/income/industry scoring events, or introduce a versioned summary event. Support existing saves without adding VP again. Do not derive historical money bonuses from later final cash. Add a safe results-view fallback for incomplete legacy breakdowns.

**Acceptance tests:** 2/3/4-player introductory games; positive/negative income; capped/uncapped money bonus; high-level industries; reload before/during review and at final revision; repeated load must preserve both component totals and total VP.

### B14 — Hard AI crashes when its manufacturer stack is exhausted

**Priority/evidence:** P1 potential. Null dereference reproduced through the public original AI decision entry point in crafted states; a natural match in which the shipped policy reaches this state has not been demonstrated.

**Rule reference:** p. 11, Develop can remove the next eligible tiles, including exhausting a stack. A permitted stock state must not stop the AI from selecting another legal action.

**Code:** [DevelopManufactureForMerchantBeer.cs:37](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.AI.RuleBasedSystem.Rules/DevelopManufactureForMerchantBeer.cs:37>) obtains the next manufacturer, then accesses `.Level` at line 53 without checking null. Its `CurrentRoundNumber < 2` condition includes the first Rail round. `HardAiBehavior.GetNextAIMove` calls the selector's applicability tests; [PlayAiTurnUseCase.cs:33](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.PlayAiTurn/PlayAiTurnUseCase.cs:33>) has no local recovery.

**Reproduction:** Three or four players, first Rail round, no manufacturers on the AI mat, a Cotton Mill/Manufacturer industry card, £0, income −9, and one free opponent iron. Higher-priority actions do not mask the rule. Both cases throw before returning an action. Moving to Rail round 2, restoring Manufacturer I, or using Easy AI avoids it. Eleven manufacturers can be removed in six Develop actions within the Canal action budget, so exhaustion is permitted; the particular AI policy's whole-game route to the full fixture is still unproven. The broad AI runs below did not encounter it.

**Repair:** Return false when the next manufacturer is null before reading its level. Continue normal selection to another legal action. An explicit Canal-only strategy restriction is a separate design choice, not a substitute for the null guard. Check analogous next-tile accesses while keeping unrelated strategy changes separate.

**Acceptance tests:** Zero/one/multiple remaining manufacturers, both eras and first/second rounds, all difficulties and player counts, matching/nonmatching cards, free/market iron and loan availability. Assert a legal returned action, completion, and ordinary confirmation through the public AI boundary.

### B15 — Random identifiers collide and can flip an unrelated tile

**Priority/evidence:** P1, rare and conditional. Duplicate IDs occur during unmodified setup; wrong-tile effects are confirmed in an arranged placement fixture. No full naturally played wrong-flip trace is claimed.

**Rule reference:** pp. 8 and 10, depletion and selling flip the corresponding industry; p. 7, only the correct flipped industries contribute their points.

**Code:** [UniqueIdGenerator.cs:5](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Utils/UniqueIdGenerator.cs:5>) uses a random integer suffix without a uniqueness guarantee. [Board.cs:176](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Board/Board.cs:176>) resolves the first industry with a matching ID. [GeneralAction.cs:75](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Actions/GeneralAction.cs:75>) applies `IndustryFlipped` through that lookup, ignoring the recorded region. Equality and other ID-based lookups have related ambiguity.

**Reproduction:** A deterministic prefilter scans the first 395 seeded random draws for seeds 0–999999, then constructs 34 candidate four-player games using the original setup. Four have actual duplicate industry IDs: seeds 445060, 677697, 754778 and 984130. This is **not** a million played games or an estimate of the failure rate in users' games. Seed 984130 gives Red Brewery III and Red Cotton Mill III the same `industry944625454` ID, independently confirmed on .NET 10 and .NET Framework 4.8. Place those original objects in an isolated Rail fixture, cotton in Birmingham and brewery at the Central farm. Original `ConsumingBeer.SupplyBeer` exhausts two brewery barrels: **the unsold cotton flips; the empty brewery does not**, while brewery income increases by five spaces. The emitted flip correctly names Central but resolves Birmingham's cotton. No IDs were changed to manufacture this collision. Twelve original-AI games at the four collision seeds completed without triggering the checked invariants; their sampled actions do not establish that both colliding tiles were used in the required arrangement.

**Repair:** Guarantee identifiers are unique and deterministic per game/category, using stable ordinals or equivalent construction. Version the scheme in saved setup. Preserve existing shared PRNG advancement, or isolate identifier generation without changing the versioned game seed behavior; simply retrying random draws changes later setup/shuffle results. Never assign new IDs to legacy objects: preserve the legacy version's ID algorithm and PRNG sequence during reconstruction. Legacy collisions make some historical events ambiguous: the region can resolve this particular flip, but not every resource event has enough context. Validate any migration, retain originals, and report ambiguity rather than silently guessing. Add duplicate-ID validation at setup and reconstruction boundaries with a recoverable error path.

**Acceptance tests:** All four known seeds under supported runtimes; unique IDs across every mat, board, removed tile, card, and slot; consume/sell/overbuild both colliding objects in legacy fixtures; original versus reconstructed events; deterministic setup and shuffles; new-game versioning and unchanged noncolliding legacy saves. A seed sweep supplements a construction-level uniqueness guarantee; it cannot replace one.

### B16 — Reloaded Rail results misattribute Canal merchant points

**Priority/evidence:** P2, reproduced persistence/presentation defect. Total VP and the winner are unchanged by this finding.

**Rule reference:** p. 7, scoring each era. The result breakdown must attribute the already awarded points to the correct era.

**Code:** [BaseEra.cs:60](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.GamePhases/BaseEra.cs:60>) copies and clears `VPFromBonusesInCurrentEra` only during command execution. [CanalEra.cs:63](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.GamePhases/CanalEra.cs:63>) reconstructs the era end without that reset; Rail's handler likewise omits it. Merchant events replay their counter increments. The final screen derives Canal VP by subtracting the Rail breakdown from total VP.

**Reproduction:** Two-player Hard AI, seed 20260927, completes Canal normally; Blue earned 4 merchant VP. Save/load at the Rail transition restores the counter as 4 instead of 0. Finish both uninterrupted and reloaded branches with legal Pass actions. Blue has **53 total VP in both**, but results report Rail bonus 0 / Canal 46 uninterrupted and Rail bonus 4 / Canal 42 after reload. Broader turn-boundary comparisons isolate the same counter drift.

**Repair:** Move counter finalization/reset into the persisted era-end event handlers, restoring historical bonus summaries from their recorded dictionaries. Do not award or subtract total VP while reconstructing these summaries. A per-era event-derived ledger can replace the fragile mutable counter if needed.

**Acceptance tests:** Canal-only/Rail-only/both-era/no merchant VP; reload before and after the era transition and during Rail; repeated load, undo and replay skip. Compare all result components as well as total VP against uninterrupted play.

## Implementation sequence

### Stage 1 — Establish a repeatable patch and test boundary

- [ ] Preserve the DLL fingerprint and a separate backup before implementation. Keep a clean stock installation available for comparisons.
- [ ] Create a small optional local patch project targeting the installed Unity/Mono environment. First prove that one method can be intercepted and disabled cleanly. The decompiled directory is inspection output, not a verified buildable Unity project; editing its `.cs` files alone will not change the game.
- [ ] Convert the retained reproductions into assertions for intended corrected behavior. The current probes record bugs rather than claim the game passes those expectations.
- [ ] Extract reusable fixture builders and state snapshots covering cards, cash, income, industries/resources, links, remaining actions, event revision, pending choices, and final ranks. Avoid sharing static mutable game state across concurrently simulated games.
- [ ] Record a patch/rules version. Decide how existing saves select their rules before testing save compatibility.

**Exit criterion:** Stock code reliably fails the targeted expectations; the patch can be loaded and removed without touching a save. Do not train a replacement AI against these known defects first.

### Stage 2 — Repair turn blockers and isolate game state

- [ ] Implement B12's game-scoped merchant-bonus state; update every static consumer together. Run replay cutoff and multi-game isolation tests before changing any event schema.
- [ ] Add B14's exhausted-stack guard and the public AI decision regressions. Preserve the crafted fixture even if natural self-play never reaches it.
- [ ] Decide B15's versioned identifier scheme and legacy-save policy before coding generation changes. Add the known-seed wrong-flip test, deterministic uniqueness invariant, and duplicate detection. Ship new-ID generation only when compatibility tests establish safe setup/replay behavior.
- [ ] Run the existing action cancellation/undo suites after the state-ownership changes; a fix to replay must preserve active human choices.

**Exit criterion:** Replay skip cannot strand End Turn; empty stock cannot crash rule selection; new-game IDs are guaranteed unique; old identifiers are never silently reassigned. B15 may require a separately versioned release rather than delaying a safe B12/B14 patch.

### Stage 3 — Repair resource legality together

- [ ] Implement B02/B03 with shared coal queries for current selection and complete affordability. A pure query must not consume resources or mutate the board.
- [ ] Implement B04 with one selected merchant per sold industry.
- [ ] Update engine validation, human availability/reason text, and every AI helper named above together.
- [ ] Add B08's occupied/empty placement distinction without weakening ordinary placement restrictions.

**Exit criterion:** UI candidate lists, AI candidates, and execution agree on each regression fixture. Invalid moves leave state and events unchanged; legal moves consume the correct resources and award income once.

### Stage 4 — Repair economy and persisted results

- [ ] Fix B01's final-round debt gate and B07's loan boundary.
- [ ] Implement B05 and B06 as one result-comparison/ranking change.
- [ ] Implement B13/B16 through event reconstruction, keeping awarded total VP separate from summary data. Verify repeated load cannot award points twice.
- [ ] Audit the final screen, game metadata, replay, and result consumers for shared-rank assumptions. Avoid contacting analytics or online services during local validation.
- [ ] Resolve introductory-mode questions below before including that variant in a rules-parity claim.

**Exit criterion:** Correct final-round state and winner/draw representation in both fresh and reloaded games, with no change to ordinary round settlement.

### Stage 5 — Repair human choices and presentation

- [ ] Implement B09 as an asynchronous/resumable flow; do not leave `EndRound` running while the human choice is pending.
- [ ] Reproduce B10 in a controlled Unity game, add its no-target branch, and verify undo/replay around it.
- [ ] Correct B11 and verify displayed values against the rules state.

**Exit criterion:** Every pending human choice can complete; only legal choices appear; replay and skip reach the same authoritative state.

### Stage 6 — Compatibility, broader regression, and packaging

- [ ] Run the targeted tests under the actual Unity Mono runtime, then complete 2/3/4-player human/AI games and both era transitions. Include deliberate debt and merchant-bonus cases, not only normal AI play.
- [ ] Extend the completed in-memory save/load and undo checks to real file-backed copies, process termination/restart, actual Unity callbacks, and every partial action boundary. Compare live state against replayed events at the same revision, including summaries and game-owned pending decisions.
- [ ] Keep original event identifiers and ordering where feasible. This engine reconstructs state from events; changing an event handler can change historical results. Do not silently rewrite finished saves to the new rules. Prefer new-game activation first, with explicit handling for older saves.
- [ ] Test multi-instance isolation before using the engine for AI search/training. B12 establishes an ordinary-play impact from the mutable singleton. Static AI helpers and incomplete copy constructors still require separate validation; they are not assumed safe cloning mechanisms.
- [ ] Package a version-gated, reversible patch with a clear affected-rule list and rollback instructions. Verify removal restores stock behavior on copies of test saves. Online compatibility remains unverified until separately tested; a local patch cannot establish server agreement.

**Exit criterion:** All targeted expectations pass; UI/AI and live/replayed state agree; supported saves reload; rollback works; the original installation backup is intact. Complete-game smoke tests alone are not a release gate.

## Coverage obtained and remaining work

| Area | Evidence obtained | Remaining validation |
| --- | --- | --- |
| Setup, 2/3/4 players | Original DLL: £17, eight cards, one initial discard, marker 10, 45 industries, 14 links per player; deck totals 40/54/64; merchants 5/7/9. Partial component/map comparison; B15 seeded-ID investigation | Independently transcribe every printed tile statistic, card multiplicity, and map edge; actual Unity seed compatibility |
| Markets and income | 832 market assertions across every coal/iron stock and quantities 1–16; 297 income/loan assertions across every marker | Post-fix loan UI boundary and rendered values |
| Build and Develop | 6,399 build, 501 development, 126 overbuild assertions; all 45 mat positions, both eras, 2/3/4 players, exact/short cash; B02/B08 retained | Unity interaction and independently certified physical-component values; synthetic matrix topology is not a played-game reachability proof |
| Network and beer | 670 candidate fixtures checked against independent graph/cost logic; 105 executed rail placements; 324 brewery/destination pairs | B03 source choice repair; Unity double-rail input/animation |
| Human Sell and action rollback | 30 original human Sell flows; 12 multi-sale sequences; 21 Network and 18 Sell cancel/undo cases using production history and memory persistence; 1,282 total assertions across this and preceding row | B04 AI repair, B10 real modal, animation interruption; these tests stub replay presentation |
| Rounds/cards | Earlier six complete Pass/Scout games: all hands/decks empty at both era ends; wild cards returned | Variants and abnormal interruption boundaries |
| Normal original-AI play | **90 complete games**, ten seeds per Easy/Medium/Hard × 2/3/4-player combination; **9,210 actions**, **320,920 repeated action/state assertions**, **519 independent era/player score comparisons**, 16 debt phases | Deliberately rare states remain necessary; no B14 natural-policy trigger found |
| Collision-seed AI controls | 12 additional complete four-player games, all three difficulties at the four B15 seeds; 1,488 actions, 58,823 assertions, 96 era/player comparisons | These matches did not reproduce the arranged wrong-flip scenario; passing is not evidence of unique IDs |
| Save/load, phases and debt | Five complete Hard-AI games; **276 completed-turn reconstruction comparisons**, including **12 completed liquidation phases**. Only B16's bonus counter differed among the compared fields | File-backed saves, interrupted writes, partial liquidations, online transactions; current debt flow is synchronous |
| Replay and result views | B12 production replay failure with two controls; B13 original/reloaded result method comparison; B16 legal divergent-continuation comparison; first-turn undo and partial-Build cancellation controls | Real Unity callback timing, screen rendering, skip clicks and controller hit targets |
| Original runtime and online | Stock DLL fingerprint checked; probes on .NET 10 / .NET Framework | Actual Unity Mono integration, network/server agreement, mobile versions are untested |

The first audit's six Pass/Scout games executed 614 actions, including 97 Scouts; its three Hard-AI matches brought that earlier action count to 921. The second pass adds the separately reported 90-game batch and 12 collision-seed controls: **102 completed games and 10,698 actions**, beyond the earlier tests. Do not add the save/load suite as independent seed coverage: some of its seeds overlap. The build/resource suite passes **8,161 assertions**, including six AI controls; the two deliberately reproduced B14 exceptions are recorded separately. The Network/Sell suite passes **1,282 assertions**. These counts describe executed checks, not thousands of independent rule scenarios or proof of exhaustive correctness.

AI progression checks require one discarded card and one net hand reduction per completed action, zero unpaid coal/iron/beer, nonnegative stocks/cash/link counts, legal income-marker bounds, and no resource-retaining flipped industry. End-of-era industry/link scoring is independently recomputed from the actual board. Scoring comparisons are conservatively omitted after any debt phase in those games; debt-specific save equivalence is tested separately. State checks alone can miss mutually consistent wrong rules, which is why the rulebook-derived boundary fixtures remain required. One early audit-run failure was the PowerShell harness omitting an optional boolean on `MakeAction`; correcting the invocation made those runs complete. It is not counted as a game defect.

Several suspicious cases were deliberately **not** reported as established bugs:

- `DevelopAction.CanDevelopGivenTypeOfIndustry` checks IronWorks regardless of its argument, but its only identified production caller requests IronWorks. The human UI uses a different correct type query.
- The partial-deck drawing helper looks risky, but legal Pass/Scout games did not expose a shortage. Do not change card arithmetic on an artificial helper call alone.
- Opponent beer being required to connect to the **second** rail agrees with p. 11. Preserve that restriction.
- The single-industry Canal limit correctly permits replacement of the player's existing tile. Empty-network placement, farm links, market access at unoccupied merchant locations, and board-first iron selection were traced without a demonstrated ordinary-play mismatch.
- Some public engine methods lack defensive validation. Treat those as hardening work unless a normal caller or an added UI path can reach an illegal state.

**Follow-up implementation review:**

1. **Introductory scoring interpretation remains unresolved.** The publisher still supplies the same rulebook; no publisher/designer clarification was found establishing whether the level-2+ bonus includes unflipped tiles or changing final-Canal income interpretation. Those disputed rules remain unchanged. The confirmed introductory reload crash (B13) is repaired. A separate reversed introductory result-sort comparator was also found and corrected.
2. **Oxford cap repaired.** Original Oxford events can temporarily set the marker above 99; completed sales ordinarily clamp it afterward. The patch clamps immediately and caps both merchant and industry-income replay targets. Actual JSON reconstruction and cap-boundary checks cover this. This is not claimed as a newly proved P1 permanent-income exploit.
3. **Pending debt is now reconstructible.** B09's interactive phase uses existing events to restore debt, sales, current decision owner, and completion state; tests cover partial saves, retries, multiple humans, and stale callbacks. Empty/shallow copy constructors were reviewed separately. No additional ordinary-play P1 or normal snapshot producer was established; they remain unsupported as general simulation-cloning APIs.
4. **New confirmed P1: local-save data loss on failed writes/retries.** Fault injection against the stock DLL truncated a 467-byte history to one byte; a metadata-failure retry duplicated revisions as `[1,2,2]`; loading empty metadata deleted history. The production cache also retained failed appends. The patch serializes before publishing, uses atomic replacement with retained backups and a checksummed recovery journal, verifies idempotent retries, locks metadata updates, and removes the unsafe offline cache layer. See [save-integrity implementation](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Persistence/README.md>) and [fault-injection tests](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/tests/BrassPatch/Persistence/PersistenceTests.cs>). All test saves are artificial; no user save was altered by validation.

## Reproduction and evidence index

From a PowerShell terminal in the game installation directory, run each script in its own process:

```powershell
pwsh -NoProfile -File .\rules-audit\baseline-probes.ps1
pwsh -NoProfile -File .\rules-audit\build_resources-probe.ps1
pwsh -NoProfile -File .\rules-audit\network-sell-probes.ps1
pwsh -NoProfile -File .\rules-audit\phases-probe.ps1
```

Each checks the assembly fingerprint and writes only its audit result file. The phase probe also invokes the endgame object by reflection to isolate ranking, and directly arranges the final-round fixture. The probes are review scaffolding, not a complete regression framework.

Second-pass commands (run as separate processes; `pwsh` is PowerShell 7, `powershell` is Windows PowerShell 5.1):

```powershell
pwsh -NoProfile -File .\rules-audit\extended-build-probes.ps1
powershell -NoProfile -File .\rules-audit\extended-network-matrix.ps1
powershell -NoProfile -File .\rules-audit\extended-phases-probes.ps1
powershell -NoProfile -File .\rules-audit\extended-phases-roundtrip.ps1
pwsh -NoProfile -File .\rules-audit\extended-ai-probes.ps1 -Players 2
pwsh -NoProfile -File .\rules-audit\extended-ai-probes.ps1 -Players 3
pwsh -NoProfile -File .\rules-audit\extended-ai-probes.ps1 -Players 4
pwsh -NoProfile -File .\rules-audit\extended-identity-probe.ps1
pwsh -NoProfile -File .\rules-audit\extended-identity-impact.ps1
```

The roundtrip script also reruns the phase reproductions it imports. AI defaults run seeds 270900–270909 for each difficulty; the three player-count invocations produce the 90-game batch. To reproduce a collision-seed control without replacing that evidence, use `-Players 4 -GamesPerDifficulty 1 -Seed 984130 -OutputTag identity-ai-984130`; repeat for 445060, 677697 and 754778. The identity prefilter defaults to one million seeds. Scripts replace only their corresponding audit output files. A process exit of zero does not mean all rules are correct: inspect the explicit failure lists and the deliberately recorded defect outcomes.

| Artifact | Purpose |
| --- | --- |
| [baseline-probes.ps1](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/baseline-probes.ps1>) / [results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/baseline-probe-results.json>) | Setup checks for all player counts |
| [build_resources-probe.ps1](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/build_resources-probe.ps1>) / [results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/build_resources-results.json>) | B02/B08, each across 2/3/4 players |
| [network-sell-probes.ps1](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/network-sell-probes.ps1>) / [results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/network-sell-probe-results.json>) | B03/B04 and partial B10 evidence |
| [phases-probe.ps1](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/phases-probe.ps1>) / [results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/phases-probe-results.json>) | B01/B05/B06/B07 and full card-cycle games |
| [2-player smoke](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/smoke-2p.json>), [3-player smoke](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/smoke-3p.json>), [4-player smoke](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/smoke-4p.json>) | Additional full Hard-AI runs using the pre-existing headless harness |
| [Build/resource notes](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/build_resources-findings.md>) | Extended source analysis and negative controls |
| [Network/sell notes](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/network-sell-findings.md>) | Extended source analysis and UI call chains |
| [Phase/economy notes](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/phases-review.md>) | Extended scoring, debt, and card analysis |
| [Verified replay source](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/ReplayView.verified.cs>) | Fresh decompilation of the B11 presentation expression |
| [Extended build notes](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-build-findings.md>), [runner](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-build-probes.ps1>), [results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-build-results.json>) | 8,161 passing checks, B14 failure traces and controls, partial component cross-check |
| [Extended network notes](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-network-coverage.md>), [runner](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-network-matrix.ps1>), [C# harness](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-network-matrix.cs>), [results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-network-matrix-results.json>) | 1,282 assertions; original human flows, graph oracle, cancellation/undo |
| [Extended phases notes](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-phases-findings.md>), [runner](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-phases-probes.ps1>), [results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-phases-results.json>) | B12/B13 reproductions, original replay and summary controls |
| [Roundtrip runner](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-phases-roundtrip.ps1>), [results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-phases-roundtrip-results.json>) | 276 comparisons, 12 liquidations, B16 final-results divergence |
| [AI runner](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-ai-probes.ps1>); [2-player](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-ai-2-results.json>), [3-player](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-ai-3-results.json>), [4-player](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-ai-4-results.json>) | 90-game batch: action/card/resource invariants and independent era scoring |
| [Identity scan](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-identity-probe.ps1>), [scan results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-identity-results.json>), [impact probe](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-identity-impact.ps1>), [impact results](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/extended-identity-impact-results.json>) | B15 original seed-generated collisions and wrong-tile flip |
| [445060 AI controls](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/identity-ai-445060-4-results.json>), [677697](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/identity-ai-677697-4-results.json>), [754778](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/identity-ai-754778-4-results.json>), [984130](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/identity-ai-984130-4-results.json>) | Twelve additional natural matches; no observed invariant failure, no claim of a complete wrong-flip trace |

**Original recommended release order:** First address B12's proven turn blocker and B14's small null guard, alongside the small B01/B07 economy corrections. Then repair B02–B04 resource legality and B05/B06 ranking. B15 needs an early versioning decision and a separate compatibility-gated identity change. B13/B16 form a focused persistence-summary repair; B08–B11 follow with their human-flow checks. The subsequent patch implements these findings; its [implementation and validation record](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/LOCAL_PATCH.md>) supersedes this historical checklist.
