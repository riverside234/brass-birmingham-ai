# Turn, economy, cards, and endgame review

Audit date: 2026-09-27. Installed Assembly-CSharp.dll SHA-256: `600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB`.

Evidence: `phases-probe.ps1`, `phases-probe-results.json`, and decompiled code in `../ai-research/decompiled`. Probes load the original assembly in a separate process and use controlled state fixtures. They do not patch the game or access saves. Complete pass/scout games are also run through normal action/turn/round APIs. The human UI was traced in source, not operated live.

Rule source: the publisher's [official English rulebook](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281), linked as 2018.11.20 from [Roxley's product page](https://roxley.com/products/brass-birmingham). Page numbers below are printed pages; the PDF contains seven sheets, mostly two-page spreads.

## Confirmed findings

### P1: Final-round debt resolution charges income that is not owed

**Rule:** Printed p6, End of Round / Take Income: the final round of the game has no income collection. The negative-income shortfall procedure is part of this income step.

**Code:** `BoardGameRules.Entities.GamePhases/GameFlow.cs:115-121` starts debt liquidation whenever money plus income is negative, without checking final game round. `EndRound` resolves this debt at lines 126-132 before deciding at lines 149-152 to end the Rail Era without an income payment. `BoardGameRules.Entities.SellIndustry/SellIndustryFlow.cs:92-106` removes VP for unresolved debt; its `IndustrySold` handler at lines 130-133 removes a selected industry.

**Reproduction:** In the last Rail round, a player has cash £0, income -1, VP10, and no industries. Finish the final player's turn using `GameFlow.EndTurn()`, then end the round. Probe result: a debt phase starts and the player finishes with VP9. Expected: VP10. With industries on the board, the same path requires their liquidation before scoring and can remove tile VP and adjacent-link VP as well.

**Fix plan:** Introduce one predicate defining whether income settlement is due this round. Use it for both debt preparation and bank payment, before any debt state or events are created. It must exclude the final Rail round in a full game. Preserve final Canal income in a full two-era game. Review the introductory variant separately: p12 instructs playing the entire Canal Era as normal, so resolve how that interacts with the final-game exception before changing its behavior. Include the introductory path's unconditional `IncomeMoneyPaid` at `GameFlow.cs:138` in that review: currently it emits income after the scoring bonus has already been computed. Resume era scoring only after income settlement when required.

**Tests:** Full-game last Rail round at cash0/income-1 with no assets; with a flipped industry and a link beside it; cash0/income+1; penultimate Rail round; final Canal round of a full game; final Canal round of introductory mode. Assert money, industries, VP, debt events, and final scoring. Test 2/3/4 players.

### P2: A legal loan at income -7 is unavailable to the human player

**Rule:** Printed p11, Loan: decrease three income levels and receive £30; only a resulting income below -10 is prohibited.

**Code:** `BoardGameRules.Entities.Actions/LoanAction.cs:36` rejects `IncomeLevel <= -7` in availability checking, while `Loan()` at line25 correctly rejects only `<= -8`. `BrassApplication.UseCases.Game.PlayerActions/PlayerActionsUseCase.cs:93-103` uses the availability checker before allowing a human to start the loan action.

**Reproduction:** Set a legal player state to income -7. Probe availability returns `INCOME_TOO_LOW`; the execution method accepts the same state and grants £30, lowering income to -10. Boundary controls: -8 is correctly rejected; -6 correctly becomes -9.

**Fix plan:** Share a single `CanTakeLoan` predicate between availability and execution, derived from current income minus three being at least -10. Keep the card-discard requirement in the existing action flow. Search AI callers for duplicated thresholds and reuse the same predicate there.

**Tests:** Income -10 through -6; positive income bands with more than one marker position per income level; human availability agrees with execution; rejected loans leave cash, cards, actions, and income unchanged. At -7 a loan yields exactly £30 and -10 income.

### P1: The first endgame tiebreak uses the marker's space instead of income

**Rule:** Printed p7, Winning the Game: tied VP is broken by income, then remaining money. Printed p6 defines income as the coin value beside the marker, not its exact space.

**Code:** `BoardGameRules.Entities.GamePhases/EndOfGame.cs:38` sorts a weighted numeric expression containing `p.IncomeMarkerPosition`.

**Reproduction:** Blue has VP100, marker12 (income £1), cash0; Yellow has VP100, marker11 (also income £1), cash10. The installed DLL ranks Blue first. Expected: Yellow, because their incomes are equal and Yellow has more cash.

**Fix plan:** Replace weighted arithmetic with an explicit descending comparison of `(VP, IncomeLevel, Money)`. Keep result presentation order separate from score/rank equality. Review save metadata, finished-game review, analytics and any online result readers so they agree on the comparator.

**Tests:** Equal VP/same income/different marker/different money; equal VP/different income/different money; different VP; positive and negative income; all 2/3/4-player order permutations. Test very large valid cash values to avoid implicit weight limits.

### P2: A complete tie is assigned a winner and loser

**Rule:** Printed p7, Winning the Game: if the stated tiebreaks still leave players tied, they draw.

**Code:** `BoardGameRules.Entities.GamePhases/EndOfGame.cs:38-43` creates an ordered list and assigns `source.IndexOf(player)` as a distinct place. Its attempted random fraction uses integer division and is zero, but changing it to floating point would still violate the draw rule. `BrassApplication.UseCases.Game.FinishedGameReview/FinishedGameReviewUseCase.cs:48-68` passes those places to the results presentation. `Frameworks.Analytics/UnityAnalytics.cs:34` selects only `EndGameRanking.First()` as winner.

**Reproduction:** Two players each with VP100, marker12, cash10 receive places0 and1 from the original DLL. Expected: both tied for first.

**Fix plan:** Group equal `(VP, IncomeLevel, Money)` results and assign equal ranks; expose a winners collection or draw flag without letting a stable display order imply a sole winner. Update the results view and result/analytics consumers. Keep existing saves readable; only recalculate a historic result if explicitly supported by the compatibility policy.

**Tests:** Two-way tie for first; tied lower places; three-/four-way ties; equal income with different marker spaces; save/load and final-screen display. Check that no random draw is consumed solely to determine result order.

### P2: Forced liquidation silently chooses industries for human players

**Rule:** Printed p6, negative income: a player may remove any of their own industries to meet a shortfall, stopping when enough money is obtained. Choosing which industry to lose is a strategic decision.

**Code / normal call path:** `BrassApplication.UseCases.Game.PlayGame/PlayGameUseCase.cs:96-100` calls the presenter to sell industries and immediately ends the round. `InterfaceAdapters.Game.PlayGame/PlayGamePresenter.cs:74-76` starts `SellIndustryFlowUseCase`. `BrassApplication.UseCases.Game.SellIndustryFlow/SellIndustryFlowUseCase.cs:30-39` unconditionally invokes an AI rule. `BrassApplication.AI.RuleBasedSystem.Rules/SellIndustryBecauseOfDebt.cs:37-48` loops through every indebted player and sells the heuristic's chosen tiles; there is no Human-vs-AI branch. This is a confirmed source-flow finding; no live UI reproduction was performed.

**Impact:** The game removes a human's preferred brewery, resource access, link-scoring support, or future sellable industry without allowing the legal alternative. The heuristic considers a rough tile/link VP value, not the player's whole strategic position, and can differ from the user's intended choice even when the immediate VP estimate is equal.

**Fix plan:** Make debt settlement a resumable phase. For humans, show eligible own industries, half-cost proceeds, debt remaining, and a confirmation before each sale; for AI, keep a selector. Stop automatically once the shortfall is covered and then apply any unavoidable VP loss. Resume `EndRound` only after all indebted players finish. Add authoritative ownership/phase/debt checks in `SellIndustryFlow.SellIndustry`; currently it checks slot existence and debt but not ownership, though the current AI caller filters by owner. Support saved/reloaded and network-controlled debt choices without allowing another player to choose them.

**Tests:** Human with two alternative industries can choose either; AI still completes; cannot select another player's tile; cannot continue selling after covering debt; surplus proceeds remain; no-assets shortfall loses at most available VP; multiple indebted humans; save/reload during a pending choice. End-turn continuation must run exactly once.

## Further verification, not counted as confirmed bugs

- `IntroductoryGameScoring.AddVPForIndustries` at lines81-98 awards additional VP to every level2+ industry without testing `IsFlipped`. Printed p12 says these industries score a second time and illustrates a flipped tile; printed p7 says unflipped industries do not score. This is a plausible defect, but the introductory wording uses "all" and should be resolved against an authoritative clarification before changing behavior. A test should compare flipped/unflipped level2+ tiles once that expected rule is fixed.
- CardDealer's helper returns no cards if fewer than the requested number remain. This looks suspicious in isolation but was not reproduced through legal play; deck/action arithmetic preserves correct exhaustion, including Scout exchanges. Do not count direct malformed helper calls as a user-facing bug.
- `SellIndustryFlow.SellIndustry` lacks explicit ownership validation. The actual current automatic caller filters owned tiles. Treat this as validation work required for the new interactive flow, not a demonstrated current ordinary-game exploit.

## Coverage and ruled-out suspects

- Six complete games through the original DLL: 2/3/4 players, with only Pass or with Scout whenever legal and Pass otherwise. Correct totals were 40/54/64 turns and 78/105/124 actions. Both eras ended with empty hands and empty draw deck. Scout runs performed24/33/40 scouts respectively and returned all wild cards to supply. No turn starvation, extra rounds, wild-card leakage, or deck shortage occurred.
- Source confirms first Canal round has one action and all later rounds have two. Scout's two-card exchange is in addition to the use-case's action-card discard, so it is not an incorrect two-card Scout cost.
- Source confirms loans move to the highest marker position of the new income level, income level mapping clamps at30, loan cash is30, and negative-income liquidation returns floor(tile cost/2), retaining surplus. Debt VP loss clamps at available VP.
- Source confirms relative order of tied round expenditures remains stable, end-of-Canal links are removed, only level1 built industries are removed, merchant beer is replenished, and level2+ flipped industries remain eligible for both-era scoring.
- These checks do not prove the entire rules engine correct. Networking, crash recovery, event replay, and graphical UI behavior were outside this sub-audit's runtime coverage.
