# Second-pass build, development, market, and AI audit

This supplements the first audit without modifying the game or its existing reports. Existing B02 (multi-mine coal affordability) and B08 (dual-space overbuilding) are not counted again.

## New finding: Hard AI decision crashes with exhausted manufacturers in the first Rail round

**Suggested severity: P1 potential gameplay blocker. Evidence: reproduced through the original public AI decision entry point in crafted 3- and 4-player states. Natural original-AI match reachability has not been demonstrated.** The failure itself is confirmed; its frequency and whether the shipped policy naturally reaches the input state remain unknown.

**Location:** `ai-research/decompiled/BrassApplication.AI.RuleBasedSystem.Rules/DevelopManufactureForMerchantBeer.cs:37,53`. `GetFirstBuildingOfType(Manufacturer)` returns null after the stack is exhausted. The conjunction later reads `.Level` without checking null. Its round condition includes the first Rail round because round numbering resets between eras (`BoardGameRules.Entities.GamePhases/RailEra.cs:44` and `GameFlow.cs:101,195`).

**Ordinary caller:** `BrassApplication.AI.AiBehaviors/HardAiBehavior.cs:24-29` calls `PrioritySelector.Select`, which calls rule applicability at `BrassApplication.AI.RuleBasedSystem.RuleSelectors/PrioritySelector.cs:24`. `BrassApplication.UseCases.Game.PlayAiTurn/PlayAiTurnUseCase.cs:33` invokes this decision during an ordinary AI turn; that method has no local recovery from this exception. The reproduction calls `HardAiBehavior.GetNextAIMove`, not the internal faulty rule directly.

**Reproduction context:**

- Three or four players; first Rail round; Hard AI has no manufacturers remaining on its mat.
- Hand contains a Cotton Mill/Manufacturer industry card. The unchanged starting hand is retained except for substituting one ordinary card of this type, leaving eight cards.
- £0 cash, income −9, and an opponent's unflipped Iron II with one remaining cube. Other industry stacks retain their canal-only first tiles (except Pottery I, which is unaffordable).
- These conditions prevent higher-priority loans/builds from masking the faulty rule while making a free Develop action possible.

**Observed:** Both player-count cases throw `NullReferenceException` from `DevelopManufactureForMerchantBeer.IsApplicable`, through `PrioritySelector.Select` and `HardAiBehavior.GetNextAIMove`. No action is returned. Changing only the round to Rail round 2, restoring Manufacturer I to the mat, or selecting Easy AI avoids the exception. Retained traces and controls are in `extended-build-results.json`.

**Minimum legally possible stock context:** There are eleven manufacturer tiles; none has the prohibition on development. Six Develop actions can exhaust the stack (five pairs and one singleton) before the Rail Era. The normal 3-/4-player Canal Era supplies 17/15 actions per player, so the stack exhaustion is not prevented by the action budget. Income −9 is reachable through three loans; the initial £17 plus £90 is enough to pay for eleven market iron even at the actual initial ladder, with remaining turns permitting debt settlement. An opponent's Iron II can survive the transition. This establishes that the relevant stock/round context is allowed by the rules, **not** that the shipped Hard AI chooses that development sequence. The fixture does not prove a natural complete-match trigger.

**Rule references:** [Official Roxley rulebook](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281), printed p. 11 (Develop permits removal of the next one or two eligible tiles), pp. 6–7 (ordinary era/turn sequence), and p. 9 (a build needs a remaining tile). This is primarily a null-safety defect in a supported game state rather than a disputed rule interpretation.

**Fix:** Immediately return false when the next manufacturer is null, before examining `.Level`. Prefer checking the required tile/level before card and strategic conditions. If the rule is intended only for the start of the Canal Era, add an explicit era restriction as a separate strategy decision; a missing-tile guard is required regardless. Let the normal selector continue to another legal rule; do not paper over the exception with an empty AI action.

**Regression checks:** First Rail round with zero/one/multiple manufacturers remaining; 3-/4-player ordinary industry cards; no matching card; free iron versus affordable/unaffordable market iron; blocked versus available loan; second Rail round; Easy/Medium/Hard difficulty. At the public `GetNextAIMove` boundary require a returned legal action and normal action confirmation. Retain targeted fixtures even if broad original-AI self-play never reaches this strategy edge.

## Additional checks completed

`extended-build-probes.ps1` calls the fingerprinted original DLL in a fresh PowerShell process and writes `extended-build-results.json`. It does not touch saves or game assemblies. **8,161 method-level assertions pass**, separately from the two reproduced AI exceptions:

| Group | Assertions | Independent expectations / scope |
| --- | ---: | --- |
| Markets | 832 | Every coal stock 0–14 and iron stock 0–10; quantities 1–16; buy/sell totals against explicit printed paired price ladders, including external £8/£6 supply and full-market sale limits |
| Income | 297 | Every marker 0–99; income level and band threshold; legal three-level loan landings against independently enumerated printed bands |
| Build | 6,399 | All 45 tile positions, both eras, 2/3/4 players, market stocks 0/1/8; exact cash versus one short; original AI build helper execution; payment completion, placement, card discard, round spending, production sales, flips, income, and brewery era quantities |
| Develop | 501 | Market stocks 0/1/2/5/10 × free iron 0/1/2 × cash 0–12; first/second-develop affordability; original helper pays for two removals; every tile's development permission and exhausted-stack rejection |
| Overbuild | 126 | All six types; own/opponent targets; old stock 0/1; market stock 0/1; strict upgrade in single-icon slots; replacement archive, discarded old resources and absence of spurious old-tile income |
| AI controls | 6 | The three non-crashing controls above, each for 3 and 4 players |

The build/overbuild matrix intentionally uses synthetic connected boards to isolate placement and payment behavior. It is not thousands of played matches or proof that every whole-board fixture is reachable with physical link supply. Build tile costs and rewards come from installed tile data; the checks verify their application, **not** independently certify every number printed on a component. The price/marker ladders and development prohibitions have separate rule expectations. Known B02/B08 topologies are excluded and remain covered by the first audit.

## Additional source review and partial component checks

- Human Build resolves iron before coal; subsequent delivery callbacks re-query available sources after depletion. Original `BuildActionAIHelper` and `DevelopActionAIHelper` also re-query board-first resources before using markets. No additional ordinary-caller bypass of iron priority was found.
- The Develop finish button starts disabled until one tile has been selected (`Frameworks.View.Game.Actions/ActionInProgressView.cs:73-77`), so merely displaying the button does not establish a zero-tile development exploit.
- The details presenter supplies archived lower-level placeholders while a stack still exists; a superficially suspicious view index does not underflow for the supported model shape. An exhausted stack uses the explicit empty-list branch. No UI crash is claimed from that code.
- Overbuilding one's own resource industry removes the old object from the board before applying production/market effects. The retained resource count on its archived object does not make it a supply source or award another income increase. The broader overbuild matrix found no new resource/payment defect.
- The price ladders, starting £17, tile counts, initial market stocks, farm brewery slots and the single three-endpoint Kidderminster–Worcester/farm connection were cross-checked against the official setup/build illustrations and pp. 4–5/9. Empty merchant locations correctly retain coal-market access. This is a **partial component/map check**, not a complete transcription of all card multiplicities, edge types, or printed industry statistics.
- Source inspection of all basic-rule priority returns found no equal priorities across different rules; the selector's use of a unique-key dictionary is a future maintenance risk, but no present collision was established.

## Limits and implementation impact

No further confirmed rule defect was found in this assigned Build/Develop/market scope beyond existing B02/B08 and the new crafted-state AI exception. No Unity UI was launched, and asynchronous animation/cancellation behavior was not runtime tested. Add the minimal null guard to the AI reliability batch, preserving the existing retained probes as stock-code evidence. These passing checks supplement the targeted failures; they do not establish exhaustive correctness or negate defects found in other subsystems.
