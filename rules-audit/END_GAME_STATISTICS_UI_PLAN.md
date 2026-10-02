# End-game statistics panel: design and implementation plan

**Status:** Implemented, qualified and installed as patch 1.4.8. All ten pages are included; the installed player passes 523 statistics checks. See [release notes](../docs/END_GAME_STATISTICS_1.4.8.md).

**Current follow-up:** The readability revision from 1.4.10 is retained in installed patch 1.4.13. See the [1 October improvement plan](END_GAME_STATISTICS_IMPROVEMENT_PLAN.md) for larger entry buttons, plot-first pages, additional metrics and an evidence-backed Match review page. That follow-up is a plan only; it does not install another patch or change the implementation described below.

**Date:** 30 September 2026.

**Implementation baseline:** Installed local-game patch 1.4.7, including the current AI, city highlights, presentation recovery and save compatibility fixes.

**Scope:** A comprehensive, interactive review of one completed local match, available through a bottom-left **Statistics** button.

**Revision:** Incorporates all nine groups of analytics suggested by the user, including turn-order tempo, cards, geography, investment conversion and player interaction. Section 4A maps each suggestion to a page and identifies measurements that require qualification.

## 1. Recommended design

Add a large, closable statistics panel with **ten pages**, persistent player/era filters, interactive charts and vertically scrolling content. Keep the current results table and board review available. The new panel explains where points came from, how each player developed their industries and network, and how money, resources, actions and turn order changed through the match.

The default page is **Overview**, with the final ranking and a readable score breakdown. More detailed pages are **Scoring**, **Industries**, **Networks & Geography**, **Economy**, **Resources & Trade**, **Actions & Cards**, **Turn Order & Tempo**, **Efficiency & Transition**, and **Timeline**. A vertical page list accommodates these choices without another long cycle-through selector. Use a separately scrollable navigation list on short displays, while keeping the page body independently scrollable.

Statistics are descriptive. Show measured outcomes and their definitions; avoid assigning a strategy grade or claiming an action caused a win. The first implementation includes all ten pages. Local exports and additional visual polish can follow separately.

## 2. Research and design basis

Internet research checked the publisher's game page, official rulebook download, an existing score tracker and a strategy article. The rules cross-check uses the [official rulebook already downloaded in this workspace](official-rulebook.pdf), with [searchable text](official-rulebook.txt). Sources are listed in section 12.

The relevant rule foundations are: era scoring separates links and flipped industries; a qualifying industry can contribute in both eras if it remains on the board; income and remaining money resolve standard-game ties, with a draw if the tie persists. Cash is not ordinary full-game VP. Resource use can help another player's industry flip. Money spent affects the following round's order. Introductory games have separate additional scoring. These distinctions determine the panel's categories. See the publisher rules, pp. 6–12.

[MeepleShelf's tracker](https://meepleshelf.com/games/brass-birmingham) offers Canal/Rail industry and link categories, which supports an era-first score overview. The proposed panel adds merchant VP, penalties, introductory components and the underlying activity records so these do not disappear into a misleading subtotal.

[Dylan Ho's strategy article](https://dlaern.substack.com/p/how-to-dominate-brass-birmingham) discusses action efficiency, grouped sales, resource supply, network timing and loans. These suggest useful measurements, rather than fixed targets or claims about the strongest strategy. Its advice is personal and focuses on four-player games; calculate statistics from actual events for each match.

The ten-page structure and interactions below are design recommendations inferred from these sources and the current code, rather than features specified by the rulebook.

## 3. Entry points and panel behavior

### Bottom-left button

- Add a rounded **Statistics** button with a small chart icon and visible text, anchored inside the bottom-left safe area. Start with a roughly 180 × 48 reference-unit target and a 20-unit inset; validate actual placement against the existing finished-review controls.
- Show it only for a completed game after the final replay/scoring presentation has finished. It stays available throughout board review. It must not replace action controls or appear during AI thinking.
- If the existing review/results button occupies that corner, arrange both in a short vertical stack with clear spacing. Avoid the player HUD, chat/log controls and safe-area edges.
- Also add **Statistics** to the existing results dialog. That dialog can block clicks on a button behind it, so the user should be able to open statistics immediately after seeing the result.
- Opening from results hides that dialog temporarily; closing statistics restores it. Opening from board review returns to board review. Never stack two active modal input blockers.

### Panel layout

```text
 Bottom-left, after the match: [ chart icon  Statistics ]

 +-----------------------------------------------------------------------+
 | Match statistics      Full game · 4 players                       [×] |
 | Players [All players v]     Era [Whole game v]       [Reset filters]   |
 +--------------------+--------------------------------------------------+
 | Overview           | Final result and score breakdown                 |
 | Scoring            |                                                  |
 | Industries         |  Player      Total   Canal   Rail   Bonuses ...   |
 | Networks & Geo.    |  ...                                             |
 | Economy            |                                                  |
 | Resources & Trade  |  Interactive charts and sortable tables          |
 | Actions & Cards    |  ...                                             |
 | Turn Order & Tempo |                                                  |
 | Efficiency & Trans.|                                                  |
 | Timeline           |                                       scrollbar  |
 +--------------------+--------------------------------------------------+
 | [Return to results]                           [Close · Return to board]|
 +-----------------------------------------------------------------------+
```

The diagram shows arrangement, not final coordinates. The footer's return/close labels adapt to the entry point so both actions have distinct destinations.

- Use approximately 90% of the safe area, capped around 1,680 × 960 reference units on large displays. At smaller resolutions, reflow content instead of shrinking all text.
- Keep the title, filters, page navigation and footer fixed. Each page has one main vertical `ScrollRect`, with a visible scrollbar, mouse-wheel scrolling and drag support. Prevent wheel input from zooming the board underneath.
- At 1,280 × 720, narrow the navigation column and stack chart cards. Wide ledgers use a deliberate horizontal scrollbar or hide optional columns behind row details; avoid clipping essential columns.
- Remember the selected page, filters, expanded row and each page's scroll position while reviewing the same match. Reset this state when the game ID changes.
- The player filter supports any subset of the actual 2–4 players. Default to all; choosing two provides direct comparison. Era choices are Whole game, Canal and Rail. Hide Rail for introductory games.
- Final ranking, total final VP and final tie-break information remain explicitly labeled **Final result**, even when an era filter is active. Flow metrics use the selected era; end-state metrics identify their boundary, such as Canal scoring, after Canal cleanup or final Rail scoring. This prevents a Canal filter from appearing to change the winner or mixing cleanup and scoring snapshots.
- Page-specific filters include industry type, resource, action type and city. Clear labels indicate which filters apply. Reset filters restores all defaults without closing the panel.

### Interaction and accessibility

Click a score segment, bar or headline metric to open its corresponding ledger/page with the relevant filters. Hover/focus shows exact values and a short definition. Clicking a chart legend toggles that player's series; the player selector remains the authoritative filter.

Tables support numeric sorting, a local search field where useful, and expandable row details. Keep column headings visible. Use text, icons and line/marker differences alongside player colors. Every chart has an equivalent values table.

Escape or the close button dismisses the panel and returns focus to its opener. Tab navigation, Enter/Space activation and keyboard scrolling work. Enter/Space must never invoke the original results dialog's exit/review handlers through the new panel. Consume clicks inside the panel and the surrounding backdrop; a backdrop click need not close it, avoiding accidental dismissal while inspecting a chart.

## 4. Page contents

### Page 1 — Overview

Answer: **Who won, and how did each player's result differ?**

- A ranked comparison table: player/avatar, total VP, Canal contribution, Rail contribution, merchant bonus VP, VP lost, final cash and final income level. Explain overlapping columns: era totals already include the bonuses/penalties shown separately.
- Show shared ranks and **Draw** when appropriate. Explain a resolved tie using final income level, then cash. Consume the patched competitive rank; never infer a unique winner from row order.
- A stacked score chart by player, grouped by era, with industries, links and merchant VP. Show penalties separately below zero. Introductory additions receive their own labeled segments.
- Compact activity cards: industries built, industries flipped, links placed, loans and goods sold. These link to detailed pages.
- Match context: player count, full/introductory mode, human/AI labels and available configured AI style/difficulty. A style is the starting configuration; do not present it as a strategy the bot necessarily followed throughout the game.

Use neutral summaries such as **Largest scoring link: 8 VP** or **Most goods sold in one action: 3** only when fully supported by the match record. These are examples of labels, not sample results to ship.

### Page 2 — Scoring

Answer: **Which assets and bonuses actually produced these points?**

| Display | Required detail |
| --- | --- |
| Era/category matrix | Canal and Rail industry VP, link VP, merchant VP, penalties and total; full-match reconciliation |
| Industry-type breakdown | Cotton mills, manufacturers, potteries, coal mines, iron works and breweries; actual scored VP by era |
| Industry scoring ledger | Player, city, type, level, build/flip position, Canal VP, Rail VP and introductory additional VP |
| Link scoring ledger | Player, route, placement position, era, actual scored VP and endpoint contributions |
| Bonus/penalty ledger | Merchant location/type or debt settlement, occurrence position and actual VP delta |

Separate **distinct industries built** from **industry scoring occurrences**. A surviving flipped industry can score twice in a full match; show one asset row with two era contributions. A replaced or liquidated tile can retain points already earned but cannot earn future points after removal.

Show points when they were awarded. Flipping a tile earns income and enables later scoring; do not add its printed VP immediately to the official running score. A chart of awarded VP therefore has legitimate era-end jumps.

For introductory games, display the recorded additional money, income and industry components separately. Use the installed game's authoritative scoring results and test against the variant rules; the statistics feature must not silently introduce a different interpretation of introductory scoring.

### Page 3 — Industries

Answer: **What did each player build, develop, flip and leave unfinished?**

- A six-type comparison: tiles built, developed from the mat, flipped, scored, overbuilt/replaced, liquidated and removed at the Canal transition.
- A level distribution and per-tile ledger with city, type, level, build/flip timing, relevant income gain, resources initially placed, resource use and removal reason.
- Development detail by industry and level, specifically including level-1/2 tiles removed and higher-level tiles reached later. Show Develop-action counts, tiles removed by normal development and tiles removed through merchant bonuses separately.
- A conversion measure: distinct built tiles that flipped ÷ distinct built tiles, with numerator and denominator visible. Count a tile once, even if it scores twice. Zero denominators display an em dash.
- Time from build to flip in **turns elapsed**, with same-action flips counted as zero. Use a monotonic match turn index across the era boundary. Do not call this real elapsed time.
- Canal carryover: level-2+ tiles remaining after maintenance, whether flipped, and their later Rail contribution. Measure actual carryover, not every level-2+ tile ever built.
- Remaining unflipped industries at the final scoring boundary. Optional printed-VP totals must be labeled **Nominal unflipped VP**, with a tooltip that this is neither earned VP nor proof those points were achievable.

Keep normal Develop actions and merchant-bonus development distinguishable. Developing a tile is not building one. Resource industries can flip through depletion without a Sell action; include those conversions.

### Page 4 — Networks & Geography

Answer: **How useful and valuable were the canals and railways?**

- Links placed and Network actions used, separately for Canal and Rail. Show single-rail versus double-rail actions and links per Network action.
- Actual link VP, average VP per link placed in that era and highest-scoring routes. Include zero-point links rather than dropping them from averages. A scored-link average may also appear in detail; distinguish its denominator when history is partial.
- An expandable route ledger showing every linked location and its contribution at the scoring boundary: the player's industries, other players' industries and merchant locations.
- A static miniature map with an era selector and clickable route rows. Selected routes get a clear outline, with their endpoint totals displayed beside the map. Use the game's route geometry, including routes with additional connected regions; do not assume every link has exactly two scoring locations.
- Toggle **Points from my industries / Other players / Merchants** to explain interdependence. This explains existing link points; it does not allocate those points to the industry owner.
- A city occupancy table/map for Birmingham, Worcester, Wolverhampton, Walsall and the other actual board locations. Show owned occupied slots / total slots as density, and owned occupied slots / all occupied slots as ownership share. Display both denominators. A Black Country comparison group, if added, must list its chosen cities explicitly; it is a display grouping, not an official scoring region.
- Compare city occupancy at Canal scoring, after Canal cleanup and final Rail scoring. Count currently occupied slots separately from the number of builds performed there throughout the match.
- **Opponent-link reliance:** For recorded operations that require a route, test connectivity at that moment with all eligible links, then with only the actor's links. Count cases where the actual recorded endpoints cease to connect. Coal deliveries, opponent-brewery deliveries and identifiable merchant sales have distinct denominators. Exclude iron and own-beer use, which do not require such a route. The result measures reliance for those recorded connections, not whether the whole action was impossible using a different supplier or destination.
- **Slot competition:** Log builds occupying the last vacant eligible slot of a type in a city, plus actual opponent overbuilds. Drill down to the public board before/after the build. Label the capacity change directly; do not call it proven space sniping or claim an opponent critically needed that slot without recorded intent.

Reconstruct the Canal board before link removal and obsolete-industry cleanup. Rail ledgers also use the Rail scoring boundary. Miniature maps use independent UI drawings/icons; do not borrow or reparent live board tiles or animation transforms.

### Page 5 — Economy

Answer: **Where did the money come from, where did it go, and when was income useful?**

- Cash and income-level charts through the match, selectable by turn or round. Keep pounds of cash and income levels on separate clearly labeled plots. Income-marker spaces can appear in details, never as the primary income measure.
- Receipts: paid income, loans, coal/iron market receipts, cash merchant bonuses and liquidation proceeds.
- Spending: industry construction, canal/rail construction and resource purchases. Show negative-income payments separately from action spending.
- Loan count, total borrowed and timing. Show actual income changes around each loan rather than estimating its total lifetime cost.
- Sum the actual income levels reduced by loan events, separately from marker spaces. Add selectable loan annotations and sale/depletion income annotations on the income curve. A double-loan turn can be labeled **Two loans this turn**; a chart must not invent a universal threshold for a loan rush.
- Cash at the Canal/Rail transition and final cash, alongside final income level.
- A round table with each player's action spending and the actual next-round turn order. Tied spending preserves the recorded order; do not invent a reordering.
- Debt settlements: industries removed, money raised, negative-income payment and VP lost.

Each cash breakdown must reconcile to the corresponding opening and closing balances. Bought-resource prices and generic money-spent events can describe the same transaction; count the cash movement once. Preserve zero-value transactions where they help explain actions.

### Page 6 — Resources & Trade

Answer: **Who supplied whom, and how were goods sold?**

- Coal, iron and beer: units placed on industries, units consumed, units sold to the market, remaining units and units removed with industries. Merchant beer is a separate source.
- Beer gets a prominent source/use matrix: own versus opponents' brewery beer versus merchant beer, crossed with Sell versus double-rail use. Distinguish **beer from a merchant** from **brewery beer used during a sale**. Count beer produced, self-consumed and supplied to each opponent.
- A **Most beer supplied to opponents** callout identifies the leading supplier by recorded barrel count, with ties. This supplies a defined version of the suggested MVP Brewery without pretending it measures overall strategic value.
- Consumption by source: own industries, another player's industries, market or merchant, as applicable to that resource.
- A supplier-to-consumer matrix for each resource, showing quantities. Click a cell to inspect the underlying actions, producers and locations. Resource transfers are not payments between players.
- Coal/iron market purchases and receipts, with actual quantities and prices. Credit market receipts to the selling industry's owner.
- Goods sold by type and level, Sell action count, tiles sold per Sell action, multi-tile sales and beer consumed for those sales.
- Merchant beer uses and recorded bonuses: extra income, cash, VP or development. Show benefits in their own units rather than treating all bonuses as VP.
- Market-impact charts record the next-unit coal/iron quote and stock before/after each purchase or market sale. Attribute the immediate quote movement to that transaction's actor/producer. Show quantities and real payments beside the quote change, with later real transactions available on the same timeline; do not claim a measured downstream cost to opponents from a hypothetical alternate history.
- Merchant-location rows show attributed goods deliveries, beer barrels used and each bonus benefit, with an **Unattributed destination** row. If a merchant beer identifies the destination, attribution is exact. A single eligible merchant location may support a clearly marked derivation; multiple eligible locations without a recorded destination remain unattributed.

Track ownership before resource depletion flips a producer or a tile is removed. The consuming actor can differ from the industry owner, who receives the associated flip income and later industry VP. Do not attribute all resource use to whoever is current after a round transition or debt phase.

The goods-sale ledger must not confuse debt liquidation with a Sell action. In this code, `IndustrySold` and `IndustrySellingEnded` belong to liquidation; goods sales require their Sell action context and flipped-industry records. Zero-beer and brewery-only sales do not necessarily select or record one destination. Show **Merchant not recorded** where needed, with coverage counts; do not arbitrarily choose one connected merchant or require extra gameplay clicks just to complete statistics.

Merchant bonus categories are Develop, Income, VP and Money. The glossary uses the actual locations: Gloucester development, Oxford income-marker advance, Nottingham/Shrewsbury VP and Warrington cash. Read VP amounts from the current rules/result records. There is no Free Network merchant bonus in standard Brass: Birmingham; Oxford's two-space advance is not automatically two income levels. See official rules p. 10.

### Page 7 — Actions & Cards

Answer: **How did each player use their available actions?**

- Counts and proportions for Build, Network, Develop, Sell, Loan, Scout and Pass, by player and era.
- A stacked action chart and a comparison table of tiles developed per Develop action, goods sold per Sell action, and rails placed per rail Network action.
- Wildcards acquired and used, separated from ordinary action counts. Scout consumes one action despite trading additional cards.
- A Build-card table separates designated Location, Industry, Wild Location and Wild Industry use. Non-Build action-payment cards get a separate Location/Industry/Wild distribution, with Scout's two additionally traded cards shown separately. Exclude setup's initial discard and era reshuffles. Call these **Non-Build payment cards**, rather than asserting they were junk.
- Wildcard conversion shows acquisition occurrences, uses for Build, uses as non-Build payment and cards held at each era end. Track each acquisition episode, since the same card ID can return to the supply and be acquired again.
- A wildcard-use ledger identifies the actual tile/location enabled. An optional deterministic **No ordinary card in hand authorized this build** flag can be derived from the pre-action hand and the same placement, using the existing card/placement rules. This explains a concrete use of flexibility; it is not proof of a game-changing turnaround.
- Whole-match **final VP per completed action**, with a visible definition. This is a descriptive ratio; it does not assign causal value to individual actions. For an era filter, use that era's net awarded VP divided by that era's actions and label it accordingly.
- Click an action category to filter the Timeline. Show merchant-bonus development within its parent sale, rather than counting a second Develop action.

Count confirmed actions from the retained history and action-completion boundaries. Exclude cancelled/undone branches. Distinguish actions from individual links, developed tiles, sold tiles and cards discarded. The six core actions plus Pass form the full distribution; show Pass separately, and use all completed actions as the default percentage denominator. An optional six-action-only view must state that it excludes passes.

Normal full games provide 31/35/39 actions per player for 4/3/2 players respectively, derived from 8/9/10 rounds per era and the first Canal round's single action. Introductory games provide 15/17/19. These are validation expectations, not hard-coded statistics or a 32–36-action assumption. Count the actual retained match actions, including passes, and show anomalies or incomplete records explicitly. See official rules pp. 6–7.

### Page 8 — Turn Order & Tempo

Answer: **How did spending affect when players acted?**

- Round-position table: rounds and percentage in each available position, by player and era. Two-/three-player games hide nonexistent seats. Denominator is the player's actual turns with known order.
- A round-by-round seat-position chart paired with action spending. Rows show the transition from spending in round N to order in round N+1, preserving recorded ties.
- **Back-to-back turns:** Count the same player being last in one round and first in the next within an era, with no intervening opponent turn. Show the count and eligible round-boundary denominator. Show Canal-to-Rail boundary sequences separately because scoring, cleanup and new hands intervene.
- Expand each sequence into its real ordered actions, such as Build → Network / Sell → Loan. Also show the immediately adjacent action pair and the resources/assets involved, rather than claiming every sequence was an intentional combo. A sequence may contain fewer than four actions near the opening or in incomplete records.
- Spending-tempo chart and numeric summary: total, minimum, maximum, mean, median, population standard deviation, zero-spend rounds and largest-spend round. Optional relative volatility is standard deviation / mean; an all-zero mean displays an em dash.

Use the measured range and volatility, not fixed £10–£20 or £50 thresholds to classify a style as balanced or volatile. Those labels would require a separately defined benchmark across comparable games. Keep loan/income timing available as chart annotations without adding income settlements to the action-spending total used for turn order.

### Page 9 — Efficiency & Transition

Answer: **How much did the player's investments and actions convert into scored assets?**

| Metric | Definition and interpretation |
| --- | --- |
| VP per completed action | Final VP / all completed actions, including Pass; era view uses era net awarded VP / era actions |
| Action cash spent per VP | Gross construction, link and resource-purchase spending / final VP; purchase costs counted once; an explicitly labeled All cash payments toggle includes negative-income payments |
| Industry realization at scoring | Awarded ordinary industry VP / total printed industry VP of tiles present at that scoring boundary; exclude links, merchant/introductory points and penalties |
| Unflipped investment | Count, printed VP, base construction cash, directly attributable bought resources, free resource quantities used and resources already supplied/market receipts generated by those remaining tiles |
| Level-1 cleanup | Tiles removed, flipped/unflipped status, Canal industry VP already scored, remaining resources removed and link-icon contributions present immediately before cleanup |
| Carryover conversion | Tiles surviving Canal cleanup, their Rail flips/removals and actual Rail VP, with the same asset identity tracked across eras |

Ratios show their numerator/denominator and definition. Cost per VP is unavailable when final VP is zero or negative. Industry realization is a snapshot of printed tile value, not the maximum attainable match score. Whole-game aggregation sums clearly labeled era scoring opportunities, so an eligible surviving tile can appear in both era denominators.

Call the unflipped figure **Investment remaining unflipped**, not cash/resources proven wasted. A resource tile can have generated market cash or supplied useful resources without fully depleting. Free resources are quantities, not fictitious cash payments, and already-scored Canal points remain earned.

Canal link scoring precedes level-1 cleanup, so removal causes **no deduction from Canal link VP already scored**. Show the removed link-icon contribution by location as board context. Any hypothetical Rail link loss would depend on later placement, new industries and opponents' play; it is not an exact waste statistic.

Do not promise a maximum reachable VP calculation. Proving what could have been earned requires a counterfactual sequence of legal actions, resources, cards and responses, beyond a factual end-game report. The printed-value realization measure provides the requested comparison with an explicit, auditable denominator.

### Page 10 — Timeline

Answer: **When did the important changes happen?**

- A chronological, searchable ledger: era, round, global turn, player, action, locations/assets, cash/income changes, resource flows and relevant score changes.
- Expand a row to see its grouped operations, card used where recorded, industry levels, supplier ownership and merchant benefit.
- Mark loans, grouped sales, double rails, important flips, liquidation, era transitions and scoring as filterable event types.
- A synchronized chart cursor reveals the state after that operation. Scoring and income-settlement rows sit between turns, with explicit phase labels instead of being attached to an arbitrary player's action.
- Optional **View on map** opens an independent historical miniature map at that boundary. It must not rewind the live board or restart the game's replay animations.

Existing events contain revisions, not reliable wall-clock timestamps. Show turns/rounds, not match duration, seconds per action or AI thinking duration. Such timing statistics would require a separate future data-collection change.

## 4A. Coverage of the user's proposed analytics

| Proposed group | Where it appears | Qualification |
| --- | --- | --- |
| I. Economic & Era Analytics | Scoring; Economy | Income-level loan reductions and actual spending are separate; market purchases are not counted twice |
| II. Industry & Building Efficiency | Industries; Efficiency & Transition | Distinct built/flipped assets, six industry types, level-specific development and real scoring occurrences |
| III. Resource & Player Interaction | Resources & Trade | Beer source and purpose are separate dimensions; supplier ranking has an explicit barrel-count definition |
| IV. Action & Deck Distribution | Actions & Cards | Six core categories plus Pass; wildcard outcomes are observed uses, with optional placement-specific feasibility |
| V. Turn Order & Spending Tempo | Turn Order & Tempo | Exact back-to-back sequences, seat shares and spending dispersion; era-boundary cases separated |
| VI. Geography & Merchant Analytics | Networks & Geography; Resources & Trade | Density denominators and snapshot phases are explicit; destination coverage and actual bonus types are shown |
| VII. Era Transition & Waste Analytics | Efficiency & Transition; Actions & Cards | Cleanup does not reverse Canal link points; unfinished investment and non-Build payment cards receive neutral labels |
| VIII. ROI & Efficiency Metrics | Efficiency & Transition | Actual action counts, one cash-spending ledger, safe denominators and a printed-value realization measure |
| IX. Counter-play & Disruption | Networks & Geography; Resources & Trade | Capacity change, required opponent routes and immediate market quote shifts are measurable; intent and downstream counterfactual harm are not assumed |

Figures are marked **Recorded**, **Derived from recorded state**, **Partial**, or **Unavailable** where appropriate. Estimates or strategic interpretations, if added later, must have a separate **Estimate** label and a visible definition. The primary panel does not present unlogged intentions as facts.

## 5. Visual style

Match the game's brass-and-dark-brown appearance, with readable fills and restrained rounded corners. The user has previously found highly transparent highlights difficult to see, so the panel and selected controls should have substantial contrast.

| Element | Starting treatment |
| --- | --- |
| Main panel | Deep warm brown `#201C17`, approximately 96% opaque |
| Cards/table rows | Slightly lighter brown `#30291F`, approximately 98% opaque |
| Main text | Warm ivory `#F3E7CC`, opaque |
| Secondary text | Muted parchment `#CBBB9F`, opaque; validate contrast |
| Selected page/control | Brass `#D2AA55` outline and clear filled selection; text/check/icon indication |
| Hover/keyboard focus | Brighter cream-gold `#F2D58B` outline, visibly distinct from selection |
| Player series | Existing player identities, plus labels and different markers/patterns |

Use roughly 8–12-unit corner radii and comfortable 44–48-unit button targets at the reference scale. Start around 18–20-unit body text and 26–30-unit headings, then qualify in actual Unity captures. Maintain visible grid lines, selected rows and scrollbars. Tooltips must remain legible over charts and never require hovering to read essential results.

Prefer standard uGUI/TMP controls and code-drawn charts using existing assets. An embedded website/browser is unnecessary. Keep effects modest and do not animate chart values so slowly that comparison becomes difficult. Respect reduced animation where available.

## 6. What the current code provides

| Source | Observed behavior | Consequence for this feature |
| --- | --- | --- |
| [FinishedGameReviewUseCase](../ai-research/decompiled/BrassApplication.UseCases.Game.FinishedGameReview/FinishedGameReviewUseCase.cs) | `ReplayCompleted` exposes review controls and opens results; `FinishGameReview` deletes local games | Enable the entry point after replay completion; capture history/results before review cleanup |
| [FinishedGameReviewController](../ai-research/decompiled/InterfaceAdapters.Game.FinishedGameReview/FinishedGameReviewController.cs) | View destruction calls `FinishGameReview` | Closing statistics must hide its own panel, never destroy the finished-review owner |
| [FinishedGameReviewPresenter](../ai-research/decompiled/InterfaceAdapters.Game.FinishedGameReview/FinishedGameReviewPresenter.cs) and [EndOfGameView](../ai-research/decompiled/Frameworks.View.Game/EndOfGameView.cs) | Existing results use rank/score tuples and Review/Exit actions | Add a statistics entry and preserve existing result/exit behavior |
| [StateFixes](../src/BrassPatch/State/StateFixes.cs) | Installed ranking compares VP, income level and cash, with shared competitive ranks | Use patched results; the decompiled stock ranking is not the active implementation |
| [GameLogUseCase](../ai-research/decompiled/BrassApplication.UseCases.Game.GameLog/GameLogUseCase.cs) and [IGameStorage](../ai-research/decompiled/BrassApplication.Persistence.RepositoryBackends/IGameStorage.cs) | Event history is loadable by revision; the stock log reads persisted events | Build a separate structured projection, including retained uncommitted events when applicable |
| [BaseEra](../ai-research/decompiled/BoardGameRules.Entities.GamePhases/BaseEra.cs) | Score events aggregate industry/link VP per player; link scoring includes connected regions | Reconstruct per-asset detail at scoring boundaries and reconcile to the aggregates |
| [CanalEra](../ai-research/decompiled/BoardGameRules.Entities.GamePhases/CanalEra.cs) | Canal links and obsolete industries are removed | The final board alone cannot supply accurate Canal statistics |
| [TurnFlow](../ai-research/decompiled/BoardGameRules.Entities.TurnPhases/TurnFlow.cs) and [GeneralAction](../ai-research/decompiled/BoardGameRules.Entities.Actions/GeneralAction.cs) | Confirmed actions end with `ActionCountDecreased`, which has no player/type payload | Classify action frames using their turn context and operations |
| [Bank](../ai-research/decompiled/BoardGameRules.Entities.NumericalResources/Bank.cs) and [PlayersTrack](../ai-research/decompiled/BoardGameRules.Entities.NumericalResources/PlayersTrack.cs) | Income payment/turn-order events have no amount/order payload | Derive results from state before and after each event |
| [LiquidationFixes](../src/BrassPatch/Liquidation/LiquidationFixes.cs) | Liquidation has its own owner, proceeds and penalty events | Give liquidation its own phase and financial category |
| [CardDiscarded](../ai-research/decompiled/BoardGameRules.Entities.Actions.Events/CardDiscarded.cs), [WildCardsTaken](../ai-research/decompiled/BoardGameRules.Entities.Actions.Events/WildCardsTaken.cs) and [ScoutAction](../ai-research/decompiled/BoardGameRules.Entities.Actions/ScoutAction.cs) | Main payment and extra Scout trades have separate records; wildcards return to reusable pools | Track card purpose and acquisition episodes rather than counting every discard as an action |
| [BaseMarket](../ai-research/decompiled/BoardGameRules.Entities.Markets/BaseMarket.cs), [CoalBought](../ai-research/decompiled/BoardGameRules.Entities.Actions.Events/CoalBought.cs) and [IronBought](../ai-research/decompiled/BoardGameRules.Entities.Actions.Events/IronBought.cs) | Resource prices follow stock and transactions carry quantities/payments | Report real market costs and immediate next-unit quote changes |
| [SellAction](../ai-research/decompiled/BoardGameRules.Entities.Actions/SellAction.cs) and [SellActionUseCase](../ai-research/decompiled/BrassApplication.UseCases.Game.Actions.SellAction/SellActionUseCase.cs) | Sale flips record the industry; selected merchant context is primarily merchant-beer selection | Full per-merchant delivery attribution is not always recoverable or defined |
| [PresentationLifecycle](../src/BrassPatch/Presentation/PresentationLifecycle.cs) and [LinkReplay](../src/BrassPatch/Presentation/LinkReplay.cs) | Current patch protects information panels and animation completion | Integrate input/cleanup without disturbing these protections |

Some event payloads are intentionally sparse. For example, `IndustryBuilt` records type and slot, not level; supply events identify the source, not the consuming player; and `IncomeMoneyPaid` has no amount. Rich statistics therefore need state-aware reconstruction rather than a count of event names.

## 7. Data architecture and correctness requirements

### Immutable match report

Create a `CompletedMatchStatistics` model containing the authoritative result, match/player identity, era summaries, asset records, action frames, card-use/acquisition episodes, resource transfers, financial entries, round orders, market quotes, timeline snapshots and data-availability flags. Bind UI only to this model. Use stable game/player IDs and colors, never mutable turn-order indices.

1. At completed-review entry, snapshot metadata and final results and copy the retained event stream before `FinishGameReview` can delete it. Merge persisted and uncommitted events up to the final `GameEnded` revision; deduplicate exact overlapping revisions, reject conflicting duplicates and validate continuity.
2. Reconstruct on an isolated rules state created from the original configuration/setup records. Apply recorded events only. Do not rerun random setup, AI decisions or action commands that could emit another sequence.
3. Resolve ownership, industry level, resource stock and monetary/score state before and after relevant events. Capture scoring-boundary board states before removals. Store minimal immutable snapshots needed by the pages.
4. Validate every aggregate and assign availability per metric. Cache by game ID, final revision and report-schema version. Opening, filtering and scrolling reuse the report.

Do not call `BrassGameHistory.MoveGameToRevision` on the live game for statistics. Do not rely on a shallow game copy, shared RNG or process-global merchant state. Qualify ownership isolation using the existing patched rules constructors/replay path. Snapshot references to mutable industry/player objects must become value records before further events are applied.

Collect from the logical completed state, not the partly animated display. Generation can be lazy after the initial capture, with a loading message and a working close button. Unity object work stays on the main thread; only detached, proven thread-safe data work may run elsewhere. Every callback checks the game/session generation, preventing a late report from another match appearing in the current panel.

### Accounting rules

- **Score:** Reconcile final VP to opening VP plus actual score deltas. Categorize industry, link, merchant, penalty and introductory components. Era summary events and `GameEnded` validate totals; they are not additional point awards. Merchant bonus summaries must not duplicate already-awarded VP. Use actual state deltas for any capped/clamped operation.
- **Money:** Reconcile opening cash plus receipts minus payments to closing cash, per player and boundary. Derive real positive/negative income settlement from before/after balances; the final Rail round has no income payment. Distinguish action spending used for turn order from income debt payments. Do not sum both a purchase price and its `MoneySpent` event as separate expenses.
- **Assets:** Use the actual industry identity when reconstructed, together with its build revision; repeated use of a slot is not one tile. Link records include era and placement revision. Separate destruction, overbuilding, liquidation and obsolete removal.
- **Resources:** Reconcile initial stock plus production/restocking to consumption, market sales, removed stock and remaining stock. Include recorded era adjustments, such as merchant-beer replenishment. Separate merchant beer and bought market resources from produced stock. Attribute consumption from its retained action frame, including when the consumer differs from the producer.
- **Actions:** Build frames around confirmed `ActionCountDecreased` boundaries within turns. Infer type from distinctive operations, with explicit handling for multi-sale merchant development. Ambiguous older records receive an unclassified action entry; never invent a type to make a chart look complete.
- **Cards/tempo:** Correlate payment cards and extra Scout trades to their retained action frames. Preserve pre-action hands only where needed for deterministic card feasibility. Record orders at each `TurnStarted`/round boundary, so temporary debt ownership is never counted as a player's normal seat. Round-order percentages and double-turn opportunities expose their denominators.
- **Interaction:** Connectivity comparisons operate on detached historical graph snapshots and actual recorded endpoints, with no legal-action execution or live-board changes. Market quote movement uses the before/after stock, independent of whether another player could legally buy there. Unknown sale destinations do not enter an exact destination-reliance denominator.

### Incomplete and older histories

If a completed game has final results but an incomplete event history, show the trustworthy result immediately. Mark affected metrics **Unavailable for this match** or **Partial history** and explain why. Missing data is not zero. Do not fill gaps by guessing from the final board or by making an uncertain category appear exact.

Games continued from an existing save can get the same report if their retained history is sufficient. This feature should initially reuse the existing save/event format. Opening and closing the panel must not trigger a save, change a revision or delete the game.

The report remains in memory during completed-game review. Existing exit behavior may remove the local game; automatic permanent match archives are separate scope. A future explicit export button can write a compact report, but should not silently change retention of saved games.

## 8. Implementation work packages

### A. Establish the statistics projection

Proposed files: `src/BrassPatch/Statistics/CompletedMatchStatistics.cs`, `StatisticsCollector.cs`, `StatisticsProjection.cs` and `StatisticsMetrics.cs`.

Define identity, availability, signed financial/score entries and action-frame contracts first. Build projection fixtures from existing native full-match, liquidation, merchant-bonus and saved-game tests. Deliver exact final-score reconciliation and reliable action/asset accounting before displaying derived ratios.

### B. Add panel routing and the entry points

Proposed files: `StatisticsController.cs`, `StatisticsPanel.cs` and `StatisticsEntryPoints.cs` in the same directory.

Attach one button/controller per finished-review instance. Add a separate result-dialog entry. Use a dedicated modal routing adapter compatible with the game's Router and the existing presentation hooks. The adapter must own visibility, keyboard focus, raycasts and return destination; merely setting a high canvas sorting order is insufficient.

Use existing UI fonts/sprites and programmatic reusable rounded buttons, chart cards, tables and scroll views. Opening/rebinding is idempotent: no duplicate callbacks, panels or blockers. Clean up on review exit, scene destruction and new-game ownership changes.

### C. Implement all pages and drill-downs

Build Overview/Scoring first, then Industries/Networks & Geography, Economy/Resources, Actions & Cards/Turn Order & Tempo, and Efficiency & Transition/Timeline. Share player/era filters, ledgers, chart legends and formatting. Every interactive chart must drill into the same records used to calculate its totals.

Implement lightweight historical miniature maps after the required tables and charts work. They can ship with the same release if qualified; the ten pages must remain fully usable through their ledgers if a map drawing is unavailable.

### D. Package and qualify

Add a narrowly scoped `src/BrassPatch/Statistics/manifest.json` and the appropriate build/installer integration. Reuse the existing patch workflow, baseline checks, release backup and rollback. Preserve all installed 1.4.6 hooks and local-only behavior. Choose the next release version during implementation, not in this plan.

Measure report generation on retained full-match histories. Use one generation per completed revision and reuse table rows/visible ledger ranges where necessary. Target a responsive first overview and smooth scrolling; report measured timings rather than claiming an untested performance guarantee. Avoid unnecessary continuous redraw or repeated repository reads.

## 9. Verification and acceptance criteria

### Statistics accuracy

- Full games with 2, 3 and 4 players; introductory games; human/AI mixtures; ties resolved by income or cash; unresolved two-/three-/four-way draws; totals beyond the printed VP track.
- Independent reconciliation of authoritative total/era VP and cash balances. Assert that opening, closing, navigating and filtering leave game state, revision, retained events and results unchanged.
- Canal assets later removed, level-2+ carryover scored in both eras, unflipped survivors, own/opponent overbuilds and repeated builds on the same slot.
- Multi-industry Sell actions, zero-beer sales, one-/two-tile Develop actions, free merchant development, Scout's extra card exchange, Pass and cancelled/undone operations.
- Single and double rail placement, route geometry with additional scoring regions, zero-point links, merchant endpoints and link points supplied by other players' industries.
- Own/opponent resource use, multiple producers depleted during one action, market purchases, automatic coal/iron market sales and merchant beer. Confirm supplier/consumer ownership before flip/removal.
- Every merchant bonus, income caps, positive/negative income, loan timing, debt liquidation and penalties. Confirm the missing final Rail income payment is not fabricated.
- Persisted/uncommitted overlap, a save continued to completion, sparse/partial history, conflicting revisions, missing setup and failure during projection. Trustworthy summary remains usable on a detailed-data failure.
- Card role/identity tests: Wild cards reacquired after return, Scout's two extra trades, ordinary/dual-industry cards, initial setup discard, era reshuffle, build outside the previous own network and incomplete pre-action hands. No arbitrary game-changing labels.
- Seat/tempo tests: 2/3/4 players, equal-spend stable order, all-zero spending, last-to-first transitions, era transitions, the opening one-action turn and debt-choice ownership. Every percentage and standard-deviation denominator is specified.
- Merchant coverage tests: merchant-beer sales, zero-beer and brewery-only sales, one/multiple eligible destinations and reused beer after the era reset. Validate benefits in their real units and display unattributed destination coverage.
- Efficiency/interaction tests: zero final VP, zero actions/empty city, a partly depleted unflipped supplier that generated cash/resources, level-1 cleanup after Canal link scoring, route alternatives, iron/own-beer exclusions, last eligible slot builds and an exhausted/full resource market. No causal/waste claim from a descriptive count.

### UI and lifecycle

- Bottom-left button appears after final replay; results-dialog entry works without click-through. Opening/closing each entry returns to the correct screen.
- Escape, keyboard focus, mouse wheel and scrollbars work. Backdrop/input blockers disappear on close. Information panels and existing results controls regain input correctly.
- Reopen repeatedly, switch all pages/filters, expand/sort/search large ledgers, start a new match, and exit to the menu. No stale report, duplicate button/listener or cross-game callback.
- Native render captures at 1,280 × 720, 1,920 × 1,080 and 2,560 × 1,440, including supported window/fullscreen and UI scaling combinations. Check long names, 4-player comparisons, signed values, tooltips and chart/table contrast.
- Verify the panel never reparents board tiles/player portraits/link tokens or alters camera/replay ownership. Rerun the relevant existing presentation, link replay, city highlight and save reconstruction regressions.
- Installed qualification confirms startup, complete-match entry, all ten pages and hashes for the new release. Keep AI search/difficulty behavior and the disabled Play now button intact.

### Completion definition

The first release is complete when the bottom-left button and result-dialog entry expose all ten navigable, scrollable pages; the nine requested analytics groups have the representations in section 4A; supported statistics reconcile to the completed game; every figure is labeled by units/era and supported by an inspectable record; partial histories are handled honestly; and the panel closes/reopens without modifying gameplay or breaking existing presentation fixes.

## 10. Delivery order

| Stage | Reviewable output | Gate to proceed |
| --- | --- | --- |
| 1 | Data contracts, projection, accuracy fixtures and an example completed-match report | Score, cash, resources and action frames reconcile |
| 2 | Bottom-left/results entry, modal navigation and Overview/Scoring | Entry, close, focus and rendering qualification pass |
| 3 | Remaining eight pages, comparison/filter interactions and ledgers | All required metrics agree with the same projection |
| 4 | Historical map views where qualified, performance and compatibility checks | Presentation/save regressions pass; partial-history behavior works |
| 5 | Versioned release, backup, installation and installed-game verification | New build and installed-game qualification pass under a subsequent implementation request |

For the present request, the deliverable is this plan. Implementation and installation follow a subsequent request to implement it; no game files are patched now.

## 11. Later extensions

Optional additions after the complete single-match panel: explicit CSV/JSON export, a persistent local match archive, cross-match trends, saved chart preferences and real timing statistics collected prospectively. AI search diagnostics and training evaluation belong in a separate advanced/debug view, rather than the normal end-game report.

## 12. Sources

Research accessed 30 September 2026.

1. [Roxley: Brass: Birmingham](https://roxley.com/products/brass-birmingham) — publisher description, game identity and official rulebook listing. The publisher's current [English PDF download](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281) exceeded the web reader's size limit; rule text was cross-checked against the existing local publisher PDF rather than claiming a new successful download.
2. [Official rulebook, local copy](official-rulebook.pdf) — pp. 6–7 for turn/round order, income, scoring and victory; pp. 8–11 for resources and actions; p. 12 for the introductory variant. Publisher-hosted [2018.11.20 rulebook reference](https://files.roxley.com/Brass-Birmingham-Rulebook-2018.11.20-highlights.pdf).
3. [MeepleShelf: Brass: Birmingham score tracker](https://meepleshelf.com/games/brass-birmingham) — precedent for separating Canal/Rail industry and link totals. Used as UI inspiration, not rules authority.
4. [Dylan Ho: How to Dominate Brass: Birmingham](https://dlaern.substack.com/p/how-to-dominate-brass-birmingham), 20 April 2025 — personal strategy observations motivating action, supply, sale and timing metrics. Used as inspiration, not a prescribed strategy or quantitative benchmark.
5. Current local code and installed patch — linked in section 6; [1.4.6 release record](../docs/LINK_PLACEMENT_REPLAY_1.4.6.md) and [installation metadata](../artifacts/patch-installation.json).

## 13. Readability refinement — 1.4.10

The subsequent request to enlarge and improve the implemented panel adds:

- Body/table text at 21 pixels, secondary labels at 18–19, section headings at 24, and page/title text at 30. A−/A+ offer a further 15% or 30% enlargement.
- Taller controls and rows, wrapped and pinned headings, wider columns, alternating row backgrounds, brighter ivory text and clearer gold selected states.
- A player/era/row-count status line, search clearing, reset of hidden chart series, collapse-all details and a Top button.
- Fully measured expanded details, adaptive tooltips, larger chart hover targets, retained horizontal table positions, keyboard focus reveal and category-menu dismissal.

The report calculations and gameplay/AI implementations remain unchanged. The readability release is qualified with completed-match/statistics fixtures and native renders at 720p, 1080p and 1440p, including Largest on every page at 720p and 1440p. Existing presentation, link replay, menu and managed regression checks run against the candidate before the atomic installation and installed-player checks.

See [the readability release notes](../docs/STATISTICS_READABILITY_1.4.10.md) for implementation, qualification and recovery details.
