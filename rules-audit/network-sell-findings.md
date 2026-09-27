# Network, selling and beer audit

Audit date: 2026-09-27. Reviewed the decompiled installed assembly and original methods using isolated in-memory fixtures. No installed game binary or save was changed. Source line references below are in `ai-research/decompiled/`. Runtime evidence: [network-sell-probe-results.json](network-sell-probe-results.json); executable probe: [network-sell-probes.ps1](network-sell-probes.ps1).

Rule reference: Roxley, **Brass: Birmingham Rulebook — 2018.11.20**, linked from [publisher product page](https://roxley.com/products/brass-birmingham), [official PDF](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281). Printed page references are used; downloaded PDF sheets are two-page spreads.

## N1 — AI consumes two merchants' beer to sell a single industry

**Priority P1. Runtime confirmed using the installed original AI helper.** Human UI has a partially correct preflight, so do not claim an isolated two-beer sale is directly selectable through the initial human action menu.

**Rules:** printed p8, Consuming Beer, and p10, Sell step 2–3. A sale is to one matching merchant; merchant beer belongs to that selected merchant. A two-beer industry therefore cannot combine two merchant barrels. Breweries can provide the additional barrel.

**Evidence and cause:**

- `BoardGameRules.Entities.Actions/SellAction.cs:142`: `CanIndustryBeSold` sums every connected matching merchant's barrel; one sale is incorrectly treated as eligible with two merchants and no brewery.
- Same incorrect count in `BrassApplication.AI.AIHelpers/SellActionAIHelper.cs:98` and `BrassApplication.AI.AIHelpers/BoardHelper.cs:498`.
- `BrassApplication.AI.AIHelpers/SellActionAIHelper.cs:19`: `FindBeerSource` searches matching merchants afresh every call. `:115`–`:119` repeats this for every barrel. No selected merchant is remembered.
- `BoardGameRules.Entities.Actions/SellAction.cs:53` delegates beer validation without checking merchant identity. `BoardGameRules.Entities.Consumables/ConsumingBeer.cs:75` awards each merchant bonus consumed.
- `SellAction.cs:95` correctly caps merchant beer to one in the human initial preflight. `BrassApplication.UseCases.Game.Actions.SellAction/SellActionUseCase.cs:80` suppresses additional merchants after a merchant barrel has been selected. Those safeguards do not protect the AI or the engine's eligible-slot list.

**Reproduction:** create a reachable level V manufacturer in Birmingham with no breweries, a Birmingham–Oxford link, and Oxford's two merchant slots containing the manufactured-goods and wildcard tiles (a possible two-player arrangement). Run the original `SellActionAIHelper.SellIndustries`.

**Observed:** preflight reports `NO_BEER`, eligible-slot query nevertheless returns true, both Oxford barrels disappear, the manufacturer flips, and income increases by **6**: its normal 2 plus two Oxford bonuses of 2. This is not merely a speculative missing guard; the shipped AI execution path performs it.

**Fix plan:** make sale eligibility and resource selection use a shared sale context containing the sold industry and chosen merchant. Until merchant choice occurs, test whether any one matching merchant plus eligible brewery supply can meet demand. Once selected, permit only its one barrel; enforce this in the rules method, then reuse it in the UI and AI. Reset context after each industry, not only after the whole Sell action. Update duplicated AI feasibility queries. Do not change merchant bonus values.

**Acceptance tests:** two matching merchant beers with no brewery cannot sell a two-beer industry; one matching merchant plus one eligible brewery can; two breweries can; multiple industries in one Sell action can use different merchants independently; bonuses awarded exactly once for each consumed merchant barrel; one-beer and zero-beer sales remain valid. Include the human continuation case where another legal sale makes the action startable while the two-beer industry is also falsely highlighted.

## N2 — Rail coal selection permits a farther mine

**Priority P1. Runtime confirmed in the original rules; human/AI candidate generation uses the same faulty predicate. No Unity click automation performed.**

**Rule:** printed p8, Consuming Coal: use the connected mine at the minimum link distance; ties are choices. Printed p11: pay coal after placing each rail.

**Evidence and cause:**

- `BoardGameRules.Entities.Actions/NetworkAction.cs:81` passes the new rail's endpoint regions to the consuming function.
- `BoardGameRules.Entities.Consumables/ConsumingCoal.cs:60` accepts a source if it is valid for **any single endpoint**. `:142` compares mines only relative to that one endpoint.
- `BrassApplication.UseCases.Game.Actions.NetworkAction/NetworkActionUseCase.cs:167` highlights the union of per-endpoint nearest mines; `BrassApplication.AI.AIHelpers/NetworkActionAIHelper.cs:27` uses the same union. `NetworkAction.cs:285` duplicates it for feasibility.

**Reproduction:** in the Rail Era, have a coal mine at Dudley, a coal mine at Redditch, and the existing Birmingham–Redditch rail. Place Birmingham–Dudley. Dudley's mine is at the new rail's endpoint; Redditch requires the existing additional rail. Distances to Birmingham/Dudley are Dudley `[1,0]` and Redditch `[1,2]`.

**Observed:** both Redditch and Dudley enter the allowed-source collection. `NetworkAction.SupplyCoal(Redditch,1)` succeeds, taking one from Redditch and none from Dudley. At the Birmingham endpoint they tie, but Redditch is farther from the newly placed rail as a whole. Board enumeration lists Redditch first, also making it the AI helper's first source in this fixture.

**Fix plan:** add one authoritative multi-region coal-source query. For each reachable nonempty mine compute the minimum distance to any endpoint of the new rail, then keep only globally minimum candidates. Use it for resource consumption, UI highlighting, AI, and affordability. Preserve singleton-region behavior for building industries. All rail endpoints, including the farm-brewery third endpoint, must participate; remove null endpoints. Query after placing each rail, and re-evaluate after depletion.

**Acceptance tests:** this fixture must permit only Dudley; equidistant mines at opposite endpoints must both be available; a depleted nearest mine must allow the next one; market must remain unavailable while any reachable mine contains coal; cover the three-endpoint Kidderminster–Worcester–farm connection and separately the first/second rails in a double action.

## N3 — Gloucester bonus can trap a human player when no tile can be developed

**Priority P2. Strong static UI finding with partial runtime confirmation; full Unity interaction has not been exercised.**

**Rules:** printed p10, Sell and Merchant Beer Bonuses. Gloucester beer gives a development bonus subject to the pottery restriction. The Sell requirements do not require a remaining developable tile. Consuming the beer must not produce an unfinishable action when there is no legal bonus target.

**Evidence and cause:**

- `BrassApplication.UseCases.Game.Actions.SellAction/SellActionUseCase.cs:130` always enters `DevelopBonus` if the flag is set; it does not check legal targets.
- `InterfaceAdapters.Game.Actions.SellAction/SellActionPresenter.cs:130`–`:135` sets every highlight index to 10 when no type is developable.
- `Frameworks.View.Game/IndustriesDetailsView.cs:103` hides the back button; `:154` marks merchant mode; `:230` prevents Escape in merchant mode.
- `BoardGameRules.Entities.TurnPhases/TurnFlow.cs:89` refuses to end a turn with the pending bonus.
- The engine already provides `DevelopMerchantBonus.FinishDevelop()` at `BoardGameRules.Entities.Actions/DevelopMerchantBonus.cs:42`, and the AI helper explicitly calls it when there is no development target (`BrassApplication.AI.AIHelpers/SellActionAIHelper.cs:160`). Human flow omits this branch.

**Runtime evidence:** an isolated no-eligible-tile state returns `NOTHING_MORE_TO_DEVELOP`, but applying Gloucester's bonus leaves `MerchantDevelopmentNeeded=true`; `EndTurn()` throws “You can't end turn before finishing develop merchant bonus”. This establishes the underlying blocked state; the UI trap follows from the static call chain, not an observed live-game freeze.

**Fix plan:** before presenting a development choice, enumerate legal targets. If there are none, finish the bonus using its existing event, then proceed with the industry's flip and normal Sell continuation. Reuse the same resolution logic for AI and human paths. Do not allow restricted pottery to be developed and do not silently skip a choice when a legal target exists. A general optional-skip rule is outside this finding; the official text does not explicitly call this bonus optional.

**Acceptance tests:** no tiles left; only an undevelopable pottery tile remains; at least one legal target exists; sale consuming Gloucester plus brewery beer; multiple sales per action; replay/undo resumes without leaving or double-applying the bonus flag. Add a UI integration test asserting the no-target case returns to Sell continuation and the turn can finish.

## Behaviors checked that agree with the rules

- Double rail is £15 total, one coal per rail, one brewery beer after the second rail. `NetworkAction.PlaceLink` charges £5 then £10 and adds coal per placement. Merchant beer is omitted from the Network UI/AI candidate list.
- **Requiring an opponent's brewery to connect specifically to the second rail is correct**, explicitly stated on printed p11. The code should not be changed to accept a brewery connected only to the first disconnected rail.
- A player's own beer is usable without a connection; an opponent's requires a connection. `ConsumingBeer.CanBeerBeSuppliedToSlotFromSource` implements this distinction. There is no nearest-brewery requirement.
- Board routes traverse any owned links, including opponents' links; `GetYourNetwork` consists of own industry locations and endpoints of own links. When neither exists, it allows any starting location/link as the rulebook permits.
- The southwest farm brewery is a third endpoint of the Kidderminster–Worcester link; the Cannock farm brewery has its own separate link, matching printed p9.
- Sell continuation supports multiple industries after one discard and rechecks after each sale. Zero-beer goods flip without requiring a barrel.
- Merchant setup and reset reserve beer for nonblank merchant tiles. Breweries flip and grant their owner income when the last barrel is consumed. Oxford bonus is two income spaces rather than two income levels; the core bonus values inspected are correct.

## Evidence limits

These probes use constructed, possible board configurations and original production methods; they do not establish frequency in naturally played games. The inspected decompiled code is review evidence rather than a buildable upstream project. UI findings require an integration harness or controlled live-game reproduction before declaring them fixed. Exposed engine methods also lack some direct-call input guards; those were not promoted to separate user-facing defects without a normal caller path.
