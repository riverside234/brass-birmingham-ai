# Build, development, and resource audit findings

Reviewed the decompiled `BoardGameRules` domain and the human-facing Build/Develop/IronDelivery use cases against the [publisher's official rulebook](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281), retrieved from [Roxley's product page](https://roxley.com/products/brass-birmingham). Printed page numbers below are the numbers printed on the spreads, not PDF sheet numbers. This is a review and implementation plan; no game code was changed.

The two defects below were reproduced by calling the installed `Assembly-CSharp.dll` in an isolated process with deliberately constructed rule states for **2, 3, and 4 players**. These are domain/predicate reproductions, not observations of clicks in the running Unity UI. The use-case source shows that the human UI calls these predicates. Both fixtures use the rail era; the faulty helpers are shared with canal-era builds.

Evidence: `build_resources-probe.ps1`, `build_resources-results.json`. The script checks the DLL SHA-256 (`600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB`) and does not read saves or modify game binaries. Run from the game directory with `powershell -NoProfile -ExecutionPolicy Bypass -File rules-audit/build_resources-probe.ps1`.

## BR-01 — Legal two-coal builds are rejected when the nearest mine empties

**Priority:** P1; confirmed installed-DLL behavior. A legal move can disappear, altering both human and AI choices.

**Rule:** Printed p. 8, “Consuming Coal”: after the nearest mine empties, the next closest connected mine supplies further required coal without payment.

**Code:** `ai-research/decompiled/BoardGameRules.Entities.Actions/BuildAction.cs:621-634`, `CostOfCoalForGivenIndustry`; callers at `421-429`, `464-480`. `ai-research/decompiled/BoardGameRules.Entities.Consumables/ConsumingCoal.cs:88-116,142-155` correctly identifies the *next* coal source, but is unsuitable for counting *total* connected coal. UI entry: `BrassApplication.UseCases.Game.Actions.BuildAction/BuildActionUseCase.cs:41-49,75-87`.

The cost estimator filters all mines through `CanCoalBeSuppliedToRegionFromSource`, so it only counts mines tied at the shortest current distance. It never simulates exhausting that distance group. It treats remaining requirements as market purchases: without a merchant connection it returns `-1`; with a merchant connection it adds a needless market price. The latter also falsely rejects a player who can pay the tile cost but not the imaginary purchase.

**Reproduction:** Prepare Pottery III as the player's next pottery, £22, a Coventry location card, and an empty Coventry pottery space. Put one remaining coal on an opponent's Coventry mine and one on an opponent's Nuneaton mine. Place the Coventry–Nuneaton rail and no rail to a merchant. There is no own industry in Coventry. Request the pottery build.

**Observed in all three player counts:** `CanUseBuildActionForGivenSlot` returns false, and `GetCantUseBuildActionReasonForGivenSlot` returns only `NO_COAL`. In the same state, directly executing the Build domain action followed by supply of one cube from Coventry and then Nuneaton succeeds: `NeededCoal=0`, money becomes £0, and both mines flip. This establishes that the availability check contradicts the actual consumption code.

**Fix plan:**

1. Separate “currently selectable coal sources” from “coal available over this entire action.” Keep closest-source filtering for the delivery UI.
2. For build affordability, total all nonempty mines connected to the target region (including the target itself), then calculate the market shortfall. Validate merchant connectivity only for a positive shortfall. All board coal is free, so a distance-ordered simulation is unnecessary for the price calculation; use simulation if sharing a more general action planner.
3. Have the boolean availability, explanatory reasons, and AI build candidate generator use the same estimator.
4. Preserve the existing consumption ordering and flip/income events; this fix must not permit taking farther coal while nearer coal remains.

**Regression/acceptance cases:** two connected mines at distances 0 and 1 with one coal each; distances 1 and 2; two mines tied nearest; one nearest mine containing both cubes; disconnected farther coal; only one connected coal plus merchant shortfall; market empty at £8; all board coal sufficient with a merchant connection and exactly tile-cost cash. Run against Manufacturer III and Pottery III/V and all player counts. Include a one-coal build control and preserve source-selection ordering after each depletion.

## BR-02 — Overbuilding an occupied dual-icon space incorrectly requires an empty single-icon space

**Priority:** P2; confirmed installed-DLL behavior. Legal upgrades can be blocked.

**Rule:** Printed p. 9, Build step 2, applies the single-icon preference to placing on undeveloped spaces; printed p. 10, “Overbuilding,” permits replacing one's own industry with a higher-level tile of the same type. Printed p. 12 repeats the own-industry overbuild allowance. The preference should not redirect an overbuild to a vacant space.

**Code:** `ai-research/decompiled/BoardGameRules.Entities.Actions/BuildAction.cs:263-270` (`IsGivenSlotTheBestOneInGivenRegionForGivenTypeStatic`) and `603-610` (`IsSingleSlotOrDoubleSlotButLastPossibleStatic`). Both check for empty single-icon alternatives without first distinguishing an occupied overbuild target. Call sites: `77-80` (execution throws), `414-417` (boolean availability), `502-505` (reason), `531` and `550` (slot enumeration). `CanBeOverbuild` independently permits the upgrade, explaining the contradiction.

**Reproduction:** In the rail era, have one's own Coal II in Cannock's Manufacturer/Coal space, with Cannock's single Coal space empty. Such a board can arise when another player's Coal I in the single space is removed at the end of canals. Prepare Coal III as the next coal tile, hold a Cannock location card, and have sufficient cash for the tile and market iron. Attempt to overbuild the existing Coal II.

**Observed in all three player counts:** the overbuild-slot predicate includes the existing mine, but the full build predicate returns false with `FIRST_BUILD_IN_SINGLE_SLOT`. Direct execution throws `Player should build on space displaying only that industry's icon.`

**Fix plan:**

1. Centralize placement validation and explicitly branch between an empty-space build and occupied-space overbuild.
2. Apply the single-icon preference only to empty-space builds. For occupied targets enforce same industry type, strictly higher level, card/network permission, era restrictions, costs, canal location limit, and the opponent-resource restrictions.
3. Route the two duplicated preference helpers and all slot enumerators through the shared validation; fixing only the highlight predicate leaves the execution exception intact.

**Regression/acceptance cases:** Cannock Coal II→III with a vacant single Coal space; Coalbrookdale Iron II→III in its Iron/Brewery space with the single Iron space vacant; full versus empty markets when overbuilding an opponent; same/lower-level rejection; wrong-type rejection; own unflipped stock discarded without an income reward; previously earned income preserved; overbuilt tile excluded from later scoring. Keep a control showing that a *new* industry still must use an empty single-icon space before an empty dual-icon space.

## Suspicions checked and not promoted to gameplay findings

- **Develop ignores its type argument:** `DevelopAction.cs:65-77` checks `IronWorks` regardless of `buildingType`, but the only production caller found (`BrassApplication.AI.RuleBasedSystem.Rules/ForceDevelopForIronOverbuild.cs:33`) passes `IronWorks`. The human Develop presenter calls `PlayerMat.CanIndustryBeDeveloped` with each actual type. Treat the unused generality as cleanup, not evidence that other industries cannot be developed.
- **Iron market before existing iron works:** `ConsumingIron.SupplyIron` lacks its own board-first guard, but `IronDeliveryUseCase.StartIronDelivery` exposes the market only if there are no unflipped iron works. No ordinary human UI path bypass was demonstrated. Add the guard in a domain-validation hardening pass only after checking all callers, and do not claim an observed UI defect.
- **Overbuilding opponents when their markets still contain cubes:** `Board.GetAmountOfCoalOnBoard` and `GetAmountOfIronOnBoard` use a binary positive-market indicator plus industry stocks, which looks odd but correctly answers the zero/nonzero question used by overbuild validation. Coal's highest on-track price is £7 and external supply £8; iron's are £5 and £6.
- **Build anywhere with an industry card when one's network is empty:** `Board.GetYourNetwork` returning all regions in that case agrees with the printed p. 9 exception. Wild Location correctly excludes the two farms; industry cards can use them.
- **Canal limit on own overbuild:** the full build predicate ignores the occupied target when checking the one-industry-per-location rule, allowing replacement rather than counting it as a second industry.
- **Sell newly produced resources before paying:** execution spends the tile cost and resolves required resources before moving a new mine/ironworks' stock to the market, consistent with the build sequence. Coal market connections intentionally include merchant locations without active merchant tiles.

## Scope limits

No claim of exhaustive rule correctness: this sub-audit did not independently validate every printed tile statistic, UI animation or interaction timing, online synchronization, save migration, or platform behavior. The synthetic fixtures establish the logic defects without establishing their frequency in natural play. Existing decompiled source was read without modification; only new audit scripts/results were written.
