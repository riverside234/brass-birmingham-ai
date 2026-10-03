# End-game Statistics: plot and data audit

**Date:** 2 October 2026.  
**Reviewed runtime:** Installed **1.4.20**, with the accepted **1.4.19** Statistics helper. All 16 working Statistics files still match the accepted source pins.  
**Status:** Audit and repair plan only. No production source, game DLL or save data was changed.  
**Related plan:** [Light-theme and factual-chart revision, revision 3](END_GAME_STATISTICS_LIGHT_THEME_REVISION_PLAN.md).

## Evidence and scope

The review covered report collection/projection, metric queries, page assembly, numerical coordinates, tooltips, chart drill targets, filter behavior and existing tests. A read-only diagnostic ran against the **installed** `BrassPatch.dll`: **11 managed-method probes** and **2 evaluations of source-verified geometry/formatting predicates**. The 13 probes reproduced **11 finding groups**; SD05 has three probes.

The diagnostic uses independently constructed report fixtures. It executes the installed query, description and point-position methods, but it does **not** launch Unity, render new screenshots or reconstruct a complete match. SD10 and SD11 evaluate the exact source predicates rather than native pointer dispatch/rendering. Geometry findings demonstrate positions outside the declared plot area, not necessarily clipping by the outer window.

Evidence:

- [Observed expected/actual results](../artifacts/statistics-audit/2026-10-02-plan-r3/observations.json).
- [Runtime hashes before/after the diagnostic](../artifacts/statistics-audit/2026-10-02-plan-r3/runtime-evidence.json): all three installed DLLs remained unchanged.
- [Reproducer source](../artifacts/statistics-audit/2026-10-02-plan-r3/StatisticsAuditProbe.cs) and [runner](../artifacts/statistics-audit/2026-10-02-plan-r3/run-probe.ps1).
- Earlier accepted [native Statistics report](../artifacts/patch-releases/1.4.19/statistics-installed.json): **2,638 checks passed, zero failures, nine matches**. This archived qualification is not a new test run. Its checks do not establish correctness for the newly probed combinations.

The runner's successful exit means the diagnostic completed. Its `matchesExpected: false` observations deliberately record current defects; it is not a successful qualification of their behavior.

## Priorities

**Eight P2 findings** affect statistical correctness or access to matching evidence. **Three P3 findings** affect plot geometry/readability. No P1 gameplay failure was demonstrated by this Statistics-focused review. That does not constitute a general gameplay/AI audit.

All eleven findings are in scope for the requested Statistics revision. Correct them before treating the new plots as validated. The seven planned additions remain unchanged; these repairs do not add more navigation pages.

## SD01 · P2 · Two wild cards share one acquisition episode key

**Sources:** [projection: acquisition](../src/BrassPatch/Statistics/StatisticsProjection.cs#L98), [projection: use association](../src/BrassPatch/Statistics/StatisticsProjection.cs#L330), [episode table](../src/BrassPatch/Statistics/StatisticsDetailMetrics.cs#L48).

A Scout event acquires Wild Location and Wild Industry at the same revision. Both uses are associated through `wild@<revision>`, and the detail query takes the first Card record with that key and player. Card type/unique ID does not participate in the lookup.

**Reproduction:** Acquire both cards at revision 10; spend only Wild Location. Expected: one used episode and one unused episode. Actual: both episode rows show the same Build use and “Unused at finish: No.” The coarse acquired-minus-used total can remain correct while the episode detail is wrong. The two acquired rows also share their revision/player row identity.

**Repair:** Give each acquisition a card-specific episode identity, incorporating player, acquisition revision and the original card unique ID. Bind the use through that identity. Give each acquisition row a distinct UI key. Preserve reacquisitions as new episodes and keep Wild Location/Industry types explicit. Apply this to the detached Statistics projection; no saved-game format change is needed.

**Regression:** Acquire both/spend one; acquire both/spend both in either order; reacquire the same returned card; leave both unused. Reconcile episode rows with aggregate counts and recorded card payments. Add at least one collected-history Scout fixture, beyond the managed report fixture.

## SD02 · P2 · Rail filtering removes all Canal carryover from its headline

**Sources:** [build-cohort filter](../src/BrassPatch/Statistics/StatisticsPages.cs#L23), [Efficiency headlines](../src/BrassPatch/Statistics/StatisticsPlotPages.cs#L68).

`assets` is selected by **build era**. A Canal-built carried tile is consequently absent when Rail is selected, yet that same filtered array supplies the headline “Canal carryover tiles.”

**Reproduction:** One Canal tile survives and scores in Rail. Select Rail. Expected carryover count: 1. Actual headline: 0. This does not prove that the tile's awarded Rail VP is absent from the scoring ledger; the demonstrated defect is the headline's population.

**Repair:** Separate build-cohort queries from Rail-entry carryover/historical-presence queries. Count carryover at the appropriate recorded boundary, respecting player and applicable industry/region scope. Label whether an outcome represents a build cohort or a historical board population.

**Regression:** Rail selection with carried tiles, Rail-only builds, a carried tile later overbuilt/removed, introductory mode, and a genuinely empty carryover population.

## SD03 · P2 · Efficiency divides total VP by filtered actions

**Sources:** [typed action filter](../src/BrassPatch/Statistics/StatisticsPages.cs#L24), [headline ratios](../src/BrassPatch/Statistics/StatisticsPlotPages.cs#L68), [older efficiency table](../src/BrassPatch/Statistics/StatisticsPages.cs#L343).

The headline retains whole-game or era VP, while its action count and default spending can be restricted by an action-kind/evidence filter. The older table uses a different player/era query. Thus the underlying calculations disagree and the headline can imply that a subset of actions produced the entire score.

**Reproduction:** Final VP 20, two actions: a £12 Build and a Sell. Select Sell. Headline: **20.00 VP/action**. The player/era ratio is **10.00 VP/action**, with **£0.60/VP** action expenditure. The managed legacy-table query returns those latter ratios. The UI normally hides the older table under typed filters, so this is not a claim that both tables are simultaneously visible.

**Repair:** Use the planned N7 shared player/era efficiency query for the plot, headline and table. Detail action/industry/evidence filters must not silently narrow only one side of a ratio. State N7's player/era scope explicitly; apply the chosen expenditure basis consistently. Do not invent an allocation of scoring-boundary VP to individual action kinds.

**Regression:** Action-kind and asset/evidence filters; Canal/Rail/whole-game selection; multiple players; all-payments mode; zero actions and nonpositive VP. Compare each displayed ratio to its visible numerator/denominator.

## SD04 · P2 · Resource detail tables ignore plot filters

**Sources:** [typed resource-flow query](../src/BrassPatch/Statistics/StatisticsMetrics.cs#L70), [market-dependence query](../src/BrassPatch/Statistics/StatisticsDetailMetrics.cs#L40). The brewery-detail branch similarly queries entries using only era/player for several totals.

The plots apply consuming-action, source industry/region and evidence scope. Some accompanying detail tables read raw entries and apply only player/era, although the panel's visible detail context is still active.

**Reproduction:** A Network uses two free coal units; a Build buys/uses five market coal units for £10. Select Network. The flow plot correctly shows two units; the dependence table still shows **5 / 7 market units** and **£10 paid**. For that action scope it should show **0 / 2** and **£0**, or explicitly identify itself as an unfiltered player/era overview.

**Repair:** Define supported scope per measure. Reuse a typed resource/transaction query for every corresponding plot, headline and table. Where source-industry filters have no meaningful applicability to market purchases, make that scope limitation visible rather than guessing an attribution. Incoming/outgoing ownership roles must remain distinct.

**Regression:** Network/Sell/Build scopes; incoming/outgoing; source industry/region; specific and empty evidence; carried suppliers. Assert units and money against exact record membership, not just table existence.

## SD05 · P2 · Generic tooltips confuse coordinates, counts and units

**Sources:** [generic description](../src/BrassPatch/Statistics/StatisticsPlotViews.cs#L101), [generic exact-data assembly](../src/BrassPatch/Statistics/StatisticsPlotPages.cs#L49).

`mark.Y` is treated as the displayed value, paired with an axis title. Its meaning varies by plot type.

**Reproductions:**

- A city with **3 tiles**, stored at normalized map Y **0.23**, is described as “0.23 · Owned tiles.”
- One Build among four actions is drawn as **25%**, but its main tooltip line says “1 · Actions (%).” The detail line may correctly say 25%, leaving contradictory descriptions.
- An industry interval from turn 2 to 8 is described as “6 · Tile” instead of a six-global-turn duration with its endpoints.

**Repair:** Separate numerical coordinates from the semantic display measure. Supply plot-specific value/unit descriptions and exact-data columns: map quantity, percentage plus count/denominator, interval start/end/duration, action lane/time/player, and scatter cost/VP. Use the same descriptor for mouse and keyboard tooltips and exact data.

**Regression:** Every supported plot family, including map occupancy/VP, percentage bars, lifespans, event lanes, signed waterfalls and the new ratios. Check formatted quantities and their units, not only finite mark coordinates.

## SD06 · P2 · Link-contribution map opens unrelated industry evidence

**Source:** [map mark target](../src/BrassPatch/Statistics/StatisticsPlotPages.cs#L145).

Every map mark opens Industries with keys for the current city's owned assets, regardless of the selected map measure. A player's link can earn points from other players' tiles or merchants while that player owns no industry there.

**Reproduction:** Birmingham contributes 4 link VP with no selected-player asset in the city. Clicking the mark targets **page 2 (Industries)** with an explicitly empty asset-key set. Expected: the corresponding link/contribution records on Networks.

**Repair:** Build drill targets from the displayed measure. Link contributions use the represented link IDs/contribution evidence; industry scoring uses awarded asset evidence; occupancy uses boundary-present asset evidence. Retain empty scope only when the selected mark truly represents zero records.

**Regression:** Own/opponent/merchant contribution, city with no owned asset, different boundaries, and an actual zero-valued city. The clicked total must reconcile to the opened records.

## SD07 · P2 · Map link totals ignore selected evidence

**Source:** [link map aggregation](../src/BrassPatch/Statistics/StatisticsPlotPages.cs#L142).

Route plots restrict by link evidence, but the map's contribution query ignores that restriction. Industry/action scope also needs an explicit applicability policy for this map mode rather than silent omission.

**Reproduction:** Link A contributes 4 VP at Birmingham; Link B contributes 10 at Walsall. Restrict to Link A. The route view selects A, but the map still shows **14 VP**, including Walsall's **10**. Expected represented contribution total: 4.

**Repair:** Resolve represented links/contributions once and use that membership for route/map totals, legends, headlines and drill targets. Preserve city/player/era/boundary meaning; require an explicit scope explanation for unsupported filters.

**Regression:** One/multiple/empty selected links; region and era scope; merchant endpoints; hidden series. Map contribution sums must match the selected contribution ledger.

## SD08 · P3 · Grouped columns extend outside their numeric category range

**Sources:** [column offsets](../src/BrassPatch/Statistics/StatisticsPlotViews.cs#L62), [column width](../src/BrassPatch/Statistics/StatisticsPlotViews.cs#L83), [axis bounds](../src/BrassPatch/Statistics/StatisticsPlotViews.cs#L137).

The categorical X range uses the first/last **centers**, then adds player offsets and bar widths without category-end padding.

**Reproduction:** Two-player/two-round plot with centers 0 and 1. The first group's outer column starts **8 pixels left of the declared plot area**. The final group has the symmetric problem. This can intrude into axis gutters even when the outer UI card does not clip it.

**Repair:** Allocate full category bands with half-band padding at both ends, deriving mark centers, grouped widths and ticks from the same categorical layout. Keep all group geometry inside the chart area.

**Regression:** One/two/many rounds; two/three/four players; default/largest text; 1080p/1200p. Check complete bar bounds and category-label alignment.

## SD09 · P3 · Overlap offsets push scatter points beyond the scale

**Source:** [scatter displacement](../src/BrassPatch/Statistics/StatisticsPlotViews.cs#L63).

Each overlapping point moves right/up by a positive offset without boundary-aware placement.

**Reproduction:** Four points at the maximum cost/VP coordinate. In a 300-pixel-high plot, the fourth center lands at **Y=303**, above the plotted maximum. More coincident points produce larger excursions.

**Repair:** Use bounded symmetric displacement or a cluster with an expandable exact-record list. Retain true data coordinates in axes, guides, tooltips and records; reserve symbol-radius space as needed. Do not increase a tile's apparent VP through unbounded jitter.

**Regression:** Repeated points at all four corners and at zero; dense overlap; hidden series; consistent selected-record access and unchanged raw coordinates.

## SD10 · P2 · Most of a wide heatmap cell cannot be clicked

**Source:** [hit predicate](../src/BrassPatch/Statistics/StatisticsPlotViews.cs#L95).

Heatmap hit testing uses a circle with radius `min(cellHeight/2, cellWidth/2)` around the center. Bar/interval hits likewise rely on proximity to the mark point rather than their full shape.

**Source/geometry reproduction:** A **300 × 50** cell has a **25-pixel** hit radius. A click 100 pixels horizontally from its center is inside the cell but fails the hit predicate. Native pointer dispatch was not exercised by this diagnostic.

**Repair:** Use the rendered rectangular cell/bar bounds and interval segment bounds for hit testing. Respect padding, viewport clipping and hidden series. Make mouse and keyboard evidence selection agree.

**Regression:** Native clicks at cell center, near every edge/corner and outside the cell; long bar middles and interval spans; horizontal scrolling; overlapping points. Clicking a visible cell must select its own record, not a nearby point's record.

## SD11 · P3 · Count axes display fractional counts

**Source:** [fixed quarter ticks](../src/BrassPatch/Statistics/StatisticsPlotViews.cs#L152).

The renderer divides every range into quarters and applies one-decimal formatting without a unit-specific tick model.

**Source/formatter reproduction:** A count range 0–1 produces **0, 0.3, 0.5, 0.8, 1**. Fractions of tiles/actions are not valid count tick labels. Fixed boxes and ellipsis further reduce numerical clarity, already addressed by revision 2.

**Repair:** Implement the approved shared numerical-axis specification: integer count/time steps, appropriate ratio/price precision, rounded bounds, measured label gutters and readable numbers. Category centers and mark positions must use those same scale results.

**Regression:** Integer ranges 0–1, 0–3 and 0–23; signed VP/income/cash; decimal prices/ratios; all-zero/single-value ranges; percent 0–100; dense categories and both display resolutions.

## Follow-up checks without an established defect

These should be tested during implementation, but are **not** additional counted bugs:

1. **Historical industry-VP map semantics:** the current map deliberately reports points from assets present at the chosen boundary. Consequently, a previously scored but removed tile is absent from the final map. Decide and label whether the revised measure means “VP from boundary-present assets” or “cumulative awards by city”; use separate queries if both are offered. Do not call absence of a removed tile a scoring error by itself.
2. **Loan recovery across eras:** the recovery query can find a later Rail recovery for a Canal loan. This may be a valid whole-match episode; the era selector must state whether it filters loan acquisition or caps the recovery observation window. Add overlapping-loan and never-recovered fixtures.
3. **Missing recorded turn positions:** `IndexOf + 1` can produce a synthetic 0th position if a player is absent from a recorded order. Normal reconstructed complete rounds should contain every player. Verify malformed/partial-data behavior before asserting this occurs in ordinary games; the planned N6 query must represent absence as unavailable.
4. **Scope propagation through new local views:** Supply/Market/Merchants controls and N7's cost basis are not implemented yet. Preserve applicable state, explicitly clear incompatible evidence, include all semantic scope in cache keys, and test repeated navigation.

## Implementation order

1. Correct SD01 identity and SD02–SD04/SD07 query populations; build shared typed queries and semantic value descriptors.
2. Correct SD05/SD06/SD10 descriptions and evidence access. Create independently expected fixtures for each finding.
3. Implement SD08/SD09/SD11 through the axis/category/selection foundation, with native geometry and pointer checks.
4. Build the seven new plots on the corrected data, remove advice and apply the isolated light theme.
5. Run the existing qualification suite plus the new independent semantic and native interaction checks. Preserve the latest accepted gameplay/AI/save runtime and Card Tracker defaults.

**Limits:** No fresh native rendering or full-game event qualification was performed in this planning audit. The installed-method reproductions and source analysis establish the cases above; implementation must still pass the specified collected-history and native UI regressions before release.

## Implementation verification — patch 1.4.21

The original audit above remains the pre-repair record. **SD01–SD11 are repaired and installed** in 1.4.21. [Release notes](../docs/STATISTICS_LIGHT_THEME_1.4.21.md) link the implemented chart changes, calculation scopes, binary fingerprints and recovery information. Independent typed fixtures, nine reconstructed completed matches and native Unity interactions passed **5,301 Statistics checks**. The installed game repeated the same checks successfully; no native errors were reported.

The implementation also checks overlapping/unrecovered loans and later-era recovery, explicitly labels VP from boundary-present map assets, represents missing turn positions as unavailable, and exercises Supply/Market/Merchants cache/scroll transitions. Native rectangular cell/bar and interval clicks include scrolling; category-bound and scatter-cluster tests retain exact coordinates. Axis, legend and endpoint layout checks cover 1200p/1080p and the text presets. The game and AI DLLs remain identical to accepted 1.4.20, with unaffected helper methods and source pins verified.

Verification reports and captures are retained in `artifacts/patch-releases/1.4.21`. The planning-only limit above describes the historical audit, not the installed release's qualification.
