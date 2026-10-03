# End-game statistics: plots, tactical review and larger entry controls

**Date:** 1 October 2026.  
**Status:** Core stages A–G implemented, qualified and installed as **1.4.17**. Stage H remains optional.  
**Planning baseline:** Qualified patch **1.4.13**; the findings below describe that original helper implementation.  
**Implementation baseline:** The then-current installed, qualified **1.4.16**. Its game and AI DLLs are retained byte-for-byte. The unchanged helper's verified 1.4.13 sources were reconstructed, then only the Statistics sources were revised. See the [1.4.17 release notes](../docs/END_GAME_STATISTICS_1.4.17.md).

**Revision requested 2 October 2026:** The Match review/advice requirement and visual design below are superseded by the [light-theme and factual-chart revision plan](END_GAME_STATISTICS_LIGHT_THEME_REVISION_PLAN.md). The user requested removal of tactical verdicts and improvement suggestions, a light panel, clearer existing plots and additional plots from the current tables. This note does not change the historical implementation record; the new revision is planned, not implemented.

### Implementation record

| Stage | Delivered in 1.4.17 |
| --- | --- |
| A | Verified the accepted 1.4.16 runtime and 1.4.13 helper-source archive; pinned units, scope, boundaries, coverage and original page identities |
| B | Enlarged both Statistics entries, retained physical sizing and modal destinations, added default Match review and collapsed/paged ledgers |
| C | Added an explicit plot model, visible labels/scales/legends, typed drill targets, exact-value tables, keyboard access and cached analytical views |
| D | Added the eleven-page primary plot matrix, grouping with exact Other membership and Show all, incoming/outgoing resource views, and historical geography with full city labels |
| E | Added typed prices, own-turn positions, mat/stock/cash boundaries, link contributions and flip triggers; added delay, batching, recovery, transition, resource, merchant and late-cash analyses |
| F | Added deterministic evidence-backed observations and next-game experiments, named comparison players, whole-match review, and missing-history/introductory safeguards |
| G | Passed native metric, UI and regression checks, atomic installer/rollback fixtures and checks against the installed files; retained the accepted AI/gameplay/save implementation and offline mode |
| H | Optional sale-feasibility probes, exact path participation and richer flow diagrams are deferred; no such conclusions are claimed |

Two pre-existing statistics errors were also corrected: action/round spending omitted market purchases, and payday records incorrectly treated the all-player income event as actor-only. Directional resource matrices, bars, headlines and source evidence now share one query and include counterpart players correctly.

Qualification: **2,793 statistics**, **539 Card Tracker**, **184 presentation**, **765 link replay**, **259 startup/menu** and **54 installer** assertions passed. The statistics and menu suites also passed against the installed release. Native captures cover all eleven pages at the three planned resolutions and text sizes. Collection, page/filter response and scroll-layout measurements are retained with hardware metadata. The release archive is `artifacts/patch-releases/1.4.17`; its manifest pins the installed binaries, compiled source snapshot, reports, tests, scripts and 119 native captures. Optional stage H is outside the completed core scope.

## 1. Intended result

Make the Statistics entry easier to see and click. Change the panel from a sequence of long ledgers into an interactive, plot-first review. Add a **Match review** page that explains each player's strengths, areas to investigate and practical experiments for the next game, supported by recorded evidence.

Keep the existing ten pages and their detailed records. Add Match review as an eleventh page, displayed first in navigation and opened by default. Players should be able to understand their result from the summary, compare strategies through readable plots, and inspect exact actions when they want more detail.

This extends the [original statistics plan](END_GAME_STATISTICS_UI_PLAN.md) and [readability release](../docs/STATISTICS_READABILITY_1.4.10.md). It applies to completed local games with **2, 3 or 4 players**, including introductory games. Card Tracker's four-player restriction does not apply to statistics.

### Completion criteria

- Both Statistics entry buttons are substantially larger and remain clear of existing controls.
- Each analytical page starts with a compact explanation, relevant headline values and at least one useful plot, rather than a long table.
- Charts have explicit axes, units, category labels and meaningful legends. Understanding a chart does not require hovering.
- Large datasets are grouped, with reversible drill-down into the contributing records.
- Match review provides evidence-backed strengths and improvement suggestions for any selected player. Missing or ambiguous information does not produce a tactical verdict.
- Existing scoring, cash accounting, filters, modal routing, large-text settings and completed-game lifecycle continue to work.

## 2. Findings in the installed implementation

The reviewed working statistics sources match the source fingerprints in the 1.4.13 archive. Existing native screenshots from the readability release confirm the table-heavy presentation.

| Current behavior | Evidence | Planned response |
| --- | --- | --- |
| Board-review Statistics button is 180 × 48 physical pixels with 20-pixel text; results button is 168 × 44 | `StatisticsEntryPoints.Ready` and `AttachResults` | Enlarge both entries and verify their placement in real finished-review/results prefabs |
| Ten pages contain extensive industry, city, resource, action, economy and transition records | `StatisticsPages` | Preserve these records, but show grouped plots before expandable ledgers |
| Plots are restricted to bars/stacked bars and step lines | `StatisticsChart` has `Lines` and `Stacked` flags; `StatisticsChartGraphic` | Introduce an explicit plot model and several appropriate renderers |
| Plot topics are score composition, cash, income, action counts and awarded VP; score composition repeats on two pages | `AddScoreChart`, `LineChart`, `Actions`, `Timeline` | Add useful plots for industries, resources, networks, timing and conversion; avoid repeating the same overview |
| The chart layout primarily shows minimum/maximum values and generic axis text | `StatisticsPanel.Chart` | Add graduated ticks, direct category labels, axis titles, distinct units and structured legends |
| Wide tables can show 60 rows before another pageful is requested; charts can be pushed far below the initial view | `StatisticsPanel.TableView` and archived screenshots | Default to collapsed ledgers and short grouped summaries; retain full records on demand |
| Search filters tables while charts retain the player/era selection | `StatisticsPages.Build` and `StatisticsPanel.Render` | Introduce typed analytical filters shared by plots, totals and their evidence tables; label text-only ledger search separately |
| Report fields already include tile costs/timing/VP, links, confirmed actions, suppliers, cash changes, round order and city snapshots | `CompletedMatchStatistics`, `StatisticsProjection` | Reuse this data first; add only the typed fields needed for reliable new analyses |
| Merchant destination can be recorded, uniquely inferred, or unattributed | `StatisticsProjection.Flip` | Preserve those distinctions in plots, coverage counts and summary advice |

The relevant implementation files are [entry points](../src/BrassPatch/Statistics/StatisticsEntryPoints.cs), [panel](../src/BrassPatch/Statistics/StatisticsPanel.cs), [page definitions](../src/BrassPatch/Statistics/StatisticsPages.cs), [widgets](../src/BrassPatch/Statistics/StatisticsWidgets.cs), [report model](../src/BrassPatch/Statistics/CompletedMatchStatistics.cs), [projection](../src/BrassPatch/Statistics/StatisticsProjection.cs) and [collector](../src/BrassPatch/Statistics/StatisticsCollector.cs).

## 3. Larger Statistics buttons

| Element | Proposed initial size | Appearance |
| --- | --- | --- |
| Finished-board entry | **240 × 64 physical pixels** | 24-pixel label, approximately 28-pixel chart icon, comfortable icon/text separation, rounded corners and a clear brass outline |
| Results-dialog entry | **240 × 60 physical pixels** | Same label/icon treatment and visible hover/focus states |

Measure these dimensions in actual display pixels, using the active canvas/camera and safe area. Avoid relying solely on reference-resolution units or `Screen.width` when rendering to a different-sized target.

Reserve at least a 20-pixel safe-area inset and a 12-pixel gap from neighboring controls. Position using the full transformed bounds of nearby review buttons, visible hand/market art and other finished-review controls. Enlarge/reflow the results popup's reserved button area if necessary; do not place the enlarged control over Review, Exit or score rows.

Retain the existing reveal point after final replay/scoring completes. Opening from results temporarily hides results, and closing returns to that entry point. Opening from board review returns there. Keyboard focus, Escape and modal input behavior remain consistent. The larger hit target must not activate underlying controls through its edges.

## 4. Panel structure and navigation

Use eleven stable page identities: the existing ten plus Match review. Preserve the current identities of Timeline and the other drill-down targets; inserting a page visually must not shift numeric targets accidentally. Replace hardcoded `10` values, page-index arrays and category-specific index checks with a shared page registry and state keyed by stable page ID.

| Page | First question it answers |
| --- | --- |
| **Match review** — new, default | What worked, what limited my result, and what should I try next? |
| Overview | How did the players' overall results compare? |
| Scoring | Which eras and categories explain the score difference? |
| Industries | Which industries converted investment into income and scored VP? |
| Networks & Geography | Which links and locations supported my result? |
| Economy | When did cash and income support or constrain my plan? |
| Resources & Trade | Who supplied whom, and how much did the market cost? |
| Actions & Cards | How effectively did I use actions and card flexibility? |
| Turn Order & Tempo | How did spending and turn position evolve? |
| Efficiency & Transition | What carried into Rail, and what remained unfinished? |
| Timeline | Which recorded decisions and phase boundaries deserve review? |

### Plot-first page layout

Each page presents: short explanation → 3–5 headline metrics → 1–3 primary plots → concise observations → collapsed evidence ledgers. The first useful plot should be reachable immediately, without scrolling through dozens of records.

Use two plot columns only when both plots retain readable axes and legends; stack them at narrow widths or larger text sizes. Keep the current large default text and A−/A+ choices. Recommended minimums are 21-pixel body text, 18–19-pixel axis/legend labels, 24-pixel plot headings and 44-pixel interactive legend/filter targets.

Replace a permanently expanded ledger with **Show data · N records**. Opening reveals its table, sorting, search and details. Large ledgers use a bounded visible row window or explicit pagination; no records are discarded. Keep an equivalent table available for every plot, including keyboard access to exact values.

Remember page scroll, expanded ledgers, typed filters and plot choices when closing/reopening the same match. Reset them for a different game ID. Preserve return-to-results and close-to-board destinations.

## 5. Plot choices and page coverage

Variety should make the data easier to understand. Use different forms for different questions, rather than changing the shape of the same count everywhere.

| Page | Primary plots | Labels, grouping and interpretation |
| --- | --- | --- |
| Match review | Score-gap **waterfall**; six-industry conversion **bullet bars** | Waterfall compares the focus player with a named reference. Bullet bars show flipped / built counts, with the denominator and unconverted portion visible |
| Overview | Horizontal **stacked score bars**; Canal/Rail **dumbbell comparison** | One named row per player, score-category legend and net total. Dumbbell endpoints show era VP; introductory mode omits Rail |
| Scoring | Player × industry **heatmap**; era/category gap **waterfall** | Cell values are actual awarded VP. Group six industries and use separate Canal/Rail panels. Include merchant bonuses, penalties and introductory additions in reconciliation |
| Industries | Built/flipped **paired bars**; build-to-flip **dot distribution** | Six industry groups, faceted by player when crowded. Individual delay dots plus median; a box summary is optional only with enough observations |
| Networks & Geography | Sorted route **lollipop plot**; independent **mini-map** | Route labels, era and actual VP. Map toggles occupancy, industry VP or link contributions, with distinct legends and scoring-boundary selection |
| Economy | Cash/income **step plots**; cash-movement **waterfall** | Separate £ and income-level axes; loan/flip/payday annotations and a Canal/Rail divider. Waterfall reconciles opening balance, receipts, payments and closing balance |
| Resources & Trade | Supplier → consumer **heatmap**; resource-source **stacked bars** | Select coal, iron or beer. Matrix cells show units with named row/column owners; market and merchant sources remain separate |
| Actions & Cards | Action-share **100% bars**; sale-batch **histogram**; optional focus-player **donut** | Actual confirmed actions, including Pass. Histogram counts Sell actions by number of industry tiles sold. Donut only for a single player with a short labeled category list |
| Turn Order & Tempo | Round × player **seat heatmap**; round spending **bars** | Cells show 1st–4th seat as applicable, with era/round labels. Mark observed last-to-first sequences; spending stays in £ |
| Efficiency & Transition | Tile cost vs. actual VP **scatter plot**; carryover/unfinished **stacked bars** | £ construction + bought resources on x, actual scored industry VP on y. Facet by player, use industry icons/colors, and label carryover outcomes |
| Timeline | Confirmed-action **event lanes**; selected-industry **lifespan chart** | One lane per player; actions aligned to recorded turns. Industry spans run from build to flip/removal/scoring, with unflipped endpoints explicit |

### Plot contracts

1. Every plot has a title, definition, units, visible tick/category labels, sample size where relevant, and an adjacent legend. Provide direct labels for player rows and small numbers of marks.
2. Category counts and money bars begin at zero. Signed waterfalls/penalties show both sides of zero. Step plots use an honest, visible scale; compare facets using the same scale unless a clearly labeled independent scale is selected.
3. Player colors always identify players. Industry colors/icons identify industry type. Resource colors/icons identify coal, iron and beer. If a plot combines ownership and type, use facets or marker shape so the legend is unambiguous.
4. Add line styles, marker shapes, hatching or text alongside colors. Heatmaps print cell values and their units; their intensity scale must be visible. Avoid using an ownership color as a separate good/bad verdict.
5. Legends wrap or stack with readable spacing. Clicking them changes visibility, not the authoritative player filter. State whether totals still include hidden series; default to fixed scales so toggling a series does not misleadingly exaggerate remaining data.
6. Hover/focus provides exact values and definitions; clicking a mark opens its contributing evidence through stable record IDs and typed filters. Hover text alone must not contain essential labels.
7. Empty data, zero denominators and unavailable records produce explicit states. Unflipped tiles do not become zero-day observations in delay plots. Coincident scatter points show a count or deterministic display offsets, without changing values or using gameplay RNG.
8. Timeline axes distinguish global match turns, the focus player's own turns and era rounds. Label the selected measure. Keep multiple operations in one turn available in detail; downsampling must retain extrema and highlighted events.
9. A donut uses mutually exclusive categories and shows counts plus percentages. Do not use it for signed score changes, incomplete totals or comparison across four tiny charts. Avoid 3D, dual-axis cash/income charts and arbitrary strategy radar scores.

## 6. Grouping large datasets

| Dataset | Default grouping | Drill-down |
| --- | --- | --- |
| Industry records | Six industries → player → era | Level, city, individual tile and build/flip evidence |
| Resource events | Resource → supplier/consumer → purpose | Source tile, action and revision |
| City/build records | Actual board **region color** → city → industry | Individual slot/tile, selected boundary and route |
| Player comparisons | Stable ownership color/name, with 2–4 rows/facets | Focus one player or compare two |
| Market movements | Resource → era/round | Individual purchase/sale, quantity, cost and quote change |
| Action events | Player → era/round → action type | Confirmed action frame, cards and ordered operations |
| Links | Era → player → route | Actual scoring locations and contributions |
| Merchant outcomes | Merchant location → accepted industry/bonus | Recorded or inferred destination, actual benefit and source action |

Use `Region.Color`/`RegionColor` for the board's existing color groups, rather than treating player ownership as geography. Collect a value-only city-to-region mapping from the detached board. Keep farm breweries and merchant locations identifiable; do not assume that every location has a city card or identical slots.

For an optional Black Country/geographic view, list its member cities explicitly and call it a display grouping. It does not create a scoring region or change connectivity.

Cap an initial dense route/city ranking at about eight groups, with a labeled **Other** group and **Show all**. Other must preserve its exact total and member IDs. Unknown/unattributed is separate from Other. Full six-industry and 2–4-player comparisons should not need arbitrary truncation.

Resource filters require explicit direction: **consumed by selected players** versus **supplied by selected players**. A selected brewer's outgoing supply must include opponents who consumed it even if they are outside the focus selection. Matrix counterpart rows/columns can remain visible as labeled context. Use the same directional scope for the plot, totals and evidence table.

## 7. Additional Brass-specific statistics

The existing panel already covers loans, flip rate, era scoring, industry types, resources, cards, turn order, merchant bonuses, carryover, unflipped investment and VP/action. The following additions deepen those analyses instead of duplicating their tables.

**Availability:** **Existing** means derivable from current typed report records. **Extension** means the detached history projection needs a small additional typed record or snapshot. **Optional** means it requires a separately validated read-only feasibility analysis and should not hold up the core panel.

| Addition | Definition and presentation | Availability / caveat |
| --- | --- | --- |
| Score-gap decomposition | Focus-player minus reference VP by era and scoring category; positive and negative contributions sum to the authoritative final VP difference, including opening VP where applicable | Existing. Explains arithmetic, not why the player lost |
| Sale batching | Sell-action histogram; tiles sold / Sell action; largest batch; number of multi-tile sales | Existing `StatisticsAction.Goods`. Count industry tiles, not crates or printed VP. No Sell actions gives unavailable ratio |
| Completion delay | Median and individual own-turn/global-turn delays, separated by industry; unflipped tiles shown separately at the boundary | Existing global timing; add explicit own-turn positions for the own-turn view. Identify the small sample size |
| Early income usefulness | Actual positive income paid in each era, when income gains occurred, and the number of subsequent actual income-payment rounds | Existing balance entries/samples. Do not invent income after the final round or claim hypothetical additional earnings |
| Loan sequence | Loan amount and observed income drop; actions until income recovers to its pre-loan level; intervening recorded income payments | Existing + typed sequence positions. Recovery is an income comparison, not repayment of a loan; an unrecovered late loan may still be effective |
| First-round Rail plan | Cash/income at Rail entry, carried tiles, and first two recorded Rail actions | Existing samples/assets/actions, with an explicit boundary snapshot preferred. Present observed sequence, not inferred intent |
| Mat readiness for Rail | Industry types whose next mat tile is Rail-ineligible at the transition; later development clearing those tiles; later builds in that industry | Extension: actual mat snapshots and tile eligibility, including type-specific exceptions. Do not equate every low-level tile with a blocker |
| Development follow-through | Types/levels developed, the next available level, and whether a later build in that industry followed before the selected boundary | Extension for mat tile identity/next level. Developing removes tiles; those removed tiles are not expected to score |
| Last-resource flip triggers | Which actor/source operation depleted each resource industry, enabling it to flip; distinguish self, opponents and automatic market sale | Derivable where the current resource/flip records correlate exactly; add typed trigger attribution for uncertain cases. Flipping enables scoring and earns income; it is not itself an immediate industry VP award |
| Brewery supply balance | Own/opponent/merchant beer consumed; brewery beer supplied to each opponent; barrels remaining/removed; Sell versus double-rail use | Existing. A quantity award such as most beer supplied is descriptive and permits ties |
| Market dependence and price | Market-bought units / all consumed units; £ per purchased unit; distributions of actual paid prices; quote changes by player | Counts/quotes existing; add typed transaction price. Resource cash must reconcile and be counted once |
| Recorded network use | Count recorded coal, opponent-beer and merchant connections using/requiring opponent links; selected-link participation if traceable | Existing endpoint reliance; optional path-use extension. Multiple possible paths prevent claiming a particular link was necessary |
| Canal investment outcome | Flipped level-2+ carryover, subsequent Rail VP, unflipped carryover later converted, and carryover removed before Rail scoring | Existing assets, evaluated at explicit boundaries. Distinguish two scoring occurrences from two industries |
| End-of-era unfinished goods | Count, type/level/city, construction cash, bought inputs, nominal printed VP and time since build for unflipped sellable tiles | Existing identities/costs plus boundary state. This is unconverted investment, not proof the nominal VP was attainable |
| Supplier investment still unfinished | Unflipped coal/iron/beer industries, stock remaining, useful resources supplied and market receipts | Existing. Do not classify these as wholly wasted because they did not flip |
| Endgame cash conversion review | Final cash alongside last own turns, actual spending and VP sources; show any tie-break role | Existing. Cash alone is not a tactical failure; introductory scoring differs |
| Wild-card follow-through | Acquisitions, actual build/payment use, unused episodes, and time from Scout to use | Existing episode references; add typed acquisition/use timing if needed. Do not call an unused wildcard proof that Scout was unnecessary |
| Merchant benefit profile | Tiles sold by merchant/industry, recorded bonuses and their cash/income/development/VP effects, with attribution coverage | Existing. Keep original units; an income/development bonus is not converted into a made-up VP value |
| Debt and recovery cost | Actual negative-income payments, liquidation proceeds, assets removed and recorded VP penalties, aligned to the economy timeline | Existing records; add explicit settlement links where needed. Separate debt payment from action spending for turn order |
| Individual sale readiness | At a selected historical boundary, whether a particular unflipped goods tile had a matching reachable merchant and enough legal beer; list identified constraints | Optional. Probe a detached state using validated rules. Individual feasibility does not prove every pending sale could be combined, or that the same action was available earlier |

### Units and attribution rules

- Use actual confirmed action counts for VP/action, including Pass. Scout's extra card trades and merchant development are not extra actions. Do not ship a fixed 31, 32 or 36-action denominator.
- Keep **built in an era**, **flipped by a boundary**, and **scored in an era** distinct. An asset built in Canal can flip or score later in Rail.
- Industry printed VP is separate from earned VP. Do not credit the same industry's score to both Build and Sell and then add those credits as independent action returns.
- Cost/VP identifies its spending definition. Gross industry cash and purchased-resource cost are measured; free cubes are reported in units, without invented cash prices. Net cash after market receipts is a separate measure and can be negative.
- Cash waterfalls use recorded balance changes once. Resource price records enrich attribution; they do not add a second expense. Cash is fungible, so do not draw a loan-to-specific-build flow unless it is explicitly an accounting convention.
- Historical industry status uses build, flip and removal revisions at the chosen boundary, rather than the final `Remaining`/`Flipped` flags alone.
- Distinguish Canal scoring, after Canal cleanup, Rail entry and final scoring snapshots. Cleanup does not subtract already earned Canal link points.

## 8. Match review: strengths, concerns and next-game experiments

### Focus and comparison

Provide a visible **Review player** selector and **Compare with** selector. Default to the sole human player when there is one; otherwise select a human in stable seat order, or the first player in an all-AI match, with the choice clearly shown. Any player, including AI, remains selectable. Use the winner as the default reference for a nonwinner, and the next-ranked player for a winner; ties allow an explicit named comparison without declaring a unique winner. A named comparison player can remain visible as reference context even when excluded from the general player filter.

Match review is clearly **Whole-match review**. Its final result and score-gap explanation do not change under an era filter. Detailed evidence links can open a particular era. Respect the focus choice when player filters change; show a selection prompt if it becomes excluded rather than silently reviewing someone else.

Suggested arrangement:

```text
MATCH REVIEW       Review player [name]    Compare with [name]
[Final VP / rank] [Canal → Rail VP] [Flipped / built] [VP per action]

[Score-gap waterfall]              [Industry conversion bullet bars]

WHAT WORKED                       REVIEW NEXT TIME
2–3 evidence-backed strengths     2–3 supported concerns/opportunities
[Show evidence]                   [Show evidence]

NEXT-GAME EXPERIMENTS
Up to three concrete habits, each connected to this match's evidence.
[Open the relevant actions]        [Explore detailed statistics]
```

### Insight contract

Build a deterministic, local rules-based `StatisticsInsights` layer over the immutable report. No AI search, external service, strategy-bank change or save-format change is needed.

Each insight contains a stable rule ID/version, player and scope, supporting record IDs, exact metric with numerator/denominator, data coverage, interpretation and a practical suggestion. Present three distinct parts:

1. **Observed:** what the match actually records.
2. **Interpretation:** what that pattern may suggest, with relevant context.
3. **Try next game:** one actionable experiment rather than a claim about the optimal move.

Clicking **Show evidence** opens the relevant tiles/actions/plot with typed filters. Evidence must remain reachable without rewinding or altering the live board. All numerical statements must be reproducible from the same records used by the charts.

### Initial insight families

| Pattern | Evidence to display | Useful next-game suggestion |
| --- | --- | --- |
| Successful sale batching | Multi-tile Sell actions and their tiles | Preserve the preparation that made batching possible; compare merchant access and beer sources in those turns |
| Strong industry conversion | Built/flipped counts, industry VP and delay, by type | Repeat the industry sequence that converted well under similar merchant/resource conditions |
| Productive resource supply | Actual market receipts, supplied units and depletion flips | Review how connectivity and market demand helped those suppliers become useful |
| Productive Canal carryover | Tiles flipped before Canal scoring and their later Rail VP | Examine the development/build timing and surviving footholds that supported Rail |
| Strong network scoring | High-value routes and their actual contribution mix | Review the adjacent flipped industries/merchants and the timing of placement |
| Unfinished goods | Boundary tile state, nominal VP, age, costs and sale-related context | Plan the merchant route and legal beer supply before committing another similar late build; inspect the specific failed conversions |
| Rail preparation delay | Rail-ineligible mat tiles and actual early Rail development/actions | Consider clearing the relevant blockers during Canal when it fits the intended industry plan |
| Heavy market expenditure | Actual purchases/prices and resource-source mix | Compare whether connecting to supplies or timing a supplier build would merit investigation in a similar position |
| Unused wild flexibility | Scout acquisition/use episodes and remaining own turns | Review whether earlier Scout timing or spending the wildcard on a key placement could help; do not assume scouting was avoidable |
| Low link VP alongside little recorded transport use | Route score and available endpoint-use evidence | Review route purpose; distinguish useful access links from routes that provided little observed benefit |
| Debt/penalty episode | Actual cash shortage, liquidation, negative-income payment and VP delta | Inspect cash reserves before that settlement and how the preceding actions created the exposure |
| Cash retained at the end | Cash curve, final balance and last actions | Review conversion opportunities before the final turns, while retaining the tie-break/introductory context |

These are candidate patterns, not unconditional warnings. A single-tile sale may be excellent; a zero-VP link may provide vital access; using opponents' resources can be effective; loans may fund a winning plan. Do not grade industry diversity, low final income, final cash or opponent-resource use as inherently bad.

### Prioritization and safeguards

- Show at most three strengths and three review items, then an expandable **More observations** list. Merge overlapping items such as unfinished goods and low goods conversion.
- Prioritize clear, material outcomes such as actual VP penalties and substantial unconverted investment, followed by repeated supported patterns. Nominal unscored VP is never labeled proven lost VP.
- Use same-match comparisons as context, with the named reference, actual sample sizes and strategy mix visible. Two to four players are not a statistical training population. Avoid universal target flip rates, loan counts, final cash targets or optimal industry mixes.
- Require a sufficient sample for comparative claims. Small samples can still show exact facts, with cautious interpretation. Do not invent numerical confidence percentages.
- Missing history suppresses affected insights. Ambiguous destinations show attribution coverage and cannot support an exact merchant accusation. If no supported concern is found, state that instead of manufacturing a weakness.
- Do not claim to know an opponent's intent, a blocked legal alternative, luck, or the exact action that caused defeat. Optional feasibility analysis can narrow a particular historical constraint; it is not an optimal replay of the match.
- Full-game advice must not appear unchanged for the introductory variant, where additional cash/income/industry scoring changes the context. Keep existing authoritative variant results.

## 9. Data and implementation design

### Reuse and extend the detached projection

Continue using `StatisticsCollector`'s detached historical replay and value-only report. Never retain live industry/player transforms, mutate the current game, execute actions through its Router, read the repository repeatedly while scrolling, or change a revision while opening statistics.

Add typed records only where current fields are insufficient:

| Proposed record/field | Purpose |
| --- | --- |
| `StatisticsBoundarySnapshot` | Named boundary/revision, cash, income, mat readiness and historical asset status; prevents using final state for Canal analyses |
| `StatisticsMarketTransaction.TotalPrice` and kind/resource/quantity | Numeric resource prices without parsing `Detail` strings; correlation to the actual cash movement |
| Own-turn/action ordinal on analytical positions | Clear delay, loan recovery, wildcard timing and action-lane labels |
| Mat tile identity, next level and eligibility on development/transition records | Development follow-through and Rail readiness |
| `StatisticsLinkContribution` | Typed city/merchant/owner contribution at scoring, including routes with extra linked scoring regions |
| Resource-industry flip trigger attribution | Exact actor/operation where supported, plus an explicit unavailable reason otherwise |
| City region-color/type metadata and value-only topology | Reliable grouping and independent mini-map drawing |

Store coverage/attribution alongside each derived view. Existing complete-history final-score and cash reconciliation remain mandatory. Optional analyses may become unavailable independently without hiding otherwise reliable results.

### Plot and filter architecture

- Replace `Lines`/`Stacked` as the main type discriminator with `StatisticsPlotKind`, explicit axis specifications, series/category identities, legend items and typed evidence targets. Adapt existing plot data through this model before adding new types.
- Separate metric computation from rendering. Proposed `StatisticsMetrics.cs` and `StatisticsInsights.cs` remain Unity-independent so formulas and advice can be tested against hand-built records.
- Split plot rendering into focused components, for example `StatisticsPlotGraphic`, `StatisticsPlotAxes`, `StatisticsPlotLegend`, `StatisticsHeatmapGraphic`, `StatisticsTimelineGraphic` and `StatisticsMiniMapGraphic`, rather than a single growing switch inside `StatisticsWidgets`.
- Use new statistics-specific surfaces/controls where necessary. Card Tracker also uses shared `StatisticsWidgets`; retain its palette, glyphs, button placement and pause/input behavior.
- Replace broad text search as a drill-down mechanism with a typed `StatisticsDrillTarget`: page ID, player/era scope, industry/resource/city/action/asset IDs, boundary and selected evidence. Keep free-text **Search ledger** available with its scope plainly labeled.
- Cache grouped views by completed revision and analytical filters. Generate expensive optional views once on demand. Pool/window large ledger rows; do not redraw plots continuously when data and pointer state are unchanged.
- Keep the existing integration-hook signatures if possible. This should allow the accepted AI and game DLLs to remain unchanged while updating the helper; verify that expectation by hashes and method comparison during implementation.

## 10. Delivery sequence

| Stage | Priority | Concrete deliverable |
| --- | --- | --- |
| A. Freeze accepted baseline and definitions | Required foundation | Verify the then-current accepted runtime/archive fingerprints (1.4.16 for implementation); pin metric units, scope, boundary definitions, coverage and page IDs |
| B. Improve entry/layout/navigation | High | Both enlarged entries, eleven-page registry, default Match review shell and collapsed ledgers, with safe modal routing |
| C. Build reusable plot infrastructure | High | Axes, legends, typed filters/drill-down, accessible hit targets and equivalent tables; migrate existing bars/lines |
| D. Add comprehensive primary plots | High | The mandatory page/plot matrix, grouping, resource-direction semantics and independent geography view; retain every ledger |
| E. Add derived metrics and typed extensions | High | Batching, delay, gap, transfer/price, carryover and economic measures; validated Rail-readiness and attribution records |
| F. Add tactical summary | High | Deterministic insight rules, evidence links and next-game experiments; whole-match scope and variant/coverage handling |
| G. Qualify and release | Required completion | Native render/interaction checks, independent metric/insight tests, regressions, verified installation and recoverable archive when implementation is requested |
| H. Optional deeper analysis | Later | Individually validated sale-readiness/path probes or a small resource-flow diagram, only where they add understanding beyond the core plots |

Stages A–G constitute the complete core implementation. Optional feasibility analysis, cross-match rankings, permanent match archives, exports and a simulated optimal alternative game are separate additions, not requirements for the first improvement release.

Choose the next patch version at implementation time and build from the then-current accepted installed release. Do not overwrite unrelated AI research or stage an older release over a newer accepted patch.

## 11. Verification and acceptance

### Independent metrics and summary tests

- Reconcile every score-gap waterfall, category stack, grouped Other total and cash waterfall against independent known totals. Include negative penalties, opening VP, all-zero data and ties.
- Check actual action denominators, Scout's card trades, merchant development, double rail, zero-beer Sell, single-/multi-tile sales and games with no Sell actions.
- Verify industry build cohorts versus flip/scoring eras, same-turn flips, later flips, carryover, removed/overbuilt/liquidated assets, and unflipped observations excluded from completed-delay medians.
- Check income levels versus marker spaces, capped gains, real income paydays, final-round exception, loans followed by multiple gains and late loans without recovery. No invented counterfactual lifetime loan cost.
- Independently check resource matrices and directional filters, market/merchant sources, self versus opponent triggers, typed market prices and expenses counted once.
- Verify geography membership from real board metadata, farm breweries, merchant boundaries, no-slot cases and links with more than two scoring locations.
- Test insights with both positive and adverse fixtures: profitable borrowing, useful zero-VP access link, partly depleted supplier, unconverted goods, successful batching, unused wildcards, debt loss, ties and introductory scoring. Assert the evidence IDs and every numerical sentence, not just the generated text's existence.
- Partial/gapped/conflicting histories and unattributed merchant outcomes retain trustworthy final results and suppress unsupported advice. Missing data is not zero.
- Opening, filtering, plotting, drilling and closing leave the live game, events, ranking, revision, RNG and save behavior unchanged.

### Native UI and interaction

- Capture all eleven pages at **1280×720, 1920×1080 and 2560×1440**, including the three text-size settings, long names, 2–4 players, day/night review backgrounds and dense histories.
- Verify entry-button physical dimensions, icon/text spacing, hit targets, safe-area placement and non-overlap with results/review/hand/market controls. Check reveal after final replay and input return from both entry points.
- Check axis/category labels, units, ticks, readable legends, color-independent encoding, exact-value access, negative values, zero/empty data, grouped totals and coincident points. No invisible legend hidden far below a chart.
- Confirm each analytical page shows a useful plot before any long ledger; confirm grouping and Show all retain every record. Drill-down and back preserve filters and scroll context.
- Verify stable page routing after adding Match review, all previous links to Timeline/Scoring, missing-reference cases, repeated reopen, scene teardown and a new match with a different player count.
- Check mouse/keyboard scrolling, plot/legend selection, focus reveal, collapsed/expanded tables and no click or wheel leakage to the board. Virtualized rows must not expose stale event handlers or tooltips.
- Rerun the accepted statistics scenarios plus new page/metric/insight coverage. Run affected finished-review/presentation, link-replay and startup/menu regressions. Because shared widgets/input routing are involved, rerun the current Card Tracker's eligibility, pause, layout and input checks.
- Preserve accepted AI behavior and rule/save compatibility by exact source/binary/method evidence and relevant existing checks. Maintain online-disabled installation, atomic replacement, baseline backup and rollback qualification.

Measure collection time, first-page display time, page/filter response and scrolling on the native player. Aim for cached page/filter responses around 100 ms and smooth scrolling; qualify measured behavior rather than claiming an untested guarantee. If optional historical probes are slow, load them on demand with an explicit status while the core report stays usable.

## 12. Rules, strategy context and references

Rules were cross-checked against the [retained official rulebook](official-rulebook.pdf) and [searchable text](official-rulebook.txt), with the publisher's current [rulebook listing](https://roxley.com/products/brass-birmingham) checked on 1 October 2026. The listing links the [English PDF](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281).

Relevant rule anchors are printed pp. 6–7 for eras, scoring, income, turn order and ties; pp. 8–11 for resource access and actions; and p. 12 for introductory scoring and publisher tips. In particular, the publisher tips motivate inspecting loans, level-2+ carryover, brewery access and mat preparation for Rail. These support the choice of review topics, not universal statistical thresholds.

[Dylan Ho's strategy article](https://dlaern.substack.com/p/how-to-dominate-brass-birmingham), reviewed on 1 October 2026, suggests action efficiency, grouped selling, supply interaction and transition planning as useful questions. Treat these as the author's strategic perspective. Do not copy fixed action-count, cash/income targets, optimal-industry claims or blanket endgame advice into the summary. Actual events and the official rules determine the measurements.

The plot choices, grouping policy and coaching templates are design proposals inferred from the current implementation, the user's goals and those rules/strategy themes. The official rulebook does not prescribe a statistics interface or an automated tactical assessment.
