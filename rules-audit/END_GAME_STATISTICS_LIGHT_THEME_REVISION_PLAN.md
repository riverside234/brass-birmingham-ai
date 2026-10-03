# End-game statistics revision: light theme and factual charts

**Date:** 2 October 2026.  
**Revision:** 3 — adds a plot/data bug audit, eleven identified repairs and their regression checks; retains the economics/VP comparison and axis-readability specification.  
**Status:** Implemented, qualified, installed and verified as **1.4.21**.  
**Runtime baseline:** Accepted patch **1.4.20**, with the Statistics helper carried forward from **1.4.19**. The revision retains the exact 1.4.20 game and AI DLLs and all non-Statistics helper source pins. The findings below describe the pre-revision implementation.  
**Display target:** **1920 × 1200 first**, with **1920 × 1080** verification. Keep the second text-size preset, **1.15**, as the default. No 720p layout work is included.

This plan supersedes the advice-page requirement and visual design in the [previous improvement plan](END_GAME_STATISTICS_IMPROVEMENT_PLAN.md). The earlier implementation record remains historical. Its factual statistics, exact-data access and accounting safeguards remain relevant.

## 1. Intended result

Opening Statistics will show an objective **Overview** and nine other factual pages. Remove **Match review**, strengths, concerns, interpretations and next-game experiments. The panel will use warm ivory surfaces, dark readable text, restrained brass accents and clearer plots. Add **seven focused plots** using information already collected and currently available in tables, including an economics/VP efficiency comparison. Improve numerical axes throughout the panel so the scale and values can be read without hovering.

The panel should answer factual questions such as where points came from, which industries flipped, how links scored, how resources were obtained and how cards were used. It should not tell the player whether a tactic was good or bad, assign a strategy grade, or propose a better move.

## 2. Findings in the current implementation

| Finding | Evidence in the reviewed implementation | Revision |
| --- | --- | --- |
| Statistics initially opens on the advice page | `StatisticsPanel.CurrentPage` defaults to 10; `StatisticsPages.NavigationOrder` begins with 10 | Default to Overview, page 0; remove Match review from navigation |
| Advice is generated separately from the factual charts | `StatisticsReviewPanel.Observations` renders strengths, concerns and experiments from `StatisticsInsights.Build` | Remove the generation and presentation of advice, including expandable observations and their buttons |
| The advice page repeats existing factual charts | Its score-gap waterfall also appears on Scoring; its build-to-flip plot also appears on Industries | Retain those charts on their factual pages; do not add a replacement review page |
| The full panel is still dark | `StatisticsSurface`, widget text, table backgrounds, scrollbars and plot mesh colors use dark-theme constants | Introduce a complete, panel-scoped light theme |
| Changing shared colors globally would also change Card Tracker | Card Tracker calls `StatisticsWidgets` and creates `StatisticsSurface` objects | Make the light theme explicit for Statistics; retain existing defaults for other callers |
| Legends occupy substantial space and can include irrelevant series | The Overview capture has ten large era/category legend buttons, including categories with no awards | Use compact, measured legend items; separate category identity from era identity; omit structurally inapplicable series |
| Many chart values require hovering | Bars, waterfalls, route points and dumbbells mainly expose exact values through tooltips/tables | Add readable totals, endpoint values and selected-point labels, with collision-aware placement |
| Numerical axes do not communicate values clearly enough | Renderer divides the raw range into four equal intervals, formats most ticks as `0.#`, uses fixed label boxes and ellipsis, and repeats top/bottom scales for horizontal plots | Introduce measured, unit-aware axes with rounded intervals, appropriate precision, larger numbers, visible tick marks and readable label margins |
| Symbols do not have explicit semantic assignments | Renderer hashes series IDs into four shapes | Use stable, named symbol/color assignments for the represented player, industry or action |
| Useful data is still table-only | Development, link contribution sources, market prices, card purpose, merchant destinations, position shares and player-level efficiency already have ledgers | Add the seven plots in section 6, backed by the same records |

Visual inspection used the accepted [Overview](../artifacts/patch-releases/1.4.19/screenshots/statistics-1920x1200-0-larger.png), [Resources](../artifacts/patch-releases/1.4.19/screenshots/statistics-1920x1200-5-larger.png) and [Timeline](../artifacts/patch-releases/1.4.19/screenshots/statistics-1920x1200-9-larger.png) captures at the default text size. These demonstrate presentation issues; they are not evidence of incorrect game scoring.

## 3. Remove the advice page completely

1. Remove the Match review navigation item and its special plot assembly. Keep the original page identities **0–9** stable, since chart drill targets already use them.
2. Set the initial page to **Overview (0)**. Normalize an old request for page 10 to Overview **before** indexing scroll positions, buttons or drill state. Other invalid targets must also be handled safely.
3. Remove `Observations`, `Observation`, `moreObservations`, strengths/concerns sections, experiment lists, advice-specific evidence buttons and their cached state. Retire `StatisticsInsights` and its observation model after checking all callers and build references.
4. Retain the factual rendering, view cache, historical-boundary selector and player selector currently held in `StatisticsReviewPanel.cs`. Rename `ReviewContent` to `PlotContent`; do not delete this partial class wholesale.
5. Rename the remaining selectors to **Player** and **Compare with**. Use “selected player” and actual player names in score-gap labels instead of “review player.” Keep these selectors on Scoring and Economy only where needed.
6. Keep factual definitions, coverage notices and chart-reading help. Remove evaluative or prescriptive copy wherever it remains. A label such as “5 tiles remain unflipped” is acceptable; “weak conversion” or “sell earlier next game” is not.
7. Bring final ranking, VP, era totals and tie-break values to the top of Overview. The current final-result ledger must remain available, including for reports with partial history. Build/flip counts, final cash and VP/action already have factual homes; do not duplicate them as a new advice summary.

**Acceptance:** Exactly ten factual navigation entries, no advice engine call during rendering, no advice text or controls in the UI, Overview opens immediately, and all retained chart drill targets still reach the correct page.

## 4. Light visual theme

Use a warm paper-and-brass palette that fits the game and the existing light Statistics entry button. Keep the panel and plot surfaces effectively opaque, so board detail cannot interfere with text.

| Token | Proposed color | Use |
| --- | --- | --- |
| Window | `#F5F1E7` | Main ivory surface |
| Plot/card surface | `#FFFCF5` | Charts, score cards, tooltips and expanded details |
| Alternate row | `#EEE8DB` | Subtle table banding |
| Primary text | `#283238` | Body, values, controls and axis labels |
| Secondary text | `#586269` | Definitions, context and placeholder text |
| Accent | `#8A672B` | Headings, active borders and selected controls |
| Selected fill | `#E9D8B4` | Current page, checked filters and active tabs |
| Border | `#C5BCA9` | Card and table boundaries |
| Grid | `#DCD6C9` | Quiet chart gridlines |
| Focus | `#286D83` | Keyboard focus ring and selected plot-point outline |

These are starting design tokens, to be verified in rendered captures. Use a roughly 45–55% dark input-shield backdrop, with a clear light window above it. Rounded corners should be consistent across window, plot cards, buttons and tooltips; avoid decorative gradients inside plots.

### Color and state rules

- Keep player identities recognizable: blue, mustard yellow, violet and red. Use darker variants on ivory, such as `#1D6F9D`, `#8B6413`, `#77539D` and `#B34F45`, with a separate fill/outline choice where required. Yellow labels must remain legible.
- Use explicit, stable mappings for scoring categories, industries, actions, resources and eras. The same action should have the same representation on Actions and Timeline. Do not hash names into colors or shapes.
- Player color identifies a player; industry color identifies an industry. State the active encoding in the legend. Where both are needed, use color plus a labeled symbol, or separate player panels.
- Use sequential light-to-dark intensity for quantitative heatmaps, a signed scale for signed differences, and discrete colors for turn positions. A last-place seat must not look like a larger quantity of VP.
- Use dark legend text with a colored swatch. Do not require the series color itself to carry small text.
- Add symbols, outlines, hatch/line styles or direct labels where color alone would be ambiguous. Cash-in/out colors describe direction, not good/bad play.
- Selected, hover, pressed, disabled and keyboard-focus states must remain distinguishable. Avoid tint multiplication that washes out light surfaces or leaves a dark first frame after enabling a control.
- Target at least 4.5:1 contrast for ordinary text; ensure essential plot marks remain distinguishable from their surface. Use automatic dark/light text selection for heatmap cells.

### Theme isolation

Add a `StatisticsTheme` palette and explicitly opt the Statistics window into it. Shared widget/surface code may accept an optional theme or resolve a scoped theme on the Statistics root, but its existing default must remain unchanged. Apply the chosen theme when dynamically creating dropdowns, tooltips, plot legends, rows and sticky headers.

Audit all hard-coded colors in `StatisticsPanel.cs`, `StatisticsReviewPanel.cs`, `StatisticsWidgets.cs` and `StatisticsPlotViews.cs`; recoloring the outer window alone is insufficient. Card Tracker's appearance and behavior must remain unchanged.

## 5. Layout and common plot improvements

### Panel organization at 1200p

- Retain the current safe-area and physical-pixel sizing behavior. Use a window approximately 94–96% of the screen; avoid scaling the entire panel down to fit more content.
- Use a compact header: title and close/text controls, a player/era filter row, then a search/detail-tools row. Target about 150–170 physical pixels at the default text size, allowing more height at the largest preset.
- Use a left navigation rail about 230–260 pixels wide with roughly 54–62-pixel page rows. Ten pages should fit comfortably at 1200p.
- Keep title, filters, navigation and return controls fixed. Scroll the factual content and keep exact-table column headings pinned as today.
- At the default preset, retain approximately 24-pixel body text, **24-pixel major axis numbers**, 21–22-pixel legends/category labels and clearly larger page headings. Axis numbers scale with the three text presets. Do not reduce text size to accommodate a long label.
- Use related small plots side by side only when each has at least about 640 pixels of usable width and all labels fit. Long route plots, geographic diagrams, timelines and round heatmaps use the full content width. Fall back to vertical arrangement at the largest text setting when needed.
- Keep the current Statistics button's size, light appearance and location below the results popup. This revision should not reintroduce overlap with Review or Exit.

### Standard chart card

Each chart should have a clear title, a one-line scope/unit summary, a compact legend, the plot, and an **Exact data** disclosure. Move long definitions to a small “About this chart” disclosure/tooltip. Keep missing-history and attribution-coverage warnings visible whenever they materially affect the chart.

- Format ticks for the measure: whole numbers for tiles/actions/rounds, `£` for cash, `%` for shares and explicit `VP` or `income levels` units. Use rounded, sensible steps rather than arbitrary quarter-range decimals.
- Start count and spending bars at zero. Preserve negative cash, income and VP differences. Add space for endpoint labels without clipping the maximum mark. Share scales across comparable player panels.
- Display totals on stacked bars, values on paired bars and dumbbell endpoints, and signed values on waterfalls. Label small segments through selection/tooltip rather than overlapping text.
- Measure legend labels and wrap compact swatch items naturally. Avoid fixed-width boxes that consume a large fraction of the chart. Keep a checked/off indicator when a legend can hide a series.
- Hidden series should not change numerical scales or recompute percentage denominators. Exact data retains all records; the legend clearly indicates hidden series. Reset filters restores visibility.
- Treat **zero**, **no matching records** and **unavailable history** differently. A ratio with no denominator shows `—`, not 0%, and unknown observations must not be drawn as zero.
- Plot selection should visibly outline the selected mark and show exact values with player, era/round and category. Dense overlapping marks need a selectable list of matching records; a single arbitrary nearest point is insufficient.
- Use bar/cell/interval hit areas for those shapes rather than only a small radius around an endpoint. Keep keyboard selection and exact-data access for dense plots.
- Every grouped value must expose its exact membership. Top-N summaries must retain the count and total of omitted records without plotting an aggregate on an individual-item scale.

### Numerical axes: explicit readability requirements

Replace the current fixed-quarter-range tick drawing with a shared axis model. Compute the scale, major ticks, formatted labels and layout together, then use the same result to position gridlines and plot marks. Axis rounding changes the displayed range, never the underlying recorded value.

#### Tick selection and precision

1. Find the full in-scope data range, including hidden series. Include zero for count, spending, efficiency-comparison and other bar/dot comparisons; preserve negative values where the metric permits them. Percentage axes are fixed at **0–100%**.
2. Select a sensible step from `1, 2, 2.5, 5, 10 × 10^n`, using the available plot width/height and measured label sizes. Counts, turns, rounds, VP totals and income levels require whole-number steps; skip candidates that would imply fractional counts. Cash prices and ratios may use decimal steps.
3. Round the outer bounds outward to the selected step, with room for the data and labels. Always include a labeled zero when it is in range. Add clear tick marks and aligned major gridlines; use a stronger zero line for signed plots.
4. Target about **6–9 major ticks on a wide numeric axis**, **4–6 on a compact axis**, and **4–7 on a vertical axis**, reducing density when measured labels would collide. Very small integer ranges may have fewer ticks. These are layout targets, not a requirement to draw overlapping labels.
5. Choose enough decimal places to distinguish adjacent ticks. For example, a `0.25` step needs `0.00, 0.25, 0.50…`, not duplicated values after one-decimal rounding. Keep exact selected-point and ratio labels at appropriate precision, usually two decimals for `VP/action` and `£/VP`.
6. Format units consistently: `£20`, `−£10`, `20%`, and whole-number VP/count labels with the unit in the axis title. Use explicit titles such as **Action spending (£)**, **VP per confirmed action** and **Cash spent per VP (£/VP)**. Do not show scientific notation or rounded `−0` for ordinary game values.
7. Handle all-zero, single-value and missing ranges explicitly. A true zero remains visible; a missing ratio has a labeled `—` row without a fabricated point, division by zero, NaN or infinity.

Illustrative targets below are examples of formatting, not measurements from a match:

| Measure / example data range | Readable ticks / labels | Exact-value presentation |
| --- | --- | --- |
| Industry tiles, 0–23 | `0, 5, 10, 15, 20, 25` | Whole counts on bars |
| Awarded VP, 0–146 | `0, 25, 50…150`, if space permits | Exact total such as `146 VP` |
| Signed cash, −£13 to £77 | `−£20, £0, £20…£80` | Exact signed movement beside a selected step |
| Share | `0%, 20%, 40%, 60%, 80%, 100%` | Count and percentage together |
| VP/action, 0–4.37 | `0, 1, 2, 3, 4, 5` | A point label such as `4.37 VP/action` |
| Cost/VP, 0–2.31 | `£0.00, £0.50…£2.50` | A point label such as `£2.31/VP` |

#### Label placement and number visibility

- Measure tick labels with the actual TMP font and text preset before assigning axis gutters. Reserve space for the widest negative/currency label, axis titles, category names and endpoint values. **Essential numerical axis labels must never use ellipsis.**
- Keep major numbers dark and approximately 24 physical pixels at the default preset. Avoid pale text on gridlines or colored series text for numerical scales.
- Increase gutters or reduce the number of major ticks when labels collide. Wrap long category names, use separate era/round label bands, or allow horizontal scrolling for dense categorical data; do not squeeze labels into fixed 56/72-pixel boxes.
- For ordinary horizontal charts, show one clear bottom numeric axis. Repeat it at the top only when a tall plot benefits from it. Avoid double labeling compact charts.
- Give bar endpoints and extreme ticks sufficient padding. Align each label with its actual tick coordinate, including nonuniform category layouts and signed axes. Labels remain inside the visible plot card and clear of scrollbars.
- Use direct values for bars, waterfalls, route points and efficiency points in addition to the axis. A tick scale alone should not force the player to estimate a value by eye.
- On selecting a curve/scatter point, show its exact coordinate values and optional guide lines to the axes. Keep exact coordinates unchanged when visually separating overlapping marks.

#### Axes by plot type

- **Time plots:** use integer global-turn ticks and Canal/Rail bands; round-category plots use real `C1…Rn` labels, with predictable thinning if dense. Do not generate fractional turn/round labels.
- **Heatmaps:** categorical row/column labels plus a clearly labeled quantitative color scale. Turn-position heatmaps use the discrete position key, not a continuous numeric intensity axis.
- **Geographic schematic:** retain city/region labels and the visible tiles/VP scale; do not expose arbitrary normalized X/Y coordinates as game statistics.
- **Small multiples:** share scales where the measure and scope are comparable. The two efficiency metrics in N7 have separate units/scales, so they must not share a numeric axis or use a dual-axis overlay.

### Improve all existing plots

| Page | Existing plots | Representation changes |
| --- | --- | --- |
| **Overview** | Awarded-VP stacked bars; Canal/Rail dumbbells | Place a compact ranked score strip first. Show Canal and Rail as separate rows/panels using the same category colors, with a concise era key instead of ten category/era colors. Show net totals and signed penalties. Label both dumbbell endpoints and the era difference; do not imply a two-era comparison when only one era is shown. Introductory mode shows its actual awards without empty Rail legends. |
| **Scoring** | Player × industry VP heatmap; score-gap waterfall | Show full industry names/icons, readable cell values and a visible scale. Include zero cells distinctly from unavailable cells. Add waterfall connectors and final-gap value; label its whole-match scope even under an era filter. Name both compared players and preserve a zero VP gap for ties. |
| **Industries** | Built/flipped paired bars; flip-delay dots | Label counts and `flipped / built` percentages, with `—` for an empty cohort. Make aggregation across selected players explicit; allow player comparison panels rather than 24 crowded series. Use non-overlapping delay dots, a median marker and sample size per industry. Unflipped tiles remain outside the delay distribution. |
| **Networks & Geography** | Route lollipops; historical topology map | Label route VP and owner; maintain sorted routes and a separate omitted-route summary. Use full readable city names, leader lines and collision-aware labels on the map. Separate merchant/farm symbols from city symbols; retain boundary and measure controls. Drill to assets for occupancy, scoring evidence for industry VP, and link contribution records for link VP. The schematic must not claim that a drawn board edge was a route actually used. |
| **Economy** | Cash/income step curves; cash waterfall | Label line endpoints and the Canal/Rail boundary. Mark loans and recorded income changes with exact event tooltips; preserve ordered revisions when changes share a turn. Keep income levels separate from marker spaces and cash. Label cash-in/out and opening/closing values in the waterfall. Do not add market costs again to action spending. |
| **Resources & Trade** | Supplier → consumer heatmap; resource-source stacked bars | Label both directions clearly and use the same query for cells, totals, bars and evidence. Add row/column totals and a numeric intensity scale. Incoming retains all suppliers; outgoing retains all consumers. Source bars show exact unit totals, with an optional percentage view and a visible denominator. Separate units from money and hide structurally irrelevant source categories. |
| **Actions & Cards** | Action-share bars; sale-batch histogram | Keep a fixed action order, stable action icons/colors, percentages and action totals. Add an absolute-count toggle. Keep Pass and any unclassified actions in the denominator. Show `1`, `2`, `3+` sales as discrete bins with counts per player; retain exact large batches in the ledger. |
| **Turn Order & Tempo** | Round-seat heatmap; spending columns | Replace quantitative seat intensity with a discrete `1st…Nth` key and numeric labels. Split round labels into Canal/Rail bands. Mark actual within-era back-to-back turns separately from era transitions. Space grouped spending columns by player; if dense, offer a line view using the same round totals. Spending in round N relates to order in N+1. |
| **Efficiency & Transition** | Construction-cost/industry-VP scatter; tile-outcome stacks | Lead with the new player-level economics/VP comparison (N7). Retain the tile scatter with explicit £ and VP scales, industry symbols and selected-tile labels; provide an overlap list while retaining exact cost/VP coordinates. Do not imply that low supplier VP measures its total usefulness. Label outcome bars as **final outcomes of the selected build cohort**, so Canal filtering does not silently mean Canal-boundary state. Show counts and explain mutually exclusive categories. |
| **Timeline** | Action lanes; industry intervals | Use explicit action symbols and colors, era bands and readable round/turn markers. Distinguish both actions in the same turn. Group industry intervals by player/industry with build, flip, removal and still-open endpoint symbols. Use paged/grouped records with a visible remaining count; do not compress unrelated omitted intervals into a fake average lifespan. Keep Show all and every original record. |

## 6. Seven additional plots from existing tables

Add these within the existing pages, not as seven new navigation pages. Their models must be pure queries over `CompletedMatchStatistics`; use typed records, not parsed display strings. Each plot and its exact ledger must share the same aggregation and filter scope.

| ID / page | New plot | Existing data | Question answered |
| --- | --- | --- | --- |
| **N1 · Industries** | Development by industry and tile level | Development ledger and follow-through records | Which tiles were removed, at which levels and by which source? |
| **N2 · Networks** | Sources of link VP | Link scoring ledger / `StatisticsLink` | How much link VP came from own industries, others' industries and merchants? |
| **N3 · Resources / Market** | Market purchase price profile | Market dependence and recorded transaction prices | How much did each player pay per coal/iron unit, and how many units were bought? |
| **N4 · Actions & Cards** | Cards used by purpose and type | Card purpose distribution | How were Location, Industry and Wild cards spent? |
| **N5 · Resources / Merchants** | Goods sold by merchant destination | Goods sales and merchant attribution coverage | Where were goods delivered, and how complete is destination attribution? |
| **N6 · Turn Order** | Share of rounds in each position | Round positions table / `StatisticsRound.Order` | How often did each player act first, second, third or fourth? |
| **N7 · Efficiency & Transition** | Economics and VP efficiency comparison | Conversion and efficiency table; action spending, actual VP and cash movements | How much VP did each player obtain per action, and how much cash did each VP cost? |

### N1. Development by industry and level

- Use stacked columns: six industry groups, segments `Level 1`, `Level 2`, `Level 3+`; exact data keeps each original level.
- Offer **Selected players total / By player**, and source selection **All / Develop action / Merchant bonus**. Limit player panels to actual selected players and use the same scale.
- Count **one tile per Development record**. `StatisticsEntry.Quantity` contains the removed tile's **level**, so summing it would incorrectly count tiles.
- Show the Develop action count separately from tiles removed. A two-tile Develop is one action; merchant development is part of a Sell action.
- Drill to original development IDs and the existing follow-through ledger. Later higher-level builds remain observations, not “development value” or a causal success rate.

### N2. Sources of link VP

- Use absolute horizontal stacked bars by player, with separate Canal/Rail rows when both eras are selected.
- Segments are **Own industries**, **Other players' industries**, **Merchants**. Values are `OwnVP`, `OthersVP`, `MerchantVP` for scored links; their sum must equal actual link VP.
- Show total link VP, number of links and VP/link next to each bar. Include scored zero-VP links in the link count and average.
- An optional percentage toggle may describe source shares, with the absolute VP total retained. A zero-VP total has no percentage denominator.
- Drill to the links and recorded endpoint contributions behind the chosen segment. Do not label this “freeloading” or infer route dependency from point attribution.

### N3. Market purchase price profile

- Use two aligned dot/lollipop panels, **Coal** and **Iron**: player rows, horizontal axis **£ per unit**.
- Calculate weighted price as `sum(TotalPrice) / sum(Quantity)` for confirmed `Market purchase` records. Do not average transaction averages or use the before/after quote as the paid price.
- Label each point with weighted price, units bought and total paid; retain a shared scale across the two resources. No purchases shows `No purchases`, not a £0 price point.
- Keep receipts from market sales in a separate transaction view/ledger. Outgoing supply must not masquerade as player purchases or be combined into a net purchase price.
- Drill to exact transactions. The total paid is already included in action spending; this chart must not increase game-expenditure totals.

### N4. Cards used by purpose and type

- Use horizontal stacked bars for **Build authorization**, **Non-Build payment**, **Scout extra trade**. Segments are **Location**, **Industry**, **Wild Location**, **Wild Industry**.
- Show selected-player totals by default, with a By player comparison view. Provide **Counts / Share**; retain total cards for every percentage bar.
- Aggregate `Kind == Card` records by recorded `Category` and `Label`. Count Scout's action payment separately from its extra trades. Setup discards, draws and reshuffles are not played cards.
- Acquiring a wildcard is a separate episode, not a played card. Keep existing wildcard acquisition/use timing in its ledger; do not claim a use caused a tactical turnaround.
- Drill to the card payment and confirmed action. Preserve the two wildcard types explicitly in labels/icons.

### N5. Goods sold by merchant destination

- Use grouped horizontal bars: merchant destinations on rows, player series with stable player colors. Keep **Unattributed** as its own visible row. Offer an industry filter for Cotton mills, Manufactories and Potteries instead of multiplying player × industry × location series.
- Count sold **industry tiles**, using `Goods sale` records (`Quantity == 1` in the current projection). Label this unit so it cannot be confused with Sell actions, merchant beer or bonus activations.
- Retain recorded and uniquely derived attribution separately in exact data and the selected-value tooltip; show the coverage totals beside the chart.
- Never redistribute unattributed sales among likely cities. If attribution is sparse, display the sparse bars and coverage rather than suppressing the uncertainty.
- Keep merchant bonus cash, marker spaces, income levels, VP and Develop uses in a concise separate ledger. Their different units must not be combined into a bonus-value score or a single pie chart.

### N6. Turn-position shares

- Use 100% horizontal stacked bars per player, with ordered segments `1st`, `2nd`, `3rd`, `4th` as applicable to the actual player count.
- Show counts, percentages and each player's known-round denominator. Omit nonexistent positions in two- and three-player games.
- Use known recorded positions; if a player is absent from a recorded order, mark it unavailable instead of generating a “0th” position or silently treating it as first.
- Use the same era scope and round population as the existing position table. Keep back-to-back frequency separate because it measures adjacent-round relationships.
- Drill to the exact matching rounds and their preceding spending records. This is an ordering distribution, not a tempo grade.

### N7. Economics and VP efficiency comparison

Use **one chart card with two aligned horizontal dot/lollipop panels**, placed first on Efficiency & Transition:

- **VP per confirmed action:** player rows; X-axis **VP/action**.
- **Cash spent per VP:** the same player rows in the same order; X-axis **£/VP**.

Player colors and names identify the rows. Display each exact ratio beside its point, with the VP, action count and cash total available in the tooltip and exact ledger. Use separate rounded numeric scales with labeled units. Keep player order stable across the two panels rather than sorting each metric independently. On a narrow content area or the largest text preset, stack the panels while preserving that order.

#### Calculation and scope

| Value | Definition |
| --- | --- |
| VP numerator | Whole game: authoritative final VP. Selected era: actual net VP awarded in that era, including recorded bonuses and penalties. Label the scope explicitly. |
| Action denominator | Confirmed actions in the same player/era scope, including Pass and recorded unclassified actions. Double rail is one action; Scout extra trades and merchant development are not extra actions. |
| Default spending | Sum of confirmed action spending in the same scope, already including market resource payments. Never add `MarketCost` again. |
| Optional all-payments spending | Sum of the magnitudes of actual negative balance-cash movements in the same scope, including payments outside actions such as debt settlement. Do not subtract receipts or include borrowed money as expenditure. |
| VP/action | VP numerator ÷ action denominator. With zero actions, show `—`. A negative net VP with a positive action count remains a signed value. |
| £/VP | Selected spending basis ÷ VP numerator. With zero or negative VP, show `—` and the actual totals rather than a misleading ratio. A real zero expenditure with positive VP may show £0/VP. |

- Reuse the existing **Action spending / All cash payments** choice, placing it **above** N7 so the current basis is visible before reading the chart. Updating it refreshes the plot, its labels, headline and exact table together.
- Support player and era comparisons. Industry, region, action-kind and individual-evidence filters must not silently divide a player's total VP by only a subset of their actions or spending. Clearly label N7 as player/era totals and show that other detail filters do not apply to this comparison; they can still apply to the tile-level charts below.
- Calculate the plot and the existing **Conversion and efficiency** table from one shared typed query. Preserve the table's industry VP, printed opportunity and realization columns as factual details; do not turn nominal printed opportunity into an optimal-score benchmark.
- Point labels might read `3.25 VP/action` and `£1.84/VP`; tooltips must state the actual numerator and denominator. Do not combine the two ratios into a rank, grade, “best strategy” badge or improvement suggestion.
- Selecting a VP/action point opens the player's matching score/action records. Selecting a £/VP point opens the matching score/spending records with the current cost basis preserved. The selected scope must remain visible after drilling.
- This is a **player-level** comparison of the existing table. It complements the current tile-level construction-cost/industry-VP scatter, which measures a different scope and excludes broader income/network benefits.

### Avoid a longer wall of content

Resources & Trade will use three local views: **Supply**, **Market**, **Merchants**. Supply retains its matrix and source bars; Market shows N3 and transaction data; Merchants shows N5 and bonus/coverage data. Keep player and era selection across these views, clear incompatible chart-specific evidence on a view change, and apply only the visible view's resource/direction controls. Include the local view in the analytical cache key and scroll state.

On other pages, keep the two current main plots prominent and place the new compact plot below them or beside a related plot when labels fit. On Efficiency & Transition, put N7 first, followed by the tile scatter and outcomes. Detailed ledgers remain collapsed and paged; clicking a mark opens its exact records immediately.

## 7. Plot/data bug audit and required repairs

The [detailed audit](END_GAME_STATISTICS_PLOT_DATA_AUDIT.md) identified **11 finding groups**: eight P2 statistical correctness/evidence-access issues and three P3 geometry/readability issues. **13 targeted probes** reproduced their conditions against the installed helper or its source-verified predicates. These are newly added diagnostic cases, not a new full-game/native-render qualification. All three installed DLL hashes remained unchanged.

| ID / priority | Concrete issue | Required repair and acceptance |
| --- | --- | --- |
| **SD01 · P2** | Both wildcard acquisitions from one Scout share a revision-based episode key; spending only one makes both detail rows look used | Card-specific acquisition/use identity and unique row keys; one used/one unused remains distinguishable, including reacquisitions |
| **SD02 · P2** | Selecting Rail removes Canal-built tiles before counting the “Canal carryover” headline | Separate build cohorts from boundary/carryover populations; an actual carried tile still counts in Rail |
| **SD03 · P2** | Whole-game/era VP is divided by action-kind/evidence-filtered actions and spending | Shared N7 player/era totals for plot/headline/table; detail filters cannot silently inflate VP/action or mix cost scopes |
| **SD04 · P2** | Resource plots honor action/source/evidence filters while some detail tables ignore them | Shared supported-scope queries, or explicitly labeled independent scope; plot/ledger totals and membership reconcile |
| **SD05 · P2** | Generic tooltips label map coordinates as tiles, raw action counts as percentages and interval duration as “Tile” | Plot-specific semantic quantities/units shared by tooltip, keyboard description and exact data |
| **SD06 · P2** | Link-contribution map clicks open Industries using possibly empty asset IDs | Drill according to the represented measure; link VP opens its actual link/contribution records |
| **SD07 · P2** | Map link VP ignores selected link evidence, including unrelated cities' contribution totals | Apply the same represented-link membership to routes, map, summaries and evidence |
| **SD08 · P3** | First/last grouped columns extend beyond the declared category-axis area | Full category bands and end padding; bar bounds and category labels align |
| **SD09 · P3** | Positive scatter overlap offsets push maximum-valued points above the scale | Bounded displacement/cluster selection with exact coordinates preserved |
| **SD10 · P2** | Wide heatmap cells are clickable only near their centers; bar/interval hits also use point proximity | Hit-test rendered cells, bars and intervals; verify native clicks across visible shapes and scrolling |
| **SD11 · P3** | A 0–1 count axis produces fractional ticks `0, 0.3, 0.5, 0.8, 1` | Implement section 5's measured, unit-aware axis specification, including integer count/time ticks |

The archived accepted Statistics suite passed 2,638 checks, but did not cover these combinations. Preserve its independent accounting checks and add the targeted semantic/interaction cases. No P1 gameplay failure was demonstrated by this Statistics-focused review.

**Repair scope:** SD01 requires a targeted change to the detached Statistics projection; it does not require a saved-game format change. The other repairs belong to query, display, scope and selection code. Historical map treatment of removed tiles, cross-era loan recovery and missing round-position behavior remain explicitly labeled follow-up checks in the audit, rather than additional asserted bugs.

## 8. Accuracy and interaction requirements

- Keep final ranking and whole-match comparison independent of era/detail filters, with explicit scope labels. Other plots must visibly state their selected player, era and relevant industry/resource scope.
- Keep **build cohort**, **flip era**, **scoring era**, **historical boundary** and **final outcome** distinct. A tile may earn industry points in both eras, but it is still one built tile.
- Use one typed query result for each chart, headline and matching ledger. If a filter is inapplicable to a plot, its control must not pretend to affect that plot.
- Keep a semantic measure separate from its plotted coordinates. Tooltip text, axis units, exact-data columns and click targets must agree on what the mark represents.
- Assign identities at the granularity of the record, including individual wildcard acquisition episodes. A shared event revision is not sufficient to distinguish multiple records emitted by that event.
- Do not recompute gameplay, inspect hidden hands, infer a better legal action, or add new save-game fields. The seven plots need no new gameplay telemetry.
- Preserve detached-report behavior: opening, filtering or closing Statistics must not mutate the live board, player money, rules, AI state or completed history.
- Match drill evidence to the represented quantity. An aggregate market point opens its transactions; a merchant bar opens its goods sales; a round-position segment opens its rounds. Existing entry IDs use revision plus player, and tile IDs are generation-specific.
- Keep text search explicitly scoped to tables. Plot membership is controlled by the visible player/era/detail controls; a search must not silently rewrite chart totals.
- Preserve scroll position, open ledgers, selected series and local view across redraws/text-size changes. Clear or remap a removed page's state safely. Repeated opening and closing must not duplicate UI objects or listeners.
- Complete history with no activity produces zeros/empty states. Partial history produces authoritative final results and clearly unavailable detailed plots; it must not generate apparently complete new statistics.

## 9. Implementation sequence and files

| Stage | Work | Main files / checks |
| --- | --- | --- |
| **A · Lock scope** | Record the latest installed accepted baseline; confirm exact helper-source pins; inventory advice references and numerical page targets | Installation/build/archive manifests; `StatisticsPages.cs`, `StatisticsPanel.cs`, `StatisticsReviewPanel.cs`, `StatisticsInsights.cs` |
| **A1 · Data/evidence repairs** | Correct SD01–SD07 and SD10 before building new charts: acquisition identities, population/scope consistency, semantic values and typed evidence selection | `StatisticsProjection.cs`, `StatisticsMetrics.cs`, `StatisticsDetailMetrics.cs`, `StatisticsPlotPages.cs`, `StatisticsPlotModels.cs`, `StatisticsPlotViews.cs`; independent fixture and collected-history checks |
| **B · Remove advice** | Remove page 10, default to Overview, normalize old targets, retire observations and rename remaining comparison controls | Page registry, plot page assembly, partial panel; update advice-specific native tests and build references |
| **C · Light theme** | Add theme tokens and scoped theme plumbing; style all static and dynamic controls; preserve shared defaults | New `StatisticsTheme.cs`; `StatisticsWidgets.cs`, `StatisticsPanel.cs`, `StatisticsReviewPanel.cs`, `StatisticsPlotViews.cs` |
| **D · Plot foundation** | Add semantic palettes/symbols, a shared axis model, rounded/adaptive major ticks, measured numeric-label margins, direct value labels, selection, correct hit areas, compact legends and density handling; repair SD08/SD09/SD11 | `StatisticsPlotModels.cs`, `StatisticsPlotViews.cs`, a small `StatisticsAxisLayout.cs`; add presentation metadata without changing stored game data |
| **E · Existing pages** | Apply the section 5 plot matrix and reorganize factual Overview | `StatisticsPlotPages.cs`, `StatisticsPages.cs`, panel layout and table styling |
| **F · Seven new plots** | Add shared typed aggregations, matched exact ledgers, N7's two-panel efficiency comparison and Resources local views on the corrected populations/identities | `StatisticsMetrics.cs` / a small `StatisticsChartQueries.cs`, `StatisticsPlotPages.cs`, filter/cache/scroll state and the existing efficiency table; preserve the collector and limit projection changes to demonstrated audit defects such as SD01 |
| **G · Qualification** | Run numerical, interaction, native-render and preservation checks; capture both supported resolutions | Statistics tests, InformationUI regression checks, existing presentation/menu/link checks, packaging tests |
| **H · Future release** | Build from the latest qualified accepted runtime, archive the exact sources/evidence, and install only when implementation is requested | A scoped statistics build/archive script; existing installer/rollback workflow and release notes |

Use links to the [Statistics sources](../src/BrassPatch/Statistics) and [Statistics tests](../tests/BrassPatch/Statistics) for the implementation. The existing `build-statistics-review.ps1` is an older release workflow and must not be run unchanged against today's runtime. The current InformationUI build is restricted to four overlay files; a broader Statistics revision needs its own explicit source scope and baseline checks.

## 10. Verification plan

### Numerical fixtures

| Case | Required assertion |
| --- | --- |
| Advice removal | No Match review page or observations; default and legacy page-10 requests open Overview safely; retained page IDs do not shift |
| Score reconciliation | VP category sums, opening VP, signed penalties and era totals match final VP; score-gap components sum to the actual gap; ties stay zero |
| Development | Removing a level-3 tile counts one tile; two developments in one action count two tiles/one action; merchant development adds no Develop action |
| Link attribution | Own + others + merchants equals link VP per link and per player/era; zero-VP scored links remain in averages |
| Market price | Unequal transaction quantities use the weighted average; zero purchases has no price; market receipts remain separate and total spending is unchanged |
| Card use | Build, other payment and Scout extras partition recorded played cards; acquiring/reacquiring a wildcard is not counted as spending it |
| Wildcard episode identity | Acquiring both wildcard types and using only one leaves one unused episode; both used/reacquired cases retain distinct rows and associations |
| Merchant destinations | Recorded + derived + unattributed counts reconcile to sold tiles; industry/player filters and evidence agree; bonus units remain separate |
| Turn positions | Counts use known positions, shares total 100% when a denominator exists, absent seats do not become zero/first, and nonexistent player-count seats are omitted |
| Economics/VP efficiency | VP/action and £/VP match the shared table query for each player/era; market cost is counted once; changing the cash basis updates chart/table together; zero actions or nonpositive VP gives the specified unavailable ratio; detail filters cannot create a mixed-scope ratio |
| Axis calculations | Tick bounds cover all marks; rounded steps are suitable for the unit; count/time ticks are integers; labels are unique after formatting; signed ranges include zero; all-zero, single-value, fractional-price and large ranges remain finite; percentage axes remain 0–100 |
| Historical boundaries | Canal-built/Rail-flipped tiles, supplier use by opponents, cleanup, overbuild and removed tiles preserve their correct timelines and previously scored VP |
| Carryover and filtered evidence | Rail carryover counts use the boundary population; resource tables respect their declared scope; selecting one link excludes unrelated map contributions and opens that link's evidence |
| Semantic plot values | Map quantity, percentage/count pair, interval start/end/duration and each other plot family display the correct units in mouse/keyboard descriptions and exact data |
| Display state | Hiding a series keeps scale/denominator stable; filtered chart values equal exact records; zero, empty and unavailable states remain distinct |

Update existing tests that intentionally expect page 10 or generated insights. Keep their independent accounting and evidence assertions, relocating score-gap assertions to Scoring. Extend the current native harness rather than replacing factual tests with screenshot checks alone.

### Native visual and interaction checks

- Capture every page and all three Resources views at **1920 × 1200** using the default and largest presets. Capture every page at **1920 × 1080** using the default preset; exercise the largest preset on the densest plots/tables at 1080p.
- Use representative two-, three- and four-player completed matches, introductory mode, partial history, duplicate/long player names, negative values, a zero-value chart and a dense match.
- Verify no clipped tick/endpoint values, hidden maximum bars, overlapping legends, truncated essential labels, unreadable yellow series, tooltip obstruction or plot content under the fixed footer.
- Measure numerical tick-label bounds at both resolutions and tested presets. Check zero, negative currency, decimal ratios, wide values and long category labels; there must be no overlapping or ellipsized axis numbers, duplicated rounded labels or fractional count/turn ticks.
- Capture N7 for both spending bases and Whole game/Canal/Rail scopes. Check separate unit labels/scales, stable player alignment, exact ratio labels, missing-ratio rows and the cost selector above the chart.
- Check complete first/last grouped-column bounds and scatter clusters at all plot corners. Use actual native pointer events near heatmap edges/corners and across bars/intervals, including after scrolling; do not substitute center-only button proxies for these checks.
- Inspect hover, focus, selection, disabled/enabled and first-open states. Check mouse wheel, nested horizontal scroll, keyboard navigation, plot selection, legend toggles, table sorting, row expansion, drill/reset and text-size changes.
- Verify the Overview result strip and a complete primary plot can be read comfortably at the default size. The new plots must appear as charts with labels/legends, not merely differently styled tables.
- Reopen Statistics from both results and board review. Confirm the existing entry location remains clear of the results popup and Review/Exit controls, and modal return destinations still work.
- Capture Card Tracker before/after and verify its colors, fixed button position, pause/turn gating, wild-card representation and text default are unchanged. Scope any shared-widget changes accordingly.

### Release preservation

For a later installation, preserve the latest accepted `Assembly-CSharp.dll`, `BrassLegacyAI.dll`, AI configuration and save implementation. Only the Statistics helper and explicitly isolated shared-widget support should change. Compare unaffected helper methods with the accepted baseline, including Card Tracker, custom AI menu, animation/presentation and link-placement recovery. Use the existing atomic installation, recovery backup and installed-file hash verification. Do not roll back a newer AI patch to recreate the older Statistics baseline.

## 11. Completion criteria

- [x] Statistics opens on an objective Overview; the advice page, verdicts, interpretations and improvement experiments are removed.
- [x] The entire panel, plots, tooltips, tables and control states use the readable light theme.
- [x] Every retained plot has suitable labels, units, scale, legend and exact-data access.
- [x] Numerical axes use readable 24-pixel default numbers, rounded/adaptive tick intervals, sufficient precision and collision-free margins at 1200p/1080p.
- [x] All seven added plots reconcile to existing typed records and ledgers, including N7's VP/action and £/VP comparison.
- [x] SD01–SD11 have independent semantic/geometry regressions; collected-history and native-pointer checks pass where specified in the detailed audit.
- [x] Grouping and local views reduce clutter without concealing unknown or omitted data.
- [x] The default text size remains 1.15; 1200p and 1080p captures pass the specified checks.
- [x] Gameplay, AI, save handling, Card Tracker and the accepted Statistics entry positioning are preserved.
- [x] Patch 1.4.21 has its own scoped build, qualification evidence, release notes and verified installation record.

## 12. Implementation record — 1.4.21

The [release notes](../docs/STATISTICS_LIGHT_THEME_1.4.21.md) describe the implemented UI, all seven chart families and SD01–SD11 repairs. The candidate passed **5,301 Statistics**, **458 Card Tracker**, **184 presentation**, **765 link-replay** and **259 menu** assertions, covering nine completed matches and ten factual pages. Native captures and actual cell/bar/interval pointer checks cover 1200p/1080p; all three text presets, partial/zero/negative history and duplicate long names are exercised.

All non-Statistics source pins and **12,067 existing method bodies** are preserved. One changed compiler-generated anonymous type is used only by Statistics; the preservation checker verifies that it has no references from other runtime methods or fields. The game and AI binaries match accepted 1.4.20 byte for byte. The new scoped build/qualification/archive workflow and atomic installer passed **58 packaging checks** using artificial files, including rollback and rejection of changed or unqualified candidates.

The installed game passed Statistics, Card Tracker and menu suites again (**5,301 / 458 / 259** checks). All three live DLL fingerprints match the qualified candidate. The verified receipt is `artifacts/patch-installation.json`; recovery files retain the previous 1.4.20 installation. The original findings and planning-audit reproductions remain historical evidence. Installed results, binary fingerprints, source snapshots, captures and recovery information are recorded in the release notes and the 1.4.21 release archive.
