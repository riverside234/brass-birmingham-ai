# Statistics revision: efficiency plots, end-game performance and overview awards

**Status:** Implemented and installed as **1.4.26** — 3 October 2026. The original design below is retained for traceability; implementation and verification details are in [the release notes](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/STATISTICS_EFFICIENCY_1.4.26.md>).

**Implementation baseline:** Accepted patch **1.4.25**. The revision applies to both ways of opening the shared Statistics panel: completed-game Statistics and Game History. Optimized and manually inspected at **1920 × 1200**, keeping the current light theme and default second text size. Existing 1080p automated checks are retained.

**Delivered:** Six efficiency plots, three internal tabs with independent scroll positions, shared exact-data/evidence queries, corrected realization/lifetime-return calculations, deduplicated ledgers, a scoped live presentation lease, stored-rank gold/silver medals and row-based result backgrounds. Runtime additions are `StatisticsEfficiencyPages.cs`, `StatisticsMedal.cs` and `StatisticsPresentationLease.cs`; there is no report schema migration or gameplay/AI change.

**Qualification:** 5,750 Statistics, 3,780 History, 646 menu, 490 Card Tracker, 10 performance and 110 packaging checks passed. All 26 planned 1200p plot/Overview captures were reviewed. The matched fixture brought live scroll p95 to 2.54 ms versus History's 2.33 ms, with the measurement limits and remaining cold-start/warm-response variation documented in the release notes. This is not a displayed-FPS claim.

## 1. Requested outcome

1. Add six useful efficiency plots, with explicit units, legends, filters and supporting records.
2. Make end-game Statistics as responsive as the same report opened from Game History.
3. Add a **gold medal for first place** and a **silver medal for second place** on Overview.
4. Give the first and second Overview result cards the same background, with the third and fourth sharing the next row's background.

Here, “tab color” is interpreted as the background of the Overview player-result cards: the current code reproduces the reported first/third grouping exactly. Player identity colors in names, score labels and chart legends remain consistent.

Keep the existing factual presentation. Do not reintroduce tactical grades, strength/weakness advice or an automatic coaching page.

## 2. Findings from the current implementation

| Finding | Evidence | Implication |
| --- | --- | --- |
| Efficiency currently has three plot models | `StatisticsPlotPages.Build`, page 8: player economics/VP comparison, individual tile cost-versus-VP scatter, and tile outcomes | Add complementary views instead of duplicating these plots. |
| Player-result cards alternate by rank parity | `StatisticsPlotControls.ResultStrip`: `surface.Alternate = p.Rank % 2 == 0` | Rank 1 matches 3; rank 2 matches 4. Change this to displayed-row grouping. |
| Result cards currently have no medal graphic | `ResultStrip` renders rank/name, final VP and totals | Reserve a small medal area without compressing the score or long names. |
| Live Statistics does not suspend the background presentation | `StatisticsPanel.Awake` sets `DisableMenusUnderneath=false`; `StatisticsEntryPoints.Open` hides the result dialog and shows Statistics | Background rendering, animation or update work is a credible cause of the reported lag. It is not yet a measured root cause. |
| Both entry points already use one panel implementation | `StatisticsViewHost`, `StatisticsEntryPoints` and `HistoryEntryPoints` | Keep one implementation and fix lifecycle differences at the host/session boundary. |
| Reports are already detached and captured once | `StatisticsEntryPoints.Capture` retains `CompletedMatchStatistics`; analytical page views are cached in `StatisticsReviewPanel.View` | Replaying/recollecting the completed match whenever the panel opens is unnecessary. Preserve this design. |
| A page redraw rebuilds its body controls | `StatisticsPanel.Render` clears/recreates the content and forces canvas layout | More plots could increase page/filter latency unless only the selected efficiency view is built and controls are reused where profiling warrants it. |
| Efficiency can append the same ledger twice | `PlayerEfficiency` adds `EfficiencyLedger`; the legacy page builder adds it again, and `Render` appends legacy tables | Remove the duplicate and ensure disclosure/sort/scroll keys identify one section. |
| Introductory bonus VP and printed opportunity use different bases | `StatisticsMetrics.AssetVP` includes `IntroVP`; `StatisticsChartQueries.Efficiency` divides that value by `CanalPrinted + RailPrinted` | A new realization chart must not turn this mismatch into a misleading percentage. Separate ordinary industry realization from introductory bonus points. |

The archived 1.4.25 native report records roughly **112 ms first-page response**, **45 ms median cached-page response**, and **0.81 ms median scroll-layout work** on the test machine. These are batch-player operation timings, **not FPS measurements** and not a matched live-versus-history benchmark. They do not establish the cause of the user's lag.

Relevant source entry points:

- [Overview cards and efficiency controls](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Statistics/StatisticsPlotControls.cs:41>)
- [Live Statistics opening lifecycle](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Statistics/StatisticsEntryPoints.cs:90>)
- [Shared panel rendering](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Statistics/StatisticsPanel.cs:250>)
- [Existing efficiency queries](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Statistics/StatisticsChartQueries.cs:26>)

## 3. Efficiency organization

Retain the existing **Efficiency & Transition** navigation entry. Add three compact internal tabs, following the Resources page's existing interaction style:

| Internal tab | Plots shown |
| --- | --- |
| **Returns** — default | Existing economics/VP efficiency; new industry VP realization; new construction return by industry |
| **Conversion** | Existing individual-tile scatter; existing tile outcomes; new completion-delay heatmap |
| **Transition & networks** | New investment remaining unflipped; new Canal carryover scoring; new VP per link |

Only the selected internal tab should construct plots and related tables. Preserve scroll position separately for each tab. Switching tabs must retain player/era selection and explain any change in metric scope. Reset filters returns to the default view; entering a drill-down opens the relevant internal tab automatically.

Keep the cost-basis control beside the player economics comparison, where it applies. It must not silently change construction-cost or link metrics. Each plot retains **About this chart**, **Exact data**, legend controls and evidence drill-down.

Use full-width chart cards by default. Use two small panels side by side only when both retain readable labels at 1200p; stack them at the largest text preset. Do not shrink text to fit six industry names or long player names.

## 4. Six additional plots

### E1 — Industry VP realized versus printed opportunity

**Purpose:** Show how much of the ordinary industry VP available at the actual scoring boundaries was realized.

- **Visual:** Horizontal bullet-style comparison, one row per player. A pale track shows recorded printed opportunity; a filled bar shows ordinary industry VP awarded. Direct labels show `awarded / printed VP` and the percentage. Reuse the bar/endpoint renderer where practical.
- **Numerator:** `CanalVP + RailVP`, restricted to the selected scoring era. This deliberately excludes `IntroVP`.
- **Denominator:** `CanalPrinted + RailPrinted`, for the same player and scoring era.
- **Carryover:** A surviving tile can contribute to both scoring opportunities; count each actual era snapshot once.
- **Introductory games:** Show the additional industry bonus as a separate, labeled value. Do not mix it into the normal realization percentage or clamp an invalid percentage to 100%.
- **Filters:** Player and scoring era. Count tiles built earlier if they contribute to that era's scoring snapshot. Detail/action filters do not narrow these player totals; state that above the chart.
- **Drill:** Tiles and scoring records supplying both totals, with Canal and Rail amounts separate.

Label this **recorded printed opportunity**, not “maximum possible score” or “optimal VP.” It excludes industries removed before a scoring snapshot and cannot represent every alternative move the player could have made.

### E2 — Construction return by industry

**Purpose:** Compare actual industry points earned for construction spending across the six industry types.

- **Visual:** Horizontal lollipop plot, six industry rows, with `VP / £ construction` values. Default to the selected players combined; allow **By player** using separate player panels with a shared numeric scale.
- **Calculation:** `sum(lifetime industry VP) / sum(BaseCost + MarketCost)` for the same set of tiles. Use the ratio of sums, not the average of individual tile ratios.
- **Scope:** Era selects the **build cohort** here. Label it explicitly, for example: `Built in Canal · lifetime industry VP through game end`. Whole game includes all built tiles.
- Include zero-VP tiles in spending. Count a carried tile's construction once even if it scored twice. Include actual introductory tile bonus VP in lifetime returns, identified in the tooltip.
- Include player, industry, region and asset-evidence filters. Action filters cannot allocate total tile return to Build versus Sell; show their inapplicability rather than silently changing the denominator.
- **Tooltip/exact data:** Tile count, actual VP, base cost, bought-resource cost, and ratio. A zero-cost denominator is unavailable, not infinite.

This measures industry points per construction pound. It excludes income, network access, link contributions and other benefits; a brewery or coal mine with a low value is not automatically a poor investment.

### E3 — Completion speed by player and industry

**Purpose:** Make the existing completion-delay table easier to compare.

- **Visual:** A compact heatmap: players in rows, the six industries in columns. Each cell shows **median own turns from building to flipping**, with the flipped sample count available directly or in a short second line.
- Use a sequential light-to-dark scale for elapsed turns. Darker means longer, without good/bad ratings. Share the scale across all visible cells.
- **Population:** Tiles built in the selected era/cohort and eventually flipped by game end. The caption says that later-era completion remains included. Show still-unflipped/removed-unflipped counts separately in the cell tooltip.
- A same-turn flip is a valid `0`; a cell with no observed flips displays `—`. Never give unfinished tiles a fake completion time of zero.
- **Drill:** The exact cohort of tiles, each tile's build/flip era and own/global turn interval. Reuse those records for the supporting delay table so chart and table agree.

Use measured delays, not a universal benchmark for how quickly a particular industry “should” flip.

### E4 — Construction investment remaining unflipped

**Purpose:** Show which industries held the most construction expenditure in unflipped tiles at the final board state.

- **Visual:** Horizontal stacked cash bars by industry: **base construction** and **bought resources**. Direct totals use `£`. Support selected-player totals and player panels.
- **Population:** Tiles with `Remaining && !Flipped`, narrowed by the selected build cohort, industry, region and player.
- Separate goods industries from resource suppliers in the grouping/labels. Include tile count and printed VP in exact data.
- Include supplied units and market receipts as supporting values; do not subtract receipts from the plotted construction cost or invent prices for free resources.
- **Drill:** The existing investment ledger, with the same tile identities and cash totals.

Call this **unflipped investment**, not “money wasted.” Partly consumed suppliers can already have provided useful resources and receipts. Removed tiles belong to transition/removal records, not final remaining investment.

### E5 — Points from tiles carried into Rail

**Purpose:** Show the scoring contribution of industries that survived the Canal transition.

- **Visual:** Stacked columns by player, split into **Canal industry VP** and **Rail industry VP**, with tile counts and total VP labels. Expand by industry if requested; keep the default concise.
- **Population:** Actual `Carried` tiles. Do not replace this with all level-2+ tiles or filter the population to tiles built during Rail.
- Era filters select the displayed scoring contribution; the cohort stays **tiles present at Rail entry**. Whole game shows both segments.
- Zero-point carried tiles remain in the cohort and tile count. Tooltips distinguish flipped, still unflipped and later removed tiles.
- **Drill:** Carryover ledger with the exact tile keys and per-era awards.
- **Introductory game:** Show a concise “No Rail transition in this match” state, not a fabricated zero-value comparison.

Reuse the existing Canal-cleanup ledger beneath this view. Its caption continues to explain that Canal links scored before cleanup; removed icons do not subtract VP already earned.

### E6 — VP per scored link

**Purpose:** Compare Canal and Rail network scoring efficiency without requiring the player to read individual route rows.

- **Visual:** Paired Canal/Rail lollipops on each player row, with different marker shapes, era colors, and direct `VP/link` labels. Use one shared numeric scale.
- **Calculation:** `sum(link VP) / count(scored links)` for that player and era. Include scored zero-VP links in the denominator.
- Label each mark with total link VP and number of links. No links means unavailable, not zero efficiency.
- Respect player and era filters. A regional detail scope may select links by recorded endpoint contribution, but count each selected link once and label this scope.
- Do not treat a two-link Network action as one link or divide a whole action's cost equally between links without recorded attribution.
- **Drill:** Existing network route/scoring evidence. Reuse its queries instead of creating a second source of link totals.

This is a VP measure; transport access, resource availability and benefits to other players remain outside it.

## 5. Calculation, filtering and label rules

Use one query result for each plot and its exact-data table. The new plots must work from the existing immutable report in both opening contexts, including older stored reports with the currently supported schema.

| Concern | Required behavior |
| --- | --- |
| VP/action and cash/VP | Preserve current player/era denominators. Confirmed actions include Pass. Borrowed cash is not income/VP; market payments are already included once in action spending. |
| Introductory realization | Keep ordinary industry awards and bonus awards explicit in E1 and the conversion ledger. Add a fixture proving the current numerator/denominator mismatch, then verify the corrected display basis. Preserve authoritative final VP. |
| Build era versus scoring era | Display scope beside every chart. E1/E6 use scoring era; E2/E3/E4 use build cohort and observed lifetime/final outcomes; E5 uses the fixed Rail-entry cohort with era-specific awards. |
| Different units | Use separate scales for pounds, VP, percentages and turns. No dual-axis overlay of unrelated measures. |
| Aggregation | Ratios use summed numerators and denominators. Medians use the full tile sample, not averages of player medians. |
| Missing/zero/negative | Keep genuine zero values distinct from unavailable history or zero denominators. Preserve valid negative net VP in the existing player-ratio plot. |
| Filters and drill-down | Apply only supported scopes and visibly identify ignored scopes. Carry exact asset/action/link keys through drill-down; an empty evidence set must not expand to all records. |
| Text search | Continue filtering ledgers, not changing the plotted population invisibly. |
| Duplicate tables | Produce one Conversion and efficiency ledger. Use stable section identities when combining analytical and legacy tables, preserving sort, disclosure and scroll state. |
| Axis labels | Currency ticks use `£`; ratios use readable 0/1/2-decimal steps; count ticks remain integers. Include axis units and direct values without clipping. Reuse `StatisticsAxisLayout`. |
| Palette | Player colors retain identity. Industry/era legends use their existing palette and symbols. Background row grouping must never recolor all series for the winner. |

## 6. Performance work

### P1 — Measure the opening-context difference

Use the **same completed four-player report**, encoded/decoded for the History case, with the same page, text preset, filters and scroll sequence. Do not compare unrelated games or different content volumes.

At 1920 × 1200 record three conditions: current live end-game viewer, live viewer with proposed presentation suspension, and Game History. Separate:

- Cold opening/report access from warm reopening.
- Idle frame time from sustained scrolling and pointer hover.
- Page, tab, legend, filter and text-size changes.
- Main-thread layout/TMP/mesh work, allocations and GC pauses, plus GPU/background rendering where the runtime supports measuring it.
- Active background animations, renderers/canvases and presentation update loops.

Use a warm-up period and repeated runs. Record median, p95 and p99 frame times; keep PNG capture/encoding outside timed intervals. Existing CPU layout samples must not be presented as an FPS guarantee.

### P2 — Suspend live background presentation while Statistics is open

Add a small, disposable **live Statistics presentation scope**, owned by the live viewer session. Archive viewing continues through the same panel without acquiring a live-game scope.

1. Acquire only after final report capture and the existing finished-review readiness boundary. Complete or explicitly manage the results-dialog transition before suspending presentation.
2. Save the original state of the specific underlying presentation objects/tweens affected. Pause only animations that were playing and belong to the underlying completed-game presentation.
3. Suppress unnecessary board/world rendering under the opaque Statistics window and use an allowlist for expensive presentation updates proven by profiling.
4. Keep the Statistics canvas, its ancestors, required camera, EventSystem, router, scrollbars, pointer input and UI transitions active. A shared camera/parent canvas must not be disabled.
5. Review `OnDisable` behavior before changing component activation: for example, `PlayerRowsLayout.OnDisable` cancels its highlight. Prefer a narrow suspension guard for such updates over a broad `SetActive(false)` on game roots.
6. Restore exactly the prior state before returning to Review/results. Release idempotently on normal Close, Escape, show failure, router/session disposal and scene destruction. Do not restart animations that had already been paused or finished.
7. Keep score state, game history and asynchronous report persistence untouched. Do not unload the completed match merely to show Statistics.

**Do not use `Time.timeScale = 0` alone as the fix:** it does not stop rendering or every `Update`/unscaled animation. Also avoid a blanket global DOTween pause, which can stall the panel/router's own show/hide sequences. If profiling requires a clock pause in addition to scoped presentation suspension, make its ownership/restoration explicit and move required UI transitions to unscaled time before enabling it.

### P3 — Bound work inside the viewer

- Build only the active efficiency tab. Add its identity to the analytical cache key and maintain per-tab scroll positions.
- Preserve detached report/page-query caching. Do not recollect, reload or deserialize a report on every tab change.
- Remove the duplicate ledger and avoid constructing large exact-data rows until requested. Retain the existing incremental row limit for expanded long tables.
- Coalesce multiple filter changes in a frame into one render. An idle frame or ordinary scroll must not rebuild the page or rescan the complete report.
- Where profiling shows redraw stalls, reuse stable plot/header controls and update their data instead of destroying the entire page. Precompute repeated mark positions/stack totals for a draw rather than repeatedly scanning all marks.
- Keep layout refreshes scoped to the affected UI. Skip unchanged sticky-header/tooltip updates; preserve the current pointer-change guard.
- Keep caches bounded by the current report and filter key. Disposal must release any presentation scope and retained Unity objects.

**Performance targets on the same machine:** aim for smooth 60 Hz interaction where hardware permits. More importantly, warm live end-game p95 frame time should be within **20% or 2 ms, whichever allowance is larger**, of the matched History case. Target warm page/tab/filter changes under **100 ms** and normal first-page presentation under **200 ms**, measured separately from report collection and disk access. Treat these as goals to verify, not results already achieved; retain traces and investigate any remaining gap.

## 7. Overview medals and card backgrounds

### Awards

- Read authoritative `StatisticsPlayer.Rank`, already stored in both live and archived reports.
- Rank **1** gets a gold medal; rank **2** gets a silver medal. Other ranks retain their numeric place without a new medal.
- Use a small native UI graphic with a circle, ribbon and readable `1`/`2`, avoiding reliance on emoji font support. Starting colors: gold `#C69736` with a darker brass border; silver `#BAC3CC` with a slate border.
- Reserve approximately 36–40 physical pixels plus padding beside the name at the default text size. Measure long names and final-VP widths; retain full details in the tooltip.
- Show “1st place”/“2nd place” text or tooltip as well as color. Respect shared ranks: every rank-1 player gets gold and every rank-2 player gets silver. Do not assign silver merely to the second displayed card in a tie.
- Unknown/unranked records get no medal and an explicit unavailable-rank label. Do not infer a winner from VP alone or change the game's tie-break rules.

### Background grouping

Keep the two-column result layout. Choose the background by **displayed row**, not rank parity:

```text
[ Gold · 1st player    ]  [ Silver · 2nd player  ]   same ivory background
[        3rd player   ]  [          4th player  ]   same alternate background
```

Use `(displayIndex / 2) % 2` for row banding or an equivalent explicit row style. With tied ranks the visual rows still share their backgrounds, while medals continue to reflect actual rank. Two-player games show one matching row; three-player games leave the unused position empty.

Keep name/VP text in the player's established color. The medal indicates placing; background indicates row grouping. Final rankings remain visible and authoritative even when player/era detail filters change.

## 8. Implementation sequence and files

| Order | Work | Principal code |
| --- | --- | --- |
| 1 | Create matched live/History performance capture and record the baseline; reproduce duplicate ledger and introductory realization cases | Existing Statistics/History native fixtures and archived 1.4.25 evidence |
| 2 | Add safe live presentation suspension and verify every close/disposal path | `StatisticsEntryPoints.cs`, `StatisticsViewHost.cs`, proposed `StatisticsPresentationScope.cs`; narrow presentation hooks only if profiling requires them |
| 3 | Add medal rendering and row-based overview backgrounds | `StatisticsPlotControls.cs`, proposed medal graphic in the Statistics namespace, scoped theme tokens |
| 4 | Add the efficiency tab state and shared calculation queries; deduplicate ledger assembly | `StatisticsPages.cs`, `StatisticsReviewPanel.cs`, `StatisticsPanel.cs`, `StatisticsChartQueries.cs`, `StatisticsPlotControls.cs` |
| 5 | Add E1–E6 and their exact-data/evidence records | Proposed `StatisticsEfficiencyPlots.cs` / `StatisticsEfficiencyQueries.cs`, with dispatch from `StatisticsPlotPages.cs` |
| 6 | Reuse plot kinds/axis tools; extend only the presentation features needed for bullet tracks, heatmap labels and efficiency ratios | `StatisticsPlotViews.cs`, `StatisticsPlotSemantics.cs`, `StatisticsAxisLayout.cs`, plot models if required |
| 7 | Profile again, address measured redraw costs, inspect actual 1200p output in both contexts | Native render and performance harness |
| 8 | Package the implementation from accepted 1.4.25, qualify, install when implementation is requested, verify and archive | Existing fingerprinted build/install/recovery workflow |

Prefer calculations over the existing report schema. If a required value cannot be derived reliably from older history, label that metric unavailable and define compatibility handling before changing storage. No gameplay or AI-search revision is part of this work.

## 9. Acceptance checks

### Data

- Whole-game, Canal and Rail values agree between each plot and its exact-data rows; copied live/history reports produce the same values and identities.
- Include carried tiles scoring twice, scored zero-VP links, two-link Network actions, market-funded construction, free resources, removed suppliers, no-build/no-link games, and same-turn flips.
- Ratios aggregate by totals, market costs are counted once, and no zero denominator produces infinity or a misleading zero.
- Ordinary realization and introductory bonuses remain separate; partial history shows unavailable detail rather than invented data.
- There is one efficiency ledger with stable disclosure/sort/scroll behavior.

### Overview

- Test two, three and four players; long/duplicate names; shared first/second ranks; and unknown ranks.
- In the ordinary four-player case, cards 1/2 match and cards 3/4 match. Gold/silver are determined by stored rank.
- Verify all three text presets at 1200p, including medal/name/score spacing. Repeat through Game History.

### Lifecycle and performance

- Open/close from both end-game results and finished-board review; return to the correct previous view with the board pose and existing visibility restored.
- Repeat at least 30 open/close cycles and exercise Escape, direct Hide/disposal, scene change and an injected show failure. No stranded pause state, lost UI input, resumed completed tween or retained scope.
- Report persistence can finish while Statistics is open. Reading archived statistics does not acquire a game suspension scope.
- Compare matched frame/response measurements at 1200p; verify no idle/scroll-triggered full-page rebuild and no growing retained-object count after warm-up.
- Manually inspect the three efficiency tabs and Overview from both entry points at 1200p. Retain normal Statistics/History, player-selection and Card Tracker regression coverage appropriate to the files changed.

Deliver the measured before/after performance summary, 1200p captures, calculation checks and release notes with the implementation. This plan makes no claim that the reported lag has already been reproduced or fixed.
