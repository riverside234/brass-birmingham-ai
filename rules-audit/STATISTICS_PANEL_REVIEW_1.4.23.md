# Statistics review for patch 1.4.23

Date: 3 October 2026 (America/Chicago).

Scope: the existing end-game Statistics viewer and that same viewer opened from Game History. Visual review is at **1920 × 1200**. This is a factual report and UI review, with no strategy grades or coaching page added.

## Reproduced display defects

| ID | Defect | Reproduction and correction |
| --- | --- | --- |
| S23-01 | Formatting-like player names alter Statistics labels | A detached completed report with a `<size=80><color=red>…` nickname was interpreted by TextMeshPro. Report text now disables rich-text parsing within Statistics only. Other panels keep their existing behavior. |
| S23-02 | Long chart row labels exceed fixed-height rows | Long duplicate names on both efficiency plots exceeded their label rectangles. Row height now uses the measured wrapped text at the active font size. |
| S23-03 | Curve endpoint rectangles extend outside the chart | The vertical-axis layout replaced the curve's reserved right gutter with a smaller constant. Both cash and income endpoints exceeded the plot width. The layout now preserves the requested gutter. |
| S23-04 | Geography labels overlap in dense southern cities | The 1200p map placed Worcester and Wolverhampton labels over each other after nearby candidates were exhausted. Placement now searches the full diagram, avoids existing labels and full city-symbol bounds, and uses light text backing. |
| S23-05 | Heatmap exact values show categorical axis names as units | The resource ledger displayed values such as `3 Supplier`; industry heatmaps similarly used `Industry`. Tooltips and exact data now explicitly say `3 units` or the actual `VP` amount. Counts and scoring calculations were already correct. |

The pre-fix native report is saved as `artifacts/game-history-build/statistics-before-plot-fixes.json`. It contains failing reproductions for S23-01–03 and expected failures for newly specified visual improvements. Map and position-key screenshots preserve the subsequent visual findings. S23-05 was identified in the rendered exact-data ledger and now has unit-label assertions. Visual expectation failures are not counted as additional data bugs.

## Visual improvements

- Repeat numeric scales and units above long horizontal charts and event/lifespan lanes.
- Add quiet alternating row backgrounds while preserving zero baselines and grid lines.
- Label sufficiently wide stacked segments with their count or percentage. Small segments keep hover/click and exact-data access.
- Use a blue shared intensity scale for industry VP, visible continuous heatmap ramps, and an explicit **Total** heading.
- Give turn-position heatmaps separate colored position keys, including **1st · first**.
- Keep all position keys inside the initial visible width even when the round grid scrolls horizontally.

All chart geometry, labels and pointer hit bounds use the same numerical axes. Legend hiding retains the original scale and denominator, as stated in the legend tooltip. Percentage totals remain based on all series.

## Calculation and filter audit

New independent checks run against each player and era in both entry paths:

- Scoring-category sums against recorded signed VP changes.
- Industry VP against Canal, Rail and introductory awards, retaining Canal-built carryover when scoring Rail.
- Confirmed action totals, including Pass, against raw action records.
- VP/action and cash/VP with both action-spending and all-negative-payments bases; no market double-counting and no invented value for a nonpositive denominator.
- Incoming and outgoing Coal/Iron/Beer quantities against raw resource-use entries; source mix and supplier/consumer matrix must agree.
- Market purchase price as total paid divided by units purchased; no purchases remain unavailable.
- Rendered player/era filters, ledger-only search, and switching Supply/Market/Merchants without stale cached plots.

The existing independent Statistics tests continue to cover development counts, card-payment/wildcard episodes, merchant attribution, turn order, link contribution sums, partial histories, zero/negative values, evidence drilldowns and pointer hit geometry. History codec tests compare every report field, backing collection and derived plot/ledger model before and after persistence.

No new calculation discrepancy was reproduced in the pre-fix audit. A passing finite fixture suite is not a claim that every possible match history has been exhaustively verified.

## Final verification

The exact candidate passed **5,613 Statistics checks** and **3,741 Game History checks**, plus the existing Menu, Card Tracker, Presentation and Link suites. The shared native review captured 30 distinct plots at 1200p through each entry path, including scrolled plots, all resource tabs and long-name fixtures. Manual inspection covered the live plots and archived examples across every page. The corrected map leaves complete city symbols visible; continuous and discrete heatmap keys and long-chart scales remain readable.

Compatibility qualification confirms **14,268 unrelated method bodies**, the AI binary and all embedded resources are preserved. The installed game passed the same **5,613 Statistics**, **3,741 History** and **492 Menu** checks. Final fingerprints, the recovery backup and release evidence are recorded in [release notes](../docs/GAME_HISTORY_1.4.23.md).
