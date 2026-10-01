# End-game statistics — patch 1.4.8

**Status:** Installed and qualified on 30 September 2026. The installed player passes all 523 statistics checks and the startup/menu smoke test.

After the final scoring replay, select **Statistics** at the bottom left of the board, or in the original results dialog.

## Included

All ten pages from the implementation plan are available: Overview, Scoring, Industries, Networks & Geography, Economy, Resources & Trade, Actions & Cards, Turn Order & Tempo, Efficiency & Transition, and Timeline.

- Compare any subset of players and select Whole game, Canal or Rail. Introductory matches omit Rail.
- Search ledger rows by city, industry, resource or action; the category menu supplies common searches. Search is explicitly separate from the player/era filters that determine chart values.
- Sort table columns, expand rows for source details, scroll wide ledgers horizontally and load more rows as needed. Column headings remain available while scrolling.
- Hover over charts for exact values, toggle player series in the legends, and click a plotted value to inspect the relevant ledger. Cash, income and awarded VP use step charts with turn-number axes.
- Retain the page, filters, expanded rows, sorting, chart visibility and page scroll position while reviewing the same match.
- Escape returns to the entry context. The footer offers explicit results and board destinations. Closing statistics preserves the finished-review owner and match; leaving review performs the existing game cleanup.

The brown, ivory and gold panel follows the game's visual style. Player colors supplement names and table markers. The panel blocks board clicks, zoom gestures and keyboard input underneath it.

## What the figures mean

The report reconstructs a detached copy of the retained event history once, before the finished-review replay can rewind presentation or delete a local match. It never rewinds the live history or reparents board objects.

Cash and VP are counted from actual balance changes, with independent resource-stock and per-asset scoring reconciliation. Market purchase prices are already part of spending; they are not added twice. Confirmed actions include Pass, and multi-sale/double-rail actions remain one action. Canal level-1 cleanup does not subtract link VP already earned.

The panel describes observed events. Last-slot builds and reliance on opponents' links do not prove tactical intent. Printed industry opportunity is not the maximum attainable game score. An unfinished industry may still have supplied resources or market receipts; its construction investment is not labeled wholly wasted.

Some original events omit a sale's merchant destination. Merchant-beer records establish a destination; a uniquely eligible merchant can be identified as **Derived**. Other sales remain **Unattributed**. Missing or inconsistent history leaves the authoritative final result available and marks detailed figures unavailable.

AI style labels describe the starting configuration, with available difficulty/time metadata. They do not assert that the bot kept one intent throughout the match.

## Verification

The candidate statistics suite passes **523 native checks** over nine completed fixtures: full and introductory matches with two, three and four players, plus two-/three-/four-way unresolved draws. It checks confirmed action counts, final results, cash and VP accounting, producer-resource conservation, overlapping history, gaps, conflicting revisions, all ten page schemas, repeated open/close, return destinations, wheel/drag input ownership, duplicate player names and stale callbacks.

All ten pages were exercised at 1280 × 720, 1920 × 1080 and 2560 × 1440. Sixteen native render captures were inspected. Report collection took approximately **15–128 ms** in the final isolated run; this is a fixture measurement, not a guarantee for every machine or older save.

Fresh managed rules/state/save regressions and the existing native rule/city/save suites passed. Additional checks passed: 2,339 AI runtime/save checks, 184 presentation checks, 765 link replay checks, 79 installer recovery assertions, and 259 startup/menu smoke checks. The installed game also passes the full 523-check statistics suite. Results and runtime fingerprints are recorded in the release archive.

The AI DLL remains byte-identical to installed patch 1.4.7. Its qualified policy, profiles, thinking allowances, node limits and resource lifecycle are preserved. The archived 108-configuration AI matrix is retained for that identical DLL; it is not claimed as a newly run matrix.

## Files and recovery

- Plan: [END_GAME_STATISTICS_UI_PLAN.md](../rules-audit/END_GAME_STATISTICS_UI_PLAN.md).
- Implementation: src/BrassPatch/Statistics/.
- Native qualification: tests/BrassPatch/Statistics/run-native.ps1.
- Release archive: artifacts/patch-releases/1.4.8/.

Installation replaces the three qualified runtime DLLs using the existing atomic installer, retains a verified backup of patch 1.4.7, and leaves online play disabled. User save files are not migrated or modified by the statistics panel.
