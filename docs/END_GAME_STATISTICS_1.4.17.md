# End-game statistics — patch 1.4.17

This release implements the core stages A–G of `rules-audit/END_GAME_STATISTICS_IMPROVEMENT_PLAN.md`. It builds on the installed **1.4.16**, preserving that release's game and AI binaries. The unchanged helper baseline is reconstructed from its verified 1.4.13 source archive, then only statistics sources are revised.

## Using the panel

- After the final scoring animation, select **Statistics** on the results screen or the larger button at the lower left of the review board. The board button is 240 × 64 physical pixels; the results button is 240 × 60.
- **Match review** opens first. Choose a review player and a reference player, read the score-gap waterfall and conversion plot, then inspect strengths, supported concerns and up to three next-game experiments.
- The ten existing pages retain their IDs and add varied plots: stacked bars, dumbbells, heatmaps, paired bars, delay dots, route lollipops, cash/income steps, waterfalls, spending columns, tile-efficiency scatter plots, action lanes and industry lifespans.
- Select plot marks or **Show evidence** to inspect exact records. **Show data** expands a ledger; long ledgers use pages of 60 rows. Legends toggle series without changing the numerical scale. Mouse, keyboard, tooltips, table sorting, sticky headings, scroll restoration and the three text sizes remain available.
- Player and era controls affect analytical views. **Filter detail** applies typed industry, action or region filters. **Search ledger** affects tables only. Whole-match score gaps, final results and cash reconciliation are explicitly labeled.
- Resources offer Coal / Iron / Beer and incoming / outgoing views. The matrix, companion bars, headlines and exact source ledger use the same direction and records. Outgoing keeps all counterpart consumers visible; incoming keeps all counterpart suppliers visible. Industry/region filters refer to source tiles and action filters to their consuming actions. Switch to incoming for the broader consumer, market-purchase and merchant profiles; outgoing retains market-sale records separately from units supplied to actions.
- Geography offers historical boundary selection and occupancy / industry VP / link-contribution views. It uses board region colors, separate farms and merchants, full city labels and horizontal scrolling on narrow screens. It is an independent topology schematic, not a record of paths taken.

## Additional analysis and accounting fixes

The detached report now records owner-turn ordinals, historical cash/income/mat/stock boundaries, typed transaction prices, typed link contributions and resource-depletion flip attribution. New ledgers cover sale batching, completion delays, income recovery after loans, actual income payments and subsequent paydays, development follow-through, Rail entry and its first two actions, unfinished investment, brewery demand, market unit prices, wildcard episodes, merchant attribution and late cash conversion.

Two existing statistical accounting problems were corrected:

1. Market purchases now contribute once to action and round spending. Their detailed resource prices are an attribution of that spending, not another expense.
2. Each income-payment event creates a payday record for every player, including zero payments. The game processes the payment event for all players together.

The review uses deterministic local rules with record IDs and versioned observations. It separates observed facts from interpretation and suggested experiments. It does not label loans, shared supplies, retained cash or zero-VP links as inherently poor tactics. Printed points on unsold tiles are nominal opportunities, not proven attainable lost VP. Missing history suppresses detailed advice; ambiguous merchant destinations remain unattributed. Introductory scoring retains its own context and suppresses Rail advice.

## Verification and installation

Release qualification is stored in `artifacts/statistics-review-build/validation.json` and the recoverable archive `artifacts/patch-releases/1.4.17`.

- **2,793 statistics assertions**, including hand-built metric/insight fixtures and nine completed two-, three- and four-player games, full and introductory variants, draws, gaps and overlapping history.
- All 11 pages at 1280 × 720, 1920 × 1080 and 2560 × 1440, with Large / Larger / Largest text; **119 native captures**, including the later plots reached by scrolling.
- Fresh Card Tracker **539**, presentation **184**, link replay **765** and startup/menu **259** checks on the release candidate.
- Game and AI DLL hashes match accepted 1.4.16. Cecil verifies all 10,543 game methods and all helper methods outside Statistics are unchanged. Shared StatisticsWidgets source is also byte-identical to the accepted helper source.
- **54 installer assertions** verify fingerprints, validation rejection, reinstallation, backups, locked-file rollback and retry with artificial files. Installation uses the existing atomic installer and retains a recoverable backup of all previous DLLs.
- The **2,793 statistics** and **259 startup/menu** assertions also passed against the installed files before the release archive was accepted.

The qualified native candidate recorded collection times of 15–102 ms across nine matches. Its first page response, including collection and initial UI, was 114 ms. Across all three resolutions, cached page responses had a 25 ms median and 51 ms maximum; the two resource/direction filter responses were 18–28 ms. Twelve expanded-ledger scroll layout updates took 0.8–2.0 ms. These are measurements on this computer in the batch player, with response samples including frame waits/layout and excluding PNG encoding. Scroll layout time is not a frame-rate guarantee. Exact samples and hardware metadata are retained in `statistics-native.json`.

Some large batch-mode render runs require cleanup of their owned Unity test process after the completed assertion report has been flushed. This is recorded as `batchShutdownCleanup`; the runner never accepts an incomplete report. It does not terminate a user's game.

The game remains local-only. AI strategies, search budgets, worker policy, gameplay, existing saves, Card Tracker pause/input behavior and city highlights are preserved.

## Optional analyses

Stage H remains optional: individual historical sale-feasibility probes, exact participation of a particular link when several paths exist, and richer Sankey/path overlays. The shipped report does not claim these analyses or infer a better legal move from their absence.

## Rebuild or reinstall

```powershell
pwsh -NoProfile -File scripts/build-statistics-review.ps1
pwsh -NoProfile -File tests/BrassPatch/Statistics/run-review-native.ps1 -Suite statistics
pwsh -NoProfile -File tests/BrassPatch/Statistics/run-review-native.ps1 -Suite tracker
pwsh -NoProfile -File tests/BrassPatch/Statistics/run-review-native.ps1 -Suite presentation
pwsh -NoProfile -File tests/BrassPatch/Statistics/run-review-native.ps1 -Suite link
pwsh -NoProfile -File tests/BrassPatch/Statistics/run-review-native.ps1 -Suite menu
pwsh -NoProfile -File tests/BrassPatch/Statistics/qualify-review-release.ps1
pwsh -NoProfile -File tests/BrassPatch/Statistics/packaging-review.ps1
pwsh -NoProfile -File scripts/install-combined-patch.ps1 -BuildDirectory artifacts/statistics-review-build
```

Close the game before installation. To verify the completed archive without changing the game or shared staging, run `scripts/stage-archived-patch.ps1 -Version 1.4.17 -VerifyOnly`.
