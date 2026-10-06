# Patch 1.4.26 — Statistics efficiency, performance and Overview

Implementation of `rules-audit/STATISTICS_EFFICIENCY_PERFORMANCE_OVERVIEW_PLAN.md`, built from the accepted 1.4.25 archive. Both completed-game Statistics and Game History use these changes.

## Player-visible changes

The shared **Efficiency & Transition** page now has three internal tabs. The selected tab remembers its scroll position, retains player/era filters, and is the only tab whose charts are constructed. Reset returns to **Returns**.

| Tab | Charts |
| --- | --- |
| Returns | VP/action and cash/VP; industry VP awarded versus recorded printed opportunity; construction return by industry |
| Conversion | Individual tile construction cost versus lifetime industry VP; final tile outcomes; median own turns to flip, with sample counts |
| Transition & networks | Construction investment still unflipped; Canal carryover scoring in each era; Canal/Rail VP per scored link |

Charts include labeled units, readable numeric axes, legends, exact-data tables and evidence navigation. Construction return and unflipped investment can be combined across selected players or shown in player panels with a common scale. The cash-basis control appears only on Returns and only changes the player economics comparison.

Overview awards a native gold medal to stored rank 1 and silver to stored rank 2, including shared ranks. Unknown ranks remain explicitly unranked. Cards share their background by displayed row: first/second together, third/fourth together. Player name and VP colors retain their player identity.

## Calculation and interaction fixes

- Ordinary industry realization excludes introductory bonus VP. The bonus has its own value. Recorded opportunity describes actual scoring snapshots, not an optimal hypothetical score.
- Construction return uses total lifetime industry VP divided by total construction cash for the same build cohort. Purchased resources are included once; carried tile construction is counted once; zero-VP tiles remain in the denominator.
- Completion medians include later-era flips of the selected build cohort. Same-turn flips are valid zeroes. Unflipped tiles do not become zero-delay samples, and medians are not summed into heatmap totals.
- Final unflipped investment excludes removed tiles. Market receipts, remaining stock and supplied units are evidence rather than deductions or invented cash values.
- Carryover uses the actual Rail-entry population, including tiles scoring zero. An industry built in Rail is not carryover merely because it is level 2 or higher.
- VP/link counts distinct scored links, including zero-VP links. A two-link Network action contributes two links. Repeated endpoint contributions do not multiply the denominator.
- Zero denominators and missing observations display an em dash. Exact-data disclosures have stable plot IDs, including player facets with duplicate names. The duplicated efficiency table is removed.

## Live viewer performance and lifecycle

Opening Statistics from the completed game acquires a disposable presentation lease. It suppresses underlying view canvases/renderers and safe-area/hover callbacks, and pauses only playing tweens whose targets belong to that presentation. Menu transition sequences, the Statistics canvas, the shared camera, input routing and EventSystem remain available. It does not change global time scale or pause every tween.

Closing, hiding, destroying or disposing the panel releases the lease and restores recorded component/tween states. Already paused animations stay paused; killed/completed animations are not restarted. A failed show releases the lease and removes the broken panel so it can be reopened. History opens the same Statistics implementation without acquiring a live-game lease.

Detached report/query caching remains bounded. Scroll and idle frames do not reconstruct pages; expanded long tables retain their 60-row increment. Report persistence is independent of the presentation lease. No report schema or game-save migration is required.

## Verification and measured performance

Candidate qualification passed with **5,750 Statistics**, **3,780 History**, **646 menu**, **490 Card Tracker** and **10 matched-performance** checks. All suites report zero failures. The preservation check confirms 14,395 existing method bodies outside the allowed Statistics changes and all 48 AI resources are unchanged.

Manually reviewed **26 native Unity captures at 1920 × 1200**: all six new plots in both entry points at the default second and largest text presets, plus both Overview panels. Automated checks cover filters, exact-data identities, player facets, shared scales, ties, unavailable ranks, duplicate names, zero/missing values and later-era completion. The existing 1080p regression checks remain. Lifecycle checks include 30 repeated openings, original component/tween state restoration, disposal, direct Hide and injected show failure followed by successful reopening.

The performance probe uses the same completed four-player report and a codec-decoded History clone, the real board prefab, Large text and explicit 1200p camera rendering. Hardware: Intel Core Ultra 5 135U / Intel Graphics. Each frame sample contains camera rendering and one frame yield, with PNG encoding excluded. The fixture omits the player HUD, so these are **CPU frame intervals, not displayed FPS, GPU-only timings or a live-session guarantee**. The 1.4.25 baseline was run from a separate copy of the accepted binaries.

| Mode | Idle p95 | Scroll p95 | Hover p95 | Warm page median / p95 |
| --- | ---: | ---: | ---: | ---: |
| 1.4.25 live baseline | 3.35 ms | 5.60 ms | 4.19 ms | 51.62 / 73.07 ms |
| 1.4.26 live, suspension disabled for comparison | 3.67 ms | 4.97 ms | 3.98 ms | 55.00 / 70.32 ms |
| 1.4.26 live, suspension enabled | 0.70 ms | 2.54 ms | 2.47 ms | 45.91 / 90.66 ms |
| 1.4.26 History | 0.65 ms | 2.33 ms | 2.82 ms | 40.49 / 73.27 ms |

Live idle, scrolling and hovering meet the plan's matched-History p95 allowance (20% or 2 ms, whichever is larger) in this fixture. Warm live tab/filter/text-size p95 values were 77.03 / 74.01 / 84.60 ms. Warm page and filter responses still vary more than History, but all measured warm interaction p95 values were below 100 ms. Rebuilding the page on those explicit changes remains; ordinary scrolling and hovering retain the chart objects.

First panel opening in each process also includes graphics/font warm-up: 205.35 ms for the old baseline and 215.50 ms for the candidate's unsuspended control, slightly above the 200 ms goal. Later new-panel openings were 63.76 ms for suspended live and 61.58 ms for History; their warmed state prevents claiming a cold-opening improvement. Full samples, medians, p95/p99 values and native logs are retained as `performance-native.json` and `performance-baseline-1.4.25.json` in the release evidence.

## Installation and preservation

Installed in the Steam game directory on **3 October 2026 at 21:17:57 CDT**. All three installed DLL hashes match the qualified candidate. The 1.4.25 game and AI DLLs are preserved byte for byte; all 48 embedded AI profiles, Card Tracker, player selection, History persistence and gameplay behavior are retained. Online remains disabled.

The installer saved the previous release under `artifacts/patch-recovery/aa304493ab624195894eaabc19bee930`. The installation receipt is `artifacts/patch-installation.json`. Packaging/recovery validation passed **110 checks using artificial files**; user saves and History entries were not used as test fixtures.

Post-install native testing passed **5,750 Statistics** and **646 menu** checks against the installed binaries. The accepted release is archived at `artifacts/patch-releases/1.4.26` with compiled source snapshots, qualification reports, native logs, reviewed captures, recovery provenance and the 1.4.25 comparison evidence. The same qualified package is staged at `artifacts/patch-build` for subsequent reinstallations.

| Installed file | SHA-256 |
| --- | --- |
| Assembly-CSharp.dll | `50BF8A2F8A167C1FE5679D37C72A5D5D1F1B01BDFA144EBE5811F5DAF7552699` |
| BrassPatch.dll | `C0D1E126B7F21132FEAAA5B8D392D47D80261D2356EFD13A74750164763A3A79` |
| BrassLegacyAI.dll | `366ADB638E64A951DB87EAA7F52222690CA62D866E64D938DDD0F0F214A69B42` |
