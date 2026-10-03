# Statistics light theme — patch 1.4.21

**Status:** Installed and verified on 2 October 2026 (Chicago time). The installation receipt records `2026-10-03T01:13:05.1240418Z`.

This revision implements [the approved revision plan](../rules-audit/END_GAME_STATISTICS_LIGHT_THEME_REVISION_PLAN.md). It is designed for **1920 × 1200**, with **1920 × 1080** qualification. The second text preset, **115%**, remains the default.

## What changes

- Statistics opens on **Overview**. The advice page, strengths/weakness verdicts and proposed next-game experiments are removed. Ten factual pages retain their existing identities and drill destinations.
- The panel uses warm ivory surfaces, dark text, brass controls and readable player colors. Tables, legends, tooltips and selected/disabled states share the same scoped theme.
- Overview leads with compact final results. Final VP has its own reserved space, so a long player name cannot hide the score. Duplicate player names are distinguished by player color.
- Numerical axes use measured labels, sensible rounded intervals and correct units. Counts and turns have integer ticks; price/ratio ticks have sufficient precision; negative cash and VP stay signed. Numerical axis labels do not use ellipsis.
- The retained plots have clearer totals, conversion percentages, curve endpoints, era differences, delay medians and resource row/column totals. Waterfalls have connectors and readable signed labels. Timeline endpoints distinguish flipped, removed and still-open tiles.
- Legends wrap to their measured height and show **On/Off** states. Hiding a series preserves its numerical scale and percentage denominator. Dense scatter marks offer a record chooser while retaining their exact coordinates.
- **Exact data** and typed evidence remain available. Mouse selection uses the displayed cell, bar or interval shape, including after scrolling. The fixed header, navigation and return controls remain visible.
- Resources has separate **Supply**, **Market** and **Merchants** views, with retained scroll positions and explicit filter scopes. Action and card charts support counts/shares.

## Seven added chart families

| Page | Chart | Unit and calculation |
| --- | --- | --- |
| Industries | Development by industry and level | Tiles removed, grouped into levels 1, 2 and 3+; one recorded removal counts as one tile |
| Networks & Geography | Sources of link VP | Own industries, other players' industries and merchants; includes zero-VP links in link counts and VP/link |
| Resources · Market | Coal and Iron purchase price profiles | Weighted total paid ÷ units bought; no purchases is unavailable, never a £0 purchase |
| Actions & Cards | Card purpose/type | Build authorization, other action payments and Scout extra trades; wildcard acquisition and draws are excluded |
| Resources · Merchants | Sold goods by destination | Industry tiles sold, with unattributed destinations explicit; merchant bonuses keep their separate units |
| Turn Order & Tempo | Turn-position shares | Recorded known positions only, with the actual two/three/four-player seat count |
| Efficiency & Transition | Economics and VP efficiency | Separate, aligned **VP/action** and **£/VP** plots; whole-game or selected-era player totals; action spending or all actual negative cash payments |

N7 uses the same calculation as its efficiency ledger. Detail filters cannot narrow only a ratio's denominator. Resource market costs are already included in action spending and are counted once. A zero action denominator gives `—`; zero or negative VP gives no £/VP ratio.

## Data and interaction repairs

The [planning audit](../rules-audit/END_GAME_STATISTICS_PLOT_DATA_AUDIT.md) records the original SD01–SD11 reproductions. This patch repairs all eleven groups:

| ID | Repair |
| --- | --- |
| SD01 | Separate wildcard acquisition episodes by card identity and acquisition revision; preserve use/reacquisition associations and distinct supporting rows, including Scout extras |
| SD02 | Count Rail carryover from the boundary population, including Canal-built tiles |
| SD03 | Use consistent player/era score, action and spending populations for efficiency ratios |
| SD04 | Align resource plots and ledgers under their declared resource/action/source/evidence scopes |
| SD05 | Show semantic units and values for maps, percentages and intervals, rather than display coordinates |
| SD06 | Open actual contributing links from a link-VP map selection |
| SD07 | Exclude unrelated contributions when link evidence is selected |
| SD08 | Keep complete first/last grouped columns inside their category bands |
| SD09 | Keep overlapping scatter marks inside the plot and make every exact record selectable |
| SD10 | Use full rectangular/interval pointer areas and expose the represented evidence |
| SD11 | Use whole-number count/time ticks with unique, measured numeric labels |

Turn-position selections also keep heatmaps, shares, spending summaries and exact round tables aligned. Empty recorded orders are unavailable and cannot fabricate a back-to-back turn. Loan recovery is explicitly observed through match end; era filtering selects the loan's acquisition era, so a later Rail recovery for a Canal loan remains valid. Historical industry-VP maps explicitly describe VP from assets present at the chosen boundary.

## Preservation and verification

Only the Statistics helper changes. `Assembly-CSharp.dll` and `BrassLegacyAI.dll` are carried forward **byte for byte from accepted 1.4.20**, preserving the current v4 masked AI bank and its two game workers. Every helper method outside Statistics is compared against that baseline. Card Tracker, custom AI menus, gameplay, save handling and the existing light Statistics entry positioning are retained. Online play remains disabled.

Qualification uses independent typed fixtures, nine completed matches, native Unity rendering and actual pointer events. It covers all ten pages, all three text presets, 1200p/1080p, all Resources views, both efficiency cash bases and all era scopes. Additional previews include smaller games, introductory mode, partial history, ties/zero values, negative cash/VP, and duplicate long names. Card Tracker, presentation, link replay and AI menus are rerun on the exact candidate DLLs.

The candidate passed **5,301 Statistics**, **458 Card Tracker**, **184 presentation**, **765 link-replay** and **259 menu** assertions, with no native errors. The preservation check verified **12,067 unchanged existing method bodies**. One changed compiler-generated anonymous type is referenced only by Statistics; other runtime methods and fields do not reference it. The installer passed **58 packaging checks**, including rejection of unqualified/tampered candidates, backup verification and rollback after a locked replacement.

After installation, the actual game files passed Statistics, Card Tracker and menu checks again (**5,301 / 458 / 259**). The three live DLL fingerprints match the qualified candidate. The installation receipt is [patch-installation.json](../artifacts/patch-installation.json); the previous 1.4.20 files are retained in `artifacts/patch-recovery/8fdfe046241546d7a7c078527100ce12`.

| Installed file | SHA-256 |
| --- | --- |
| Assembly-CSharp.dll · unchanged 1.4.20 | `7595C64939F000CC1A68BBC6F6B3C9B7324919E5F3E4A6AA49D9564246ECE35C` |
| BrassLegacyAI.dll · unchanged 1.4.20 | `C785597B44463C5BD0C64EE49404014532D0B8B200FF0C908099106A543984A0` |
| BrassPatch.dll · revised Statistics | `68614CE9C3E91490266D486FDF0650706CA4268C2D0526E90D320DD1D674F950` |

Exact qualification, installed reports, source snapshots, native captures and the checked release inventory are stored under [artifacts/patch-releases/1.4.21](../artifacts/patch-releases/1.4.21). The installer checks the live baseline, preserves save paths, retains a recovery backup and publishes the combined package atomically. Gameplay and AI binaries remain the accepted 1.4.20 files; online play remains disabled.
