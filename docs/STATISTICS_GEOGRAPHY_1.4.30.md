# Patch 1.4.30 — Larger, smoother geography map and compact city tags

Date: 9 October 2026  
Base: accepted patch **1.4.29**  
Status: **installed and verified**.

## Map

- The diagram uses a **1.75:1** aspect instead of 2.7:1. At 1920 × 1200, its measured height is **512 px** at the default second text preset and **469 px** at the largest preset. Reduced header text and a shorter tag reserve make room for the larger board.
- Re-spaced the northern, central and southern cities to follow the printed board's arrangement. Adjusted the southern branch and longer corridors to match the new positions. All 27 locations and existing Canal/Rail connections are preserved.
- Retained the two era views, player-colored links, owner shapes, optional printed routes and label-free initial map. Era switching does not rearrange or rescale the board.

## Aliasing: cause and solution

The previous route renderer made each segment from two solid triangles, with an abrupt opaque-to-transparent boundary. At diagonal angles that boundary steps across the pixel grid. Its circular markers were also 16-sided polygons, which made their outlines look faceted. Enlarging the layout alone does not fix either issue.

The map now uses a **0.9-screen-pixel alpha fade** around its geometry, retaining a solid center. Connected segments share their edge geometry at bends, and the stroke ends are rounded. Circles use 48 segments and the same smoothing; merchant/farm/player shapes keep their distinct silhouettes. Map, tag and legend symbols account for the viewer's Canvas scale. This is local to Geography, with no global graphics-setting changes or additional render texture.

Alpha-faded boundary geometry follows the approach described by Unity's engineer for custom UI geometry. [Unity discussion: smoothing custom geometry](https://discussions.unity.com/t/how-to-anti-alias-vertexhelper/884886).

Other solutions considered were camera MSAA or rendering the panel into a higher-resolution texture and downsampling. MSAA's effect depends on the rendering path, while either alternative adds graphics cost beyond these map edges. The local geometry solution works even with a one-sample target. [Unity quality settings and MSAA](https://docs.unity.cn/2020.3/Documentation/Manual/class-QualitySettings.html).

## City tags

- A warm, light card contains a **two-column tile grid**. Each tile shows an owner shape/color, industry symbol, short industry name, level number and flip check/dash.
- The header shows the city, vacancy count and neutral slot dots: filled means occupied; a ring means vacant. Counts always include all players.
- The compact **Links** footer shows owner shapes/colors and directly incident link counts. Its colors match the player key above the board.
- Occupancy tags omit the old VP summary and repeated explanatory text. Select **Industry VP** or **Link contributions** to show that era's selected-player value. The separate introductory bonus remains labeled.
- **Map key** explains symbols, flip state, level numbers, ownership and VP scope on demand. Exact records remain available by clicking a city or opening the snapshot ledger.
- Tags remain below their source dots. They reuse the same UI objects, do not intercept pointer input, and support keyboard focus.

The same implementation serves end-game Statistics and Game History. Existing report calculations, History schema 1, the latest-10 limit, gameplay, AI, Card Tracker and player selection are preserved. Game and AI DLLs are byte-identical to 1.4.29. The assembly comparison preserves **14,843 existing methods** outside the permitted map presentation changes, with embedded resources unchanged.

## Verification and delivery

Candidate end-game Statistics: **5,805 checks passed**. Game History: **3,999 checks passed**. The final run against the installed game files also passed **5,805 checks**. Installer rejection and recovery checks: **166 passed**. Manually inspected **15 native captures at 1920 × 1200**, covering both viewers, both eras, second/largest text, populated and merchant tags, both VP views and the Map key. Exact tested-file hashes and results are recorded in the release archive.

The new checks verify physical pixel dimensions across both viewer canvases, city spacing, compact tag bounds, occupancy/flip symbols, selected VP values, input behavior and stable object counts during repeated hover. A native pixel test renders the shipping line code at eight angles with MSAA disabled, checking blended boundary pixels, solid centers and preserved apparent width; it also tests circle edges. Existing scoring, filtering, codec and viewer-lifecycle checks continue to run.

Build: `scripts/build-statistics-geography-refinement.ps1`  
Candidate: `artifacts/statistics-geography-refinement-build/`  
Release archive: `artifacts/patch-releases/1.4.30/`  
Installation and recoverable prior-version backup: `artifacts/patch-installation.json`

Only generated test matches and artificial History/installer fixtures are used. Prior menu, tracker and AI execution evidence retains its original tested-file hashes; preservation checks establish that their code and resources are unchanged.
