# Statistics geography map revision plan

Date: 7 October 2026  
Status: **Planned — implementation and installation have not started.**  
Reviewed baseline: installed patch **1.4.27**, current Statistics source, the existing 1920 × 1200 geography capture, and the locally archived official board/rulebook.  
Applies to: the shared Statistics viewer opened from **end-game results and Game History**. Optimize the visual design and inspect rendered results at **1920 × 1200**.

## 1. Intended result

The Geography view becomes a clean, recognizable board diagram. Initially it shows city dots, distinct farm/merchant symbols, and network routes, with **no permanent city names, numerical labels, or city tags**. Hovering a city opens one compact tag **directly below its dot**, with a visible gap so the tag never covers the inspected dot.

The player can switch between **End of Canal** and **End of Rail**. City positions remain identical between these views. Owned links use their owner's player color. Tags show the city's vacant slots, its actual industries and flip status at that era boundary, and the players owning links that directly touch it.

Keep the existing light Statistics style, text-size presets, exact-data access and shared History implementation. This is a presentation revision supported by the existing recorded match data.

## 2. Findings from the current implementation

| Current behavior | Evidence | Required change |
| --- | --- | --- |
| City names and values are always rendered; dense labels can be displaced far from their dots. | `StatisticsPlotViews.MapLabels`; existing [1200p geography capture](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/artifacts/card-tracker-player/artifacts/patch-build/live-plot-1200-geography.png>). | Replace permanent labels and their leader lines with one reusable anchored hover tag. |
| The general plot tooltip appears at the lower-right of the Statistics window. | `StatisticsPanel.Tooltip`. | Give Geography a dedicated anchored tag; preserve ordinary plot tooltips elsewhere. |
| Edges are drawn from `StatisticsLocation.Neighbors` in one neutral color. This adjacency list combines board routes rather than identifying a placed link and its owner. | `StatisticsProjection` builds neighbors from all region links; `StatisticsPlotGraphic.OnPopulateMesh` connects neighboring marks. | Render owned links from `StatisticsLink` records and use era-specific topology for any faint background routes. |
| The map defaults to `Final`; historical boundaries are hidden in a multi-choice menu. | `StatisticsFilters.Boundary`, `StatisticsReviewPanel.BoundaryChoice`. | Put the two requested era-end choices directly beside the map. |
| Some positions differ materially from the printed board. | Current `CityPositions` places `Central` east of Cannock and Gloucester west of Worcester; the [official board illustration](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/artifacts/reference-review/components-official-board.png>) places them on the opposite sides. | Replace the coordinate manifest with board-relative positions and deliberate spacing. |
| The report already preserves build, flip and removal revisions, industry/slot identities, link ownership/endpoints, and scoring-boundary occupancy. | `CompletedMatchStatistics`, `StatisticsProjection`, schema-1 History codec. | Derive the map from these value records; both viewer entry points can use the same query and renderer. |
| Current visual tests require permanent labels not to overlap. Empty label collections would satisfy those checks accidentally. | `StatisticsHistoryReviewChecks.CheckPlot` map branch. | Replace those assertions with real hover, anchor, ownership and boundary checks. |

The existing broad boundary selector already provides some historical occupancy. The revision makes the two era ends explicit and adds the detailed historical tag and owned-link rendering that are currently missing.

## 3. Page layout and era selection

Keep the **Networks & Geography** navigation entry. Inside it, use two compact local tabs consistent with the existing Resources and Efficiency tabs:

| Tab | Contents |
| --- | --- |
| **Geography** — default | Era-end switch, concise legend, full-width map, collapsible exact snapshot data. |
| **Network results** | Existing route rankings, link contribution charts and supporting network tables. |

This gives the map enough space without requiring the player to scroll past a long route chart. Construct only the active tab and preserve its view state separately.

On Geography:

- Show **End of Canal / End of Rail** as two visible segmented choices. Default to Rail in a full game and Canal in an introductory game. Remember the choice while moving between tabs in the same report.
- Replace the shared Whole game/Canal/Rail chart selector with these map choices while this tab is active. Store a separate `MapEra` so a map snapshot cannot conflict with an inherited chart-era filter; restore the chart selector on other views.
- Explain the boundary in one short caption: **“Board at era scoring, before links are removed.”** Canal also precedes level-I cleanup.
- Disable Rail for introductory matches, with **“No Rail era in this match.”** If an era's history is unavailable, show that explicitly rather than filling the map with zeros or substituting the final state.
- Keep the existing three measures available in a compact **Measure** selector: **Occupancy**, **Industry VP**, **Link contributions**. All values refer only to the selected era end. Occupancy is the default; numerical values appear in the hover tag and exact data.
- Keep Setup, Rail entry and After cleanup records available in the supporting snapshot ledger. They do not need to compete with the two primary map choices.

The map has no numerical x/y axes. Its coordinates describe board placement, not distance or quantitative values.

## 4. Board arrangement and city dots

### Stable board-relative layout

Create a checked-in `StatisticsBoardLayout` containing normalized node coordinates and route paths. Use the official board illustration and the game's board definitions as the reference, without depending on live board transforms or loading a match in History.

Cover all **20 cities, five merchant locations and two farm breweries**, with stable identifiers. Preserve these recognizable relationships:

| Area | Arrangement to preserve |
| --- | --- |
| North | Warrington to the northwest of Stoke-on-Trent; Leek north/east of Stoke; Belper northeast; Nottingham on the eastern edge. |
| Upper middle | Stone west of Uttoxeter and Derby; Stafford south of Stone; Burton upon Trent below Derby. |
| Central/Black Country | Cannock below Stafford; the central/northern farm **west of Cannock**; Wolverhampton and Walsall below; Tamworth east of Walsall. |
| Southwest | Shrewsbury west of Coalbrookdale; Kidderminster south of Coalbrookdale; Worcester below Kidderminster; the southwest farm beside their shared route. |
| Southeast | Birmingham southeast of Walsall and east of Dudley; Nuneaton northeast of Coventry; Redditch below Birmingham; Oxford southeast; Gloucester **east of Worcester**, near the southern edge. |

Use controlled spacing to separate the dense southern and central nodes while retaining the board's order and recognizable shape. Do not use an automatically rearranging graph, alphabetical layout, or real-world latitude/longitude. Filtering, changing era, resizing text and hovering must not rearrange cities.

Use a fixed diagram aspect and layout rules; avoid stretching the graph arbitrarily across the entire panel width. Extra horizontal room can accommodate tag width and edge padding. Unknown locations belong in an **Unmapped records** detail list; remove the current arbitrary fallback that places unknown names near farm positions.

### Dot treatment

- City: a compact, light-centered circular dot with a clearly visible rim in the board region color. Use the same base size for the default occupancy view.
- Farm brewery: a small distinct farm/brewery symbol. Merchant: a small diamond/merchant symbol. Neither has a permanent name.
- Hover/focus: add a crisp outer ring and brighten incident owned links. Do not enlarge the dot enough to overlap neighboring hit targets.
- Target approximately **18–22 px visible city diameter** and **32–40 px hit diameter** at 1200p, with non-overlapping hit regions in dense clusters. These are initial layout values to verify in native rendering.
- If a VP measure uses a magnitude halo, bound its size and explain it in the legend. Keep the city center and hit behavior stable; a zero-value city must remain visible and inspectable.
- Keep any regional tint very faint. The legend must distinguish **city rim = board region** from **link color = player**.

## 5. Player-colored network links

Draw a very faint era-specific printed-route layer for orientation, then the links actually owned at that era's scoring boundary. Label the neutral layer **Printed routes**, so it cannot be confused with ownership or a legal-build suggestion; provide a small visibility toggle.

Use the existing `StatisticsTheme.Player` palette consistently:

| Player identity | Link color |
| --- | --- |
| Blue | `#1D6F9D` |
| Yellow | `#8B6413`, a gold tone readable on the light background |
| Violet / Purple | `#77539D` |
| Red | `#B34F45` |

Map colors must follow the recorded `PlayerColor`, never rank, seat, iteration order or the city's region. Use a compact legend with each player's actual name and a colored line sample. Add a small repeated owner glyph or marker shape as a second cue so color is not the only identifier.

Start with approximately **3–4 px owned strokes**, a thin light separation stroke where routes cross, and restrained background routes. Render nodes above links. Curved or bent paths should follow the printed arrangement and avoid passing through unrelated city dots. A geometric crossing must not look like a junction.

Important topology requirements:

- Canal and Rail routes are separate. Do not display a Rail-only background route as a Canal route.
- Reuse the same geometry for a corridor present in both eras, but read ownership separately for each era.
- The **Kidderminster–Worcester–southwest farm** connection is **one link with three endpoints**. Draw one connected branched path, all in its owner's color. Count it once; do not draw a triangle of three independent links.
- The Cannock–central farm link remains a separate route.
- Include owned links with **zero scored VP**. A zero score does not mean no link existed.
- Do not infer ownership or route usage from adjacency, industrial ownership, merchant deliveries, or the `StatisticsConnection` dependency checks.

## 6. New city tag

Follow the requested three-part design: **city/slots header → industry rows → connections footer**. Use a warm ivory surface, rounded corners, a fine taupe border, a restrained shadow, and dark text matching the current Statistics theme. Keep the tag sufficiently opaque for small text to stay readable above links.

Illustrative layout, using industry types that can actually occupy Birmingham:

```text
                         ◉  city dot
                         │  10–12 px clear gap
        ╭─────────────────────────────────────────────╮
        │ Birmingham                 ○  1 of 4 open  │
        ├─────────────────────────────────────────────┤
        │ ● Yellow   [cotton] Cotton mill Lv.2   ✓    │
        │ ● Yellow   [goods]  Manufactory Lv.3   —    │
        │ ● Violet   [iron]   Iron works  Lv.3   ✓    │
        ├─────────────────────────────────────────────┤
        │ Industry VP  …    Link contribution VP  …  │
        ├─────────────────────────────────────────────┤
        │ Connections          ● Yellow · 2 links    │
        ╰─────────────────────────────────────────────╯
```

The values above illustrate the layout, not a recorded match. Brewery, coal mine and pottery rows use the same design in cities/farms that contain them. The implementation always renders the report's real tile types and levels.

### Header: name and vacant slots

- City name on the left; **`1 of 4 open`** with an outlined slot symbol on the right. For zero vacancies, show **Full**; do not use a lock suggesting the city can never be changed.
- “Open” means physically vacant industry slots at that snapshot. It does **not** promise that a particular player has the required card, network, resources, permitted industry or a legal build/overbuild action.
- Calculate vacancies from total city occupancy across **all players**, even when highlighting only one player. Never sum the duplicated city-capacity records stored once per player.
- Use neutral slot symbols, visually distinct from the colored player dots. Wrap long city names rather than truncating or reducing the text below the selected preset.

### Body: one row per industry tile present

Each row contains an owner-color marker with a short owner label, a recognizable industry icon, the industry name, **Lv.N**, and a fixed-width status column.

- **✓ Flipped** and a muted **— Unflipped** have a visible legend; provide the full status text for focus/accessibility. Do not use green/red alone to communicate the state.
- Use `FlipRevision` at the selected boundary. A brewery that flips during Rail must still show unflipped in the Canal tag.
- Order rows consistently by slot identity, keeping an industry's row stable between eras where possible. Replaced tiles occupy the same slot row; removed tiles are absent from the later snapshot.
- Use native industry sprites if they are available independently of a live match. Otherwise provide lightweight consistent vector glyphs. Do not depend on emoji fonts, which may show missing or mismatched symbols in this Unity build.
- Empty city: **No industries at this era end**, followed by its connections footer.
- Show a small summary of actual industry VP and link-contribution VP for the chosen era. Keep introductory bonus VP separately labeled. Explain that link contributions can reward multiple incident links and are not additional industry VP.

### Footer: direct link owners

Use **Connections**, followed by one chip per player with a link touching the city, for example **● Yellow · 2** and **● Blue · 1**. Deduplicate players and count distinct link records, including a branched link only once at each endpoint.

The footer means **owners of incident canal/rail links at the selected era end**. It does not mean every player reachable through the transport system. Brass's formal “your network” also includes a city containing your industry; that is already visible in the industry rows and must not be silently added to a footer claiming link ownership. If there are no owned incident links, show **No owned links**.

Merchant tags show their name, a Merchant subtitle and connections; industry capacity is **not applicable**, not `0 of 0 open`. Farm tags show their single brewery slot and use the same historical industry/connection rules. Do not fabricate historical merchant beer or merchant-tile state that is not present in the report.

## 7. Hover placement and interaction

Use one pooled tag per map view, anchored to the inspected dot, rather than building and destroying UI objects on every pointer move.

1. Initial state: all tags hidden. Entering a city's hit region shows its tag after a short delay of about **100 ms**, reducing flashes while crossing the map. Keyboard focus provides the equivalent inspection behavior.
2. Place the tag's top edge **10–12 px below the bottom of the visible dot/halo**. Center it beneath the city unless horizontal edge clamping is needed. A small pointer/notch may indicate the source without touching the dot.
3. Reserve space for the largest measured city tag **below the lowest dot** when laying out the map stage. Derive this space from the selected text preset and the maximum row count, rather than a fragile fixed height. Size the diagram inside the remaining area before interaction starts.
4. Keep the complete dot-plus-tag within the visible map stage at 1200p. Clamp horizontally within the panel, but never flip the tag above the dot or clamp it upward over the dot. Do not pan or reshuffle the diagram in response to ordinary hovering.
5. Make the tag non-interactive and non-blocking for pointer raycasts; clicking a city can continue to open its exact records through the existing drill mechanism. The city tag takes precedence over link inspection in the same hit area.
6. Hide it on pointer exit, page/tab/era changes, filter changes, scrolling, resize/text-size rebuild, viewer close and disposal. Re-evaluate the hit after scrolling settles, including when the pointer itself has not moved.
7. Only one tag is visible. A stale tag must not survive a redraw or keep showing Canal data after switching to Rail. No persistent city labels are added by selection.

Suggested starting tag dimensions are **380–440 px wide**, with approximately **24 px title** and **20–22 px body** at the normal 1200p viewing preset. Measure actual wrapped text and tile rows. Support the largest existing text preset without clipping, overlap with the source dot, or text shrink-to-fit.

A wide city tag may temporarily cover other routes; the inspected dot remains visible above it. Avoid unnecessary card width or empty rows that obscure more of the diagram than needed.

## 8. Era data, filters and correctness

Build a detached `StatisticsMapSnapshot` for each report and era. Keep this as a derived view model, outside the serialized History DTOs. The current report fields are sufficient for the requested core design, so retain **History schema 1** and existing archived matches.

| Item | Source and rule |
| --- | --- |
| Snapshot revision | The report's `Scoring` boundary for the selected era; validate that player boundary records agree on the revision. This is recorded before link removal. |
| Industry present | `asset.Revision <= boundary.Revision` and `RemovedRevision == 0 || RemovedRevision > boundary.Revision`, matching `StatisticsMetrics.At`. |
| Industry flipped | Present at the boundary and `FlipRevision > 0 && FlipRevision <= boundary.Revision`. The final `Flipped` getter alone is insufficient. |
| Occupancy | Distinct occupied slots from all present assets, reconciled against the matching scoring `StatisticsCity` capacity/occupancy record. Capacity must agree across its per-player copies. |
| Links | Recorded links belonging to the selected era and placed by the scoring boundary, retaining their owner, key, endpoint identities and recorded scoring. Never inspect the cleaned-up live board for them. |
| Industry VP | `CanalVP` for Canal, `RailVP` for Rail, on the tiles present at that boundary. A carried Canal tile can contribute in Rail. Introductory bonus is separate. |
| Link contribution VP | Matching `StatisticsLinkContribution` records for the selected era and represented link identities. Do not add earlier-era contributions to the Rail snapshot. |
| Historical stock, if disclosed | Use the matching `StatisticsStock` boundary record, not the tile's final `Stock`. Stock is optional detail, not required clutter in every row. |

**Filter behavior:**

- Keep the entire board geometry and its actual occupancy visible. Player selection emphasizes that player's industries and links; others become subdued context rather than disappearing and making occupied slots look vacant.
- Keep all occupants and direct link owners in the tag, marking subdued rows as outside the selected-player emphasis. Any numerical metric limited to selected players must say **Selected players** and show the matching total consistently in exact data. Vacancy counts always remain all-player counts.
- Region selection emphasizes a region without moving/removing other cities. Preserve endpoint context for routes crossing its boundary.
- Industry, action and evidence filters from other analytical pages must not silently remove pieces from a historical board. Scope them to a labeled detail/evidence view or explicitly suspend them on Geography. An empty evidence selection must remain empty in drill-down results.
- Text search remains a ledger filter. It does not invisibly change map occupancy, ownership or slot availability.
- Clear hover/selection state on relevant filter changes and include map era, measure and emphasis state in the appropriate cache keys.

**Missing or inconsistent evidence:** retain the existing limited-history explanation. A missing boundary, invalid slot occupancy or absent link endpoints must yield an unavailable/partial-data message for the affected content, not plausible-looking zeros, invented owners, guessed positions or a final-state substitute. Surface contradictions in diagnostics rather than silently clamping them away.

## 9. Implementation and code reuse

| File/component | Work |
| --- | --- |
| New `StatisticsMapQueries.cs` | Derive immutable era snapshots; group city occupants and incident links; reconcile occupancy; provide the same values to the map, tag and exact-data table. |
| New `StatisticsBoardLayout.cs` | Explicit node positions and era-specific route paths, including the three-endpoint farm route; keyed by stable board identities. |
| New `StatisticsMapView.cs` | Dedicated node/link graphic, hit testing, emphasis and map-stage layout. Reuse Statistics surfaces and palette. |
| New `StatisticsCityTag.cs` | Reusable three-part tag, icon rows, measured below-dot placement and clean lifecycle. |
| `StatisticsPages.cs`, `StatisticsReviewPanel.cs`, `StatisticsPlotControls.cs` | Geography/Network results local tabs, separate map era, visible era controls, measure control and state/cache integration. |
| `StatisticsPlotPages.cs`, `StatisticsPlotModels.cs`, `StatisticsPlotViews.cs` | Route Geography to the dedicated snapshot view; remove permanent `MapLabels` and adjacency-only drawing from that path. Keep unrelated plot rendering intact. |
| `StatisticsTheme.cs`, `StatisticsWidgets.cs` | Reuse colors, light surfaces and focus treatment; add only shared primitives needed by the map/tag. |
| `StatisticsViewHost` / History entry points | Continue using the same panel and report. Do not create a second map implementation for archived games. |

Keep existing report collection and persisted DTOs unchanged unless a demonstrated data gap makes that necessary. The codec requires its scalar fields explicitly, so adding a property to a serialized DTO without a compatibility strategy is not a harmless UI change.

Cache the two map snapshots once per report. Hover should update only tag contents, anchoring and local highlights. Avoid full page rebuilds, report replay, file access or scene lookups during hover. Preserve the live Statistics presentation suspension introduced in 1.4.26; Game History continues to work without an active board scene.

## 10. Verification and acceptance criteria

### Data checks

Use focused synthetic reports plus an existing completed report, and repeat important queries after a schema-1 History encode/decode roundtrip.

1. A level-I tile present at Canal scoring appears in Canal and is absent after cleanup/Rail.
2. A level-II+ carried tile is present in both snapshots even though its build era is Canal.
3. A tile unflipped in Canal and flipped in Rail shows the correct different statuses and era VP.
4. Overbuild and liquidation before/after the boundary select the correct occupant, with no duplicated slot.
5. Changing player emphasis never changes the city's real occupied/vacant counts.
6. Per-player city snapshot copies do not multiply slot capacity or total occupancy.
7. A corridor built by different owners in the two eras changes color correctly; the previous era's links do not leak into Rail.
8. Zero-VP links remain visible. Three-endpoint links remain one recorded link and appear in all three endpoint footers.
9. A player with only an industry and no touching owned link appears in the body but not in the Connections footer.
10. Rail metrics exclude Canal scoring; city contribution totals reconcile with the recorded era link contributions without inventing extra industry VP.
11. Empty cities, empty player selection, 2/3/4-player reports, introductory games and limited-history reports have intentional states.
12. The same report produces identical snapshot/tag/exact-data values from end-game Statistics and Game History.

### Native visual and interaction checks at 1920 × 1200

- Capture both era maps with **no hovered city**: no city names, values or tags remain on the diagram.
- Inspect both eras with dense central/southern cities, a full four-slot city, an empty city, a farm, and edge merchants. Inspect the southwest branched route specifically.
- Capture hovered tags near the left, right and bottom edges at the default second text preset and largest preset. Verify the whole tag is readable, its top is below the dot's bottom, and the selected dot stays unobscured.
- Confirm no hover flicker or raycast interception; stable city hit targets; prompt switching between adjacent dots; and cleanup on scroll, era change, text change, tab change and close.
- Verify each player's links, tag swatches and legend agree. Check the gold/yellow route against the light surface and the non-color owner cues.
- Compare positions with the official board, especially Central/Cannock, Gloucester/Worcester and the southern farm. Check that route crossings do not imply false junctions.
- Open from a fresh Game History menu with no live game loaded: all icons and tags must still render.
- Repeat the same interaction sequence through both viewer entry points, retaining the existing live-view performance protections.

Replace the old permanent-label assertions with explicit visibility/anchor tests; do not count an empty collection of old labels as proof that the new map is correct. Measure hover and era-switch work with image encoding outside timed intervals. Repeated pointer movement must not trigger whole-page rebuilds or steadily increase allocated UI objects.

## 11. Delivery order

1. **Snapshot correctness:** implement and verify the era queries and shared exact-data model first.
2. **Recognizable map:** add the coordinate/path manifest, region dots, era controls and player-owned links.
3. **City tag:** implement the three-part design, below-dot placement, measured safe area and hover/focus lifecycle.
4. **Viewer integration:** add local tabs, emphasis rules, caching and History compatibility; remove obsolete labels and map-specific tooltip use.
5. **Qualification:** run focused data/native checks, inspect the requested 1200p captures, and verify unchanged Statistics pages and viewer lifecycle with the appropriate existing checks.
6. **Future patch release:** reverify the installed baseline, build a candidate, qualify it, then install using the existing backup/receipt/archive workflow when implementation is requested. Preserve the current Card Tracker, AI and gameplay patches and unrelated working-tree changes. Choose the release number from the verified release state at that time.

This request produces the plan only. No game binary, save, history record or installed patch is changed.

## References inspected

- [Current map query and positions](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Statistics/StatisticsPlotPages.cs:118>)
- [Current map/link renderer and pointer handling](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Statistics/StatisticsPlotViews.cs:94>)
- [Existing era-boundary controls](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Statistics/StatisticsReviewPanel.cs:82>)
- [Recorded scoring boundaries](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Statistics/StatisticsProjection.cs:50>)
- [History codec contract](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/History/GameHistoryStatisticsCodec.cs:21>)
- [Game board slot and route definitions](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardGameRules.Entities.Board/Board.cs:436>)
- [Official board illustration, archived locally](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/artifacts/reference-review/components-official-board.png>)
- [Locally archived official rulebook text — era scoring/removal and Your Network](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/official-rulebook.txt:614>)
