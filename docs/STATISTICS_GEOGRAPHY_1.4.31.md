# Patch 1.4.31 — Occupancy-sized geography dots and stable map layout

Date: 12 October 2026  
Base: accepted patch **1.4.30**  
Status: installed and verified.

## Dot size shows occupied industry slots

The city core grows with the number of industries physically present at the selected era's scoring boundary. **Flipped and unflipped industries count equally**, across all players. Taking a player out of the emphasis filter does not change that physical occupancy or resize the city.

| Occupied slots | Visible city diameter at 1200p |
| --- | ---: |
| 0 | 24 px |
| 1 | 28.8 px |
| 2 | 33.0 px |
| 3 | 36.7 px |
| 4 | 40 px |

Each industry adds the same amount of area; a base ring keeps an empty location visible. These sizes are in display pixels and do not change with the text preset. Merchant diamonds remain a consistent size; farm symbols grow with their industry occupancy. Region colors and the existing anti-aliasing are retained.

The map includes a short **Dot size · occupied slots** key. VP views add their separate, pale value halo around the occupancy-sized core. Pointer targets and the distance to the hover tag account for each marker's actual size.

## Stable geometry when enlarging text

Previously, increasing text enlarged the controls and tag reserve, and the map was fitted into the remaining space. Its diagram height fell from roughly 512 to 469 pixels at 1200p.

The new layout reserves a common header band and sizes the diagram independently of text. Changing the text preset enlarges labels and tags while preserving diagram dimensions, city screen positions and occupancy dot sizes. Native measurements at 1200p confirm a **955.5 × 546 pixel** diagram for all three text presets in both viewers. The layout still responds to the actual viewport size.

## City arrangement

- Spread the central and southern clusters so larger city dots and pointer targets stay separated.
- Keep the northern sequence, Black Country, western cities and southern merchants in their recognizable board-relative arrangement.
- Reposition the southern branch junction and long route bends to keep routes clear of unrelated dots.
- Keep identical coordinates between Canal and Rail. All 27 locations and the existing era-specific topology remain present.

The same renderer serves end-game Statistics and Game History. This revision changes only `StatisticsBoardLayout.cs`, `StatisticsMapView.cs` and `StatisticsMapPanel.cs`. Gameplay and AI assemblies remain byte-identical to 1.4.30. The build uses the frozen release sources with only these three overrides, so unrelated training work in the checkout is excluded.

## Verification and delivery

Native checks exercise the text buttons while the map is open, including returning to the original preset; compare every city's physical screen position and marker size; check the larger hit targets; verify occupancy ordering and player-filter independence; and inspect tag bounds for every location in both eras and VP views.

Completed candidate qualification:

- **6,686 Statistics checks** and **4,880 History checks** passed, with no native errors.
- **16 final native captures** manually inspected at 1920 × 1200, covering both viewers, all three text presets, both eras, occupancy and VP views, populated city tags and the southern edge tags.
- All 27 city positions and occupancy marker sizes remain stable across text presets and player emphasis changes. Flipped and unflipped industries contribute equally to occupancy.
- Minimum city-center separation is **82.04 pixels**. The closest printed route to an unrelated city center is **37.93 pixels**, leaving clearance around enlarged markers.
- **14,962 existing methods** and the protected resources are preserved outside the permitted Geography changes. The gameplay and AI assemblies are unchanged from 1.4.30.
- **190 installer checks** passed using artificial files, including rejected bad evidence, mismatched files and recovery from a locked destination.
- **6,686 Statistics checks passed again against the installed game files**, with no failures or native errors.

Patch 1.4.31 is installed. Results, reviewed captures and tested-file fingerprints accompany the release archive.

Build: `scripts/build-statistics-geography-polish.ps1`  
Candidate: `artifacts/statistics-geography-polish-build/`  
Archive: `artifacts/patch-releases/1.4.31/`  
Installation and previous-version recovery backup: `artifacts/patch-installation.json`

The 1.4.30 rollback files are retained in `artifacts/patch-recovery/e0570c02cdfc4bad8a4d729d46dd44a6/`. Online play remains disabled.

Verification uses generated matches and artificial History/installer fixtures.
