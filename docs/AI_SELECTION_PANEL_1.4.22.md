# AI selection panel — patch 1.4.22

**Status:** Implemented, qualified and installed as **1.4.22** on **2 October 2026** (installation receipt: `2026-10-03T03:38:49.1282215Z`). Gamma remains unavailable.

Custom Game now opens a centered **AI selection panel** with two pages, **V4** and **Gamma**, using the installed Statistics light palette, rounded surfaces and dark text. The second text preset, **Larger / 115%**, is the default. Search, separate page positions, keyboard navigation, fixed controls and full descriptions replace the small style popup.

- **V4** offers the six existing styles plus Random. Clicking an available card selects it and closes the panel. The player's difficulty remains independent at **Easy 2s / Mid 5s / Hard 12s**.
- **Gamma** displays **16 model slots plus Random**. All 17 choices remain unavailable. The panel explains that trained models and their game backend are not installed; browsing or forcing a disabled button callback cannot change a player's assignment.
- At **1920 × 1200**, all 16 Gamma cards and Random fit at default text size. At **1920 × 1080** or larger text, the grid scrolls when needed while navigation, search, Random and Close stay fixed.
- Custom-game rows use larger light AI buttons and show the selected family. Controls bind correctly while the menu is inactive and appear on its first visible frame. Continue summaries remain read-only and identify older saved AI versions from their saved runtime.
- The modal blocks underlying controls directly, including controls whose CanvasGroups ignore ancestors. Start/Continue and the dismissal frame have additional guards. Closing restores the previous menu state and focus.

The new typed family/choice catalog is an interface for future integration. It does **not** load NN models, register an inference backend or extend saved assignments. Existing V4 menu preferences, Random resolution, saved profiles, adaptive intent, search limits and worker settings are retained. The stale “v3 AI” picker footer is removed.

## Qualification

- **492 native menu assertions:** first-open/cached/Continue layouts, seven V4 choices, all unavailable Gamma callbacks, family identity, search, per-page state, input blocking, cancellation, unchanged difficulty, save/continue smoke checks and **12 layout cases** covering both pages, both resolutions and all three text presets.
- A deliberately isolated future-roster fixture selects each of 16 typed identities, including duplicate display names. Fixture catalogs are rejected outside self-test mode. This demonstrates UI plumbing only; it is not evidence of trained Gamma gameplay.
- **5,301 native Statistics checks** and **458 native Card Tracker checks** pass against the candidate.
- **492 menu assertions pass in the installed game**, with no native errors. Its tested DLL fingerprints match the qualified candidate. The installer also passes **58 packaging/recovery checks** using disposable installation fixtures.
- Method comparison preserves **14,266 existing method bodies** outside the AI menu/selection scope. The game assembly is byte-identical to 1.4.21, all existing BrassPatch methods are preserved, and all **48 embedded AI profile resources** match. Broader accepted search/rules/save evidence is retained on that basis, not presented as a fresh strength tournament.
- Native screenshots at 1200p and 1080p verify the Statistics window palette and readable layouts. Tests use disposable saves and isolated menu preferences; personal saves are not opened or modified.

## Implementation scope

The release builds from the accepted **1.4.21** archive, overlays six UI/catalog source files, and recompiles the two helpers. Source fingerprints and the method comparison reject changes to gameplay, search, Statistics, Card Tracker or persistence. Gamma activation, model exports and inference qualification remain future work.

See the [implementation plan](../rules-audit/AI_SELECTION_PANEL_V4_GAMMA_PLAN.md). Release binaries, source snapshots, screenshots, installed checks and recovery evidence are archived under `artifacts/patch-releases/1.4.22` after installation.

## Installation and recovery

The three installed DLLs match the release fingerprints in [the installation receipt](../artifacts/patch-installation.json). Online play remains disabled. The preceding installed **1.4.21** package and receipt were backed up to:

`artifacts/patch-recovery/2612eb7d831f460d8dfc871b40b723d8`

The accepted 1.4.21 archive is retained as well. Native verification uses disposable saves and isolated menu preferences; personal games are unchanged. See the [1200p V4 capture](../artifacts/patch-releases/1.4.22/screenshots/ai-selection-1920x1200-V4-1.png) and [1200p Gamma capture](../artifacts/patch-releases/1.4.22/screenshots/ai-selection-1920x1200-Gamma-1.png).
