# AI menu initialization fix — 1.4.5

**Date:** 30 September 2026.  
**Status:** Installed and verified in the Steam game directory.

Opening Custom Game now shows the style chooser and Easy/Mid/Hard controls immediately. A click or other menu refresh is no longer needed. This also applies when Continue is available and when reopening a cached menu.

The Unity view factory disables a newly created menu before its controller calls `OnViewCreated`. Patch 1.4.4 searched only active ancestors when deciding whether to attach `AiSetupRow` and when finding its owning menu. Both lookups now include inactive parents. Controls are configured while hidden and become editable when the menu is shown. The old labels and arrows are hidden during that same initial binding.

The earlier test enabled the menu before configuring it, which missed the production ordering. The revised test calls the actual controller's `OnViewCreated` while the view is inactive, checks controls before any interaction, reopens the cached view without refreshing, and creates a fresh menu with a generated saved game. The test reproduced the missing controls against archived 1.4.4 binaries and passes against 1.4.5.

Fresh native testing passed **239 assertions**, including first-open visibility, Continue summaries, all six styles plus Random, difficulty selection, popup input and cancellation, menu fade cleanup, the synthetic 16-style roster and resolution checks. First-open screenshots were inspected. The tests use isolated preferences and disposable saves.

Only the two ancestor lookups changed in game-runtime source. Qualification checks that exact source scope, byte-identical engine/helper assemblies, unchanged AI execution and save code, and frozen research profiles. The rules, 108-turn AI matrix and save/presentation evidence from 1.4.4 are retained on that basis; they are not claimed as newly rerun for this menu-only fix. Search allowances remain **2/5/12 seconds** for new-game difficulties and **10 seconds** for older local-AI saves. Step 5's frozen v3 training allowance is unaffected.

## Installation record

Upgrade and interrupted-install recovery passed **79 assertions**. The installed game passed the same **239 native menu assertions**, with no unexpected native errors. Installed fingerprints, all **126 existing hooks**, offline mode and the original backup were verified. Personal saves were not opened or modified.

The installed AI DLL SHA-256 is:

`5DEEB3802216D96725B6BA838B18F4BC372CDFD738CB5EE6CA96D8031E1A0BB7`

Source snapshots, binaries, first-open screenshots and qualification reports are archived in `artifacts/patch-releases/1.4.5`. The previous release is retained in `artifacts/patch-releases/1.4.4`, along with the installer's recovery copy.
