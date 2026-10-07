# Patch 1.4.27 — Compact Card Tracker below Game Log

The Card Tracker button is right-aligned with the native **Game Log** button at the top right, with an **8-pixel gap beneath it**. Its physical size is now **152 × 36 pixels**, reduced from 200 × 50. The light background uses **76% opacity** (24% transparency), increased from 4% transparency. Its label remains opaque and readable.

The anchor follows the fixed Game Log HUD control. Moving or animating the hand, resource markets or board cannot move it. If the summary HUD has not yet been created, a temporary top-right anchor is used until its control is available.

While Game Log is shown or transitioning, the tracker canvas inherits the board's layer and the button blocks no pointer input. Game Log therefore draws above it and receives clicks at its position. Cached Game Log roots can remain active after closing; the visible flag and popup alpha distinguish that state from a real open/closing log. Closing the log restores the button's normal appearance and eligible human-turn input. Paused card inspection retains its existing input shield and Resume controls.

## Verification

- **512 native Unity checks passed**, with zero failures or native errors, at 1920 × 1200 and 1920 × 1080.
- Checked button dimensions/alignment, translucency and opaque text, Game Log show/hide transitions, pointer ordering, restored input, AI thinking/animation disablement, fixed position during market/hand animations, paused map inspection, card counts and pause restoration.
- Manually inspected 1200p captures of the normal button, open Game Log, AI thinking and paused tracker.
- The build overlays only `src/BrassPatch/CardTracker/CardTrackerOverlay.cs` onto the accepted 1.4.26 source snapshot. **14,838 existing method bodies outside the allowed overlay changes and all 48 embedded AI resources remain unchanged.** `Assembly-CSharp.dll` and `BrassLegacyAI.dll` remain byte-identical to 1.4.26.
- Statistics, History, menu and AI evidence is retained from 1.4.26 with its original fingerprints, supported by the method/resource preservation check. Those suites were not relabeled as new executions.

## Installation

Installed on **7 October 2026 at 12:41:40 CDT**. All installed DLL fingerprints match the qualified build. Post-install verification passed the same **512 native checks**, with zero failures or native errors. **106 packaging and recovery checks passed using artificial files.**

The prior 1.4.26 release is backed up at `artifacts/patch-recovery/487d71885fe447a4b4c31f712ea99909`. The receipt is `artifacts/patch-installation.json`; the helper DLL SHA-256 is `C70E0ED0657B7ABA91971B0420B2ED810277E18B4B600DF039E9F1E7A09265C7`.

Release source snapshots, baseline provenance, reports, native logs and reviewed images are archived at `artifacts/patch-releases/1.4.27`. The qualified reinstall package is staged at `artifacts/patch-build`; the build candidate is also retained at `artifacts/card-tracker-log-button-build`. User saves and training files are outside this patch's runtime source scope.
