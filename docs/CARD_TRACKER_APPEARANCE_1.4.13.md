# Card Tracker appearance — patch 1.4.13

This revision updates the four-player Card Tracker introduced in 1.4.12. It makes its surfaces slightly translucent, improves the city dots, and keeps the bottom button and legend above the hand cards.

**Installed and verified:** 1 October 2026. The installed game folder passes all 539 tracker checks and 259 startup/custom-game/resource checks. All three installed DLL fingerprints match the qualified candidate, a verified 1.4.12 recovery backup is retained, and online play remains disabled.

## Visual changes

- Header background: 86% opacity. Player-chip backgrounds: 78%. City capsules: 84%. Other tracker surfaces: 90%, with a modest hover increase. Text and symbols stay opaque; the separate statistics panel is unaffected.
- Unseen city cards use jade (`#74D3AE`) with a dark rim, soft shadow and small highlight. Played cards use slate-gray rings with a diagonal slash, so their meaning also differs by shape. Dots retain a consistent physical size at different display resolutions. Zero unseen copies retain the crossed-card symbol; this does not mean building in that city is prohibited.
- The bottom button and adjacent legend move at least 48 pixels above their previous minimum position. When visible hand artwork extends higher, both controls move above it with a clearance margin. Layout uses all four transformed corners of each card, including raised or tilted cards, without reading any private card contents. The map fit and detail popup reserve the same raised footer area.

Public counting, the four hidden Canal setup discards, shared industry pools, wild ownership, four-player eligibility, human-turn gating and pause/restore behavior continue as described in the [1.4.12 release notes](CARD_TRACKER_1.4.12.md).

## Verification

- 539 native tracker checks pass on the candidate: public projections, input, pause/restore, lifecycle, day/night rendering, hand clearance, raised/tilted cards, background opacity and opaque text. Screenshots cover 1280×720, 1920×1080 and 2560×1440, plus depleted dots, larger text, list and details.
- 259 startup/custom-game/AI resource checks pass on the candidate.
- Both `Assembly-CSharp.dll` and `BrassLegacyAI.dll` are byte-identical to qualified 1.4.12. Source fingerprints limit runtime changes to `CardTrackerOverlay.cs` and `CardTrackerWidgets.cs`. Cecil confirms 11,921 existing method bodies are unchanged; changed and regenerated methods are confined to Card Tracker.
- The accepted 1.4.12 statistics (688), presentation (184) and link-animation (765) checks are retained for their unchanged implementation. Accepted AI and managed rules/persistence evidence is also retained. These are prior evidence, not newly run suites for this appearance revision.
- 46 installation fixture checks pass: fingerprint rejection, backup verification, offline status, retained save paths, failed replacement recovery and reinstalling. They use artificial files, without accessing the player's saved matches.

Reports are in `artifacts/card-tracker-build`. Final binaries, sources, reports, screenshots and the accepted 1.4.12 baseline are preserved in `artifacts/patch-releases/1.4.13`.

## Installation and recovery

Dedicated staging keeps unrelated ongoing AI research out of this patch. Install the qualified candidate with the game closed:

```powershell
./scripts/install-combined-patch.ps1 -BuildDirectory ./artifacts/card-tracker-build
```

The installer verifies the prior release, keeps all three previous DLLs together in a verified recovery backup, publishes the candidate and records its fingerprints in `artifacts/patch-installation.json`. Online play remains disabled. This revision does not change save formats or AI behavior.

To reinstall the archived release, use `scripts/stage-archived-patch.ps1 -Version 1.4.13`, then `scripts/install-patch.ps1`, with the game closed. Restore the previous release as a coherent set of three DLLs if recovery is needed.
