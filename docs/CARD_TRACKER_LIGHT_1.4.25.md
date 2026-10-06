# Card Tracker appearance — patch 1.4.25

**Installed and verified:** 3 October 2026 (Chicago), built from the accepted 1.4.24 installation.

## Changes

- Raised the fixed bottom-right Card Tracker button by 72 pixels. Its bottom edge is now 220 pixels above the screen bottom, with the same 24-pixel right margin. It clears both the original Thinking panel and the V4 strategy status.
- Applied the Game History/Statistics light palette to the tracker: warm ivory panels, dark text, brass accents, subtle borders and rounded corners. The header, wild-card holders, industry counts, card list, details, legend and toggle share this style.
- Reworked city badges with a location-card symbol, red unseen dots, pale crossed gray played dots and a slashed-card indicator when all copies have been seen. The legend explains the symbols.
- Expanded placement choices around crowded cities and added spacing between badges, city names and industry slots. Thin leader lines keep displaced badges associated with their cities.
- Alternating card-list rows and red Unseen values make the count table easier to scan. The second text size remains the default.

## Counting and interaction

The tracker remains available only in four-player local games, during the human player's turn between actions. Opening it pauses the game; closing restores the previous animation speed and map position. It remains disabled during AI thinking, replay and turn animations.

Red means **unseen**, not guaranteed to be in the draw pile. Canal counts retain the four hidden setup discards. Crossed gray dots show publicly played/discarded cards; an exhausted city-card category does not rule out builds using industry or wild cards.

## Scope

Only `CardTrackerOverlay.cs` and `CardTrackerWidgets.cs` are overlaid on the archived 1.4.24 runtime sources. No shared Statistics/Game History widgets are modified by this release. The light theme is attached to the tracker root, so it does not recolor the board.

`Assembly-CSharp.dll` and `BrassLegacyAI.dll` are copied unchanged from 1.4.24. Gameplay, AI search and all 48 AI profile resources, saves, public-card counting, pause/input eligibility, Game History, Statistics and the Human/V4/Gamma player-selection panel are preserved. Online play stays disabled.

## Validation and installation

Native qualification passed **10,490 checks** with no unexpected Unity errors:

| Suite | Checks |
| --- | ---: |
| Card Tracker | 490 |
| Player selection/menu | 646 |
| End-game Statistics | 5,613 |
| Game History | 3,741 |

Manual visual review covered actual 1920 × 1200 renders: human-turn button placement, original and V4 thinking text, day/night maps, exhausted city dots, scrolling card list, pinned detail and largest text. The V4 thinking label ends at approximately 140 pixels above the bottom; the raised button begins at 220 pixels. Automated geometry/input checks also retain 1080p coverage.

The runtime comparison verified **14,761 existing method bodies** unchanged outside the permitted tracker presentation methods, with all 48 embedded AI profiles preserved. Installer/recovery fixtures passed **94 checks**. The installed game then independently passed **490 tracker checks** and **646 menu checks**; all three installed DLL fingerprints match the qualified build.

Evidence: `artifacts/card-tracker-light-build`. Accepted release archive: `artifacts/patch-releases/1.4.25`. The previous 1.4.24 binaries are backed up under `artifacts/patch-recovery/7ea91f1423ea4690a55a668d962be6e3`. Personal save files were not edited.

| Installed file | SHA-256 |
| --- | --- |
| Assembly-CSharp.dll | `50BF8A2F8A167C1FE5679D37C72A5D5D1F1B01BDFA144EBE5811F5DAF7552699` |
| BrassPatch.dll | `5F5BF897641E665F4D57FEC2E4B66C5FF7837938E063B1DD22DFEBFA9A986161` |
| BrassLegacyAI.dll | `366ADB638E64A951DB87EAA7F52222690CA62D866E64D938DDD0F0F214A69B42` |

## Reproduce this release

1. Run `scripts/build-card-tracker-light.ps1`.
2. Run `tests/BrassPatch/InformationUI/run-native.ps1` for `tracker`, `menu`, `statistics` and `history`, each with `-BuildDirectory artifacts/card-tracker-light-build`.
3. Review the actual Unity captures at 1920 × 1200, then run `tests/BrassPatch/CardTracker/qualify-card-tracker-light.ps1`.
4. Run `tests/BrassPatch/InformationUI/packaging.ps1 -BuildDirectory artifacts/card-tracker-light-build`.
5. With the game closed, run `scripts/install-combined-patch.ps1 -BuildDirectory artifacts/card-tracker-light-build`.
6. Run the tracker and menu native suites with `-Installed`; archive with `scripts/archive-card-tracker-light-release.ps1`, then stage the accepted 1.4.25 archive.

Native fixtures use disposable games. The installer verifies the previous release, backs it up, atomically publishes the qualified binaries and records their fingerprints.
