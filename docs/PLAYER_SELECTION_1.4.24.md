# Player selection — patch 1.4.24

**Installed and verified:** 3 October 2026 (Chicago), patch **1.4.24**, built from accepted **1.4.23**.

## Changes

- Each active New Game seat has one light player-selection button, including Human seats. Its second line shows either Local player or the AI difficulty and maximum thinking allowance.
- CHOOSE PLAYER has Human, V4 and Gamma on the left, in that order. Human supports local hotseat play.
- V4 contains the existing six styles plus Random. Easy (2 seconds), Mid (5 seconds) and Hard (12 seconds) appear inside the page, above search and the cards.
- Gamma shows its 16 planned models plus Random. Models, Random, difficulty and Apply remain unavailable. Disabled options do not display a selected checkmark.
- Card and difficulty changes remain a draft until Apply. Cancel, Escape, × and the backdrop discard edits. Switching pages preserves the V4 draft within that panel session.
- Applying Human preserves the seat's last committed AI style and difficulty. Unapplied AI edits are discarded. Player names and colors are retained.
- Continue shows read-only saved assignments. Random still resolves once at successful game creation. The existing requirement for at least one Human now has an explanation beneath New Game.

## Appearance

The panel reuses the Statistics/Game History ivory surfaces, dark text, brass selection and teal focus outline. The default text preset is Larger (115%). At 1920 × 1200, all 16 Gamma cards fit without scrolling at that preset. Larger text can scroll the cards while the page navigation, difficulty, search and Apply/Cancel remain fixed.

## Scope and preservation

Only four runtime source files are overlaid on the archived 1.4.23 sources: `AiSetupRow.cs`, `AiSelectionPanel.cs`, the new `PlayerSelectionDraft.cs`, and four new setup-validation/apply methods in `LocalMenu.cs`.

`Assembly-CSharp.dll` and `BrassPatch.dll` remain byte-identical to 1.4.23. The existing `LocalMenu` method bodies, gameplay rules, V4 search/execution, saves, Statistics, Game History, Card Tracker, two-worker setting and all 48 embedded AI profiles are preserved. No Gamma backend is installed. Offline-only mode remains active.

## Validation

Native tests passed **10,458 checks** with no unexpected Unity errors:

| Suite | Checks |
| --- | ---: |
| Player selection/menu | 646 |
| End-game Statistics | 5,613 |
| Game History | 3,741 |
| Card Tracker | 458 |

The menu suite includes 27 dedicated draft/role checks and 18 page/text-size layouts. Manual review covered all three pages and text presets at 1920 × 1200, plus the split Continue/New Game screen. The preservation checker confirmed **14,686 existing method bodies** and all **48 AI profile resources** unchanged outside the permitted panel/row changes.

Installer/recovery fixtures passed **86 checks**. The installed game independently passed the **646-check menu suite**, and all three live DLL fingerprints match the qualified candidate. Evidence is saved under `artifacts/player-selection-build` and archived under `artifacts/patch-releases/1.4.24`. Native tests use disposable game/history data and isolated AI preferences.

The menu checks exercise Human/V4 switching, draft Apply/Cancel, difficulty-only edits, removed/rebound slots, setup identity validation, all-Human/all-AI lineups, two/three/four-player menus, saved Continue summaries, exact 2/5/12-second assignments, Random resolution, keyboard/input blocking, Gamma rejection and 18 automated page/text-size layouts. Visual review focuses on 1200p, including the split Continue/New Game screen.

## Build and install

1. Run `scripts/build-player-selection.ps1`.
2. Run `tests/BrassPatch/InformationUI/run-native.ps1` for `menu`, `statistics`, `history` and `tracker`, using `-BuildDirectory artifacts/player-selection-build`.
3. Review the captured 1200p images, then run `tests/BrassLocalAI/qualify-player-selection-release.ps1`.
4. Run `tests/BrassPatch/InformationUI/packaging.ps1 -BuildDirectory artifacts/player-selection-build`.
5. With the game closed, run `scripts/install-combined-patch.ps1 -BuildDirectory artifacts/player-selection-build`.
6. Verify the installed menu with the native runner's `-Installed` switch, then archive with `scripts/archive-player-selection-release.ps1`.

The installer keeps a recoverable copy of the previous installed release and records exact installed fingerprints. Personal saves are not edited by this revision.

## Installed fingerprints (SHA-256)

| File | SHA-256 |
| --- | --- |
| Assembly-CSharp.dll | `50BF8A2F8A167C1FE5679D37C72A5D5D1F1B01BDFA144EBE5811F5DAF7552699` |
| BrassPatch.dll | `7191226A323C6ED4E1C949BDF1464902EC48DF3DC6BD986ECF7131D7B42F9069` |
| BrassLegacyAI.dll | `366ADB638E64A951DB87EAA7F52222690CA62D866E64D938DDD0F0F214A69B42` |

The previous 1.4.23 binaries and receipt are recoverable in `artifacts/patch-recovery/6e92c6fd9ff542508f76095cae3dc8ac`. The accepted 1.4.23 release archive is retained separately.
