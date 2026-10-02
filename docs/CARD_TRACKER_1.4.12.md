# Card Tracker — patch 1.4.12

The tracker adds paused public card information to four-player local games, including the introductory game. It retains the rules, save compatibility, city highlights and statistics improvements from earlier patches, together with the accepted 1.4.11 AI repair.

**Installed and verified:** 1 October 2026. The installed game folder passes the same 515 tracker checks and 259 startup/custom-game/resource checks. All three installed DLL fingerprints match the qualified candidate; online play remains disabled.

## Controls

The **Card Tracker** button sits immediately right of the iron market. It starts closed. It is enabled only during a human turn, between actions, after replay and animations finish. It is disabled on AI turns and during thinking, action resolution and scoring. Two- and three-player games and online games do not show it.

Opening pauses Unity gameplay and DOTween animations and captures a public snapshot. The map fits below the tracker header; drag it or use the mouse wheel to inspect cities. Closing restores the exact previous map position, zoom, scrolling state and game/animation speeds.

- City capsules show one dot per ordinary city card: filled teal means **Unseen**, hollow gray means **Played**. A crossed-card symbol indicates zero unseen ordinary copies.
- Industry badges show unseen and played counts beside all six industry rows. Cotton Mill and Manufacturer show the same shared eight-card pool, marked with `=`.
- The four player chips show Wild Location with a pin and Wild Industry with a factory. Gold means held; gray and a diagonal mark mean not held. Clicking identifies the player, type and state.
- Click a city/industry badge to pin its definition, or choose **Card list** for all 25 physical ordinary categories in a scrolling table.
- **A+** cycles normal, 15% larger and 30% larger tracker text.
- **Resume**, the bottom **Resume game** button or Escape closes the tracker. Escape first dismisses a pinned detail, then the card list, then the tracker. Gameplay keys and clicks cannot pass through the paused overlay.

## Public counting

**Played** counts ordinary cards publicly discarded this era, including filler action cards, passes and both Scout trade cards. **Unseen = total − played**; it is not the draw-deck count. The ordinary four-player set is 41 city cards and 23 industry cards, 64 total. Wild cards are separate.

Four-player Canal setup still includes four face-down discards, one per player. Initial public counts therefore show 64 unseen, although only 60 ordinary cards can be played in that era. The tracker does not inspect the hidden discards, draw order, opponent hands or shuffled seed. The Canal legend and each detail explain the uncertainty. Rail reshuffles all 64 ordinary cards and resets the counts; it has no new hidden setup discards.

Zero unseen city cards only removes the ordinary city-card route. Wild Location cards and legal industry-card builds can still reach that city. The badge does not claim the city is safe or unavailable for construction.

Wild ownership is reconstructed from public Scout acquisitions and subsequent wild-card discards. Reopening or continuing a match rebuilds the currently presented history prefix. Duplicate event delivery is counted once. Incomplete, conflicting or unavailable history shows question marks instead of misleading partial counts. No save-format change is required.

Rules and deck evidence are retained in the [implementation plan](../rules-audit/CARD_TRACKER_UI_PLAN.md), [official rulebook](../rules-audit/official-rulebook.pdf) and [four-player deck probe](../rules-audit/card-tracker-deck-evidence.json).

## Implementation and verification

The change adds one BoardPresenter observer and composes paused input handling into the existing StatisticsEntryPoints handler. It does not own replay completion, action confirmation or an AI cancellation callback. The AI DLL remains identical to qualified 1.4.11. Cecil compares existing game/helper method bodies; the two integration points are the only existing bodies changed. This preserves all previous rules, AI bridge, persistence, city locator and presentation recovery code.

Qualification uses the real Unity prefabs and in-memory games, without reading or changing the player's saved matches:

- 515 tracker checks: complete Canal/Rail and introductory histories, actual physical deck totals, Scout trades/wild spends, shared pools, duplicate/revision/incomplete-history handling and independence from private cards.
- Native eligibility, pause/restore, modal pointer/keyboard input, hook attachment, map/list interaction, owner cleanup and two-/three-player exclusion.
- Actual screenshots at 1280×720, 1920×1080 and 2560×1440, day/night, plus larger text and scrolling details. Checks include control bounds, reserved header/footer space and industry row alignment.
- Existing native regressions pass on the complete candidate: statistics 688 checks, presentation 184, link-animation 765 and startup/custom-game/AI resource checks 259.
- 46 installation fixture checks cover publication, hash/evidence rejection, baseline backups, offline status, retained save paths and interrupted replacement recovery.

The final reports and screenshots are archived with the release. Accepted baseline AI and managed rules/persistence evidence is explicitly retained for the unchanged implementation; this release makes no new AI playing-strength claim.

## Installation and recovery

Build and qualification use `artifacts/card-tracker-build`, separate from the shared AI staging directory. Install with the game closed:

```powershell
./scripts/install-combined-patch.ps1 -BuildDirectory ./artifacts/card-tracker-build
```

The installer checks the qualified baseline and candidate fingerprints, retains a complete previous-release backup under `artifacts/patch-recovery`, publishes the dependencies and game DLL atomically and keeps online play disabled. The installation record is `artifacts/patch-installation.json`. It rejects an unrelated later installed release so that concurrent changes cannot be silently overwritten.

To reinstall the archived release, verify/stage `1.4.12` with `scripts/stage-archived-patch.ps1`, then run `scripts/install-patch.ps1` with the game closed. Preserve the three previous-release DLLs together if restoring 1.4.11. Do not restore stock binaries over saves that require the combined patch.
