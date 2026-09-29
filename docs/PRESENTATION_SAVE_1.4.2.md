# Presentation, save/continue and AI allowance — 1.4.2

**Installed and verified:** package **1.4.2**, 28 September 2026 at approximately
22:42 America/Chicago (29 September 03:42 UTC), in the Steam game directory.

This cumulative local-game patch retains the previous rules fixes, city highlights,
six v3 strategies, Random selection, adaptive intent and disabled Play now control.
Online play remains disabled. The original game and the previously installed 1.4.1
DLLs are preserved for recovery.

## Repairs

| Area | Finding and repair |
| --- | --- |
| Main player panels | Current-player animation created a detached inner tween, so cancelling the returned animation left movement running. A HUD refresh also failed to reactivate occupied rows previously hidden for a smaller player count. Movement now belongs to one cancellable sequence; refresh restores occupied rows and recalculates vertical layout independently of horizontal animation. A small layout component repairs overlapping rows if a stale layout occurs later. |
| White industry tiles | The flip callback appended `Flipped` to a mutable sprite name that could already end in `Flipped`, producing a missing sprite and a white image. It now animates a detached display snapshot, uses the correct owned sprite, restores transforms when interrupted, and cancels obsolete flips before a slot is rebound. |
| Information panels | The replay overlay's full-screen FastForward raycast blocked the development board, income track and game log close buttons. Open inspection panels now remain above replay and era/game-end presentation, with matching keyboard input order. Confirmation and action-selection dialogs keep their priority. AI Develop replay does not animate or retain the user's open inspection board, so closing it cannot break that replay. |
| AI thinking allowance | The live v3 S1 executor grants 15,000 ms per turn and at least 1,024 work units. Preparation remains outside the clock; both actions and replanning share the allowance. Search may finish sooner when its work limit or search is exhausted. No style, fallback, intent, evaluator or frozen research profile changed. |
| Per-player replay progress | All stock local seats use network ID `Player`. Saving last-seen progress previously updated the first matching seat. Writes now resolve the immutable starting seat and color as well as the ID; ambiguous or cross-game updates are rejected without changing the save. |
| Failed turn saves | The stock save presenter swallowed failure instead of notifying the patched turn coordinator, making its retry branch unreachable. It now dispatches local failure to the existing retry flow on the main thread. The turn stays blocked until saving succeeds; stale callbacks from another game cannot open a retry dialog. |
| Continue | A newer completed game could hide an older unfinished one. Continue now selects the newest ready, unfinished game. Existing corrupt-save preservation and incomplete-creation checks remain. |

The exact natural trigger for the reported player-panel pile-up was not reproduced.
The detached-animation and row-reactivation defects were reproduced; the vertical
repair was tested by deliberately collapsing the real prefab's row positions.
This provides recovery for that layout symptom without changing portrait, money,
income or turn-order data.

## Validation

- Native Unity 2019.1.7f1 presentation regression: **184 assertions**, including all
  six industry types in all four colors, before/already-flipped states, interrupted
  animation and slot rebinding. It checks real close-button raycasts during
  thinking, replay, replay reordering, era-end and game-end, actual close routing,
  inspection-board closure during Develop, and modal priority.
- The archived 1.4.1 baseline reproduced white sprites, blocked close-button hits,
  hidden occupied HUD rows and failure to recover collapsed row positions.
- Fresh managed rules gates include **90 completed games** across 2/3/4 players,
  the build/resource/Scout/state/AI matrices, **84 persistence assertions**, and
  **57 liquidation/presenter assertions** including actual save-failure dispatch.
- Runtime/S1/save suite: **1,030 assertions**, including shared offline IDs,
  strategy/intent preservation and Continue selection behind a completed game.
- Six complete scripted local games cover 2/3/4 players and introductory/full
  rules, with a save/reload after each nonfinal turn. Human seats are scripted,
  not a physical mouse playthrough.
- Genuine generated saves made by each archived **1.3.2, 1.4.0 and 1.4.1** runtime:
  all six strategies load, act, save and reload; **48 assertions per producer**.
- Native AI matrix: **36 turns / 26,166 assertions**, all six strategies across
  2/3/4 players and introductory/full configurations; every turn received a
  15,000 ms allowance. Click guards, single completion and persisted intent passed.
  Longest observed turn was **15.42 seconds** including main-thread overhead.
- All **10 existing native Unity suites** passed, including **46,779 city-highlight
  checks** and the revised save-failure presenter tests; no unexpected native errors.
- Six complete generated-save games passed **1,397 assertions and 231 reloads**.
- Upgrade and interrupted-install recovery: **79 assertions** passed. Installed
  startup, menu choices and game creation: **57 assertions** passed. All three
  installed DLL hashes match the qualified candidate. All **116 previous hooks**
  remain; this package has **125 hooks**, with the same six bonus-state rewrites.

Tests use generated fixtures. No personal save was opened, migrated or overwritten.
The save format and v3 policy identity remain compatible; this release changes the
package version and live allowance, not the stored strategy/profile identities.
Existing interrupted-write recovery, checksummed journals, idempotent retry and
metadata locking were rerun. The review did not establish a normal gameplay
producer for the theoretical concurrent live-event mutation/acknowledgement race.

## Source and evidence

- `src/BrassPatch/Presentation/PresentationLifecycle.cs` and `manifest.json`
- `src/BrassPatch/Persistence/PersistenceFixes.cs` and `SaveCompletion.cs`
- `src/BrassPatch/Liquidation/LiquidationFixes.cs`
- `src/BrassLocalAI/Unity/LocalMenu.cs` and `LocalTurnBridge.cs`
- `tests/BrassPatch/Presentation/` and `tests/BrassLocalAI/qualify-presentation-release.ps1`
- `artifacts/patch-build/validation.json`, `presentation-lifecycle.json`,
  `local-ai-qualification.json`, `installation-verification.json`

Every qualifying report checks the candidate DLL fingerprints. Rules and native
helper evidence is fresh because those binaries changed. Policy/host source bytes
are checked against archived 1.4.1, and embedded profiles against the frozen banks.
The earlier research admission is preserved; these lifecycle tests do not make a
new claim about AI playing strength at the longer allowance.

## Installed fingerprints

| File | SHA-256 |
| --- | --- |
| `BrassPatch.dll` | `211A4365A0E833535EE756985140495C92C121BC665467C6C2513D60CEBC2413` |
| `BrassLegacyAI.dll` | `75495FF17A483A2F4AFA01E1D9330A449EA9344836B88BCD14FE9157031BCAEC` |
| `Assembly-CSharp.dll` | `A65623AA0D7926845BC444529D89BB9AA5A88A54E4A2B96CF00AF97DEEEBDF97` |

The prior release is archived at `artifacts/patch-releases/1.4.1`; this release's
sources, DLLs and qualifying reports are archived at `artifacts/patch-releases/1.4.2`.
The installer also keeps a verified recovery copy and the original stock assembly.
