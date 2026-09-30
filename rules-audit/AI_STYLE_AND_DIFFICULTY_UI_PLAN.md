# AI style picker and difficulty selection plan

**Status:** Implemented, installed and verified as package 1.4.4.  
**Date:** Planned 29 September 2026; implemented 30 September 2026.  
**Baseline:** Installed local AI package 1.4.3, with six v3 S1 styles plus Random and a 10-second search allowance.

**Implementation record:** The direct chooser, larger-roster filter, per-player difficulty buttons, versioned execution assignments and older-save compatibility are installed. All 108 native AI configurations, 12 managed regression suites, 10 native rules/UI suites, presentation tests, archived-save compatibility, installer recovery and installed startup passed. See the [release record](../docs/AI_STYLE_AND_DIFFICULTY_1.4.4.md) for evidence, installed fingerprints and verification limits. The design below is retained as the implementation plan; display testing used actual Unity render targets rather than a physical playthrough of every Windows DPI/fullscreen combination.

Replace the previous/next style arrows with a button that opens a chooser. Put three difficulty buttons immediately below it on each AI player's row: **Easy 2s**, **Mid 5s**, and **Hard 12s**. Selecting a style takes one click to open the chooser and one to choose; changing difficulty takes one click.

The chooser must accommodate today's seven choices and a future roster of 16 styles plus Random. Difficulty changes the current v3 AI's search time. It preserves the research bank's node setting and each specialist's adaptive intent and fallback behavior.

## Proposed layout

Keep each player's portrait, name, color and human/AI controls. Use the space currently occupied by the AI label and arrow selector for two clearly labeled control rows. The diagram shows arrangement, not final pixel dimensions.

```text
 [Portrait]  Eleanor Coade                         AI

             Style       [ Goods/Beer          v ]
             Difficulty  [ Easy ] [ Mid  ] [ Hard ]
                           2s       5s      12s
                                           selected
```

Difficulty sits directly below Style because the current menu is narrow when Continue and New Game appear together. Keep this arrangement consistent when the Continue panel is absent. Allow a modest increase in player-row height or panel width if needed; do not shrink labels until they become difficult to read. Reposition the introductory-game checkbox and New Game button as part of the same layout pass.

Human rows hide both AI controls. Read-only Continue rows show the saved style and difficulty as plain text, with no arrows or inactive buttons that suggest editing is possible. Keep the existing control for changing a player between human and AI accessible.

### Style chooser

Clicking anywhere on the style button opens one shared popup above the menu. Its heading identifies the player being edited. Anchor it near that player's row when space permits, and clamp it to the visible safe area when the row is near the bottom or right edge.

```text
 Choose AI style — Eleanor Coade                 [x]

 [ Random                                          ]
   Choose a starting style; plans can adapt.

 [ Balanced               ] [ Rail Tempo            ]
   Adapts to opportunities.   Favors useful links.

 [ Cotton                 ] [ Goods/Beer          ✓ ]
   Prepares cotton sales.     Goods and brewery support.

 [ Supplier               ] [ Pottery Flex          ]
   Supplies coal and iron.    Adapts pottery investments.
```

- Show Random first in a full-width row, followed by a two-column grid of styles in catalog order. Each tile has a name and a short description. Show the selected style with a checkmark and a visible border.
- Today's seven choices fit without scrolling at the primary reference resolution. With more than eight concrete styles, reveal a **Find style** field and use a bounded scrolling grid. Keep Random and the search field above the scrolling area. Support at least 16 concrete styles plus Random; do not hard-code seven or 16 children.
- Filter names and descriptions locally. Preserve catalog order, show a clear empty result, and provide a clear-filter button. Opening the picker again clears the filter and reveals the selected style.
- Clicking a tile immediately updates that player's style and closes the popup. There is no extra Apply step. Escape, the close button, or clicking outside cancels without changing the selection. Consume the outside click so it cannot also start a game or change another control.
- Support keyboard focus, arrow navigation between tiles, Enter/Space to select, and Escape to close. Return focus to the originating style button. Keep the full description available on focus/hover if a compact tile must abbreviate it.
- Show only installed, supported styles. Future roster placeholders belong in layout tests, not in the playable menu. Hide internal profile IDs and micro-variant names. A small AI version label in the popup footer can identify the installed generation without repeating “v3” on every tile.

### Appearance

Match the game's dark brown and brass palette. Use a mostly opaque dark panel so the illustrated background cannot obscure text. Start with warm ivory text, a muted gold border for selection, and a lighter gold hover/focus outline. Use restrained rounded corners on the new controls.

Difficulty buttons use the same palette and treatment as the style button. Selected state needs a border, fill and clear text; color alone must not carry it. Avoid turning Easy/Mid/Hard into the old green/orange/red stock-AI selector. Keep seconds visible below the label inside each button, and use comfortably clickable targets rather than small arrow icons.

These colors and dimensions require screenshots in the actual Unity menu before release. The current native captures were inspected for this plan: both the single New Game layout and the split Continue/New Game layout.

## Difficulty behavior

| Display | Stable setting ID | Search allowance per AI turn | Current v3 work setting | Current bank reply setting |
| --- | --- | ---: | ---: | ---: |
| Easy · 2s | `easy` | 2,000 ms | 240 | 2,048 |
| Mid · 5s | `mid` | 5,000 ms | 240 | 2,048 |
| Hard · 12s | `hard` | 12,000 ms | 240 | 2,048 |

Each AI player has an independent setting. Changing difficulty must preserve its chosen style, and changing style must preserve difficulty. A table can therefore contain Cotton/Easy, Supplier/Mid and Random/Hard.

**Recommended default:** Hard, 12 seconds, when no valid preference exists. This is the closest of the requested options to the installed 10-second setting. Remember both settings per setup slot after successful game creation. Preserve the current Balanced default and existing saved style preferences. Switching a row to human and back, changing color, or temporarily removing and restoring that slot should retain its selections.

Tooltip wording: “Allows up to 2/5/12 seconds of search for this AI's turn. It may finish sooner. Preparing the board and playing animations can add time.” The allowance covers both planned actions and any replan within that turn; action two must not receive a fresh allowance. Retain the existing boundary that starts search timing after the initial live-board capture.

All three choices run the current v3 S1 implementation. They do not select the abandoned Easy/Medium/Hard algorithms or change evaluation weights, beam widths, intent, fallback, cash policy or conversion policy. Leave `work_limit` at 240 and `reply_work` at 2,048 as specified by the current bank. Only the detached execution configuration receives the selected time allowance; frozen research profiles stay intact.

The labels describe the requested time presets. Longer search is not a measured guarantee of stronger play: the 240-node limit can make two settings finish with the same result. Qualification should check that the shorter setting still produces coherent complete turns, and report any observed strength differences without claiming the future research difficulty calibration is already implemented. Three Hard opponents can use up to 36 seconds of allocated search between human turns, plus preparation and animations.

Random chooses a concrete starting style once per bot when a game is created, independently of difficulty and the game's shuffle. Its persistent intent can change and its specialist can use the existing fallback. Refreshing the menu, continuing a game, or changing difficulty must not reroll an already created game's style.

## Implementation changes

### Catalog and setup state

Add a small `DifficultyCatalog` defining the three stable IDs, labels and milliseconds. Represent menu selection as a pair of style ID and difficulty ID attached to the actual player metadata. Clone both into the new game's assignments before creation. Never derive player identity from current turn order or a filtered list position.

Use catalog entries with stable IDs, display names and descriptions for the picker. Replace selection by numerical array position with selection by ID. Reuse the existing profile validation and supported-style list. Generate Random's description from the catalog, removing the hard-coded “six” so future additions do not leave stale text.

The newly edited [personality research plan](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/PERSONALITIES_AND_DIFFICULTY_PLAN.md>) proposes 16 visible archetypes with eight hidden variants each. This UI should accept such a future catalog without displaying 128 entries. Training, variant sampling and calibrated neural difficulty remain separate work; this change exposes the six currently installed v3 styles and Random.

**Critical routing requirement:** Keep the engine's existing `AIType.Hard` marker for every custom v3 bot. The new `easy`/`mid`/`hard` values belong to separate custom execution metadata. Writing `AIType.Easy` or `AIType.Medium` would risk routing a selected v3 bot to the original implementation.

### Unity controls and lifecycle

Create reusable `AiSetupRow` and `StrategyPicker` components in the local AI Unity integration. Reuse available fonts, button materials and UI sprites. Add controls through the existing row hook rather than editing unrelated game screens.

The current `SetPlayer` hook is used by more than one kind of row. Bind custom controls only to supported editable custom-game rows; render Continue as a summary; preserve tutorial and other original menu behavior. Hide the old previous/next arrows and remove them from keyboard navigation in the custom editor. Retain their hook compatibility where other original screens still use them.

Refresh must be idempotent: one component, one set of children and one callback per control. Refreshing a label must not emit another edit event. Clean up the popup, raycast blocker and tooltip on selection, cancel, player removal, human/AI change, menu exit and scene destruction. Invalidate the popup binding before a metadata replacement, then require reopening it; a stale callback must never edit another player. Only one popup can be open.

Build an explicit focus order through each row's style and difficulty controls. The overlay must receive pointer input above the menu and disappear completely when dismissed. Avoid introducing any full-screen input blocker during gameplay.

### Runtime allowance

Read the validated difficulty from the current actor's saved assignment in `LocalTurnBridge.PlayAITurn`. Replace the global live time choice with that actor's allowance. Keep the existing shared clock, remaining-time calculation, detached observations, stale-result checks, complete-turn incumbents and single-commit behavior.

Update the thinking label to a compact form such as **Goods/Beer · Mid · Thinking…**, with **Playing…** during application. Continue to keep Play now disabled. Clicking elsewhere or closing an inspection panel must not shorten or restart search.

Test node accounting as well as the configured value. The current `StartRequest` can create another coordinator after a stale continuation; each planner session owns its node counter. Merely asserting `work_limit == 240` does not prove an aggregate 240-node ceiling across a rare replan. Record actual accounting, preserve the bank's definition, and avoid multiplying the configured work allowance by difficulty. Any necessary correction to enforce an intended shared turn limit must be explicitly identified and tested rather than hidden in a label change.

### Saved games and preferences

Persist a versioned execution setting alongside each AI assignment, separate from its immutable profile, for example:

```json
"execution": {
  "version": "local-ai-execution/1.0.0",
  "difficulty": "mid",
  "search_ms": 5000
}
```

Validate the ID/time pair against the versioned catalog and include this object in the assignment integrity hash. Keep difficulty independent of the profile ID used by persistent intent. Do not change the saved profile's `milliseconds`: current saved-profile validation compares the entire profile with its frozen catalog snapshot.

Use an explicit new extension-format version for newly created games and retain a reader for the existing format. Test format dispatch and integrity checks before enabling the new writer. A current-format assignment with missing or invalid difficulty is an error; only a recognized older format receives compatibility behavior.

| Case | Proposed behavior |
| --- | --- |
| New custom game | Save the selected difficulty/time with each bot; restore it exactly on Continue. |
| Existing local-AI save without difficulty | Continue with the current installed compatibility allowance of 10 seconds; display **Existing · 10s** in the read-only summary. |
| Original game save without local-AI metadata | Preserve its original bot path and honest Original AI label; do not claim a new time preset applies. |
| Invalid saved execution data | Preserve the save and show the existing recoverable configuration error. Do not silently substitute a difficulty or stock AI. |
| Missing or obsolete menu preference | Use Hard/12s for the next new game, while leaving saved games untouched. |

Older saves did not record the live allowance, so 10 seconds is compatibility with today's installed behavior, not a claim to recover the allowance used when that game began. Keep the original saved format and profile bytes on its ordinary save path unless a separately tested migration is required. Continue rows must not overwrite new-game preferences. Test with disposable saves and copied compatibility fixtures.

## Files and responsibilities

Paths below identify existing integration points; the named new components are proposals.

| File or area | Planned change |
| --- | --- |
| [LocalMenu.cs](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLocalAI/Unity/LocalMenu.cs>) | Direct style selection, paired setup state, per-seat preferences, row binding, creation and Continue summaries. |
| [ProfileCatalog.cs](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLocalAI/Policy/ProfileCatalog.cs>) and proposed `DifficultyCatalog.cs` | Display catalog and strict difficulty mapping; preserve frozen profile content and identities. |
| Proposed `Unity/AiSetupRow.cs` and `Unity/StrategyPicker.cs` | Layout, popup grid/filter, selected states, keyboard focus and lifecycle cleanup. |
| [LocalGameState.cs](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLocalAI/Policy/LocalGameState.cs>) | Versioned execution assignments, hash coverage, strict new-format validation and old-format compatibility. |
| [LocalTurnBridge.cs](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLocalAI/Unity/LocalTurnBridge.cs>) and [AiTurnControls.cs](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLocalAI/Unity/AiTurnControls.cs>) | Per-actor time allowance and accurate thinking text. |
| [Unity manifest](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLocalAI/Unity/manifest.json>) | Change hooks only if row/menu lifecycle integration requires it; retain existing routing protections. |
| [LocalAiUnityTests.cs](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/tests/BrassLocalAI/LocalAiUnityTests.cs>), [LocalRuntimeTests.cs](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/tests/BrassLocalAI/LocalRuntimeTests.cs>) and save compatibility tests | Replace arrow-cycle assumptions, add selection/budget/state checks and qualify actual native execution. |
| Release scripts and notes | New candidate version, qualification matching the changed scope, rollback files and installed verification when implementation is requested. |

## Validation and release sequence

1. **Build the menu in the isolated Unity test player.** Capture four-player New Game and split Continue/New Game layouts, the open chooser, all three selected difficulty states, and a synthetic 16-style roster plus Random. Check the actual canvas scale before fixing dimensions. Start with 1600×900, then verify 1280×720, 1920×1080 and 2560×1440, windowed/fullscreen and representative Windows scaling. If a layout cannot fit, use popup scrolling and responsive spacing rather than unreadable type.
2. **Verify interaction and identity.** Exercise 2/3/4-player setups, introductory/full games, color swaps, human/AI switches, removing/readding slots, reopening the menu and reopening the app. Select every actual style directly and the last item in the expanded test catalog. Cover mouse, wheel and keyboard; popup cancel; clicking over New Game behind the overlay; repeated refresh; and destruction while open. Confirm style changes never change difficulty, and vice versa.
3. **Verify persistence.** Round-trip mixed difficulties and Random through game creation, save, exit and Continue. Verify the same player/style/time/intent after color changes and turn reordering. Cover the previous format, original-AI saves, corrupted new fields and hashes, missing preferences, and unsupported IDs. Ensure reading a resume row neither fires editing callbacks nor writes preferences.
4. **Verify native AI behavior.** Extend the existing 36-turn matrix to six styles × three difficulties × three player counts × two rule modes, giving 108 native configurations. Record each actor's actual allowance, bank work settings, complete legal result and time spent. Probe input guards and human-turn transitions for each difficulty. Add focused low-budget, second-action, stale-continuation and cancellation cases. Show preparation/search/application timing separately; assess permitted completion overhead rather than treating animation time as search time. Preserve legal incumbents and recoverable error handling when a budget expires; never force a pass merely to meet the label.
5. **Qualify and install only during the implementation task.** Use the existing build, isolated runtime, packaging/rollback and installed-startup workflows. The 1.4.3 budget-only qualifier cannot establish correctness for this wider UI/save change; update qualification scope and pin the new candidate hashes. Confirm existing rules, city highlights, player panels, industry sprites, inspection-panel closing, online disablement and disabled Play now are retained. Reuse prior evidence only for provably unchanged components, and run focused fresh regressions for affected integration paths.

Acceptance requires visible style and difficulty settings for every AI, direct selection without cycling, no clipped or overlapping controls, exact per-player 2/5/12-second configuration, intact research profile settings, reliable old/new Continue behavior, and no recurrence of click-to-skip AI turns.

## Reviewed baseline

The plan was checked against the baseline `LocalMenu`, `ProfileCatalog`, `LocalGameState`, `LocalTurnBridge`, `Coordinator`, native menu tests, decompiled player-row/menu views, and native menu screenshots. The [1.4.3 budget record](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/AI_BUDGET_1.4.3.md>) documents the previous installed settings. All six `*-v3.json` profiles specify `work_limit: 240`, `reply_work: 2048`, and a frozen research allowance of 3,000 ms; the live override at that baseline was 10,000 ms.

The earlier [AI integration plan](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/LEGACY_AI_AND_UI_INTEGRATION_PLAN.md>) remains a historical implementation record. Package 1.4.4 replaces its cycling-control design. It does not change the in-progress research documents or implement their future neural personality system.
