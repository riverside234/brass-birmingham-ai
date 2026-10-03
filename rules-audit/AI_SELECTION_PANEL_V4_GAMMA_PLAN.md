# AI selection panel — V4 and Gamma

**Date:** 2 October 2026.  
**Status:** Proposed implementation plan. No code changes or installation are part of this request.  
**Verified baseline:** Installed patch **1.4.21**, including the current V4 AI and Statistics light theme. The relevant working UI/catalog/theme sources match the archived installed sources.  
**Primary display:** **1920 × 1200**; also qualify **1920 × 1080**. Do not compromise the 1200p design to accommodate 720p.

## 1. Intended result

Replace the small AI style popup with one centered **AI selection panel**, using the same ivory surfaces, dark text, brass accents, rounded controls and typography as the installed Statistics panel.

The panel has exactly two family pages:

| Page label | Choices | Purpose |
| --- | --- | --- |
| **V4** | The six existing V4 styles **plus Random** | Select the installed rule based AI |
| **Gamma** | **16 NN AI entries plus Random**, for **17 choices** | Select a trained NN AI when its game runtime and model are available |

Choose an AI independently for each computer player. Once Gamma is qualified, allow V4 and Gamma opponents at the same table. Do not combine all 24 choices into one long list or turn the 16 Gamma entries into 16 additional pages.

This plan reserves the Gamma UI and integration boundary now. Completing NN training, exporting the 16 models and qualifying a Unity inference backend are separate prerequisites for enabling Gamma play. Layout fixtures or placeholder labels must not masquerade as playable trained AIs.

## 2. What the current implementation needs to change

The reviewed production sources are:

- [AiSetupRow.cs](../src/BrassLocalAI/Unity/AiSetupRow.cs): the custom-game style button and separate difficulty controls.
- [StrategyPicker.cs](../src/BrassLocalAI/Unity/StrategyPicker.cs): a popup up to 740 canvas units wide, a pinned Random option and a two-column grid; search appears above eight concrete styles.
- [LocalMenu.cs](../src/BrassLocalAI/Unity/LocalMenu.cs): player-bound selections, preferences, editing guards, Random resolution and game creation.
- [ProfileCatalog.cs](../src/BrassLocalAI/Policy/ProfileCatalog.cs): six installed V4 profiles, with older runtime/profile compatibility.
- [DifficultyCatalog.cs](../src/BrassLocalAI/Policy/DifficultyCatalog.cs): Easy 2s, Mid 5s and Hard 12s.
- [LocalGameState.cs](../src/BrassLocalAI/Policy/LocalGameState.cs) and [LocalTurnBridge.cs](../src/BrassLocalAI/Unity/LocalTurnBridge.cs): saved assignments and turn dispatch currently assume the existing profile-based backend.

Three limitations require explicit treatment:

1. The picker contains a stale **“v3 AI”** footer even though its displayed profiles are V4. Replace that footer as part of the panel change. Identify the family through the **V4** page and custom-game summary rather than repeating a version suffix on every card.
2. Selection is currently a single strategy string. A family/model choice cannot safely be squeezed into the existing ProfileCatalog lookup, or inferred from a card's display text.
3. A training export is not already a game backend. Current NN contracts and portable exports are explicitly **four-seat**. Gamma needs checked capabilities and dispatch; adding a visible button alone must not enable it for unsupported games.

The previous [AI picker plan](AI_STYLE_AND_DIFFICULTY_UI_PLAN.md) remains a record of patch 1.4.4. This plan supersedes its popup layout, dark appearance and single-family selection design. Preserve the installed difficulty behavior and the [first-open initialization repair](../docs/AI_MENU_INITIALIZATION_1.4.5.md).

## 3. Visual design: use the installed Statistics style

Use [StatisticsTheme.cs](../src/BrassPatch/Statistics/StatisticsTheme.cs) and the installed [1.4.21 screenshots](../artifacts/patch-releases/1.4.21/screenshots/statistics-1920x1200-0-larger.png) as the visual reference. These are the actual palette values:

| Use | Color |
| --- | --- |
| Window | `#F5F1E7` |
| Cards and primary surfaces | `#FFFCF5` |
| Secondary surfaces | `#EEE8DB` |
| Main text | `#283238` |
| Secondary text | `#586269` |
| Brass accent / selected border | `#8A672B` |
| Selected fill | `#E9D8B4` |
| Border | `#C5BCA9` |
| Keyboard focus | `#286D83` |

Use the same **12px window / 7px control** corner treatment and **2.2px selected border**, adjusted through the same physical-pixel-to-canvas conversion. Put a 50% black input-blocking backdrop behind the opaque light panel. Do not allow menu artwork to show through text cards.

Use the existing game TMP font with the Statistics **second text preset, 115%, as the default**. Start from the same base sizes: approximately 30px title, 24px card title, 19–21px descriptive text and control labels. Offer the same Large / Larger / Largest text controls. Measure wrapping and enlarge/scroll content at larger sizes instead of shrinking text through automatic sizing.

Selected cards need a checkmark and the text **Selected**, as well as the brass border/fill. Keyboard focus needs a distinct teal outline. Hover, focus and selection must remain distinguishable. Unavailable Gamma cards use muted secondary surfaces, readable text and a written reason; avoid making them so faint that their names cannot be read.

Reuse the color tokens and simple rounded-surface geometry through a small neutral light UI utility. Do not couple the AI panel to Statistics reports, filters or game-over routing. StatisticsWidgets is currently internal, and its light theme is opt-in through a parent StatisticsTheme; simply using those widget names from the AI assembly would not reproduce the window correctly. Expose only the needed neutral primitives, including an explicit window/card surface role instead of relying on the object name “Statistics window.” Retain existing Statistics appearance and keep the new theme opt-in on the AI panel and its setup controls.

Use code-native icons: a restrained network/gear mark for V4, a small node-network mark for Gamma and dice for Random. Icons supplement the text labels. No portrait artwork, fabricated strength ratings or decorative plots are needed.

## 4. Panel layout

At 1920 × 1200, target a centered window around **1720 × 1100 physical pixels**, bounded by the actual safe area. Final dimensions should be qualified in Unity. Keep the top header compact; give most of the space to AI cards.

~~~text
┌────────────────────────────────────────────────────────────────────────────┐
│ CHOOSE AI · Eleanor Coade · Blue                  A−  Larger  A+       ×  │
│ Choose a computer opponent for this player.                               │
├──────────────┬─────────────────────────────────────────────────────────────┤
│              │ Gamma                         16 models · 16 available    │
│  V4          │ [ Find AI…                                      ] [Clear] │
│  6 + Random  │                                                             │
│              │ [ Dice  Random                                            ]│
│  Gamma       │   Choose an available Gamma model when the game starts.     │
│  16 + Random │                                                             │
│              │ [ Gamma 01 ] [ Gamma 02 ] [ Gamma 03 ] [ Gamma 04 ]        │
│              │ [ Gamma 05 ] [ Gamma 06 ] [ Gamma 07 ] [ Gamma 08 ]        │
│              │ [ Gamma 09 ] [ Gamma 10 ] [ Gamma 11 ] [ Gamma 12 ]        │
│              │ [ Gamma 13 ] [ Gamma 14 ] [ Gamma 15 ] [ Gamma 16 ]        │
│              │                                                             │
├──────────────┴─────────────────────────────────────────────────────────────┤
│ Current: V4 · Adaptive Advanced Book       Difficulty: Hard · 12s         │
│ Selected/focused AI description or availability reason.          [Close] │
└────────────────────────────────────────────────────────────────────────────┘
~~~

The diagram's **16 available** state represents the future completed roster, not the current installation. Initially Gamma says **“Models not installed”** and its cards show **“Awaiting model”**.

### Fixed controls and scrolling

- A narrow left navigation column, approximately 200–220px, has only **V4** and **Gamma**. Match Statistics navigation's spacing and selection treatment.
- Keep the header, family navigation, search field, Random strip and footer fixed. Only the concrete-AI grid scrolls vertically.
- **Gamma:** four columns × four rows, in stable catalog order. Give cards enough room for a two-line name, a short description and availability/selection status. All 16 cards and Random should be visible at 1200p with default text size.
- **V4:** three columns × two rows for the six longer existing names, with the same card styling and a pinned Random strip.
- At 1080p or Largest text size, allow the grid to scroll if its measured height exceeds available space. Keep all labels readable, the last model reachable and the close control fixed. No horizontal scrolling.
- Recalculate from the canvas transform, physical viewport and safe area when the resolution changes. Do not anchor the panel to a moving player row.
- Show a scrollbar when needed. Wheel, drag and keyboard focus reveal operate on the grid; they do not scroll the underlying custom-game menu.

### Card contents

Each card shows its display name, one concise description and an explicit state. The footer shows the full description of the hovered/focused card, so important text is not available only through an ellipsis. Retain full names through wrapping or the detail area.

Preserve the six current V4 identities and descriptions. Remove only the redundant “v4” suffix within the V4 page:

| Stable existing key | Card name |
| --- | --- |
| `balanced` | Adaptive Advanced Book |
| `rail-tempo` | Coventry Goods Opening |
| `cotton` | External-Beer Three Cotton |
| `goods-beer` | Worcester Iron-Cotton |
| `supplier` | Zero-Loan Industry |
| `pottery-flex` | Advanced Pottery Variants |

Gamma 01–16 are temporary UI slot labels until the trained roster provides real display names and descriptions. Model manifests provide those details. Do not invent specialist behavior or “strongest model” badges before it is measured.

### Search

Search within the active page by display name, description and public model label. Keep the page's catalog order. Preserve separate search/scroll state when switching pages during an open panel session; opening a fresh session clears search and reveals the assigned AI.

Show **“N of 16 models”** for a filtered Gamma grid, plus a clear empty state. Random stays visible and always samples all compatible available models in that family, independent of the search filter. State this in its help text. Filtering must never change the assigned AI.

## 5. Selection and custom-game behavior

Keep the current quick selection interaction:

1. Click the player's **AI** selection button to open the panel on the assigned family.
2. Click **V4** or **Gamma** to browse that page. Page changes do not modify the player.
3. Click an available AI card or Random to commit the family and choice **together**, then close the panel. No extra Apply click.

Hovering, focusing, searching, changing text size or scrolling is read-only. Escape, Close, the × control or clicking the backdrop dismisses without changing the AI. An unavailable card cannot commit; its reason remains accessible on hover, and a page-level message explains any family-wide restriction.

The custom-game row should use a larger, comfortably clickable light button rather than arrow cycling. Show a compact family summary such as:

~~~text
AI          [ V4 · Adaptive Advanced Book                  ▾ ]
Difficulty  [ Easy · 2s ] [ Mid · 5s ] [ Hard · 12s ✓ ]

AI          [ Gamma · Model display name                   ▾ ]
Difficulty  [ Easy · 2s ] [ Mid · 5s ] [ Hard · 12s ✓ ]
~~~

Keep difficulty immediately below the AI button and independent of the selected family/style. The panel footer repeats it as read-only context. Use a measured two-line row summary when a real model name is long; do not make it illegible to force one line.

Human rows hide AI controls. Preserve the previous bot choice when switching to human and back. Continue rows show the actual saved family, resolved AI and difficulty as read-only text. An old save's AI version must not be relabeled V4 just because its family now uses the V4 page for new games.

Keep bindings attached to the actual PlayerMetadata and active setup owner. Color changes, nickname changes, reorderings and temporarily removed slots must not transfer a choice to another player. Close an open panel if its owner is refreshed, hidden, removed, becomes human or is replaced. Reject stale callbacks rather than applying them to a new row.

Preserve the inactive-parent lookups from patch 1.4.5: build/bind the new controls before the cached menu is shown. The old labels and arrows must be hidden on the very first visible frame.

### Keyboard and modal input

Tab / Shift+Tab cycle within the panel; arrow keys navigate cards and family controls; Enter/Space selects an available focused card; Escape closes. Reveal a focused offscreen card. Return focus to the originating AI button on close if that row remains editable.

Block pointer and keyboard activation of underlying New Game, Continue and other player rows while the panel is open. Consume the closing click and Back/Escape for the frame that dismisses the panel so the same event cannot also start a game or leave the menu.

## 6. Family-aware catalog and selection state

Introduce a small menu catalog separate from the research ProfileCatalog. Proposed concepts:

| Concept | Required information |
| --- | --- |
| AiFamily | Stable IDs `v4` and `gamma`; labels exactly V4 and Gamma |
| AiChoice | Family + stable entry ID, or the explicit Random choice |
| AiCatalogEntry | ID, display name, description, order, family, availability and supported game configuration |
| Gamma model descriptor | Logical model ID, immutable model/weights identity, export manifest, compatible observation/action/search contract and runtime requirements |
| Player AI selection | Family + choice + independent difficulty, bound to player metadata |

Use `(family, entryId)` or an equivalent typed key throughout the UI, preferences and game-start validation. Do not use filtered-grid indexes, displayed names or legacy strategy keys as Gamma identities. Two models may have the same friendly name; distinguish them with a short public model label, while preserving unique internal IDs.

Preserve existing V4 preferences by importing `BrassLocalAI.v1.Strategy.<seat>` and difficulty values into the V4 selection. For a player without a preference, retain the existing `balanced` / Hard defaults. Do not overwrite a valid legacy preference merely because the new preference schema is absent.

Reserve 16 Gamma catalog slots for this roster. When a trained model is installed, its validated descriptor fills the corresponding slot. Slot labels/order are presentation; model identity and content fingerprints determine execution. Do not automatically populate the roster from every training checkpoint or a mutable “latest” filename.

### Random semantics

| Page | Random behavior |
| --- | --- |
| V4 | Choose uniformly from the six installed compatible V4 profiles |
| Gamma | Choose uniformly from the compatible, qualified Gamma models currently available for this setup |

Resolve Random **once per bot at successful game creation**, independently of the game's shuffle and other bots. Allow different bots to draw the same AI; sampling is with replacement. Persist the concrete resolved identity. Menu refresh, difficulty changes, continuing a game and subsequent turns do not reroll it.

V4 Random chooses a starting profile; its persistent intent and fallback remain adaptive during play. A fixed model identity similarly does not imply a fixed sequence of tactics. Random never mixes the V4 and Gamma families and never falls back silently from Gamma to V4.

If fewer than 16 Gamma models are ready, show the availability count and leave other slots unavailable. Random describes the actual eligible count. With zero eligible models it is unavailable; never call random selection on an empty collection.

## 7. Gamma availability, difficulty and execution prerequisites

### UI can be delivered before the models

In the first UI release, V4 is fully usable and Gamma is a visible future page with **16 unavailable entries plus an unavailable Random option**. If a qualified Gamma game backend and models are supplied at implementation time, enable only entries that pass the checks below.

“Model present” is insufficient to make a card playable. Require a compatible game adapter, checked artifact identity, correct tensor/feature/action contracts and supported game settings. Show concise reasons such as **Awaiting model**, **Runtime unavailable** or **Requires 4 players**. Keep technical paths and detailed validation diagnostics in developer logs.

The local [masked search manifest](../configs/contracts/masked-ismcts/manifest.json) uses four-player observation/model contracts. [export.py](../src/brass_ai/learning/export.py) exports and validates `four-seat-onnx/1.0.0`, including four-seat output heads. Therefore initially restrict Gamma to **validated four-player configurations**. Two/three-player or introductory support must be explicitly qualified before being advertised. V4 keeps its currently supported configurations.

Revalidate when the player count, introductory setting or model availability changes. Close/rebind stale panel sessions. If an already assigned Gamma choice becomes incompatible, mark that specific row and disable Start with an explanatory message; do not silently select V4 or another model.

### Preserve the existing search presets

- V4 keeps **Easy 2 seconds, Mid 5 seconds, Hard 12 seconds**, its accepted V4 search implementation, worker settings, work limits, profile snapshots, adaptive intent and fallback.
- This UI task does not change training settings or reinstate the old S1 runtime. Historical comments or prior UI plans are not authoritative for the installed V4 execution configuration.
- Gamma may expose the same time presets when its game search adapter supports the declared timed-search contract. Bind them to one turn deadline covering both actions and replanning, according to that backend's qualified contract.
- Do not promise identical strength across families for the same time label. If a future Gamma model only supports direct policy inference, do not simulate a 12-second wait; its release must define appropriate difficulty behavior before enabling those controls.

### Dispatch boundary

Existing LocalGameState creates `kind: legacy` assignments and LocalTurnBridge always loads ProfileCatalog.ForPlay. A Gamma selection must have its own checked assignment and backend dispatch. It must never be passed through that legacy path with a fabricated strategy name.

Use the existing [IAiExecutionBackend / IAiTurnResources lifecycle](../src/BrassLocalAI/Execution/AiExecutionScheduler.cs) for a future adapter. Preserve cancellation, revision/generation checks, legal-action validation, one active turn and proper CPU/device completion before releasing turn buffers. Model loading must not block pointer handling or run on every hover/click; shared loaded weights belong to the backend, not an individual turn job.

Browsing Gamma may read a bounded catalog/availability snapshot. Any asynchronous validation result must check its setup/session generation before updating the panel. A closed panel or replaced player cannot receive an old validation callback.

## 8. Persistence boundary

The UI-only first release should retain the current V4 game assignment format and resume validation. New family-aware menu preferences can exist without rewriting existing games. No broad save-system revision is required for this UI plan.

Before actual Gamma gameplay is enabled, extend assignments with a versioned family/backend identity and resolved model identity, export fingerprint and execution settings. Preserve old `legacy` assignments, exact profile snapshots and older-save time allowances through their existing validation path. Do not rewrite or reroll saved V4/older-runtime assignments just to display them in the new UI.

Game creation must validate the entire lineup and freeze Random/model choices before publishing a ready game. Missing/incompatible Gamma artifacts must stop creation or continuation with a clear recoverable message, preserving the previous save. Never substitute an arbitrary newer checkpoint or the current V4 bot for a missing saved model.

This is an integration requirement for family selection, not a claim that a new save bug has been reproduced.

## 9. Implementation sequence and files

| Step | Work | Main files / proposed additions |
| --- | --- | --- |
| 1 | Pin installed source and establish regression captures; verify current V4 identity and configuration | Installed release archive and patch installation receipt |
| 2 | Add the family-aware menu catalog, typed choice and availability state; import existing menu preferences | New Policy/AiSelectionCatalog.cs and Policy/AiSelection.cs; LocalMenu.cs |
| 3 | Expose a minimal neutral light-theme/surface utility using the existing Statistics tokens | Small addition under BrassPatch UI; preserve StatisticsTheme/StatisticsSurface behavior |
| 4 | Replace the picker with the fixed two-page panel, card grid, search, Random strip and text sizing | New Unity/AiSelectionPanel.cs; retire StrategyPicker's old layout; AiSetupRow.cs |
| 5 | Bind custom-game rows, first-open initialization, family summaries, atomic selection and lifecycle guards | LocalMenu.cs; AiSetupRow.cs; tooltip/focus helpers |
| 6 | Add a future Gamma descriptor registry and readiness checks; keep placeholders unplayable until qualified | New Gamma catalog/availability adapter, with real model integration deferred if unavailable |
| 7 | When ready, implement checked Gamma assignments, dispatch and runtime qualification | LocalGameState.cs; LocalTurnBridge.cs; model adapter and existing execution interfaces |
| 8 | Qualify native UI, catalog behavior and compatibility; archive and install a complete package | Local AI tests, native preview harness, patch build/qualification/install tools |

Steps 1–6 can deliver the requested panel while training continues. Step 7 is necessary before any Gamma option can actually play. Do not claim a neural integration is complete merely because a stub passes UI tests.

The two existing menu helpers are delivered inside BrassLegacyAI.dll despite their BrassLocalAI source paths. Stage changes from the accepted release sources, with a method/source allowlist for the AI selection work. Avoid rebuilding unrelated research work into the shipped game by accident. Preserve existing gameplay hooks, Statistics, Card Tracker, city highlighting and offline-only mode.

At implementation time, recheck the installed version. Use the latest accepted package if it has advanced since this plan; do not roll the game back to 1.4.21.

## 10. Verification and acceptance criteria

### Functional qualification

- [ ] Exactly two family pages labeled **V4** and **Gamma**; no combined long family list or extra model pages.
- [ ] V4 has six concrete profiles plus Random, with unchanged execution identities and difficulty settings.
- [ ] Gamma has **16 concrete slots plus Random**. Qualified models are selectable; unavailable slots and empty-family Random cannot start a game.
- [ ] A 16-model fixture proves that every card, including number 16, selects its stable family/model identity after search, page switching and scrolling.
- [ ] Uniform Random uses the eligible family roster, excludes incompatible entries and is independent of search filters; deterministic injected draws verify mapping and boundary cases without flaky frequency thresholds.
- [ ] Random is resolved once at game creation, supports duplicate draws across bots and is not rerolled by UI refresh, continue or difficulty changes.
- [ ] Page changes, hover and keyboard focus do not change a player. A successful card click changes family/choice together and closes the panel without changing difficulty.
- [ ] Human/AI toggles, nickname/color changes and removed/readded slots preserve the correct player's settings. Read-only Continue rows cannot edit assignments.
- [ ] Fresh/cached menus and menus with Continue visible render the new controls on the first visible frame, while their parents are initially inactive.
- [ ] Closing, hiding or refreshing a menu cleans up the panel, backdrop and tooltips; stale player/model-validation callbacks are rejected.
- [ ] Mouse, keyboard and outside clicks cannot activate underlying Start, Continue, another player row or menu Back through the modal.
- [ ] Gamma capability tests cover 2/3/4 players, introductory/full games, empty/partial/full rosters, duplicate friendly names, missing files and mismatched model identities.
- [ ] V4 preferences and representative existing assignments/resume summaries retain their original identities and time allowances. Gamma assignment roundtrips are required only when real Gamma play is enabled.

### Visual qualification in the native game

Capture and inspect actual Unity renders at **1920 × 1200** and **1920 × 1080**, at all three text presets:

- [ ] Default **Larger / 115%** text and the exact installed Statistics palette, with matching rounded corners, selection, focus, buttons and scrollbar treatment.
- [ ] At 1200p default, all 16 Gamma cards plus Random visible, readable and clickable; no need to cycle through options to reach the last model.
- [ ] At 1080p / Largest, any necessary scrolling preserves readable text, fixed controls, keyboard access and the last card.
- [ ] V4's long names, Gamma's longest actual name, duplicate model names, long player names, unavailable reasons, search-empty state and footer details are measured and readable.
- [ ] Selected and focused states are distinct; unavailable choices remain legible; search input and tooltips do not obscure necessary controls.
- [ ] Correct geometry under the game's CanvasScaler, window/fullscreen changes and safe-area conversion. Test the top and bottom player rows and the split Continue/New Game layout.
- [ ] Existing Statistics screenshots still match their accepted style; Card Tracker and other menu/game controls have no unintended global recoloring.

Synthetic Gamma captures validate layout and selection plumbing only. Label them as fixtures in the evidence; they do not demonstrate trained NN behavior or runtime readiness.

### Release qualification

- [ ] Relevant native menu and managed catalog/lifecycle tests pass, plus required patch-build and compatibility checks for the touched modules.
- [ ] Verify the allowed source/method scope. If the real Gamma backend is added, run separate legal-action, cancellation, deadline, mixed-family game and model parity qualification before enabling it.
- [ ] Archive the source, binaries, checked catalogs, screenshots and concise release notes with an accurate readiness statement.
- [ ] On a later implementation/install request, use the existing recoverable installer, verify installed fingerprints and launch/first-open behavior. Record the backup so the preceding installed patch can be restored.

## 11. Completion definition

The UI phase is complete when selecting V4 uses the new Statistics-style panel, Gamma presents its planned 16+Random roster with honest readiness states, existing settings/resume behavior remain compatible, and native 1200p/1080p qualification passes.

Gamma gameplay is complete only after the 16 trained entries have checked model artifacts and a qualified game backend. Adding or updating roster metadata should then populate the existing Gamma page without requiring a new UI layout.
