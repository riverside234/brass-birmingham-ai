# Player selection panel revision — Human, V4 and Gamma

**Date:** 3 October 2026 (Chicago).  
**Status:** Implemented, installed and verified as **1.4.24** on 3 October 2026 (Chicago). See [release notes](../docs/PLAYER_SELECTION_1.4.24.md).  
**Implementation baseline:** accepted **1.4.23**. The relevant setup, panel, catalog and shared-theme sources match its accepted source fingerprints.  
**Visual target:** **1920 × 1200**, default **Larger / 115%** text. Retain existing automated layout regressions; manual visual inspection for this revision can focus on 1200p.

This revision replaces the interaction and setup-row layout in the [V4/Gamma panel plan](AI_SELECTION_PANEL_V4_GAMMA_PLAN.md). Its AI identities, availability rules, Random behavior and future Gamma integration requirements continue to apply.

## 1. Requested result

Each active player in **New Game** gets **one player-selection button**. That button opens a shared panel for choosing Human or an AI, including its difficulty.

| Area | Revised behavior |
| --- | --- |
| Player row | One button summarizing Human, or AI family/style/difficulty |
| Left navigation | **Human**, **V4**, **Gamma**, in that order |
| Human page | Local human control; no difficulty controls |
| V4 page | Existing six styles plus Random, with difficulty controls inside the page |
| Gamma page | Existing 16 slots plus Random, with its own difficulty area; all remain unavailable |
| Appearance | Existing Statistics/Game History light palette, rounded controls, typography and focus treatment |

“One button” applies to the **player-control choice**. Portrait/color selection, name editing and adding/removing a player keep their existing functions. Remove the old Human/AI cycling control and all separate row difficulty controls from this selection flow.

## 2. Verified code constraints

| Current implementation | Consequence for this revision |
| --- | --- |
| [AiSetupRow.Bind](../src/BrassLocalAI/Unity/AiSetupRow.cs) shows its custom controls only for bots | The replacement button must also appear on Human rows, including the first visible frame |
| The row creates one style button and three difficulty buttons | Replace that cluster with one summary button and rebuild its keyboard navigation |
| [LocalMenu.IsEditable](../src/BrassLocalAI/Unity/LocalMenu.cs) explicitly rejects Human players | Add a guarded setup-slot editing path that permits Human; preserve the existing AI-only guard for callers that require it |
| [AiSelectionPanel.Commit](../src/BrassLocalAI/Unity/AiSelectionPanel.cs) closes immediately after a card click | Use a draft selection and Apply so AI choice and difficulty can be set together |
| The panel identifies its page with `AiFamily` | Human needs a UI page/controller role, not a fabricated AI family or strategy key |
| `EditDifficulty` rejects edits while the modal is open | The panel needs one validated apply operation, rather than calling separate existing edit callbacks |
| [DifficultyCatalog](../src/BrassLocalAI/Policy/DifficultyCatalog.cs) defines Easy 2s, Mid 5s, Hard 12s | Keep these allowances and reuse the catalog for labels and validation |
| `LocalMenu.CanStart` requires at least one Human | Preserve that rule and display a clear explanation when every active slot is AI |
| Continue summaries use saved runtime/profile settings | Keep them read-only, with their actual saved version and allowance |

The baseline Gamma catalog has zero available entries. Browsing this page, changing page layout or adding a Human choice must not enable trained models or bypass its availability checks.

## 3. One button in each player row

Use the same light control for Human and AI, positioned consistently beneath the player's name. A small person or AI icon and a right chevron make its purpose recognizable.

~~~text
Player name / portrait
[ Person   Human                                  ▾ ]
           Local player

Player name / portrait
[ V4       Worcester Iron-Cotton                   ▾ ]
           Easy · up to 2s
~~~

Both lines belong to **one button and one hit area**. The example's second line is informational, not another control. Prefer a roughly 60–68px physical height where the current row geometry permits; measure the split Continue/New Game layout before fixing dimensions.

- Use `Human` / `Local player` for a human slot; do not show a stale AI difficulty.
- For AI, show family plus style on the first line and difficulty/time on the second. `V4 · Random` remains Random until game creation.
- Wrap long names when space permits; otherwise use a measured ellipsis and a full-label tooltip. Reserve room for the icon and chevron.
- Keep nickname, portrait and button aligned for every role. Recompute offsets from the original row geometry, so rebinding does not accumulate vertical movement.
- Hide the stock type labels/type arrows and old difficulty labels/arrows/buttons on the first frame, including when the parent menu is inactive during binding. Identify controls by verified bindings, not translated text.
- A removed or inactive player slot has no active selection button.
- Continue rows retain a compact read-only summary, with no chevron or clickable selection control.

## 4. Panel layout at 1200p

Rename the visible title to **CHOOSE PLAYER**. Show the player's name and color so the user always knows which slot is being edited. Keep the current centered, screen-anchored window near **1720 × 1100 physical pixels**, constrained by the safe area.

~~~text
┌──────────────────────────────────────────────────────────────────────────┐
│ CHOOSE PLAYER · Eleanor Coade · Violet           A−  Larger  A+      ×  │
├──────────────┬───────────────────────────────────────────────────────────┤
│ Human        │ V4                         6 styles + Random              │
│ Local player │                                                           │
│              │ Difficulty  [Easy · 2s] [Mid · 5s] [Hard · 12s ✓]         │
│ V4           │ [Find AI…                                       ] [Clear]│
│ 6 + Random   │ [Random                                                 ]│
│              │                                                           │
│ Gamma        │ [Adaptive Advanced Book] [Coventry Goods] [Three Cotton] │
│ 16 + Random  │ [Worcester Iron-Cotton]   [Zero-Loan]      [Pottery]      │
│              │                                                           │
├──────────────┴───────────────────────────────────────────────────────────┤
│ Selection: V4 · Worcester Iron-Cotton · Hard · 12s      [Cancel] [Apply] │
│ Full description of the focused choice.                                 │
└──────────────────────────────────────────────────────────────────────────┘
~~~

Names are abbreviated in this diagram only; cards retain their actual catalog names.

### Left navigation

The first item, at the top left, is always **Human**, followed by **V4** and **Gamma**. Human uses a simple person icon, V4 and Gamma retain their existing icons. Keep a written label for each; color alone does not identify a role.

Opening the panel selects the player's current page: Human for a human, V4 for the installed AI, and Gamma only for a future supported assignment. Human stays first in navigation regardless of which page opens.

### Human page

Show one comfortable card with a person icon, **Human**, and **“You control this player locally.”** Include a short note that multiple Human slots support local hotseat play. Do not show search, Random, AI descriptions or difficulty here.

Selecting this page prepares Human as the draft choice. The footer says **Selection: Human**. Apply confirms it; merely browsing the page does not change the game configuration. The last committed AI style and difficulty remain remembered for switching back.

### V4 page

Place a **Difficulty** segmented control directly below the page heading and above search and Random:

| Label | Allowance |
| --- | ---: |
| Easy | Up to 2 seconds |
| Mid | Up to 5 seconds |
| Hard | Up to 12 seconds |

These are maximum thinking allowances, not mandatory delays. Keep the current search algorithm, work limits and worker settings. The selected segment uses the existing brass fill, checkmark and clear text.

Retain the six current styles in a **three-column × two-row** grid and the separate Random strip. Difficulty stays visible when the AI grid scrolls. It is not a footer-only control or a separate modal.

### Gamma page

Keep **16 cards in four columns × four rows**, with Random above them. Add the same dedicated Difficulty area below the Gamma heading, in the same position as on V4.

For this revision, Gamma models, Random, difficulty controls and Apply on the Gamma page remain **disabled but readable**. Show **“Gamma is not available yet.”** at page level and an explanation that its difficulty controls become available with the game backend. Do not let disabled controls change the remembered V4 difficulty.

The displayed 2/5/12-second layout is a UI reservation. Actual Gamma difficulty remains subject to the future backend's qualified execution contract; this revision makes no claim that neural strength or timing has been implemented.

### Space and scrolling

Make room for the new difficulty band by simplifying existing header/footer spacing, rather than shrinking card text. A practical 1100px-high starting layout is:

| Section, relative to the window | Proposed vertical placement |
| --- | --- |
| Global title/text controls | 16–86px |
| Page heading | 100–138px |
| Difficulty | 146–192px |
| Search | 204–248px |
| Random strip | 260–328px |
| Card grid | 340–976px |
| Fixed footer | Final 108px |

Four 146px Gamma rows with three 12px gaps require **620px**, fitting the approximately 636px grid at default text size. These are design targets; verify actual font metrics and canvas conversion in Unity. At larger text sizes, grow cards and scroll the grid. Keep navigation, difficulty, search, Random and Apply/Cancel fixed. Long footer descriptions may scroll within their own bounded area.

## 5. Selection behavior

Use one explicit **Apply** action because the panel now edits role, AI choice and difficulty together.

1. Open the player's single summary button. Copy committed settings into a temporary draft.
2. Browse Human/V4/Gamma. Page navigation changes the draft target only; it never writes player metadata or preferences.
3. On V4, choose a card or Random and set difficulty in either order. Card clicks show **Selected** and keep the panel open. Changing difficulty alone is sufficient; reselecting the AI card is unnecessary.
4. Apply validates and commits the active page's complete choice, closes the modal and refreshes the row once.
5. Cancel, ×, Escape or clicking the backdrop discards the entire draft, including difficulty changes.

Additional rules:

- V4 opens with its remembered style and difficulty selected. A slot with no preference uses the existing **Adaptive Advanced Book / Hard** default.
- Store draft AI settings independently from the Human draft. Visiting Human and returning to V4 during the same session preserves draft edits; Cancel still restores the original committed state.
- Browsing Gamma cannot replace a valid Human/V4 draft with a playable Gamma assignment. Its Apply button is disabled, with the reason visible. Returning to V4 restores its draft.
- Difficulty is one per-player setting under the current contract; do not introduce a silent new preference per family. Page controls share that setting when a future family becomes usable.
- Applying Human preserves the last committed AI settings. Unapplied V4 edits made before choosing Human do not replace those remembered settings.
- Reopening shows committed settings. Search and per-page scroll remain temporary UI state.
- Random still resolves uniformly once per bot at successful game creation, independent of search filtering and shuffle RNG. Panel actions never reroll it. The chosen V4 starting profile retains its adaptive intent and fallback.
- Preserve the minimum-one-Human rule. If the lineup has no Human, New Game remains disabled and explains **“Choose at least one Human player.”**

## 6. Implementation approach and code reuse

Extend the existing components; keep one panel implementation and the current close/input facade.

| File / component | Planned change |
| --- | --- |
| `Unity/AiSetupRow.cs` | One summary button for all editable active slots; Human summary; remove row difficulty controls; consistent geometry and focus |
| `Unity/AiSelectionPanel.cs` | Human/V4/Gamma navigation, draft state, page-local difficulty controls, Apply/Cancel, conditional Human content and updated sizing |
| `Unity/LocalMenu.cs` | Active-slot edit validation including Human; atomic role/AI/difficulty apply; refreshed summaries and last-Human explanation |
| Proposed small selection draft type | Separate Human/computer role from `AiChoice`, with validated difficulty and the originating player/setup identity |
| `Policy/AiSelectionCatalog.cs` | Keep V4/Gamma identities and availability; Human does not enter either AI roster |
| `Policy/DifficultyCatalog.cs` | Reuse keys, labels, allowances and validation; no timing change |
| `Unity/StrategyPicker.cs` facade | Retain existing modal close/ownership integration; adapt only if the shared panel contract needs it |
| `BrassPatch/UI/LightUi.cs` | Reuse current opt-in primitives; no global theme change |
| Native menu tests and release tooling | Replace obsolete difficulty-button/instant-commit expectations and add unified-selection coverage |

The first six paths are under `src/BrassLocalAI`. The Unity helpers are compiled into **BrassLegacyAI.dll** despite that source directory name.

Keep `AiFamily` as V4/Gamma. Add a distinct page identifier such as `Human/V4/Gamma` for navigation, plus a role in the draft. Human maps to `AIType.Human`; V4 continues to use the engine's `AIType.Hard` with the separate selected difficulty allowance. Easy/Mid must not re-enable the game's obsolete Easy/Medium AI implementations.

The new apply path must validate the exact setup owner, active `PlayerMetadata` reference, editable state, player count, available choice and difficulty before changing anything. Trace and preserve any required side effects of the stock Human/AI setter. Do not implement role switching by cycling through old choices or by assigning filtered card indexes.

Commit all fields before one presenter refresh; a failed validation leaves the original assignment intact. Preserve preference-write timing and existing saved assignments. No new save schema or Gamma execution path is needed.

## 7. Input, lifecycle and compatibility

- Allow opening from Human rows without weakening Continue/tutorial/network editing restrictions.
- Guard both pointer and keyboard activation of underlying New Game, Continue, Back and other player controls while the panel is open. Keep the existing dismissal-frame guard.
- Tab/Shift+Tab stay within the modal. Arrow keys navigate the left pages, difficulty segments and card grid; Enter/Space selects a draft card or activates the focused Apply/Cancel button. Enter on a card does not apply the draft implicitly.
- Cancel/Apply returns focus to the originating row button when it still exists. Rebinding, hiding/removing a slot, changing the setup or destroying its owner closes the panel and discards pending edits.
- A stale callback cannot apply to another row after a color/name/count change. Settings stay attached to the actual player metadata, not a visible row index.
- Keep nickname/description rendering literal. The existing panel already disables rich text; use that behavior for the new Human and summary labels.
- Continue labels preserve the actual saved family/version and time allowance, including older 10-second assignments. They are not forced into the new-game Easy/Mid/Hard defaults.
- Build from the latest accepted release when implementation begins. Preserve 1.4.23 Game History, shared Statistics fixes, Card Tracker, gameplay repairs, city highlights and offline-only mode. Exclude unrelated research/training changes from the shipped package.

## 8. Appearance

Reuse the installed `LightUi` / `StatisticsTheme` tokens:

| Element | Appearance |
| --- | --- |
| Window / cards | Warm ivory `#F5F1E7` / `#FFFCF5` |
| Secondary / unavailable areas | `#EEE8DB`, with readable text |
| Main / secondary text | `#283238` / `#586269` |
| Selection | Brass `#8A672B` border and `#E9D8B4` fill, plus checkmark/text |
| Keyboard focus | Teal `#286D83`, distinct from selection |
| Corners | Existing 12px window / 7px control treatment |
| Backdrop | Existing 50% black modal shade |

Use the existing game font and default 115% preset, with Large/Larger/Largest controls. Match the established title/card/body hierarchy. Keep the header concise and retain generous spacing between navigation, difficulty and the grid. Use code-native icons consistent with the existing panel.

## 9. Verification before installation

### Functional and data checks

- [x] Every active editable Human/AI row has exactly one player-selection button, including cold/cached menus and the split Continue/New Game layout.
- [x] Old type-cycle and row difficulty controls are hidden and noninteractive from the first visible frame; names and portraits retain their own behavior.
- [x] Human is the first left-navigation item; opening selects the current role/family.
- [x] Human → V4 and V4 → Human apply to the intended slot; remembered committed AI settings survive the switch.
- [x] Card clicks keep the panel open. Changing only difficulty works. Apply commits role/style/difficulty together; all dismissal paths discard the full draft.
- [x] Page switching, search, resizing and text changes do not commit anything or erase the appropriate draft.
- [x] Easy/Mid/Hard create new V4 assignments with **2000/5000/12000ms** allowances while retaining the existing work limits, workers and backend.
- [x] Two/three/four-player setups, all-Human hotseat and mixed Human/V4 lineups behave correctly; an all-AI lineup follows the existing minimum-Human rule.
- [x] Gamma's 16 entries, Random, difficulty controls and Apply remain unavailable, including forced callbacks; the current assignment stays intact.
- [x] Random is not resolved by opening, selecting, applying or cancelling; it resolves once at game creation and does not reroll on Continue.
- [x] Name/color changes, slot removal/readdition and stale callbacks cannot transfer or partially apply another player's settings.
- [x] Keyboard focus, underlying-input blocking, dismissal-frame protection and cleanup pass for all three pages.
- [x] Continue summaries remain read-only and preserve saved AI identity/allowance. Native tests use disposable data and isolated preferences.

### 1200p visual checks

- [x] The single-button rows align for Human and AI, including long names and all four slots in the Continue/New Game split.
- [x] Human, V4 and Gamma are inspected at all three text presets; default is 115%.
- [x] The difficulty band is within each AI page, always visible, with readable labels and a clear selected state.
- [x] All 16 Gamma cards plus Random fit at default size; larger text scrolls without moving Apply/Cancel or clipping the last card.
- [x] Full descriptions, selected/focused states and unavailable explanations remain readable. Footer controls do not overlap the selection summary.
- [x] Existing Statistics/Game History appearance is preserved; there is no unintended global recoloring or font change.

### Release checks

- [x] Pin the then-current accepted binaries/sources and enforce a narrow setup/panel method allowlist; preserve embedded AI profiles and unrelated execution/gameplay code.
- [x] Run relevant native menu, Statistics, Game History and Card Tracker regressions, plus installation/recovery fixtures for the exact candidate.
- [x] Record native captures, test reports, source snapshots and concise release notes in a recoverable release archive.
- [x] On implementation/install authorization, use the combined installer, verify live fingerprints and repeat the menu smoke checks against the installed game.

## 10. Delivery order

1. Pin the accepted release and capture the current first-open/custom/Continue layouts.
2. Add the guarded setup-slot draft and atomic apply path.
3. Replace the row controls with one Human/AI summary button.
4. Extend the existing panel with Human first, page-local difficulty and Apply/Cancel.
5. Refine 1200p spacing, text, navigation and disabled Gamma presentation.
6. Qualify, archive and install when implementation is requested.

Completion means a player can be configured entirely through one row button, with Human first on the left and difficulty inside each AI page, while retaining the accepted light visual style and current gameplay behavior.

## 11. Completed implementation

Installed patch **1.4.24** includes all three pages, a single player-selection control per active editable seat, page-local difficulty, draft Apply/Cancel and unavailable Gamma controls. Native verification passed **646 menu**, **5,613 Statistics**, **3,741 Game History** and **458 Card Tracker** checks. Installer/recovery fixtures passed **86 checks**; the installed menu passed **646 checks** again. Manual image review covered the three pages at all three text presets at **1920 × 1200**, plus the split setup/Continue layout.

The accepted game and helper DLLs are byte-identical to 1.4.23. The narrow runtime comparison preserved **14,686 existing method bodies**, including every existing LocalMenu method, and all **48 embedded AI profiles**. Gamma remains unavailable, and gameplay/search/save behavior is retained. Source snapshots, evidence and screenshots are retained in `artifacts/patch-releases/1.4.24`; see the [release notes](../docs/PLAYER_SELECTION_1.4.24.md).
