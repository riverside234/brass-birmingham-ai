# City-card location highlight improvement

**Status:** Implemented, installed and verified in combined patch **1.2.0**.  
**Date:** 27 September 2026.  
**Baseline:** Upgraded the combined local-only rules patch from 1.1.0; its binaries and validation records are backed up under `artifacts/patch-releases/1.1.0`.

**Completion record:** All 79 earlier method hooks and six merchant-bonus rewrites are retained. Nine UI hooks bring the combined total to 88. The installed game passed all ten Unity suites, including 46,767 city-locator assertions on the shipped board prefab, and the existing managed gates including 90 complete AI games. Upgrade/restore packaging passed 28 checks. The rendered fixture produced 18 day/night images at three zoom levels and passed 12 native pointer-raycast checks. Mixed and all-white results were visually inspected. No user saves were opened or migrated; online play remains disabled.

[Mixed green/white example](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/artifacts/city-locator-player/artifacts/patch-build/day-0.064-mixed.png>) · [All-white night example](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/artifacts/city-locator-player/artifacts/patch-build/night-0.100-white.png>) · [Verification scope](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/tests/BrassPatch/CityLocator/UNITY_TEST_NOTES.md>)

The implementation follows the behavior below. Two details were refined during delivery: the white frame fits the actual slot border because the green sprite has transparent glow padding; all new behavioral and native checks live in one Unity test assembly rather than a separate managed mock suite.

Clicking a city card should make its city easy to locate even when none of its industry spaces currently receives a green box. Keep every existing green box and add a white outline to each remaining industry space in the selected city.

## 1. Required behavior

Use the same card click that currently moves the map. Hovering the card continues to use its existing behavior; this change does not introduce a new hover trigger.

| Space in the clicked city | Result |
| --- | --- |
| Receives a green outline under the current preview logic | Keep the existing green outline exactly; add no white outline. |
| Empty and does not receive a green outline | Add a white outline. |
| Occupied and does not receive a green outline | Add a white outline around the space, with the industry and resource icons still visible. |
| Occupied but eligible for an existing green overbuild preview | Keep green; occupancy alone must not force white. |
| Space belongs to a different city | Add no white outline. |
| Inactive placeholder in the board prefab | Add no outline and do not activate it. |

Consequently, a mixed city shows green and white boxes together; a city with no green spaces shows white boxes around all its active industry spaces. “Other blocks” is interpreted as all remaining industry spaces in that city, including occupied spaces. Merchant spaces, links and the city name are outside this change.

White is a location marker. It does not grant build permission or become a new selection control. Preserve the existing region click, tooltip and card-selection behavior. In particular, clicking a space may still enter the game's existing region/build flow, whose existing validation decides whether construction is possible.

Keep the locator visible after the pointer leaves the card and while the player pans or zooms. Replace it when another city card is clicked. Clear it when gameplay moves into an action, another board presentation replaces it, the turn ends, or the board closes. No timeout, flashing effect or new sound is needed.

## 2. What the code currently does

The cause is in the presentation path, rather than the camera movement:

1. [HandOfCardsUseCase.OnCardSelected](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Game.HandOfCards/HandOfCardsUseCase.cs:47>) forwards a card to an active action's selection delegate first. Only the ordinary city-card preview calls `OnLocationCardClicked`. Preserve this ordering so Build, Scout and discard selection continue to work.
2. [HandOfCardsPresenter.OnLocationCardClicked](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/InterfaceAdapters.Game.HandOfCards/HandOfCardsPresenter.cs:28>) scrolls to the city and builds the board view model. Its line 48 predicate sets `Highlighted` only when the space belongs to the clicked city **and** appears in the supplied buildable-space list. Other spaces therefore have no outline.
3. [BoardView.HighlightBuildingSlots](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BoardView.cs:110>) sends that presentation to the regions and slots. [BuildingSlotView.UpdateSlot](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BuildingSlotView.cs:73>) uses the boolean to switch the outline's alpha between visible and invisible.
4. The original outline is also animated by [StartHighlightResourceAnimation](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BuildingSlotView.cs:178>). Recoloring or reusing that same object for the locator would couple the new display to resource animations.
5. [RegionDetailsView.UpdateRegionDetailsView](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/Frameworks.View.Game/RegionDetailsView.cs:46>) clones the map's region object and calls [RegionView.TurnOffOutlines](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/RegionView.cs:70>). Currently that method only hides the child named `Outline`; a new locator must also be cleared in the copy.

Asset inspection confirmed that `UpdateSlot` already assigns the original image a white RGB tint. The shipped `BuildingSlotOutline` sprite uses the default UI material and contains the green artwork and glow padding. Changing only its tint to white would not produce a white border. The implementation uses a separate white graphic.

The decompiled files above are reference material. Implement through the existing `BrassPatch` helper and patch manifests, rather than recompiling or editing the decompiled game tree.

## 3. Implementation sequence

### UI-01 — Add white outlines after the existing city preview

Add a postfix hook to `HandOfCardsPresenter.OnLocationCardClicked`. Let the original method perform its camera movement, green highlighting and interaction setup first, then apply white locator outlines to the remaining spaces in that city.

The helper should:

1. Resolve the existing board view through the presenter's router and obtain the requested region using `GetRegionViewForGivenRegionName`.
2. Read each active slot view's current `BuildingSlot` binding. Confirm that it belongs to the selected region in the supplied board.
3. Use the exact `buildingSlotsWhereBuildIsPossible` argument and its existing membership semantics. Add white only for spaces absent from that list. Do not run a second affordability calculation, reinterpret overbuilding, or redefine which spaces receive green.
4. Match current bound slot objects rather than identifying them solely by names or saved IDs. This avoids introducing ambiguity for legacy games with duplicate identifiers.
5. Apply the white display immediately after the original method returns, without adding delayed camera callbacks. Rapid city-card clicks must not allow an older callback to restore an obsolete locator.

Do not modify `BuildingSlotViewModel.Highlighted`, `IsInteractive`, action delegates or the rule engine. This design adds no serialized fields and changes no saves or replay events.

### UI-02 — Render a separate, non-interactive frame

Create a reusable child named `BrassPatchCityLocator` inside each slot that needs a white outline. Use the existing outline's normal bounds, inset and visual weight as the reference. Preserve the existing green outline object, sprite, material, alpha, transform and animation ownership.

The new frame should have:

- A transparent center and a white border, positioned around the industry space rather than covering its contents.
- The same board/slot parent so it follows map scrolling, zooming, layout and clipping naturally.
- `raycastTarget = false`, with no button, pointer handler or independent interaction component.
- No layout participation and no changes to existing child names or ordering. Append the new child; existing animations use some fixed sibling indices.
- One reusable instance per visited slot, created lazily and hidden when unused. Repeated clicks must not accumulate children or materials.

The delivered renderer is a procedural `MaskableGraphic` with white edges and a transparent center. It stretches to the actual slot border rectangle. The green image's 1.4 scale includes glow padding, so copying its full bounds would oversize the white box. Using the slot rectangle also avoids borrowing a transient resource-animation transform. No shared green asset is modified.

Only the new white frame may be adjusted for visibility. Check it on the game's day and night maps and against occupied spaces. Keep the frame static and visually distinct from the existing green action highlight.

### UI-03 — Clear the locator at presentation boundaries

Maintain transient locator state per board view, containing only the active city and its locator components. Avoid a process-wide remembered city. Clearing must be safe to call repeatedly and must affect only the new white frames.

| Boundary | Planned integration |
| --- | --- |
| Another city card is clicked | Clear the previous white frames during the normal highlight refresh; the city-preview postfix applies the new set afterward. Clicking the same card refreshes one set without duplication. |
| Board state is refreshed, including undo/replay/era changes | Prefix `BoardView.UpdateBoardView` to clear locator state. |
| Another action highlights spaces, resources, links or merchants | Prefix `BoardView.HighlightBuildingSlots`, `HighlightLinks` and `HighlightMerchantSlots` to clear locator state before the existing presentation runs. |
| An action takes ownership of hand-card clicks | Clear when `HandOfCardsUseCase.SetOnCardClickActionDelegate` receives a non-null delegate, using its presenter's existing router and only an already cached board view. Resetting the delegate to null must not resurrect an old locator. |
| Action entry before a board update or card delegate assignment | Verify each local action's entry path; attach cleanup to its existing action-in-progress presentation boundary wherever the hooks above do not clear before the next frame. Avoid modifying domain action methods for UI cleanup. |
| End turn | Clear at the presentation handoff, before another player's hand or turn screen appears. |
| A region is copied into the city-details panel | Extend `RegionView.TurnOffOutlines` with cleanup that discovers and hides locator children on that region, including newly cloned children that are not in the board's state table. |
| Board is disabled or destroyed | A board-owned lifecycle component hides locators and releases references. Reopening starts without a selected city. |

Cancellation should restore the normal board presentation, with no stale white boxes. Hover tooltips and ordinary pan/zoom must not clear the locator. Keep all cleanup on Unity's main thread and do not create a hidden board just to clear it.

### UI-04 — Integrate with the combined patch

Delivered files:

| File | Responsibility |
| --- | --- |
| `src/BrassPatch/CityLocator/CityLocatorFixes.cs` | City-preview postfix, slot classification and targeted cleanup hooks. |
| `src/BrassPatch/CityLocator/CityLocatorView.cs` | White-frame creation/reuse and board lifecycle cleanup; optional procedural border graphic. |
| `src/BrassPatch/CityLocator/manifest.json` | Exact target signatures and prefix/postfix registrations. |
| `tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs` | Behavioral cases, shipped board prefab, event/raycast checks, rendered captures, frame reuse and cleanup. |
| `tests/BrassPatch/CityLocator/run-unity.ps1` | Isolated player runner and tested-candidate fingerprints. |
| `tests/BrassPatch/CityLocator/verify-release.ps1` | Installed hashes, preservation of earlier hooks, original backup, offline hook and completed validation. |

Register the new tests with the existing validation and isolated Unity self-test runners. Build the combined candidate from the pinned original assembly, retaining every current rules, save, Scout, AI and offline-mode hook. The existing weaver supports the required prefix/postfix hooks; no new view-model fields or general weaver redesign should be necessary.

## 4. Acceptance tests

Use behavioral tests for the new feature and native Unity checks for rendering/input. A managed mock alone cannot prove that the frame is white, correctly aligned or transparent to pointer input.

| Scenario | Required result |
| --- | --- |
| Click a city with no green spaces | Camera follows its existing behavior; every active industry space in that city receives white. |
| Click a city with mixed eligibility | Existing green spaces remain visually identical; every other active space is white; no space has both frames. |
| Click a city where all spaces receive green | No white frames appear. |
| Occupied spaces and legal overbuilding | Ineligible occupied spaces receive white; spaces already eligible for green retain green. Tokens, resource counts and link-VP symbols remain readable. |
| Different city and rapid repeated clicks | Only the latest city's white frames remain; the component count is bounded by visited active slots. |
| Pointer leaves card; map pans or zooms | Locator remains and stays attached to the correct spaces. No new camera behavior occurs. |
| Build, Network, Sell, Develop, Loan, Scout and Pass | Action entry clears white; original legal targets and card-selection delegates behave as before. Clicking a city card while an action owns card input does not start the locator. |
| Industry and wild cards | Existing behavior is unchanged; they do not create a city locator. |
| Click or hover a white-outlined space | Overlay receives no raycasts; existing region clicks/tooltips still work. The white frame never bypasses build validation. |
| Resource-consumption animation after preview | White clears; original green/resource outlines animate without a changed material, transform or color. |
| Open city-details panel | Its cloned region contains no visible locator frames. |
| Cancel, undo, replay, turn/era transition, close/reopen | No stale white outline or retained city state survives the corresponding presentation change. |
| Two-, three- and four-player boards; both eras | Correct active slots are outlined; inactive placeholders and unrelated cities stay untouched. |
| Day/night map and supported UI scales | White frame remains visible, correctly clipped and aligned; green boxes match the baseline. |
| Game-state comparison before/after preview | Cards, money, industries, resources, action counts, revision and events are unchanged. |

Record comparison screenshots of a mixed city and an all-white city using the actual gameplay prefab. Check minimum and maximum supported zoom as well as the normal playing view. The existing headless Unity self-test can verify component state and input routing, but visual acceptance requires rendered gameplay inspection.

## 5. Completion and later installation

Implementation is complete when the acceptance cases pass, the combined patch validation succeeds, and the actual game's green/white appearance has been inspected. Keep the existing release validation requirements, including the prior fixes' regression gates; do not replace them with only the new locator tests.

When implementing and releasing this plan, follow the [existing local patch build/install process](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/LOCAL_PATCH.md:5>): build a new combined candidate, validate its recorded hashes, preserve the current release for rollback, install the matching assembly/helper pair with the game closed, and verify the installed files and Unity self-tests. Keep online play disabled as already requested. This UI change should require no save migration or rules-version change.

The patch guide now records the installed 1.2.0 fingerprints and validation. The implemented hooks are: `HandOfCardsPresenter.OnLocationCardClicked` (postfix), `BoardView.UpdateBoardView`, `HighlightBuildingSlots`, `HighlightLinks`, `HighlightMerchantSlots` (prefixes), `RegionView.TurnOffOutlines` (postfix), `HandOfCardsUseCase.SetOnCardClickActionDelegate`, `ActionInProgressUseCase.SetCancelActionDelegate`, and `PlayerActionsPresenter.EndHumanTurn` (prefixes). A board-owned component handles disable/destruction. No slot initialization hook is needed because the frame follows the slot's stable rectangle.

The rendered examples are artificial fixtures using the actual gameplay prefab and a test router. Native input routing is checked; a complete physical-mouse playthrough and every display/DPI configuration are not claimed.
