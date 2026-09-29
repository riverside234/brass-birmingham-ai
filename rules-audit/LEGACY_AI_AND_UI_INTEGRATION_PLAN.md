# Six-strategy AI integration and city-outline refinement

**Status: Implemented; follow-up patch 1.3.2 installed and verified.**
**Date:** 28 September 2026.  
**Starting point:** Installed local-only rules/UI patch **1.2.0** and the R3 legacy AI source.  
**Scope:** The original planning request was followed by authorization to implement and install both changes.

## Implementation record

The [1.3.2 follow-up](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/LOCAL_AI_1.3.2_RELEASE.md>) clarifies Random's adaptive intent and fallback, restores human city-click building after AI turns, guards historical replay input and fixes final introductory AI settlement evaluation. It passed 252 fresh complete simulations, six live in-memory games, 36 native turns and installed-file checks. Dedicated save-game work was excluded; unchanged rules/helper evidence is retained with explicit scope.

The [1.3.1 follow-up](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/LOCAL_AI_1.3.1_RELEASE.md>) adds Random, guards live AI turns from human input and replay skipping, removes Play now, grants a ten-second/1,024-node search allowance and strengthens rounded city outlines with a halo and fill. The six deployed S1 v2 profiles were verified against the H5 bank. The following original implementation record and design describe 1.3.0 where superseded by these changes.

The design below is implemented. See [release notes and verification](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/LOCAL_AI_1.3_RELEASE.md>) and [deployment source notes](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLocalAI/README.md>) for the final architecture and test limits.

- AI-01–AI-07 and UI-01–UI-02 are addressed: six per-player strategies; two/three/four players and introductory games; immutable saved profiles; cancellation and single action confirmation; native thinking/Play now controls; rounded translucent ivory city locators.
- The research Host/AI source and frozen bank remain separate. Deployment uses a new fork, observation/runtime identity, and embedded profiles. Mutable intent is stored inside the same metadata extension as assignments, so the existing events/metadata transaction publishes them together.
- New-game creation retains previous saves. Incomplete creations are ignored by Continue. Corrupt or incompatible saves are preserved with a visible error.
- Native tests exposed and fixed the resume row's missing nickname callback. Full mixed-game reload tests also exposed and fixed the brewery liquidation callback serialization defect.
- Completed qualification includes 252 strategy simulation games, 6 full mixed games with 231 reloads, 36 native strategy turns, every legal player-color permutation, the existing managed rules suites, all ten Unity suites, and 18 rendered city comparisons. Exact candidate hashes and the distinction between unchanged simulation code and final native binaries are recorded in `artifacts/patch-build/local-ai-qualification.json`.
- Automated fixtures and scripted human seats are distinguished from physical mouse playthroughs and untested display/DPI settings. No user save was opened, migrated, or deleted during testing.

The remaining sections preserve the original design rationale. Their baseline statements describe the installation inspected before implementation.

## 1. Intended result

The local custom-game screen will let each computer player use **Balanced, Rail Tempo, Cotton, Goods/Beer, Supplier or Pottery Flex**. Remove Easy, Medium and Hard from the new-game choices. Show the selected strategy when continuing a game, and keep it attached to that player through color changes, saving, loading and application restart.

City-card previews will retain the current green build highlights. The additional locator outlines around the other spaces will become **rounded, translucent warm ivory**. This is a refinement of the existing white locator, including occupied spaces; it does not change build eligibility.

Preserve the installed rules fixes, save recovery, asynchronous human liquidation, card interactions and local-only configuration. Preserve the current player-count and introductory-game controls; their AI compatibility needs explicit work below because R3 currently implements four-player games.

## 2. What was verified for this plan

The installed assembly hashes match the 1.2.0 installation record:

| Installed file | SHA-256 |
| --- | --- |
| `BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll` | `89EAC250DDA83EEB4285BA7D6486D32CB147B7E2EF226A256286918E27580D0A` |
| `BrassBirminghamUnity_Data/Managed/BrassPatch.dll` | `7FC0EE609F45835BC2694B918536E6A7EBFA2823E8DAC41FBA0C783354BF2C41` |

Neither `BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll` nor `configs/legacy-ai/live.json` is currently present. The new AI source exists, but the optional bridge is not installed or enabled in this game.

The R3 guide and exit record describe a separate four-player runtime based on patch 1.1.0. Their statements about the installed game being unchanged are historical; the current installation is 1.2.0. Its 88 method hooks and six merchant-bonus rewrites must be retained in the combined release.

Source inspection established these integration risks. They are implementation blockers or failure paths, not claims that each has already occurred in normal play.

| Priority / ID | Evidence in current code | Required response |
| --- | --- | --- |
| P1 / AI-01 | `build-legacy-overlay.ps1` pins 1.1.0 dependencies; `restore-legacy-overlay.ps1` restores 1.1.0. | Build against the current combined patch and use release-aware installation/recovery. Never bypass the fingerprint check or install the old overlay over 1.2.0. |
| P1 / AI-02 | `UnityBridge.TryStart` requires four players and `AIType.Hard`; otherwise it returns to the stock AI. | Route explicit per-game strategy assignments. A selected new strategy must either run that strategy or show a recoverable error, never silently run a stock bot. |
| P1 / AI-03 | `CheckSidecar` recreates expected assignments from the current global `live.json` array, indexed by player color. | Make each game's saved assignments authoritative. Menu row, sitting order, color and current turn order must not be treated as interchangeable indices. |
| P1 / AI-04 | Global JSON parsing and some reflection happen outside the bridge's guarded startup. Assignment sidecars use direct `WriteAllText`; only the intent journal uses atomic replacement. | Validate before starting a game; contain startup failures and publish complete, recoverable profile state. |
| P1 / AI-05 | `CreateOfflineGameUseCase.StartGame(int)` deletes existing games before `CanStartGame` and creation. `UpdateResumeOfflineGameView` calls `DeleteAll` after listing/loading exceptions. | Fix those local menu paths before adding profile validation. Failed AI setup or resume must preserve existing saves and recovery records. |
| P1 / AI-06 | Host creation, contracts, beliefs, evaluation and several arrays assume four players. Observations do not carry the introductory-game setting. | Port and qualify the supported configurations; merely removing the bridge's four-player guard is insufficient. |
| P1 / AI-07 | `PlayGamePresenter.StartAITurn` has a delayed asynchronous continuation; the bridge commits live actions and writes intent afterward. | Test cancellation before startup as well as during search; reject stale results and make action/intent/save ordering explicit. |
| P2 / UI-01 | New-game callbacks cycle Easy/Medium/Hard; the presenter derives text from that enum and the row colors it green/orange/red. | Replace both selection and presentation, including saved defaults and resume rows. Renaming the labels alone would not change the AI. |
| P2 / UI-02 | `CityLocatorFrame` draws four opaque white rectangular strips. | Replace its mesh and color only; preserve its placement, input behavior and lifecycle. |

## 3. Strategy catalog and custom-game UI

### 3.1 The six selectable strategies

Use the six **S1 v2** profiles as the initial four-player behavior baseline:

| UI label | Existing profile file | Existing profile ID | Short explanation |
| --- | --- | --- | --- |
| Balanced | `configs/legacy-ai/balanced-v2.json` | `legacy-hard-balanced-v2` | Adapts to the best available opportunities. |
| Rail Tempo | `configs/legacy-ai/rail-tempo-v2.json` | `legacy-hard-rail-tempo-v2` | Favors useful rail links and supporting industries. |
| Cotton | `configs/legacy-ai/cotton-v2.json` | `legacy-hard-cotton-v2` | Develops cotton investments and prepares profitable sales. |
| Goods/Beer | `configs/legacy-ai/goods-beer-v2.json` | `legacy-hard-goods-beer-v2` | Combines manufactured goods with brewery support. |
| Supplier | `configs/legacy-ai/supplier-v2.json` | `legacy-hard-supplier-v2` | Builds coal and iron when supply and demand justify them. |
| Pottery Flex | `configs/legacy-ai/pottery-flex-v2.json` | `legacy-hard-pottery-flex-v2` | Pursues feasible pottery opportunities and adapts when needed. |

All six currently specify a 3,000 ms allowance for the whole turn and a 240-node work limit. This is a soft search allowance, not a guaranteed three-second response. Preserve a complete legal incumbent and the existing main-thread commit boundary.

These are strategies, not six difficulty levels. Do not expose S0/S1/S2, profile filenames or bank IDs in the normal selector. The separate 10-second S2 variants have not established an equal-time advantage and should not become the default. Pottery Flex's held-out status in training does not exclude it from this local game UI.

Create a versioned catalog containing stable strategy keys, display strings, descriptions, supported configurations and exact profile fingerprints. Package validated profiles with the release. Do not discover UI choices by enumerating every JSON file in `configs/legacy-ai`, which also contains experimental variants.

### 3.2 Player rows

- Reuse the current previous/next buttons to cycle the six strategies in the table's order, wrapping in both directions. Example: `AI   ‹  Rail Tempo  ›`.
- Keep the Human row and the existing requirement for at least one human. Tutorial and diagnostic all-AI creation paths retain their own behavior.
- Use one restrained, warm neutral text color for every strategy, consistent with the game's existing typography. Remove the difficulty color scale. Keep full text visible; do not use color alone to communicate selection.
- Show the short explanation in the existing tooltip style on hover/focus. Verify `Goods/Beer` and `Pottery Flex` fit without truncation at supported window sizes.
- Default new AI slots to Balanced. Remember the selected strategy for each setup row in versioned preferences. Removing and re-adding a slot may restore that row's valid choice; a genuinely new slot starts Balanced.
- For existing saved defaults containing Easy/Medium/Hard, map computer slots to Balanced **for the next new game only**. Preserve human slots, names, colors, avatars and player count. Do not reinterpret an existing saved match this way.
- Changing a portrait, name or color must not change the row's strategy. Two bots may use the same strategy, with independent search and intent state.
- Continue-game rows show saved strategy names and are read-only. Old games without a new-AI assignment show `Original AI`; retain their stored vendor behavior. Remove the old difficulty names from the custom screen without removing their serialized enum values or implementations.

The relevant entry points are `CreateOfflineGameUseCase.CreateNewGameConfiguration`, `PresentGameConfiguration`, `NextAiDifficulty`, `PreviousAiDifficulty`, `IncreaseNumberOfPlayers`, `DecreaseNumberOfPlayers` and `StartGame(int)`, plus `CreateOfflineGamePresenter` and `PlayerCreateGameView.SetPlayer`.

Patch the runtime through helpers/weaving. The files under `ai-research/decompiled` are references, not the source tree to edit and compile as the game.

## 4. Per-game strategy identity and save compatibility

### 4.1 Separate setup choices from saved assignments

Keep a setup context keyed to the current creation use case and player row. Use stable row/sitting-order identity while editing, with reference identity where appropriate; do not rely on mutable `PlayerMetadata.Equals` or nickname. At creation, resolve every active row to its final player color and sitting order. Persist that explicit mapping under the newly assigned game ID.

Keep vendor `AIType` values intact. New strategy-controlled bots can retain `AIType.Hard` and the stock behavior object as compatibility markers, while an explicit saved assignment selects the new turn executor. `AIType.Hard` alone must never opt an old save into the new AI.

Recommended storage design:

1. Add a plain JSON extension, such as `brasspatch_ai`, to local `metadata.storage` through the existing persistence helper. Include a schema version, game ID, rules/observation contract identity, runtime identity, player count, introductory setting and the complete immutable player-to-profile assignments.
2. Store the selected profile parameters and their verified fingerprints with that manifest, so editing a global profile file or changing next-game preferences cannot alter a running match. Record source-profile provenance separately from the new deployment identity.
3. Parse this extension as validated JSON data. Keep vendor `GameMetadata`, `PlayerMetadata` and behavior type names compatible; do not serialize custom executor objects or enable type-based deserialization for the extension.
4. Preserve the extension in **all** metadata write paths: initial save, full save, metadata-only save, replay's `SavePlayerMetadata`, backup recovery and any clone/resume operation. The current clone/serialize path would otherwise drop an unknown JSON field.
5. Publish the metadata extension with the existing events/metadata recovery journal. Keep mutable per-seat intent in a versioned journal associated with the same game and saved manifest. Put live state beside the game's persistent save, not only under the installation's `artifacts` directory.

This makes a new-AI game distinguishable from an old Hard game even if an intent file is missing. Absence or corruption of required new-AI state must not silently turn it into an original-AI game. Profile/contract mismatch shows a readable resume error and preserves the files.

The metadata extension is a planned schema addition, not yet proven compatible. Add serialization, cloning, asynchronous metadata-update and old-save fixtures before adopting it. Use the existing persistence lock/recovery mechanism; avoid a second unsynchronized writer to `metadata.storage`.

### 4.2 Create, load and undo safely

- Validate the selected profiles, runtime compatibility, player mapping and writable destination before an overwrite can remove the current game. Keep the existing overwrite prompt; make the replacement fully recoverable before retiring the previous save. On failure, leave Continue pointing to the previous valid game.
- Replace both local resume `DeleteAll` catch paths with a non-destructive error state. Do not report an unreadable game as an empty slot that can be casually overwritten.
- Load assignments from that game's metadata before displaying resume labels or scheduling an AI turn. The optional global `live.json` can be a development override, but must not select strategies for production saved games.
- Do not automatically migrate the old experimental sidecar format or every stock Hard save. A separate explicit importer, if ever needed, must verify the original game ID, profiles and runtime before producing a new manifest.
- Scope intents to game, player identity, profile fingerprint and contract. Retain the existing per-seat separation and revision cutoff. Add an event-prefix/branch identity so an undo and new branch cannot reuse intent from a different history at the same revision.
- Commit only the intent belonging to an applied complete action. After a crash/reload, ignore unpersisted future records; after a branch, discard abandoned future records before another decision. No queued second action survives reload or undo.
- If intent persistence fails after an action has committed, stop visibly and preserve recovery data. Do not retry that action, invoke stock AI or claim that `CancelCurrentAction` rolled it back.

## 5. Runtime integration

### 5.1 Build on the installed patch family

Produce one combined release, provisionally **1.3.0**, from the verified stock baseline plus all existing rules/UI helpers. Then compile and weave the updated AI integration against that staged result. Never use an arbitrary installed or previously woven assembly as the unverified starting point.

Keep the assembly dependency one-way: the AI module already uses `BrassPatch.ResourcesFixes`. Put shared JSON save-extension APIs/data in `BrassPatch`; keep strategy execution and its Unity adapter in `BrassLegacyAI`. Do not introduce a reverse assembly reference from `BrassPatch` to `BrassLegacyAI`. UI-specific AI hooks can call the AI assembly through the second weaving stage.

Extend the AI weaver with an explicit list of allowed UI/turn methods, exact signatures and duplicate-weave rejection. Compare the final method changes against that list and verify every existing 1.2.0 hook and bonus rewrite remains present. The existing two-method check is no longer sufficient once custom-game UI is included.

Publish new dependency and deployment manifests instead of editing `runtime-dependencies-1.1.0.json` or overwriting frozen bank `bank-804c85204da65824cb9b`. Keep the research runtime reproducible. Where rules semantics remain identical, document that compatibility separately from binary identity; where observations/configurations change, issue a new contract. Do not globally relabel old training data or its `Wire.Rules` identity.

### 5.2 Turn routing and lifecycle

Replace global Hard-only opt-in with lookup of the active game's explicit assignment. Stock saves without an assignment keep the original execution path. A new-AI game with invalid assignment or unavailable runtime enters an error state before any move.

Keep and strengthen the existing bridge guarantees:

1. Capture allowed observations/legal semantic choices on the game thread. Give the worker detached data only, excluding live Unity objects, opponent hands and the live RNG.
2. Search asynchronously. Retain one whole-turn budget across the first and second actions; do not grant another full allowance after the first action or Play now.
3. Bind each result to game instance, load/undo generation, revision, actor, phase, request ID, profile fingerprint and contract/runtime identity.
4. Revalidate a complete action against the current state immediately before applying it on the game thread. Confirm it exactly once. The stock `PlayAITurn` loop also confirms actions, so it must be bypassed for new-AI turns.
5. Revalidate any planned second action on a later frame. If stale before mutation, discard/replan within the remaining budget. If an unexpected exception occurs after mutation begins, stop and preserve the last safe save; do not apply another executor.
6. Let the existing patched presenter/use case finish the turn, save, resolve debt and advance the round. Preserve human liquidation choices; ordinary AI planning must not take control of a human debtor. The current guarded liquidation policy can remain the common AI debt policy for this integration.
7. Cancel on menu return, view destruction, new game, loading, undo and application shutdown. Guard the presenter's delayed `StartAITurn` continuation too, so leaving during its delay cannot start a new worker afterward.
8. Place parsing, dependency resolution and startup reflection inside guarded initialization. On every error, dispose/cancel the worker and show one readable in-game message; technical details go to a diagnostic log.

Replace the bridge's fixed-position `OnGUI` box/button with controls in the existing AI-turn panel. Show the player and strategy, a thinking indicator and **Play now**, enabled only after a complete incumbent is available. Respect canvas scaling, input focus and cancellation. Avoid exact progress percentages because search completion is not predictable.

## 6. Player counts and introductory games

Treat four-player normal play as the first integration milestone. For the completed custom-game update, retain two-/three-player and introductory options by implementing and testing their compatibility; R3 does not establish that compatibility today.

Required changes extend beyond the Unity guard:

| Area | Required work |
| --- | --- |
| Setup / contracts | Carry player count, active player identities, sitting order and introductory setting into observations and reconstructed worlds. Build the correct board, deck, merchants and starting state. Version the extended contract. |
| Identity mapping | Replace `players[color]` assumptions with explicit lookups. Two-/three-player tables may use non-contiguous colors, including Yellow; the number of players is not the largest color ID plus one. |
| Beliefs / card accounting | Produce public composition data for each supported setup; preserve own hand, public discard knowledge, hidden discard counts and wildcards. Audit each constant separately rather than replacing every `4` with player count. |
| Search / evaluation | Enumerate actual participants for opponent comparisons, turn scheduling, terminal rankings and intent state. Carry the introductory game's terminal/scoring behavior into simulation and public final-score projection. Compare it to the patched engine. |
| Live capture / restore | Reconstruct tile identities from the matching setup, import correct merchants and preserve the same player mapping after save/load. Test both Canal and Rail for full games. |
| Profiles | Keep the existing four-player v2 files unchanged as source baselines. Publish separately identified compatible configuration variants for other setups; the user still sees the same six strategy names. |

Relevant files include `BrassHost/Host.cs`, `Observation.cs`, `Position.cs`, `Model.cs`, `BrassLegacyAI/LiveHost.cs`, `Belief.cs`, `Planner.cs`, `Evaluator.cs`, `Tactics.cs` and the harness/contracts. Audit shared `ReplySearch.cs` assumptions even though S2 remains outside the UI release.

Until a configuration passes qualification, reject starting that combination with a clear explanation **before any save is overwritten**. Preserve the selected setup so it can be corrected. Do not silently change player count, turn the introductory option off, downgrade to old AI, or claim this interim restriction completes the full compatibility work.

## 7. City-outline appearance

The existing rendered day/night fixtures were inspected. The current opaque square edges stand outside the rounded slot artwork and appear particularly stark on the night board. Use this initial style specification:

| Property | Planned value | Purpose |
| --- | --- | --- |
| Color | Warm ivory **`#F2E8D5`** | A softer near-white that suits the illustrated board, without resembling green build permission or a player color. |
| Opacity | **0.55 alpha** / **45% transparent** | Reduce the present 100% opaque border. This affects the locator only, not the tile or its icons. |
| Corners | Radius about **8% of the shorter slot dimension** | Follow the rounded slot frame instead of making a sharp square. |
| Border width | Initially retain **2.5% of the shorter dimension** | Maintain visibility when zoomed out; tune only after rendered comparison. |
| Interior | Fully clear | Keep industries, ownership colors, resources and empty-slot icons unobscured. |
| Effects | Static border; no pulse, heavy glow or sound | Keep green action highlights visually prominent. |

Warm ivory is the proposed replacement for the added white frames. The existing green sprite, glow, alpha, animation and membership logic remain unchanged. Final radius and opacity should be tuned against the shipped board at all three tested zoom levels; an attractive outline must still locate a city with no green spaces. Start with one consistent color/alpha across day and night.

Modify `CityLocatorFrame` in `src/BrassPatch/CityLocator/CityLocatorView.cs`:

- Replace the four edge strips with a hollow rounded-rectangle mesh using matched inner/outer contours. Keep a consistent stroke through corners, correct triangle winding and joined arcs without gaps or overlapping alpha.
- Clamp radius and stroke for small/non-square rectangles; handle zero dimensions without invalid vertices. Use enough arc segments to avoid visible facets at maximum zoom. Check antialiasing in the shipped Unity renderer before deciding whether a narrow alpha fringe is needed.
- Continue sizing against the actual slot `RectTransform`. The green sprite has extra transparent glow padding and is scaled up; copying its full bounds previously caused overlap.
- Keep `MaskableGraphic`, `raycastTarget = false`, transparent center, layout exclusion, slot parenting and lazy reuse. Avoid per-frame mesh rebuilding and new materials for every slot.
- Preserve all current cleanup hooks, cloned-region cleanup, rapid card switching, zoom/pan persistence and active-action precedence. The refinement requires no game-rule, event or save-format change.

Update the existing test that requires `frame.color == Color.white`: assert the new RGB and alpha instead. Add geometry checks for rounded cutouts and a clear center, while keeping the eligibility and raycast assertions. Do not weaken those behavior checks to accommodate the new style.

## 8. Implementation sequence and acceptance gates

| Step | Deliverable | Completion gate |
| --- | --- | --- |
| 1 | Baseline archive, new combined build/dependency manifests and exact hook inventory. | A staged build retains every existing rules/UI hook; installed 1.2.0 remains untouched during development. |
| 2 | Validated six-profile catalog, saved assignment schema and non-destructive create/resume handling. | Profile/setup failures cannot delete the previous game; all metadata write paths retain assignments. |
| 3 | Hardened four-player bridge and native AI-turn controls. | Each selected strategy executes through the intended executor; one-shot commit, cancellation, error recovery and shared turn budget pass. |
| 4 | Six-choice custom-game rows, preference migration and saved-game labels. | Every choice maps to its actual profile; color/avatar changes and repeated profiles preserve independent player assignments. |
| 5 | Two-/three-player and introductory compatibility. | Each supported setup passes reconstruction, legality, lifecycle and terminal-score checks under its new contract/profile identity. |
| 6 | Rounded translucent ivory locator. | Day/night visual checks pass with unchanged green highlights, eligibility, clipping and input behavior. |
| 7 | Combined regression and rendered acceptance. | Required gates below pass against the exact staged release; failures remain failures, not successful game outcomes. |
| 8 | Transactional combined installation and recovery documentation. | All required files match the release manifest after installation; rollback restores a coherent prior release without deleting saves. |

### Automated validation

- **UI mapping:** Cycle both directions through exactly six strategies; verify IDs and descriptions. Cover restored old preferences, a new slot, remove/re-add, color/portrait swaps, repeated strategies, canceled setup, mixed humans/bots and read-only resume rows. No Easy/Medium/Hard option remains in the custom screen.
- **Creation and persistence:** Inject invalid/missing profiles, incompatible runtime, malformed metadata extensions, corrupt intent, unwritable storage and interrupted creation. Prove the old game survives. Test initial save, full save, metadata-only replay update, restart and journal recovery; next-game choices must not change the old match.
- **Identity and branching:** Use all legal color subsets for two/three players and color permutations for four. Exercise undo followed by a different branch at the same revision, reload at an older action boundary and two games with the same strategy assignments.
- **Executor safety:** First Canal turn, two-action turns, Play now before/after an incumbent, zero/expired budget, rapid repeated input, stale revision/actor/request, menu during the presenter's startup delay, menu during search, a committed first action and a stale second action. Count confirmations and turn advancement to detect duplicate execution.
- **Rules and fairness:** Check live/search action parity for Build, Network, Sell, Develop, Loan, Scout and Pass; merchant bonuses, multi-resource payment, debt and era transitions. Repeat hidden-world/RNG invariance checks so presentation integration or smaller-player reconstruction does not expose private state to the worker.
- **Complete games:** For each supported player count, run all six strategies in every participating seat over at least two deterministic setup seeds, plus identical-strategy and mixed-strategy tables. Repeat introductory scenarios separately. Use fixed-work runs for reproducible regression and foreground timing samples for responsiveness; report these as reliability evidence, not a playing-strength study.
- **Existing patch:** Rerun the ten existing Unity suites and relevant managed rules/persistence regressions against the combined assembly, including prior whole-game parity/scoring checks. Preserve tests for human liquidation and offline restrictions.
- **Locator:** Retain active-city/player-count/era and no-green/mixed/all-green coverage, occupied spaces, modal/action transitions, cloned views and pointer routing. Add mesh bounds, rounded corners, clear center, clipping and expected color/alpha checks.
- **Packaging:** Test clean 1.2.0 upgrade, repeat installation, interruption at every publication boundary, missing/tampered assembly or profile, read-only files, uninstall and recovery from a known old overlay. No mixed-version installation may be reported successful.

### Rendered acceptance

Use a disposable game copy and temporary save root first. Existing R3 tests use a mock router/scripted human metadata; they do not prove the real menu, animations or interactive bridge work.

1. Open the shipped custom-game screen, select each of the six strategies and start actual human-versus-AI games. Check full labels/tooltips, arrows, display scaling and resume labels.
2. In normal four-player play, run a human with three differently configured bots through Canal and Rail; cover the remaining strategies in subsequent fixtures. Validate every additional supported count and introductory mode too.
3. Return to menu during startup and search, reload, use Play now repeatedly and verify normal interaction resumes. Observe Scout, resource selection, merchant bonuses, multi-action turns, debt prompts and end-game presentation.
4. Render baseline/locator-only/mixed cities on day and night maps at zoom **0.028, 0.064 and 0.100**, including a zero-buildable city, occupied tiles and densely adjacent slots. Verify the new borders stay readable and fit the native rounded frames.
5. Compare green-highlight pixels/behavior to the baseline, click through the locator and inspect region-detail clones. Confirm no clipping, obstructed icons, rectangular corner remnants or lingering outlines.

The earlier 1.2.0 city report records 46,767 assertions, 18 rendered fixtures and 12 native pointer-raycast checks. The new geometry adds 12 assertions, bringing the actual-board checks to 46,779; the final-build rendered and input checks were repeated. Full current evidence is in the release record linked above.

## 9. Release and installation design

Extend the existing checksummed installer transaction to cover `Assembly-CSharp.dll`, `BrassPatch.dll`, `BrassLegacyAI.dll`, the versioned strategy catalog and all required profile/contract data. Prevalidate the whole package before replacing anything. Journal the operation so a crash can recover to a coherent release.

Keep immutable packaged AI data separate from user preferences and game state. Back up the current 1.2.0 release, installation record and dependency inventory. Verify the game is closed before publication and retain local-only settings. Install and rollback must preserve user saves and strategy/intent records.

Replace the old overlay restore workflow with a release-aware path; restoring the AI overlay must not unexpectedly downgrade the rules/city patch to 1.1.0. Before allowing an older runtime to resume a new-strategy game, require an explicit compatible migration or block that resume. Merely retaining a stock Hard behavior object does not make an older runtime strategy-compatible.

Only mark the later implementation installed after checking the exact installed hashes, loading every helper assembly and passing a startup/selection smoke check from the installed location. Update `docs/LOCAL_PATCH.md` and create a release validation record listing what was actually tested, any unsupported configuration and remaining limitations.

## 10. Source map

| Reference | Relevance |
| --- | --- |
| [R3 implementation guide](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/R3_LEGACY_BANK.md>) | Existing adapter, budgets, live-action boundary and outstanding interactive gates. |
| [Tactical profiles](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/R3_TACTICAL_BANK.md>) and [exit checks](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/docs/R3_EXIT_CHECKS.md>) | Six styles, frozen-bank identities and limits of existing qualification. |
| [Unity bridge](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLegacyAI/Unity/UnityBridge.cs>) and [intent journal](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassLegacyAI/IntentJournal.cs>) | Routing, global assignment dependency, worker lifecycle and persistence. |
| [Custom-game use case](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/BrassApplication.UseCases.Menu.OfflineGame.CreateOfflineGame/CreateOfflineGameUseCase.cs>) | Setup defaults, difficulty cycling, player mutations, destructive error/overwrite paths. |
| [Custom-game presenter](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/InterfaceAdapters.Common.CreateGame/CreateOfflineGamePresenter.cs>) and [player row](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/ai-research/decompiled/PlayerCreateGameView.cs>) | New/resumed game labels, delegates, colors and row layout. |
| [Save helper](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/Persistence/PersistenceFixes.cs>) | Existing lock, serialization and recovery transaction to extend. |
| [AI overlay builder](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/scripts/build-legacy-overlay.ps1>) and [weaver](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/scripts/LegacyWeaver.cs>) | Current 1.1.0 pinning and two-method routing overlay. |
| [City locator renderer](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/BrassPatch/CityLocator/CityLocatorView.cs>) and [existing city plan](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/rules-audit/CITY_CARD_HIGHLIGHT_UI_PLAN.md>) | Target of the visual refinement and behavior that must remain intact. |
| [City regression fixture](</C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs>) | Existing behavior, actual board prefab and render/input checks to extend. |

Implementation follows this design with the refinements recorded at the top. Installation status and exact deployed fingerprints are recorded separately in `artifacts/patch-installation.json` and the release notes.
