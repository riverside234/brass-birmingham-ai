# Game History — latest 10 completed matches

**Date:** 3 October 2026 (Chicago time).  
**Status:** Implemented, installed and verified as **1.4.23**; see [release notes](../docs/GAME_HISTORY_1.4.23.md).  
**Implementation baseline:** accepted **1.4.22**, including the Statistics light theme and V4/Gamma selection panel. Relevant Statistics, offline-menu and shared light-UI sources were pinned to the accepted release fingerprints.  
**Display target:** manual plot inspection at **1920 × 1200**, as requested. Existing 1080p automated layout regressions are retained. Default text preset: **Larger / 115%**.

## 1. Requested behavior

Replace the main menu's **Online Game** caption with **Game History**. The button opens a new local history page containing the **10 most recently completed games**, newest first. Each game appears as one full-width selectable bar in a vertically scrolling list.

Selecting a game opens the **existing Statistics panel**, with its same ten factual pages, plots, legends, filters, exact-data views, text controls and light appearance. Closing returns to the same history list position. The history page's Back control returns to the main menu.

Reports are recorded automatically when a local match finishes, including when the player never opens Statistics. On successful addition of the eleventh completed match, remove the oldest history report. This feature manages statistical reports; it does not delete saved games, event histories or Continue data. Network play remains disabled.

## 2. Verified implementation facts

The reviewed files and installed method calls establish these integration points:

| Existing component | Current behavior | Required change |
| --- | --- | --- |
| `Frameworks.View.Menu.MainMenuView.OnOnlineGamesClicked` | Calls the main-menu controller | Retain the existing button location and replace its caption/meaning before its first visible frame |
| `MainMenuUseCase.ShowOnlineGamesView` | Already redirected to [OfflineMode.ShowOnlineDisabled](../src/BrassPatch/State/OfflineMode.cs) | Route this menu entry to the local history page; keep other network guards |
| `PlayGameUseCase.PlayGame` / finished review | A completed game automatically enters `StartGameReview`, which captures Statistics before replay rewinds presentation | Record the completed report here, independently of clicking Statistics |
| [StatisticsEntryPoints.cs](../src/BrassPatch/Statistics/StatisticsEntryPoints.cs) | Captures a report and binds a viewer session to `FinishedGameReviewUseCase` and Router; disposal destroys that session | Add durable report capture and a history-owned viewer session that does not require a finished-review owner |
| [CompletedMatchStatistics.cs](../src/BrassPatch/Statistics/CompletedMatchStatistics.cs) | Contains values and collections, with no live game, repository or Unity references | Store a complete, versioned snapshot and reconstruct this report for the existing viewer |
| [StatisticsCollector.cs](../src/BrassPatch/Statistics/StatisticsCollector.cs) | Reconstructs detailed reports from detached events; falls back to an explicit summary if history is incomplete | Preserve collection, coverage and warning behavior; do not reconstruct game events every time a history row opens |
| [StatisticsPanel.cs](../src/BrassPatch/Statistics/StatisticsPanel.cs) | Accepts a report; its footer currently offers Results and Review board | Add an opening context so archived reports return to History and cannot access nonexistent board/results sessions |
| [StatisticsTheme.cs](../src/BrassPatch/Statistics/StatisticsTheme.cs), [LightUi.cs](../src/BrassPatch/UI/LightUi.cs) | Provide the accepted light palette and scoped rounded geometry | Reuse these tokens and primitives for the history page |

Before this feature, the report was held in memory without a durable last-ten archive. Existing example JSON exports are test evidence, not a production history store.

There is no declared completion timestamp in the reviewed `GameMetadata` or `GameEnded` members. Event metadata exposes revision plus an arbitrary properties dictionary, not a guaranteed completion-time contract. Use the time of the first completed capture and label it **Recorded**; do not present a file modification time as the actual finish time of an older game.

## 3. Recording and retention rules

| Case | Policy |
| --- | --- |
| Completed local full game | Record it, for two, three or four players |
| Completed local introductory game | Record it with an Introductory label |
| Hotseat / human-versus-AI / all-AI local game | Record the actual player lineup and final result |
| In-progress or abandoned game | Do not add it to completed history |
| Tutorial or network game | Exclude it from this local match history |
| Test or preview match | Use an injected disposable store; never write to personal history |
| Finished game reopened or continued | Update the same record if needed; preserve its original recorded time and ordering |
| Repeated completion/replay callbacks | Produce one entry for the game, rather than one entry per player, callback or revision |
| Detailed event history unavailable | Retain the final result as a clearly marked **Summary only** report; keep unknown metrics unavailable |

Use the stable **GameId** as the logical match identity. A terminal revision and report-format version describe the snapshot; they are not extra games. An equal or weaker recapture must not replace an already complete report with a partial summary. A validated later capture can upgrade a summary without changing its position.

Assign a durable increasing **completion sequence** at first successful completed capture. Sort by that sequence descending, not by last view time, filesystem timestamps, current clock values or file enumeration order. This also resolves identical timestamps and a clock moving backwards.

Keep minimal deduplication bookkeeping for previously recorded GameIds and their original sequence. It contains no player names, scores, plots or event data. This prevents opening an already-evicted old game from inserting it again as the newest match. Once outside the retained ten, its statistical snapshot is deleted and is not recreated by ordinary review callbacks.

History starts recording with the implementation patch. Do not scan or replay every personal save during installation/startup to invent a previous ten-game archive. An older completed save encountered through ordinary Continue may be captured if it has not previously been recorded; label its first observation **Recorded**, without claiming a known historic finish date. Games already deleted or overwritten cannot be recovered from the current in-memory Statistics feature.

## 4. Durable history storage

Create a dedicated directory beneath `Application.persistentDataPath`, provisionally:

~~~text
GameHistoryStatistics/
  history.json              # atomic bank: format, sequence, minimal dedup data,
                            # and at most 10 complete report envelopes
  history.pending.json      # flushed replacement; recovered or removed on reopen
~~~

For the current installation this is beneath `C:/Users/BobXu/AppData/LocalLow/Cublo/Brass Birmingham`. Keep it separate from save repositories and patch archives. Tests must inject another root; production history must not depend on the Steam installation being writable.

### Snapshot contract

Use an explicit, versioned envelope and codec with `TypeNameHandling.None`. Include:

- Report schema version, producer patch version, collector version, GameId, terminal revision, original recorded UTC time and completion sequence.
- Player identities/colors, displayed names and avatars, opponent descriptions, mode/player count, final ranks, VP, money and income.
- The entire report: players, assets, links, actions, entries, samples, rounds, cities, connections, boundaries, mat tiles, link contributions, locations, stocks, warnings, history coverage and availability text.
- Action notes and individual card lists, each round's player order and **per-player spending**, location neighbors, evidence keys, acquisition/flip/removal positions and every field used by current chart/ledger queries.

Do not assume `JsonConvert.SerializeObject(report)` followed by default deserialization preserves this model. Its setters are internal, collections are backed by internal readonly lists, action details/cards are computed from private lists, and round spending is exposed through `Spent(player)` rather than a serializable public property. A naive roundtrip can silently lose data while final scores still look correct.

Implement explicit snapshot DTOs/collection mapping and a checked reconstruction into `CompletedMatchStatistics`. Preserve signed numbers, null/unknown attribution and empty collections. Do not serialize Unity objects, domain events, AI runtime objects or CLR type names. Do not persist the viewer's filters or temporary plot selection as match data.

### Atomic publication and pruning

1. Collect the finalized report before presentation replay; validate completion and identity.
2. Serialize the incoming report and create a candidate bank containing the latest ten snapshots plus minimal deduplication metadata.
3. Write the complete bank to `history.pending.json`, flush it, then atomically replace `history.json`. The incoming admission and old-report eviction commit together.
4. The replacement contains no evicted snapshots. Remove the transient pending file after publication; keep no permanent statistics backup of older games.
5. On reopen, validate a pending bank's sequence and incoming/changed reports before publishing it. Preserve unchanged unavailable older rows. Invalid/truncated pending data is discarded; I/O failures retain the recoverable pending transaction for retry.

**Implementation refinement:** a single atomic bank replaces the proposed two-file index/record transaction. This makes retention and publication one operation and removes the risk of orphan reports after a failed prune. The small latest-ten bank is read on a background task; there is no save-directory scan or game replay.

After a successful transaction and recovery, the canonical bank contains at most **10 statistical snapshots**. A crash or locked file can temporarily leave one pending replacement bank; recovery retries that transaction. There is no accumulating set of orphan report files.

Restrict deletion to the fixed store-owned pending filename under the resolved history directory. Report keys are GameId hashes, never paths. Reject reparse points and paths outside this directory. Never enumerate saved-game folders as pruning targets. Tests verify fixture sentinel files outside the store remain intact.

Reject unsupported schemas or invalid/corrupt reports with an explanatory unavailable row; do not generate zero-filled statistics. Validate identities, player colors/counts, terminal revisions, collection bounds, reference/evidence keys and payload integrity. Other valid rows remain usable. A damaged or unsupported bank produces a read-error/Refresh state; preserve its canonical file rather than inventing reports. Log precise I/O diagnostics and show a concise failure message; a history write failure must not prevent game completion or overwrite the existing ten reports.

## 5. Main menu and history page design

### Main menu entry

Change the existing button's displayed caption to exactly **Game History**, preserving its menu position and hit area. Locate it by its actual persistent `OnOnlineGamesClicked` binding, not by searching for the translated words “Online Game.” Give this entry the light Statistics entry-button treatment and a restrained chart/history icon.

Bind the text and click destination during main-menu setup, including while cached parents are inactive. Handle the label's localization component deliberately so localization refresh cannot restore the old Online Game caption. Repeated setup must not add duplicate controls/listeners. The first visible frame must show the new caption and open History directly.

The button remains usable when the archive is empty. Do not open the old login, connection, online-game list or disabled-online notice from this button.

### Page layout at 1200p

Use the same warm ivory window, dark text, brass borders, teal focus, 12px window/7px control corners and three text presets as the existing Statistics panel. Reference the [accepted Statistics release](../docs/STATISTICS_LIGHT_THEME_1.4.21.md). The new page can use the neutral `LightUi` primitives with explicit Window/Card roles; it must not depend on Statistics widget object-name heuristics or change other menus' themes.

Target a centered window approximately **94–96%** of the safe area. Keep its header and footer fixed and scroll only the game list.

~~~text
┌────────────────────────────────────────────────────────────────────────────┐
│ GAME HISTORY                         A−   Larger   A+                 ×    │
│ Latest 10 completed games · 8 recorded                                     │
├────────────────────────────────────────────────────────────────────────────┤
│ Recorded 3 Oct, 18:42 · Full game · 4 players                 Open ›       │
│ 1st Eleanor · Blue · 146 VP   2nd Isambard · Red · 131 VP                   │
│ 3rd James · Yellow · 117 VP   4th Sir Henry · Purple · 109 VP               │
├────────────────────────────────────────────────────────────────────────────┤
│ Recorded 2 Oct, 21:18 · Introductory · 3 players             Summary only  │
│ Player names, colors, final ranks and VP                       Open ›      │
├────────────────────────────────────────────────────────────────────────────┤
│ More match bars…                                                          │
│                                                         vertical scroll  │
├────────────────────────────────────────────────────────────────────────────┤
│ Back to main menu                 Older statistics are removed after 10.  │
└────────────────────────────────────────────────────────────────────────────┘
~~~

Each bar is one large button. Give it a 2.2px selected/focus treatment consistent with Statistics, a visible hover state and a minimum **140–160px** row height at default size. Increase height based on measured names/wrapping at larger text sizes. Scrolling through ten readable rows is preferable to compressing them all onto one screen.

Use a compact first line for recorded local date/time, Full/Introductory, player count and coverage badge. Show all players in a reserved score area below, with rank, readable player-color swatch/label and explicit `VP`. Distinguish duplicate names by color; do not use color alone to identify a winner. Preserve ties from the report's rank values and do not infer a new winner from VP alone.

Keep scores and Open visible even with long names. Show full names/opponent descriptions on focus or disclosure. Escape user text so rich-text markup cannot alter the row. Do not add tactics advice, strength/weakness grades or charts to the list; those were removed from the current factual Statistics design.

For zero games, show: **“No completed games yet. Finish a local game to record its statistics.”** Keep Back visible. For loading/recovery, show a short local loading state; for unreadable data, explain availability and offer Retry without trapping the user.

Support mouse wheel, scrollbar, pointer drag, Tab/arrows, Page Up/Down, Home/End, Enter and Escape. Bring keyboard-focused rows into view. Preserve scroll offset and focus when returning from a report, resizing or changing text size. At 1080p/Largest, keep the same readable layout with fewer visible rows rather than smaller fonts.

## 6. Reuse the existing Statistics viewer

Add a small **opening context** to the viewer/session: `LiveCompletedGame` or `ArchivedHistory`. Preserve the current initialization signature as a compatibility overload if needed.

| Behavior | Live completed game | Opened from history |
| --- | --- | --- |
| Report | Current captured report | Deserialized stored report |
| Initial view | Overview, all players/whole game | Same defaults when selecting a different game |
| Factual pages/plots/filters/legends/exact data | Existing behavior | Exactly the same code and behavior |
| Default text | 115% | 115% |
| Footer | Results / Review board | **Back to history**; no live-game controls |
| Close / Escape | Existing live return behavior | Return to the history list at the previous row/scroll position |
| Game/repository dependencies | Existing session | None; no game load, RNG, event replay or AI execution |

Retain these ten pages: **Overview; Scoring; Industries; Networks & Geography; Economy; Resources & Trade; Actions & Cards; Turn Order & Tempo; Efficiency & Transition; Timeline**.

Add a compact recorded-date/mode context without crowding the current Statistics header. Do not fork the plots into a second history-specific implementation or recalculate archived ranks. Missing-history coverage notes must remain visible, and a summary-only report must retain the existing unavailable-metric behavior.

A history session must be independent of the `FinishedGameReviewUseCase` weak-table entry. Its close callback must never call `ShowGameSummaryTable` or return to a board that has been disposed. Keep one viewer active per router; selecting another game must replace/reset report-specific filter, legend, evidence, table and tooltip state.

Share viewer creation/input/session cleanup where practical, while preserving existing live-review callers. Register/hide/remove custom views correctly in the Router's stack, cache, animation and key registries. Closing/destroying the page removes detached UI and subscriptions. Guard asynchronous report loads with a session/generation identity so an earlier row load cannot overwrite a later selection or reopen a closed page.

Keep underlying menus disabled while the history page or viewer is active, including nested controls that ignore parent CanvasGroups. Guard the dismissal frame against click/held-key leakage. Returning from History must not invoke Quit, Start, Continue or the former network action.

## 7. Implementation sequence and files

| Step | Work | Main files / proposed additions |
| --- | --- | --- |
| 1 | Recheck the installed accepted release; pin baseline and native menu/Statistics captures | Release archive and installation receipt |
| 2 | Add explicit snapshot contract, roundtrip mapper, terminal validation and stable completion identity | New `src/BrassPatch/History/GameHistoryRecord.cs` and `GameHistoryStatisticsCodec.cs` |
| 3 | Implement a separate atomic report bank, dedup metadata, exact-ten retention and recovery | New `src/BrassPatch/History/GameHistoryStore.cs` |
| 4 | Persist completion reports automatically before replay, with duplicate/summary-upgrade guards | `StatisticsEntryPoints.cs`; proposed `HistoryRecording.cs` |
| 5 | Rename and bind the main menu entry before first display; route its existing patched action locally | `State/OfflineMode.cs`; proposed `HistoryEntryPoints.cs`; a narrow main-menu setup hook/manifest |
| 6 | Build the light history page, large scrollable bars, accessibility and list return state | Proposed `GameHistoryPanel.cs`; existing `UI/LightUi.cs` |
| 7 | Add the archived opening context and shared viewer lifecycle without changing chart calculations | `StatisticsPanel.cs`, `StatisticsEntryPoints.cs`; proposed viewer-session/context helper |
| 8 | Qualify data, retention, actual native input/rendering and existing features | New history tests; existing Statistics/menu/Card Tracker tests |
| 9 | Archive, install recoverably and verify installed fingerprints/first-open behavior | A scoped build/qualifier, existing combined installer and release tools |

Build from accepted source snapshots and add a source/method allowlist for this feature. Carry forward the accepted **BrassLegacyAI.dll** byte for byte where possible. Any necessary game-assembly edits must be confined to verified menu/view hooks; do not rebuild unrelated current AI research into the game. Preserve V4 search/difficulty/workers, unavailable Gamma choices, gameplay repairs, Card Tracker, city highlights and offline-only mode.

The implementation rechecked and pinned accepted 1.4.22. The subsequent request authorizes implementation and installation as 1.4.23; the final status and verification are recorded in the release notes.

## 8. High-priority failure cases to prevent

These are implementation risks to test, not newly reproduced defects in installed 1.4.22.

| Priority | Failure | Prevention |
| --- | --- | --- |
| P1 | Pruning deletes a game save or a file outside the history directory | Dedicated storage, derived filenames, resolved-path checks, protected-file fixtures |
| P1 | Incoming write failure evicts a valid old report | Validate and atomically replace one bank containing the admission and eviction; failure/recovery tests |
| P1 | Reopened games or hotseat replay callbacks consume multiple history slots | GameId identity, original sequence, persistent dedup and idempotent publication |
| P1 | An evicted game returns as newest when reviewed again | Retain only minimal identity/sequence bookkeeping after deleting its statistics |
| P1 | JSON roundtrip loses action/card evidence or round spending, producing wrong charts | Explicit codec and independent full-field/derived-metric roundtrip comparisons |
| P1 | Capture occurs after replay rewinds the state, storing incomplete final scoring | Capture from terminal state before replay; include persisted and uncommitted events through the existing collector |
| P1 | History viewer calls a disposed game/results owner or mutates the current game | Distinct archived context, data-only report, guarded return/session/input lifecycle |
| P2 | First-frame caption is Online Game again, or localization restores it | Initialize inactive/cached menu bindings; native first-open and localization-refresh checks |
| P2 | A corrupt/unsupported record breaks the entire history page | Per-record validation/unavailable state, Retry and intact valid rows |
| P2 | Slow report loads or stale callbacks open the wrong game | Bounded latest-ten bank, background I/O, generation checks and selected-key validation |

## 9. Verification and acceptance criteria

### Storage and data checks

- [x] Insert 0, 1, 10, 11 and 25 completed fixture games; the visible order is correct and only the latest ten snapshots remain after publication/restart.
- [x] Repeated captures, per-player review callbacks, reopening and continuing produce one record; its original sequence/date are unchanged. Reopening an evicted game does not reinsert it.
- [x] Same timestamps and a reversed system clock do not change completion order; storage uses the bank sequence, not directory enumeration.
- [x] Roundtrip actual full/introductory two/three/four-player reports, including all collection/scalar fields, action card/notes lists, round spending and typed evidence associations.
- [x] Compare all ten pages' ledger values, chart models, efficiency denominators, wildcard episodes, resource attribution and link contributions before and after storage. Final-VP agreement alone is insufficient.
- [x] Preserve ties, signed values, zero denominators, partial reports, missing supplier/destination data, long/duplicate names and older saved AI descriptions.
- [x] A full report survives a weaker recapture; a summary upgrade retains identity/order. Unknown versions and corrupt/truncated/mismatched payloads never render fabricated zero metrics.
- [x] Inject faults before writing, after flush, before publication and after publication; verify complete recovery and preservation of the previous bank before publication. Exercise a locked canonical file and recovery after the lock is released.
- [x] Malformed IDs/paths and deletion attempts cannot affect fixture saves or files outside the resolved store. Personal saves/history are never test inputs or targets.
- [x] Restart preserves reports even when their original game save is absent. No repository or game replay is required to open an archived report.

### Native UI and integration checks

- [x] The first visible main-menu frame says **Game History** and its click opens the new page directly; cached reopen/localization refresh does not restore Online Game or add duplicate listeners.
- [x] Empty, unsupported-report and write-failure states remain usable; a ten-game list scrolls by wheel/drag/keyboard, and the tenth bar and reserved score/Open controls fit at all presets.
- [x] Capture actual Unity layouts at both existing regression resolutions and all three presets; manually inspect charts at **1200p** only, with the accepted Statistics light theme and 115% default.
- [x] Selecting each row opens the existing Statistics viewer with that game's complete data, not another row's data or the latest game by mistake.
- [x] All ten Statistics pages, plot selection, legends, player/era/detail filters, exact data, scroll controls and text presets work from a stored report.
- [x] Back/Close/Escape restores the history row/list position and main-menu input. Row changes reset filters, layout rebuilding preserves list state, and stale callbacks cannot reopen a disposed history session.
- [x] Underlying menus/game controls cannot receive mouse/keyboard input through History or Statistics, including on the closing frame.
- [x] Complete a match and exit without opening Statistics; reopen the main menu and confirm it is recorded once. Verify final live scores against its archived Overview.
- [x] Existing live end-game Statistics, AI menu/Gamma availability, Card Tracker, presentation and link replay pass their relevant native suites. Continue/save methods remain unchanged in the binary comparison.

### Release checks

- [x] Compare changed source/methods against the explicit allowlist; preserve accepted AI profiles/resources and all unrelated gameplay methods.
- [x] Qualify the exact combined candidate, including storage failure/recovery tests and disposable native fixtures. Archive screenshots, source, binaries, report-format contract and release notes.
- [x] On implementation/install authorization, use the recoverable combined installer, verify installed fingerprints, native first-open/history/statistics behavior and the preceding-release backup. Retain offline-only status.

## 10. Completion definition

The feature is complete when **Game History** opens the styled scrollable last-ten list; completion automatically records durable, faithful reports; older statistical snapshots are removed safely; and every retained game opens the existing interactive Statistics panel with a reliable return to History. Native regression qualification, 1200p plot inspection, persistence/recovery checks and installed-package verification must pass before the implementation is reported complete.
