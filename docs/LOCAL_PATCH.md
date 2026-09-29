# Local Brass: Birmingham rules and UI patch

**Package 1.4.1 is installed and verified** (28 September 2026, 20:07 America/Chicago). Its [release record](LOCAL_AI_V3_CONVERSION.md) gives the game checks and user-accepted limited research qualification. The six Custom Game strategies gain cash protection, industry-completion deadlines, merchant-route planning and complete beer-source comparisons. Random still chooses a style per bot. Normal thinking remains ten seconds / 1,024 nodes, with two-, three- and four-player full/introductory support. The engine and helper DLLs are byte-identical to package 1.4.0, preserving city highlights and the earlier rules/UI repairs.

The [1.4.0 cash repair](LOCAL_AI_V3_CASH_MANAGEMENT.md), [1.3.2](LOCAL_AI_1.3.2_RELEASE.md), [1.3.1](LOCAL_AI_1.3.1_RELEASE.md) and [1.3.0](LOCAL_AI_1.3_RELEASE.md) notes preserve earlier validation and implementation history. Current checks include generated saves from 1.3.2 and 1.4.0; real user saves are not test fixtures.

## Build and restore

The build uses the installed game's original assemblies, pinned Roslyn 4.12.0, and Mono.Cecil 0.11.6. Downloaded packages are checked against fixed SHA-256 values. It writes a candidate under `artifacts/patch-build`; it never recompiles the decompiled source tree.

```powershell
pwsh -NoProfile -File scripts/build-patch.ps1
# Run the managed, AI, native UI, complete-game and packaging qualification
# scoped sequence documented in LOCAL_AI_V3_CONVERSION.md before installing this build.
```

Close the game before installing or restoring. The installer refuses unknown game versions, changed candidates, and missing or mismatched validation. The untouched original assembly is retained at `artifacts/patch-baseline/Assembly-CSharp.dll`. Installed hashes and version are recorded in `artifacts/patch-installation.json`.

The previous 1.0.0, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.3.2 and 1.4.0 binaries and installation/validation records are retained under `artifacts/patch-releases/`. Current packaging tests exercise upgrading from 1.4.0. Historical save-compatibility tests retained three artificial 1.0.0 saves under `artifacts/patch-upgrade-tests`; those tests were not rerun for 1.3.2. Version 1.3.0 preserved existing event types and identifiers, added a JSON AI metadata extension for new games, and excluded a nonserializable brewery callback. Later releases retain those rules and saved-profile identities. The historical artificial-save checks loaded older games, continued play and reloaded successfully. No real user save was opened or migrated.

To restore the original game:

```powershell
pwsh -NoProfile -File scripts/uninstall-patch.ps1
```

Restoration changes the three package files only. It does not delete saves. **Games created with this patch require the patch to continue**, because they use versioned, collision-free identifiers and may have saved strategy assignments. Restoration is blocked while new-strategy games or pending recovery journals exist. Reinstall a matching combined patch to continue those games. Steam file verification or a game update can replace the patched assembly; rebuild/install only when the original version fingerprint is supported.

Complete any pending save recovery with the patch before restoring stock files. The restore script checks the game's recorded save location for pending journals and stops if it finds one; stock code cannot recover that journal.

Installation and restoration record their intended changes before replacing files. If interrupted, rerun the installation or restore script; known original and patch fingerprints remain recoverable. Save-folder locations persist in the installation record independently of disposable test results. If running these scripts from a different Windows account, pass the playing account's Unity save folder with `-SaveDataPaths`; this installation records Bob's folder. The isolated self-test's reported profile is not a save location because that test skips normal storage startup.

## Implemented changes

| Audit ID | Change | Implementation |
| --- | --- | --- |
| B01 | Final Rail round skips debt preparation and income settlement | `src/BrassPatch/State/StateFixes.cs` |
| B02 | Multi-cube build affordability includes subsequent connected mines | `src/BrassPatch/Resources/ResourcesFixes.cs` |
| B03 | Rail coal sources use the nearest distance across the placed link's endpoints | Resources |
| B04 | Each industry sale can consume beer from only its selected merchant | Resources |
| B05/B06 | Results compare VP, income level, cash; complete ties share ranks and show joint winners | State; weaver introductory comparator |
| B07 | Income −7 permits a loan | State |
| B08 | Vacant single-icon preference does not block a valid occupied-space overbuild | Resources |
| B09 | Humans choose liquidation tiles; settlement saves/resumes, validates ownership, and completes the round once | `src/BrassPatch/Liquidation/LiquidationFixes.cs` |
| B10 | Gloucester with no legal development target completes the bonus without a dead-end dialog | Resources |
| B11 | Merchant VP replay displays the accumulated total | `src/BrassPatch/PresentationFixes.cs` |
| B12 | Merchant bonus flags belong to their game, including replay and undo | State; six original singleton read sites rewritten by the weaver |
| B13 | Introductory score components reconstruct from persisted events | State |
| B14 | Exhausted manufacturer stacks cannot null-dereference Hard AI applicability | State |
| B15 | New games use guaranteed-unique deterministic identifiers without additional random draws; legacy IDs remain unchanged | `src/BrassPatch/IdentityFixes.cs`, `State/IdentityCompatibility.cs` |
| B16 | Era-end replay restores bonus summaries and resets the current-era bonus counter | State |
| R01 | Scout callbacks are bound to their exact game/action/attempt; cancellation and destruction invalidate them; completion confirms only its captured turn flow | `src/BrassPatch/Scout/ScoutFixes.cs` |
| R02 | Scout rejects duplicate, foreign, stale and excess selections; the domain validates the complete exchange before its event; native cards lock immediately | Scout; `src/BrassPatch/Scout/ScoutViewFixes.cs` |
| R03 | The opening brewery AI rule rejects late Canal/Rail rounds and empty brewery stacks before dereferencing a tile | `src/BrassPatch/AI/AiFixes.cs` |
| UI-01–UI-04 | City-card previews preserve green and add white locator frames to remaining active city spaces; frames clear on presentation/action/turn/lifetime changes and in cloned city details | `src/BrassPatch/CityLocator/` |

Scout now owns and cancels its animation sequences, restores hand layout and card positions on cancellation, and resets the view when reopened during its pending hide animation. Stale callbacks cannot hide a new exchange or confirm a replacement action. Invalid action confirmation is rejected before an event revision is allocated. Existing replay handlers remain unchanged.

Old saves retain their identifier algorithm and referenced identities. A legacy flip uses its recorded region. Ambiguous context-free legacy resource lookups are rejected before mutation rather than selecting the wrong industry. The patch does not attempt to invent missing history for an already corrupted save.

## Additional findings addressed

- **Income cap and replay:** Oxford's intermediate marker and income animations now clamp to the printed maximum. The original cap overflow was temporary in normal completed sales; this was not promoted to a new P1 outcome defect.
- **Introductory result order:** The original introductory result comparator displayed poorer ranks first. It now orders results consistently with the normal summary.
- **Failed-save data loss (P1):** Stock-code fault injection truncated an existing event file from 467 bytes to one byte; a metadata-failure retry produced revisions `[1,2,2]`; loading empty metadata deleted its history directory. The repair serializes before replacing files, keeps backups, uses a checksummed recovery journal, and makes retry appends idempotent. Tests use artificial save directories only.

The follow-up review did not find a publisher clarification resolving introductory unflipped-tile scoring or final-Canal income interpretation. Those disputed rules remain unchanged. The empty/shallow cloning helpers are not used as an advertised cloning API; no additional ordinary-play P1 was established there. Pending liquidation, which is now an actual resumable flow, reconstructs from its existing events.

## Validation and limits

The validation script executes targeted production-method tests, the existing resource/network matrices, and 90 complete original-AI games across all difficulties and player counts. It records hashes for the exact candidate. The explicit Unity self-test starts the installed player in batch mode, skips normal framework/storage/network startup, and runs managed and shipped-popup checks inside Unity's Mono runtime. It writes `artifacts/patch-build/unity-selftest-result.json` and its log.

The historical **1.2.0** validation on 27 September 2026 passed the checks below. Current 1.4.1 evidence is linked above.

| Suite | Result |
| --- | --- |
| Identifiers, JSON compatibility, Oxford cap, VP replay | 1,081 assertions across 132 setup seeds |
| State, replay, scoring, original AI boundaries | 315 assertions, 15 replay cutoffs, 5 complete games |
| New brewery AI boundaries | 3,490 assertions; 296 executed public AI decisions; 3 legal exhaustion histories and their event reconstructions |
| Scout lifecycle, selection and persistence | 125 assertions, including real replay/hand binding, cancellation, reentrancy, foreign same-ID cards and filesystem reload |
| Previous-patch save compatibility | 21 assertions across 3 artificial 1.0.0 games; unchanged initial state and successful continued play/reload |
| Targeted resource and bonus fixes | 99 assertions across all player counts |
| Existing network/sell/undo matrix | 1,282 assertions |
| Human liquidation, reload, retry, stale callbacks | 56 assertions |
| File-backed save failure and recovery | 46 assertions |
| Existing build/resource matrix | 8,161 assertions; both crafted Hard-AI crash cases now return without exceptions |
| Complete original-AI games | 90/90 completed; 9,210 actions, 320,914 repeated state assertions, 519 independent era/player scoring comparisons |
| Upgrade/restore packaging | 28 assertions in an isolated installation, including interrupted upgrade recovery and pending-save-journal refusal |
| City locator, actual board prefab | 46,767 repeated assertions over all active cities, 2/3/4 players, both eras, occupied spaces and UI cleanup/input boundaries; zero native errors |
| City locator, rendered verification | 18 day/night captures at minimum/middle/maximum zoom; 12 native pointer-raycast assertions; mixed and all-white displays visually inspected |

The AI batch covers Easy, Medium, and Hard with two, three, and four players, ten seeds per combination. These counts describe executed checks, not independent rule interpretations or exhaustive game coverage. Raw post-patch matrix evidence is under `artifacts/patch-audit/rules-audit`; the original stock evidence under `rules-audit` remains unchanged.

**Historical 1.2.0 Unity runtime validation: passed on the installed files.** Unity 2019.1.7f1 / Mono runtime `4.0.30319.42000` completed all ten self-test suites. This includes the previous nine suites and the city-locator suite using the shipped board prefab. The existing 128 native Scout assertions use real Scout/card components with an in-memory hierarchy; full Scout gameplay-prefab wiring remains outside that suite. City-locator rendered checks use the actual board prefab with a test router and artificial game state, including native GraphicRaycaster comparisons. They do not certify every physical mouse sequence or display/DPI configuration. Details are in `tests/BrassPatch/Scout/UNITY_TEST_NOTES.md` and `tests/BrassPatch/CityLocator/UNITY_TEST_NOTES.md`.

Validated binary fingerprints:

```text
Assembly-CSharp.dll  89EAC250DDA83EEB4285BA7D6486D32CB147B7E2EF226A256286918E27580D0A
BrassPatch.dll       7FC0EE609F45835BC2694B918536E6A7EBFA2823E8DAC41FBA0C783354BF2C41
Original backup     600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB
```

**Installation and recovery: verified.** The real game was upgraded from 1.1.0 to the validated 1.2.0 pair, and all ten Unity suites passed afterward. Installed hashes match the fingerprints above; the record reports 1.2.0 and online disabled. Inspection of the installed menu method confirms that online selection calls the disabled-play notice. Separate packaging fixtures exercised upgrading from 1.1.0, exact stock restoration/reinstallation, mixed-DLL interrupted-upgrade recovery, save-path preservation, pending-journal refusal and changed-candidate rejection. These restore tests used an artificial installation and save directory. Evidence is in `artifacts/patch-build/packaging-results.json`, `validation.json`, `installation-verification.json`, `city-native-result.json` and `city-render-result.json`. Rendered images are retained under `artifacts/city-locator-player/artifacts/patch-build`.

The tests cover API paths and controlled callbacks, including actual Unity popup components. They do not establish every visual layout, manual input sequence, hardware combination, or unsupported online protocol. Tests do not read, rewrite, or migrate the user's existing saves.

Source files, build scripts, and tests are authored changes. Stock game code, game assets, tool packages, generated binaries and artificial save data stay excluded from Git. Dependencies: [Mono.Cecil](https://github.com/jbevain/cecil), [Microsoft compiler toolset](https://www.nuget.org/packages/Microsoft.Net.Compilers.Toolset/4.12.0). Rule authority: [Roxley's official rulebook](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281).
