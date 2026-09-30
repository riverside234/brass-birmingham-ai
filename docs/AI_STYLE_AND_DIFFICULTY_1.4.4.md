# AI style picker and difficulty settings

**Package:** 1.4.4.  
**Status:** Installed and verified in the Steam game directory.  
**Date:** 30 September 2026.

**Follow-up:** [Patch 1.4.5](AI_MENU_INITIALIZATION_1.4.5.md) fixes first-open initialization, where the stock controls could remain until a click refreshed the menu. Its regression test follows the production order of configuring the menu before enabling it. The results below are the historical 1.4.4 release record.

Custom Game now has a direct style chooser and separate difficulty buttons for each AI player. Click the style button to see Random and the six installed styles, with descriptions and a checkmark beside the selected style. Selecting a tile applies it immediately. Escape or an outside click closes the chooser without changing the selection or activating the menu underneath.

The chooser uses catalog entries and supports larger rosters. Above eight concrete styles, it adds a filter and a bounded scrolling area. Native layout tests use 16 styles plus Random; those synthetic entries are not included in the playable catalog. The future neural personality system and its internal variants remain separate research work.

| Difficulty | Search allowance |
| --- | ---: |
| Easy | 2 seconds |
| Mid | 5 seconds |
| Hard | 12 seconds |

Hard is the default for a new setup without an existing difficulty preference. Each bot remembers its own style and difficulty after successful game creation. Color changes, human/AI switches and temporarily removing a player preserve the setup selections. Continue displays saved settings as read-only text.

All three presets run the current v3 S1 AI. They change the execution time allowance; the frozen profiles, evaluation, intent, fallback and research-bank settings remain intact. Random chooses a starting style once at creation. Its intent can adapt and its specialist can use the existing fallback during play.

One clock covers the whole AI turn, including the remaining allowance for a second planning request. Search can finish sooner, and board preparation, mandatory legal completion, validation and animation can add overhead. The bank's `work_limit` remains **240** and `reply_work` remains **2,048**. These are the existing planner settings: the planner permits bounded completion work after exhaustion, and a rare new planning request has its own node counter while sharing the turn's remaining time. They should not be described as an absolute 240-node total across replans. No new AI strength or tier-calibration claim is made.

New games use save-extension format `brass-local-ai-save/1.1.0`. A versioned execution object stores the difficulty and allowance in each bot's assignment, covered by the assignment integrity hash. It is separate from the immutable profile and persistent-intent identity. Loading rejects an inconsistent ID/time pair rather than substituting another bot.

Existing local-AI saves retain their prior format and profiles and use the installed compatibility allowance of **10 seconds**, shown as **Existing · 10s**. Earlier saves did not record their live time setting; this preserves the behavior of package 1.4.3. Original game saves without the local-AI extension keep their original AI path. Qualification uses generated saves, including saves produced by archived binaries, and does not open personal games.

The controls use rounded dark surfaces, warm ivory text and gold selection borders. The popup owns pointer and keyboard navigation while open. Refreshing a row invalidates its old popup binding; menu fade-out, scene changes and destruction clean up detached controls and tooltips. Play now and online play remain disabled.

## Validation and installation

All release gates passed against the three final candidate DLLs. Installation completed on 30 September 2026 at 05:31 UTC, followed by tests using the actual installed game executable and assemblies.

| Check | Result |
| --- | --- |
| Managed rules, resources, state, liquidation, persistence and AI regressions | 12 suites passed. |
| Local runtime, S1, difficulty contracts and save validation | 2,339 assertions passed, including real timed searches at 2/5/12 seconds and legal two-action completion at zero remaining allowance. |
| Native AI and menu integration | 108 configurations passed: six styles × three difficulties × 2/3/4 players × introductory/full games; 31,727 assertions and no unexpected native errors. |
| Stale second-action continuation | Forced replan retained the original 5-second deadline; no fresh allowance was granted. |
| Existing native rules and UI | 10 suites passed, including 46,779 city-highlight checks. |
| Player panels, industry sprites and inspection-panel lifecycle | 184 assertions passed; no failures. |
| Complete games with save/reload | Six games passed: 456 actions and 231 reloads across all player counts and both rule modes. These use a small work setting for lifecycle coverage, not strength or timing measurement. |
| Genuine older save producers | Packages 1.3.2, 1.4.0 and 1.4.3 each passed 54 assertions covering all six styles. |
| Upgrade, interrupted installation and rollback | 79 assertions passed in disposable fixtures. |
| Installed game startup, menus and creation | 153 assertions passed, with no unexpected native errors. |
| Patch preservation and provenance | All 125 previous hooks retained; 126 total hooks. Six merchant-bonus rewrites, original backup, offline mode, six v3/six compatibility profiles, 39 research source files and 112 frozen runtime files verified. |

The native menu checks cover direct selection, separate per-player difficulties, color/seat changes, human/AI switching, removal/readdition, read-only Continue, filtering a synthetic 16-style roster, empty results, keyboard focus/cancel, outside-click interception, menu fades and tooltip cleanup. Render captures cover 1280×720, 1600×900, 1920×1080 and 2560×1440, including the narrow Continue/New Game layout. Test setup preferences and saves are isolated; personal games were not opened or modified.

Timed-search accounting is recorded in `local-ai-runtime-0.json`. This run used 22, 54 and 126 planner nodes for Easy/Mid/Hard on the same two-action fixture, with unchanged `work_limit: 240` and `reply_work: 2048`; exhausted-budget completion used seven nodes. These figures describe this fixture and machine load, not a universal node count or a comparison of playing strength.

Installed SHA-256 fingerprints:

| File | SHA-256 |
| --- | --- |
| `BrassPatch.dll` | `211A4365A0E833535EE756985140495C92C121BC665467C6C2513D60CEBC2413` |
| `BrassLegacyAI.dll` | `42560334C2084CEDE7B8DFC7D1C9168726FBE39D4A46D29F50317A9353D7A934` |
| `Assembly-CSharp.dll` | `9E6DB4DBFEB710A2958DBB448E7963EDB8B081F6C06D70AD358D8EA10F418B39` |

The helper DLL is identical to package 1.4.3, retaining its prior gameplay, presentation and save repairs. Source snapshots, binaries, screenshots, installation record and qualifying reports are archived under `artifacts/patch-releases/1.4.4`. Previous releases and the installer's verified recovery copy are retained.

Resolution checks use the actual Unity controls rendered to native test targets. They do not constitute a physical mouse playthrough at every monitor DPI or a measured comparison of playing strength.
