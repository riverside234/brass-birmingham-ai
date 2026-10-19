# Patch 1.4.32 — V4 search time

Date: 19 October 2026  
Base: accepted patch **1.4.31**  
Status: installed and verified.

| V4 difficulty | Previous allowance | New allowance | Workers |
| --- | ---: | ---: | ---: |
| Easy | 2 seconds | **6 seconds** | 2 |
| Mid | 5 seconds | **10 seconds** | 2 |
| Hard | 12 seconds | **15 seconds** | 2 |

The V4 selection buttons, description, selection summary and player-row summary all read from the same difficulty catalog. They display the new allowances immediately. The default remains Hard; all six strategies and Random remain available, and Gamma remains unavailable.

Each named difficulty keeps its existing search breadth and opponent reply horizon. The two-worker coordinator, search algorithm, work guard, finishing margin formula, strategies, embedded profiles, persistent intent, shared turn deadline and replanning rules are preserved. Preparation and animations can add time outside the search allowance; a completed search can finish sooner.

New difficulty records use `local-ai-execution/1.1.0`. Previous `1.0.0` records remain readable with their original signed bytes and difficulty IDs. When resumed, their named difficulties receive the new allowance. Older saves without a named difficulty retain the existing 10-second compatibility setting and its search shape.

Only three runtime sources change: `DifficultyCatalog.cs`, `ProfileCatalog.cs` and `LocalTurnBridge.cs`. The game assembly and Statistics/UI helper remain byte-identical to 1.4.31. Sources are built from the frozen release with those three overrides; unrelated training work is excluded.

## Verification and installation

Candidate verification passed:

- **2,718 native checks**, including **343 search-allowance and compatibility checks** across all 108 player-count/mode/style/tier configurations.
- **Eight actual V4 turn cases**, covering all six styles, all three allowances, two/three/four-player games, input guards, shared-deadline replanning and worker cleanup.
- **18 native menu layouts** covering Human/V4/Gamma at 1080p and 1200p with all three text presets. Five captures were manually reviewed, including all three V4 presets at 1200p and the initial rows displaying Hard · up to 15s.
- **14,992 existing methods** and all embedded resources preserved outside the four changed timing/compatibility methods and two added adapter methods. The coordinator and search engine are unchanged.

All **78 installer rejection/recovery checks** passed using artificial files. The final menu run against the installed game files passed **998 checks**, including the new allowances, old-record compatibility and current UI labels/layouts, with no native errors.

Patch 1.4.32 is installed. Online play remains disabled. The installation receipt identifies the retained 1.4.31 rollback backup. The release archive contains the compiled source snapshot, reviewed captures, validation fingerprints and candidate/installed test reports.

Candidate: `artifacts/v4-search-time-build/`  
Release archive: `artifacts/patch-releases/1.4.32/`  
Installation and previous-version recovery backup: `artifacts/patch-installation.json`
