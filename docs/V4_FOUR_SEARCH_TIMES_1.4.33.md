# Patch 1.4.33 — Four V4 search-time settings

Date: 24 October 2026  
Base: accepted patch **1.4.32**  
Status: installed and verified.

| Setting | Search allowance | Workers |
| --- | ---: | ---: |
| Easy | **4 seconds** | 2 |
| Mid | **8 seconds** | 2 |
| Hard | **10 seconds** | 2 |
| Max | **12 seconds** | 2 |

The V4 page displays four buttons in one row. Its help text moves beside the expanded row. Button labels, the selected-player summary and the Continue rows share the same difficulty catalog. The default remains Hard, now at 10 seconds. Gamma retains its unavailable state.

Easy, Mid and Hard keep their existing search breadth and opponent reply horizon. Max uses Hard's search shape with a longer allowance. The two-worker coordinator, algorithm, finishing margin, work guard, strategy profiles, persistent intent and shared turn deadline remain intact. Preparing the board and playing animations can add time outside the search allowance; a completed search may finish sooner.

New execution records use `local-ai-execution/1.2.0`. Both earlier execution versions remain readable, with their exact recorded time/ID pairs validated and signed assignment data preserved. Named Easy/Mid/Hard saves use the new 4/8/10-second allowances when resumed. Max is valid only in the new execution version. Unnamed legacy saves retain their 10-second compatibility setting.

The build starts from the frozen 1.4.32 release and overrides only the difficulty catalog, search-shape documentation, turn-time documentation, selection panel and menu validation message. The game assembly and Statistics/UI helper remain byte-identical. Unrelated training and personality work is excluded.

## Verification and delivery

Candidate verification passed:

- **3,110 native checks**, including **683 time-setting and compatibility checks** across all 144 player-count/mode/style/time combinations and 216 compatibility cases for signed records from the two previous execution versions.
- **Nine real V4 turn cases**, covering all four allowances, all six styles, two/three/four players, shared-deadline replanning, input guards and worker cleanup.
- **18 native layouts** across Human/V4/Gamma, 1080p/1200p and all text presets. Four-button labels fit and the controls stay separated. Max applies immediately and survives save/Continue.
- **Seven native captures manually reviewed at 1200p**, including all V4 text presets, selected Max, the updated player row, Continue and disabled Gamma controls.
- **14,991 existing methods** and all embedded resources preserved outside the permitted timing/UI changes. The coordinator, algorithm, two-worker setting and original search-shape method are unchanged.

All **86 installer rejection/recovery checks** passed using artificial files, including rejection of a missing fourth control or stale time label. The final menu run against the installed game passed **1,502 checks**, with no native errors.

Patch 1.4.33 is installed. Online play remains disabled. The installation receipt identifies the retained 1.4.32 rollback backup. The release archive contains the frozen source snapshot, reviewed captures, validation fingerprints and candidate/installed test reports.

Candidate: `artifacts/v4-four-times-build/`  
Archive: `artifacts/patch-releases/1.4.33/`  
Installation and previous-version recovery backup: `artifacts/patch-installation.json`
