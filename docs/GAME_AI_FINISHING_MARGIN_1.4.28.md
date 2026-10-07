# Game AI finishing margin — 1.4.28

Installed on 7 October 2026 over the qualified 1.4.27 package. This changes only the game coordinator's search timing and publication guard. V4's six strategies, two private workers, 2/5/12-second menu options, existing 10-second saves, cancellation, live action validation and resource lifecycle remain intact. Training source, configuration, frozen opponents and checkpoints are unchanged.

## Execution

The coordinator reserves part of the **existing whole-turn allowance** for cooperative worker retirement, the existing RootForest merge and detached plan publication. Both roots receive the same earlier cancellation signal and reduced remaining work allowance. The scheduler still waits for both owners to retire; cancellation does not pretend they are finished. Neither actions, subchoices nor replanning acquire a fresh turn clock. Initial admission/preparation behavior is preserved.

| Menu allowance | Finishing margin | Latest simulation cutoff from request start |
| --- | ---: | ---: |
| 2 seconds | 250 ms | 1.750 s |
| 5 seconds | 300 ms | 4.700 s |
| 12 seconds | 400 ms | 11.600 s |

The margin interpolates between these points, caps at 400 ms and scales down to zero with an exhausted request. Existing 10-second saves reserve approximately 371 ms. Dispatch/snapshot time consumes the same allowance; workers receive only what remains until the shared cutoff.

The existing forest's best completed legal incumbent remains available while search runs. An on-time merged plan replaces it as before. A late result cannot overwrite an already published fallback, including when the deadline expires during detached copying. If initial mandatory incumbent construction itself overruns an exhausted request, its first complete legal plan is retained; no partial action is fabricated. This exceptional case is reported rather than hidden. The duplicate final publication formerly performed by the coordinator is removed: RootForest already publishes the merged result.

No tactical scoring, exploration, pruning, strategy, profile, shared search implementation or saved identity is modified. In particular, the two game workers and four lab workers remain independent.

## Laptop measurements

Native Unity Mono, 30 FPS, two workers, three four-player positions: middle Canal, early Rail and late Rail. The positions were generated using the installed game's host and reused for both measurements. The baseline uses the installed RootForest with detached publication; the revised run uses the actual coordinator, including snapshot/queue overhead. These are small timing samples, not strength results.

| Allowance | Baseline deadline overrun | Revised finalization after cutoff | Latest revised publication | Latest revised worker/merge completion |
| --- | ---: | ---: | ---: | ---: |
| 2 s | 54–164 ms | 96–156 ms | 1.905 s | 1.906 s |
| 5 s | 104–187 ms | 50–115 ms | 4.813 s | 4.815 s |
| 12 s | 19–228 ms | 63–151 ms | 11.749 s | 11.751 s |

All **9/9 revised searches** retired and published before the whole-turn deadline, with no search failures or ignored late publications in these timed cases. Finalization is measured from the shared simulation cutoff through retirement, merge and publication; it includes the tail of any in-flight non-preemptible computation. Copies alone accounted for less than a millisecond per baseline turn. The margin therefore primarily covers worker retirement rather than serialization. Normal OS scheduling can still cause an exceptional overrun; the complete-plan fallback and actual-retirement guards remain necessary.

| Allowance, three positions each | Completed rollouts before → after | Search nodes before → after |
| --- | ---: | ---: |
| 2 s | 4 → 0 | 148 → 117 |
| 5 s | 6 → 6 | 365 → 338 |
| 12 s | 36 → 31 | 788 → 744 |
| Total | 46 → 37 (−19.6%) | 1,301 → 1,199 (−7.8%) |

Measured total elapsed time fell from 58.04 to 55.00 seconds. Including finalization, completed rollouts per second fell from 0.793 to 0.673 (−15.1%); nodes per second fell from 22.42 to 21.80 (−2.8%). Short searches cross discrete completion thresholds: the three revised 2-second cases used their fully completed and legally replayed heuristic incumbents, with no completed MCTS rollouts. This cost is reported explicitly; no playing-strength improvement is claimed. The more generous settings retain more search evidence.

## Checks and preservation

- **164 native deadline checks:** nine timed searches, shared two-worker/whole-turn bounds, complete legal plans, immutable observations, active-search cancellation, full worker retirement and no detached work.
- **17 boundary checks:** zero remaining allowance, retained fallback despite a late publication, both actions replay legally, consume once, cancellation before admission, resource release.
- **1,550 native integration checks:** six styles, existing time settings, live turns, saved execution settings, stale input/cancellation, clock carry-over and resource restoration. Includes 20 resource checks; all 18 submitted jobs completed, maximum one turn job running, zero remaining.
- **4 packaging checks:** accepts the qualified build and rejects late deadlines, changed worker count and expanded source scope.
- **Preservation:** 14,882 existing method bodies outside the three modified coordinator methods are unchanged. All 48 embedded profiles are identical. `Assembly-CSharp.dll` and `BrassPatch.dll` are byte-identical to 1.4.27, preserving the current UI/rules patches. Training/shared-search diffs are identical before and after this task.

Evidence is under `artifacts/game-finishing-margin/`: `baseline-native/report.json`, `candidate/report.json`, `comparison.json`, and the candidate build's boundary, native integration, preservation and qualification reports. The installed package is archived under `artifacts/patch-releases/1.4.28/`; `installed-verification.json` records exact installed hashes and the recovery backup. Native tests ran on the qualified candidate before installation, followed by installed byte verification.

Rebuild and repeat only the affected checks with `scripts/build-game-finishing-margin.ps1`, `tests/BrassLocalAI/run-finishing-margin.ps1 -Candidate -BuildDirectory <new-build> -Label <new-label>`, and the native quick mode in `tests/MaskedISMCTS/run-era-package.ps1`. The native timing runner expects the retained native `deadline-fixtures.json` in `artifacts/game-finishing-margin/`. Use a new build directory for another qualified candidate; keep existing evidence.
