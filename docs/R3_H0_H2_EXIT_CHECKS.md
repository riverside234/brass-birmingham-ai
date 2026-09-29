# R3 H0–H2 exit checks

27 September 2026. Scope: four-player baseline diagnosis, shared scored complete actions and S1 complete-turn legacy candidates. Rules patch `steam-brass-patch-1.1.0`; observation/action contract `0.2.0`; worker protocol `legacy/0.1.0`. See [implementation and usage](R3_LEGACY_BANK.md).

**Local implementation delivered; deployment qualification remains open.** H0/H1 correctness checks and H2's combination, restoration and runtime checks pass. Balanced and Rail Tempo remain candidate profiles. A small stock comparison is a pilot, not H5 qualification, calibrated difficulty or evidence of human enjoyment. The optional Unity overlay is built separately and has not been installed in Steam.

## Executed correctness checks

| Area | Evidence and scope |
| --- | --- |
| H0 stock controls | Original and guarded stock each completed two four-player games, seeds 9301–9302. The seed-9301 instrumented and plain controls have identical action/event traces and outcomes. Original and guarded rules identities remain separate. |
| H0 independent diagnosis | Three guarded-stock runs at seed 9301: plain, recorded and independently diagnosed. All 124 action/state/outcome traces agree, selected rules are unchanged, and all 124 diagnostic decisions contain multiple complete alternatives. The live canonical state and RNG are unchanged by diagnosis. |
| H1/H2 focused tests | **17 tests pass** against the pinned executable. These cover first-turn action count, all-seat hidden-world invariance, deterministic fixed work, valid complete incumbents, stale/duplicate/partial rejection, complete-turn replay, economic scoring and tactical combinations. |
| H1 complete-game reliability | The final-build L1 Balanced profile completes 64 turns / 124 actions with four L1 seats, seed 9803, at its 250 ms turn search allowance. Scores are 0, 4, 10 and 74 VP: this inexpensive configuration is not a recommended playing opponent. |
| Enabling combinations | Explicit fixtures retain loan/develop/Scout-enabled builds and necessary second-action cards. A double-Rail fixture actually exercises two links with legal beer payment. Search remains bounded; these are not exhaustive strategy tests. |
| Scoring | Banked Canal points and merchant bonuses are counted once; border regions score two link points; supply saturation affects expected completion; final cash is not invented VP. The last Rail turn uses public final scoring and official VP/income/cash tie-breaks. |
| Coordinator and adapter | **17 bridge assertions pass** in .NET and actual Unity Mono: background-state isolation, complete incumbent, Play Now, one-shot execution, stale game, cancellation, worker failure, confirmation ownership, persisted public-history parity and rejection of unsupported player counts. |
| Full application-game lifecycle | Each runtime completes 64 turns / 124 actions, including 32 Rail turns, 15 vendor-repository save/load cycles and eight debt episodes. Publicly projected final scores match the actual result for all four seats. One seat has Human metadata but its moves are scripted; this is not a human playtest. |
| Unity AI routing | Actual woven `PlayAITurn` and `MonoBehaviour.Update` run with a mock view router. Duplicate turn requests are rejected, menu cancellation leaves the game unchanged, and Update continues while the worker thinks. This does not validate rendered controls, animations or user interaction. |
| Existing R1/patch regressions | R1 runtime check passes: 46 restored checkpoints, 5,393 replayed decisions and injected transaction rollback. Core 1,081, State 315, Resources 99, Liquidation 56 and Scout 125 assertions pass; State includes 15 replay cutoffs and five games. The pinned R1 executable is unchanged. |
| Existing R2 regressions | All 23 R2 tests pass on Windows CPU. No model training or R2 pool admission is implied by this check. |

The focused privacy tests reconstruct different compatible hidden worlds for every acting seat and compare the policy's selected plan and work count. The policy receives the allowed observation and semantic choices, never the live game's seed, hidden hands, deck or trusted checkpoint. The historical stock policy itself is not certified as a strict-observation policy.

The full application-game check uses in-memory instances of the vendor repository and factory. It verifies saved public history, developed-tile reconstruction and both eras. It does not qualify every interactive save boundary or migrate old pre-patch saves. Mixed stock/replacement harness traces also do not provide a portable, resumable replacement for R1's canonical journal.

## Four-seat pilot

Final worker, seeds **9801 and 9802**, four candidate seats per setup, one candidate against three guarded stock opponents. All **16 games** completed 64 turns / 124 actions without execution failures or rejected search candidates. [Machine-readable results and raw-report hashes](R3_PILOT_RESULTS.json) preserve the final measurements.

| Profile | Wins / games | Mean rank | Mean VP | Mean margin over best rival | Wins by setup |
| --- | --- | --- | --- | --- | --- |
| L2 Balanced | 7 / 8 | 1.125 | 141.875 | +30.875 VP | 9801: 3/4; 9802: 4/4 |
| L2 Rail Tempo | 6 / 8 | 1.250 | 138.750 | +25.625 VP | 9801: 2/4; 9802: 4/4 |

These are encouraging pilot results, with only **two independent setup groups per profile**. They do not establish population win rates or H5 strength qualification. Six selected plans reached a fully known final board in these matches; all six projected VP/placement/win-share results match the eventual engine result. Other turns retain explicitly approximate heuristic values.

| Profile | Recorded decision/execution time per turn, median | p95 | Maximum | Mean reconstruction nodes |
| --- | --- | --- | --- | --- |
| L2 Balanced | 3.192 s | 3.316 s | 3.402 s | 95.73 |
| L2 Rail Tempo | 3.189 s | 3.334 s | 3.466 s | 95.55 |

The nominal **3,000 ms** allowance governs search. Completing, validating and committing a legal incumbent adds overhead. These recorded per-turn totals sum decision/execution timers; they **exclude outer observation capture, IPC and rendered UI presentation**, and are not full user-visible turn latency. Runs overlapped other test workers, so this is not an isolated performance benchmark. Strict total-turn limits and three-opponent waiting time require the later rendered-game measurements.

Each setup is played four times: one candidate in each seat against three guarded stock opponents under the same patched rules. Seat assignments within a setup are correlated. The second setup was added after inspecting the first; these are exploratory pilots, not a prespecified qualification tournament. Wall-clock search can explore different candidates under different machine load; fixed-work mode is the deterministic regression mode.

Earlier development runs are retained locally, with their own runtime manifests. They are **superseded**, not pooled into final-build results. In particular, the 9501–9502 and 9701 pilots preceded the correction for merchant bonuses already included in banked VP. Checking the 9701 Rail report exposed a projected-placement mismatch; the summary tool now rejects such mismatches. Early one-second experiments also played poorly. L1's inexpensive all-candidate completion test establishes reliability, not equivalent playing strength.

## Fingerprints and evidence

| Component | SHA-256 |
| --- | --- |
| R3 worker `BrassLegacyAI.exe` | `babb221e432fd2df78a5bd8a16bfb45185fafaad075ff21607e55b6247af608f` |
| Optional `BrassLegacyAI.dll` | `5ec0c143a358081f3dcaa4b0ec3f1c1aeb086ee8ff0e013f1f445e19a90d40e8` |
| Installed `Assembly-CSharp.dll`, unchanged | `1c8cef281b81474200268eed23448faa3f5247f8ea49409a69efc3982c5a2d0a` |
| Installed `BrassPatch.dll`, unchanged | `8315180fba0a7d94d757bac270c8944c56cfba315f15db2537aceaca13f6eefb` |
| Existing R1 `BrassHost.exe`, unchanged | `4b6468f5ff8b0da8d8527be6292474165046faa227a38804b54242d9a37050b5` |

The [candidate registry](../training/opponents/registry.json) pins all source, profile, contract and runtime dependency hashes, and explicitly sets candidate benchmark/training eligibility to false. The worker manifest pins 14 host/policy sources and 112 executable/dependency files. The overlay weaver verifies that only its two intended routing method bodies change; its manifest records `installed: false` and `qualified: false`.

Generated local evidence, excluded from Git:

- `artifacts/legacy-ai/h0/result.json` and its plain/recorded/diagnosed traces.
- `artifacts/legacy-ai/bridge-result.json` and `unity/legacy-unity-result.json`.
- `artifacts/legacy-ai/unity/overlay-ui-result.json` and Unity logs.
- `artifacts/legacy-ai/verified-balanced-3s/`, `verified-rail-3s/`, `verified2-balanced-3s/` and `verified2-rail-3s/`, including each run's complete source/runtime manifest and individual match reports.
- `artifacts/legacy-ai/verified-l1-reliability/` for the final-build four-L1 reliability game.
- `artifacts/r1-host/dotnet-result.json` and existing patch regression outputs.

Tests ran on Windows .NET Framework and Unity **2019.1.7f1** Mono. These are CPU results. Linux/Mono on `/data/not_backed_up/yxu209/ai-training/` has not been qualified, and neither have integrated-GPU inference, lab throughput or sustained laptop UI performance.

## Reproduction and remaining gates

From this checkout, with the matching private runtime/tool dependencies already available:

```powershell
./scripts/build-legacy.ps1
./scripts/capture-h0.ps1 -Seed 9301 -Games 2
python scripts/diagnose-legacy.py --root .
python -m unittest discover -s tests/legacy-ai -v
./scripts/build-legacy-overlay.ps1
./scripts/check-legacy-bridge.ps1 -Overlay
./scripts/check-legacy-bridge.ps1 -Unity -Overlay
# Wait for the isolated Unity process to exit and inspect its new result file.
./scripts/check-legacy-overlay-ui.ps1
# Again wait for the process and inspect the newly written result.
python scripts/run-legacy.py --root . --profile balanced-v1 --seeds 9801 9802 --seats 0 1 2 3 --milliseconds 3000 --output new-balanced-pilot
python scripts/run-legacy.py --root . --profile rail-tempo-v1 --seeds 9801 9802 --seats 0 1 2 3 --milliseconds 3000 --output new-rail-pilot
python scripts/summarize-legacy.py artifacts/legacy-ai/new-balanced-pilot artifacts/legacy-ai/new-rail-pilot --output artifacts/legacy-ai/new-pilot-summary.json
python scripts/register-legacy-candidates.py --root .
```

Use new output directories for a repeated pilot. The local Python checks used the bundled interpreter, `PYTHONPATH=src` for R3 and the existing pinned R2 dependency directory for R2. Original game files, decompiled source, generated DLLs and local `configs/legacy-ai/live.json` remain ignored. No installation, commit or public upload was performed by this delivery.

Remaining acceptance is explicit:

1. **H2 deployment:** rendered one-human/three-bot sessions, actual Play Now controls, menu/restart while thinking, animations, bonuses, debt presentation and supported interactive save/load boundaries. The automated Unity tests cover only part of this gate. Strict advertised whole-turn caps also need overhead reservation and measurement while the rendered game runs.
2. **H5 qualification:** frozen, prespecified independent setup groups; grouped uncertainty intervals; mixed tables and opponent arrangements; held-out tactical families; latency/failure and human-enjoyment evaluation. No general “stronger Hard,” new-Easy-above-stock-Hard or calibrated personality/difficulty claim is made yet.
3. **Later requested milestones:** H3's remaining four profiles and persistent intent, H4/S2 opponent replies, and eventual R2 training-pool integration. These are outside this H0–H2 implementation.
