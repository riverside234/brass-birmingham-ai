# Local Rust backend release — 1.4.36

**Current installed update: 1.4.37.** The [city cancellation and checkpoint hotfix](TURN_HISTORY_HOTFIX_1_4_37.md) fixes Cancel/Undo turn boundaries and saves once after each completed human or AI turn. The original 1.4.36 measurements and receipts below retain their historical scope.

Date: **2026-10-26** (America/Chicago). **Installed and verified; Part 1 local migration complete (Full local release). All 106 existing save files are unchanged.** Release qualification and installation verification are recorded in `artifacts/rust-backend/20261026-local-release/`. The final status is the [installed verification](../artifacts/rust-backend/20261026-local-release/installed-verification.json) and [release acceptance](../artifacts/rust-backend/20261026-local-release/release-acceptance.json).

## Release scope

This completes the **local game backend implementation** requested by the user: Rust rules/simulation, Rust Gumbel IS-MCTS, same-process ONNX inference, detached C# validation and guarded main-thread commits. The production runtime binaries are byte-identical to the corrected L15 candidate. The release adds verified packaging/upgrade/recovery and additional qualification; it does not change search mathematics or trained weights.

The UI still calls the NN option **Alpha Test**, because **g890 remains the temporary test model**. ONNX Runtime **1.22.1 / CPUExecutionProvider / ort 2.0.0-rc.10** stays selected. There are two independent search sessions with one compute thread each. Difficulty remains **4 / 8 / 10 / 12 seconds per seat**; different times can share a table, while V4 and NN cannot.

The supported NN configuration is a standard four-seat local table with at least one human. Existing V4 two/three/four-seat support is retained. Qualification covers this Windows x64 Intel Core Ultra 5 135U machine. Linux/GPU training, a different CPU and the final NN require their own relevant checks.

The user's request to finish and install the local release is implemented with the existing short-test constraint. The [scope protocol](../artifacts/rust-backend/20261026-local-release/protocol.json) separates local engineering acceptance from statistical playing-strength certification. Original failed numerical screens remain failed. Statistical noninferiority, population latency tails and full live-Unity allocation totals are **not newly claimed or marked measured**. They are limitations of this local release, not unfinished runtime implementation or permission to deploy training.

## L14 allocation measurement completed for the validation stages

The earlier live-Unity callback was installed after managed code had started. Mono's public API requires allocation instrumentation to be enabled before that point. The new diagnostic host calls that API before initialization and loads the **exact shipped `mono-2.0-bdwgc.dll` and candidate assemblies**. No runtime patch, profiler or instrumentation is installed in the game. See the [Mono API implementation](https://github.com/mono/mono/blob/main/mono/metadata/profiler.c).

Both the main thread and a worker pass a mandatory known-allocation self-test. The comparison uses identical Canal/Rail fixed replies, two measured repetitions after warm-up, counterbalanced arm order, eight measured actions per arm, the existing scheduler, and exact resulting-state comparison. The old arm runs the existing three-validation call chain. The new arm counts snapshot construction, detached validation and guarded commit separately.

| Counted managed object bytes | Old validation chain | Retained optimization |
| --- | ---: | ---: |
| Eight matched actions | 742,149,300 | 383,083,234 |
| Reduction | | **48.4%** |

These are allocation traffic counts, not retained memory or uninstrumented timings. They exclude Unity presentation/engine activity, scheduler bookkeeping and ORT/native allocations. Actual whole-player allocation totals remain unavailable. This controlled exact-runtime measurement supplies the missing validation allocation comparison; it does not relabel the failed live profiler. [Results and pinned sources](../artifacts/rust-backend/20261026-local-release/allocation-final/receipt.json).

Earlier measured benefits remain applicable to the unchanged runtime: about **49% lower median worst-frame time** from detached validation, and **6.9% faster fixed work / 28.8% fewer Rust allocated bytes** from immutable decision-data reuse. Their sample limits, per-fixture regressions and the user's optimization-retention decision remain in the [L14 report](RUST_BACKEND_L14_REPORT.md).

## L15 complete normal-clock games

The new games use the real Unity turn bridge, production search, current validation and commit code. **No 450 ms clock injection is used.** Every AI turn receives the normal **four-second whole-turn allowance**, with the existing shared action-two clock. Human input is scripted Pass.

Each invocation executes at most eight turns. A trusted test-only event journal reconstructs the next segment at an ordinary turn boundary. These journals stay separate from permitted policy inputs and gameplay saves. Fixture initialization is recorded separately. During play, every segment asserts one checkpoint after human End and none during AI actions, AI End or debt. Cold worker initialization occurs again in each segment, so this is not uninterrupted-game memory evidence.

| Complete game | Turns / actions | Actual AI search turns | Human-End checkpoints | Segment execution |
| --- | --- | ---: | ---: | --- |
| Human + three NN seats | 64 / 124 | 48 | 16 | Eight segments, each 23.8–27.9 s |
| Human + V4 | 40 / 78 | 20 | 20 | Five segments, each 17.6–18.1 s |

Both games reach Canal, Rail and final scoring without invalid actions, unexpected Unity errors or hangs. All **48 native replies are retired, timely within their native deadline and search-supported**. The NN plans contain 35 builds, 32 networks, 13 loans, eight developments and five sales. This single seed per family is a mechanics/coverage result, not a win-rate comparison against V4. NN versus V4 seats remain prohibited in the deployed game.

Caller delivery includes work beyond native search. NN median delivery is **4,085 ms**, observed p95 **5,884 ms**, maximum **6,157 ms**, including repeated cold initialization. V4 median is **4,263 ms**, maximum **4,506 ms**. These are visible measured overruns at the caller boundary; the four-second setting is not a guaranteed four-second animation/initialization/validation completion time. No population-tail pass is inferred. [Full game receipts](../artifacts/rust-backend/20261026-local-release/complete-games.json).

The fresh normal-clock games did not enter debt. The unchanged L15 runtime's earlier 14 combined debt phases, focused liquidation cases, 818 lifecycle checks, 45 validation/event checks, 65 checkpoint checks, nine real-process tests and 11 golden NN positions remain linked evidence. Existing tests cover mixed seat times, stale replies, cancellation, action two, save/load, family rejection, worker failure and disable. The original remaining-budget regression remains fixed. See the [L15 report](RUST_BACKEND_L15_REPORT.md).

## Installation and recovery

[`install-rust-local-release.ps1`](../scripts/install-rust-local-release.ps1) upgrades the installed L13 package directly. It requires the sealed release and exact L13 base, verifies managed/native hashes, retains the old package and receipt, stages native files, publishes managed replacements atomically and records an installation intent. It rejects unqualified or altered packages. Both native-publication and managed-publication interruption points have explicit recovery tests in a disposable game directory. Successful upgrade and exact rollback are tested there before touching the real installation.

Rollback restores the complete prior L13 runtime and receipt. It does not convert an NN seat to V4 or modify saves. To roll back after closing the game:

```powershell
& .\scripts\install-rust-local-release.ps1 -Action Rollback
```

If publication is interrupted, use `-Action Recover`. Both operations validate their recorded recovery files and reject unrelated changes. L13's prior recovery remains in the restored receipt, so its existing whole-installation V4 rollback stays available.

[`finalize-rust-local.py`](../scripts/finalize-rust-local.py) records the complete-game evidence, identical runtime provenance, qualification, installation verification and final seal. `measure-mono-allocation.py` reproduces the short allocation probe in a fresh output directory. Historical L13/L14/L15 receipts are preserved unchanged.

**Part 2:** the local source baseline may now be used to begin the separately requested lab migration. No lab installation, training change, self-play run or teacher admission is performed by this release.
