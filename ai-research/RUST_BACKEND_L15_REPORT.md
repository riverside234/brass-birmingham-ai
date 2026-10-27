# L15 — bounded local acceptance work

**Superseding release, 2026-10-26:** L14/L15 local migration is complete and installed as **1.4.36**. See the [full local release report](RUST_BACKEND_LOCAL_RELEASE.md) for the controlled exact-Mono allocation comparison, normal-clock complete games, upgrade/recovery tests and installed verification. The earlier results and limits below remain historical evidence; their then-open local promotion status is superseded by the final local release scope. Statistical strength, broader hardware and lab deployment are not claimed.

Date: **2026-10-26**. The L15 test/replay tooling, failure diagnostics and candidate freeze are implemented. Short mechanical checks pass. **Full local release acceptance and Part 2 admission remain open.** The installed L13 game is unchanged.

The candidate retains **ONNX Runtime 1.22.1 / CPUExecutionProvider / ort 2.0.0-rc.10**, two independent sessions with one compute thread each, and the unchanged temporary g890 export. OpenVINO evaluation remains stopped. The production difficulty catalog remains **4s / 8s / 10s / 12s**, independently selectable per seat. V4 and NN cannot share a table.

## Runtime change and reproduced regression

An L15 complete-game smoke exposed a real remaining-budget error: root preparation used the search-work deadline even when only the reserved completion window remained. An ordinary request with 449 ms remaining at a longer difficulty failed immediately, before bounded completion could construct a legal action.

`native/brass-ai-worker/src/pair.rs` now bounds root identity/legal preparation by the narrowed delivery deadline. NN evaluation still runs under the stricter work deadline. This permits legal bounded completion after search closes; it grants no extra time and counts no late search work. The regression verifies zero trajectories and zero NN positions with a closed work deadline, delivery before the original narrowed deadline, and retirement of both owners. Process checks cover this boundary at all four difficulties.

The original permitted request, failed package and reproducing reply are retained in `reserve-failure-bundles/`, `candidate-before-reserve-fix/` and `reserve-replay-before.json` under the [L15 evidence directory](../artifacts/rust-backend/20261026-l15/). Successful full-game and focused checks use the corrected candidate. The 118 rules/core/inference/vendor files compared with the source archive of L14 v5 are byte-identical; that archive's full source fingerprint is also checked against its original build manifest. The correction is in the two-session worker boundary.

Production failure bundles contain the permitted request and recall, full ticket, policy/model/readiness identities, available reply and selected semantic path, error and elapsed time. Compact replies now include per-owner completed/attempted counts, quota/snapshot evidence, root Gumbels and independent seed derivation. No trusted full state, live deck or opponent hands are captured. Interrupted workers may have no reply; timed scheduling is explicitly not deterministic. Bundles are atomically published, limited to 16 files of at most 2 MiB each, and never overwrite earlier evidence or replace the original runtime error. They are diagnostics, not game saves.

[`replay-native-failure.py`](../scripts/replay-native-failure.py) checks bundle framing/digests and can replay it against the exact pinned package in a separate process. It never commits to Unity. Both the original reserve failure and a current disabled-request bundle were exercised.

## Short coverage and its limits

The complete NN games exercise the real Unity bridge, detached validation, fresh action-two validation, main-thread commits, events, debt and settlement. To honor the requested short tests, their **test-only in-memory old-save clock records start each ordinary turn with 450 ms remaining**. No game save is written to inject this fixture, and no shipped difficulty or policy is changed. These cases primarily exercise bounded completion; they are not full-allowance strength matches. Focused production-process tests separately exercise 4/8/10/12 seconds, with four seconds retained for matched comparisons.

| Final smoke case | Turns / actions | Game execution | Human-End saves | Debt phases |
| --- | --- | --- | --- | --- |
| Human + three NN seats, mixed difficulties | 64 / 124 | 31.93 s | 16 | 5 |
| Four NN seats, diagnostic fixture | 64 / 124 | 31.56 s | 0 | 9 |
| Four-seat V4 family control | 64 / 124 | 2.54 s | 16 | 0 |
| Three-seat V4 family control | 54 / 105 | 2.27 s | 18 | 0 |
| Two-seat V4 family control | 40 / 78 | 1.96 s | 20 | 0 |

Every case reaches Canal and Rail, then final settlement. The human driver uses scripted Pass input. **The V4 controls use scripted Pass actions for all seats**, testing table/save/settlement behavior rather than V4 search quality. Existing V4 profiles and protected runtime methods pass preservation checks.

The four-NN case is a research fixture. The supported game UI continues to require a human; its terminal review looks up that human. The fixture records and skips only the known missing-human review lookup after domain settlement, without changing a seat or saved assignment. It does not qualify an all-AI UI configuration. Two other development failures were harness errors: checking liquidation before its asynchronous work retired, and reading a null exception stack in a Mono exception filter. Both are fixed and their failed runs remain in the evidence directory.

Additional checks passed:

- **818 Unity lifecycle checks:** setup/family restrictions, differing seat difficulties, legacy execution mappings, shadow/live turns, same-turn clocks, cancellation, stale actor/revision/load, action-two validation, worker failure, history/menu navigation, debt and resource retirement.
- **9 real-process tests** in 43.41 seconds, including all four allowances, complex/staged/debt/forced roots, model admission, malformed inputs, cancellation, duplicate roots, shutdown and the reserve regression. Processes use the package directory as PATH and no Python inference service.
- **11 golden positions / all five output heads**, including mixed batches; original tolerances pass on CPU. The maximum absolute error across these outputs is approximately `4.58e-5`.
- **4 native worker tests**, including the new reserve regression, continuation/no clock replenishment and owner panic/retirement.
- **45 validation/event checks + 65 checkpoint checks**, including failed save retry and old mid-debt compatibility. Runtime saving remains once after human End, never during AI play.
- A live disable test commits nothing, preserves the NN family, retires resources and writes a bounded failure bundle. No seat silently switches to V4.
- In a disposable deployment directory, the installer rejects the unqualified L15 candidate without mutation. An explicitly authored rollback fixture then verifies the current candidate bytes can be retired and the exact pre-Alpha V4 binaries/receipt restored, while preserving an NN-save sentinel. Interrupted L13 publication recovery also passes. This is rollback evidence, not an L15 installation/admission receipt.
- Rust formatting and worker Clippy pass; existing upstream vendor warnings remain unchanged. Test sources, compiled test DLLs and the tested candidate manifest are captured per final Unity invocation.

## Frozen artifacts and reproduction

The [acceptance receipt](../artifacts/rust-backend/20261026-l15/acceptance.json) and [verification](../artifacts/rust-backend/20261026-l15/final-verification.json) pin source/configuration/model/runtime identities, Windows binaries, executed tests, historical parity/performance references, issue disposition, support matrix and rollback evidence. `native-candidate-windows-x64.zip` contains the native package, corresponding Rust source and notices; it contains no proprietary Steam assemblies. The candidate is under `candidate/` and is **not installed**.

The receipt intentionally records:

```json
{"smoke_passed": true, "local_release_qualified": false, "part2_admitted": false, "installed": false}
```

[`qualify-rust-release.py`](../scripts/qualify-rust-release.py) provides separate `freeze`, `build`, `unity --case ...`, `native`, `contracts`, `seal` and `audit` invocations. Use a fresh output directory and labels; failures remain immutable evidence. `test-rust-release-deployment.ps1 -StudyDirectory <directory>` runs the disposable deployment check. Test players have 90-second watchdogs, and complete-game mechanics have a 75-second internal bound. Build compilation has a separate bound. No long game/evaluation campaign is scheduled.

The remaining release gates are explicit: L14 actual Unity managed allocation totals, statistical strength and population latency tails, and full-allowance complete NN/V4 search matches. The current short tests do not establish those results or a broader CPU/platform support matrix. The measured host is the local Windows x64 Intel Core Ultra 5 135U. A final model, lab migration or wider release requires its own admission evidence.
