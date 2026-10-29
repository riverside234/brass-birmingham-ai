# NN full search allowance and fixed delivery grace

Installed release: **1.4.45-rust-local**, 2026-10-29.

The selected 4/8/10/12-second allowance is now available to NN search. Every K choice (8, 16, 32) receives one additional delivery second, giving total turn-clock deadlines of 5/9/11/13 seconds. This replaces the previous delivery reserve deducted from search time. The AI selection panel explains the additional second.

The search clock retains its existing boundary: model startup and initial live-board snapshot preparation happen before it starts. These totals therefore describe the running AI turn clock, rather than time from clicking a menu button. A selected allowance is a maximum; search can finish earlier.

## Deadline and correctness contract

- Rust freezes the search cutoff and the total deadline before starting either owner. NN simulation cannot consume the delivery second. Native retirement, completion, IPC and host delivery share the remaining time.
- Unity accepts delivery only until the earlier of the original total deadline or one second after receiving the result. It requires 100 ms of measured deadline headroom before starting a main-thread commit.
- Retries and replans keep the same running clock and frozen total deadline. Loading a legacy partial-turn clock uses its saved remaining time without adding another second. A new completed-turn save starts a new turn normally.
- Identity, generation, revision, actor, cancellation, detached legality checks and second-action revalidation remain in place. An expired result stops the NN turn before mutation.
- This update uses search_v1, V205 disabled, the existing g890 test NN and ONNX Runtime CPU inference with two independent sessions and one compute thread per session. K and temperature choices remain available.

## Short qualification tests

All 12 combinations of K8/K16/K32 and 4/8/10/12 seconds passed. Cases cover Canal, branching Rail and late Rail with a long event history. The probe pauses only the test bridge until the search cutoff; for an already-completed reply it sets test-only delivery timing at resume. Original native timing and the unmodified turn clock are recorded separately. These are delivery stress probes, not strength matches or a population latency guarantee.

| Search | K | Delivery after cutoff | Clock headroom |
| --- | --- | --- | --- |
| 4 s | 8 | 176 ms | 808.4 ms |
| 8 s | 8 | 133.7 ms | 860.5 ms |
| 10 s | 8 | 428.4 ms | 566.4 ms |
| 12 s | 8 | 460.9 ms | 511.4 ms |
| 4 s | 16 | 178.2 ms | 801.2 ms |
| 8 s | 16 | 123.3 ms | 872.1 ms |
| 10 s | 16 | 337.6 ms | 652.6 ms |
| 12 s | 16 | 175.6 ms | 809.1 ms |
| 4 s | 32 | 181.3 ms | 807.5 ms |
| 8 s | 32 | 117 ms | 880 ms |
| 10 s | 32 | 356.6 ms | 631.2 ms |
| 12 s | 32 | 568.3 ms | 421 ms |

The slowest measured delivery tail was 568.3 ms; minimum remaining clock headroom was 421 ms. An additional expired-reply probe crossed the total deadline and committed no actions. Ordinary 4-second K8 and K32 turns also passed.

The exact shipped Mono runtime passed 64 budget/clock checks. Native reserve tests passed all 9 cases. Unity lifecycle tests passed 839 checks, including cancellation, stale replies, actor/revision changes, second-action validation, save/load and completed-turn checkpoints. The panel screenshot was inspected. Disposable installation tests verified interrupted-install recovery and exact rollback to 1.4.44.

## Installed game verification

| Search | K | Probe | Total delivery | Clock headroom |
| --- | --- | --- | --- | --- |
| 4 s | 8 | Normal | 4358.2 ms | 713.2 ms |
| 10 s | 16 | Delayed delivery | 10582.5 ms | 513.4 ms |
| 12 s | 32 | Delayed delivery | 12315.7 ms | 748.8 ms |

All installed smokes executed both actions and verified save/load against the installed DLL and native binary. Installed runtime hashes match the qualified candidate. All 142 user save/history files remain byte-for-byte unchanged. Registry-backed preference measurement was unavailable in the sandbox; the installer does not write preferences and test menu settings use the existing in-memory self-test store.

Evidence: artifacts/search-delivery-grace/20261029/assessment.json, release-1.4.45/qualification.json, upgrade-test.json, installed-verification.json and user-data-verification.json. Test source snapshots, test DLL hashes and raw results are retained alongside each case.

## Recovery

Close the game before recovery. Run scripts/install-search-delivery-grace.ps1 with -Action Rollback to restore the exact previous 1.4.44 DLL, native package and installation receipt. Use -Action Recover if an installation was interrupted. The installed receipt identifies the retained backup at C:\Program Files (x86)\Steam\steamapps\common\Brass Birmingham\artifacts\patch-recovery\search-delivery-grace-0fa3d0c9859e46718bd7d174daf37b98.
