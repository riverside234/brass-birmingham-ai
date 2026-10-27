# L14 — Local profiling and validation optimization

**Superseding release, 2026-10-26:** L14/L15 local migration is complete and installed as **1.4.36**. See the [full local release report](RUST_BACKEND_LOCAL_RELEASE.md) for the controlled exact-Mono allocation comparison, normal-clock complete games, upgrade/recovery tests and installed verification. The earlier results and limits below remain historical evidence; their then-open local promotion status is superseded by the final local release scope. Statistical strength, broader hardware and lab deployment are not claimed.

Updated 2026-10-26. **Background validation and the additional Rust decision-data reuse optimization are retained. The bounded follow-up is complete; full L14 qualification remains open for actual Unity allocation totals and deferred statistical strength/tails.** Candidate v5 is built and tested in isolation. The installed game, final training model and lab deployment are unchanged.

The measured build is `artifacts/rust-backend/20261026-l14-candidate-v3/`; the frozen experiment, raw trials and profiles are in [`20261026-l14-study-v2`](../artifacts/rust-backend/20261026-l14-study-v2/). The comparison uses the exact sealed L13 package, g890 export, K=8, c1=50, c2=0.1, cap/pending=1, immediate flush, **two independent sessions with one compute thread each**, and **one four-second ordinary-turn allowance**.

The original default-enabled build is `artifacts/rust-backend/20261026-l14-candidate-v4/`. It adds the explicit default-enabled/comparison-only switch. Its method bodies match the measured build, apart from the compiler renumbering the unchanged `CheckPath` lambda and its delegate-cache field; both were checked separately. The [binary comparison](../artifacts/rust-backend/20261026-l14-study-v2/retained-vs-measured-il.json) and [lambda check](../artifacts/rust-backend/20261026-l14-study-v2/retained-lambda-equivalence.json) preserve that distinction. All 845 Unity checks, 45 event/identity checks and 65 checkpoint checks also passed on v4. The latest retained v5 additionally contains the Rust optimization described below.

## Implementation

`NativeValidation` owns private deep copies of the permitted validation observation, composition and complete semantic path. The existing execution scheduler runs reconstruction and full-action validation after the native job retires. The background validator holds no Unity objects, live host, application or mutable board references.

Immediately before mutation, the main thread checks the full game/lifetime/generation/request/model/profile identity, current revision, actor, phase, cancellation, action path, composition and exact current observation. The successful receipt is single-use and restricted to its capturing thread. Each semantic choice is still resolved against the current legal choices while applying it. The second action receives a fresh snapshot after the first action's actual events. Debt retains its existing validation route; V4 search is unchanged.

The equivalence argument is specific: the old complete-action validator reconstructs from the same observation, composition and seed, then runs the same `SearchApply` path. Equality of those inputs, the boundary checks and current semantic legality make its duplicate reconstructions redundant. `SearchApply` does not advance the turn or reveal a new draw. If this contract changes, the final validation must be reconsidered.

The baseline performed **three full validations per ordinary action**. The candidate performs **one full validation in the background**, followed by the final main-thread checks and mutation. The diagnostic eight-turn runs recorded 48 baseline full validations for 16 actions, versus 16 detached validations and 16 guarded commits.

## Save/load rule

Following the user's corrected instruction, gameplay writes one automatic checkpoint **after the human clicks End Turn**, after applying that turn's end/next-player/round events and before starting AI play. Initial game creation still saves. AI search, validation, individual actions, AI End Turn and liquidation continuation do not save. A failed human checkpoint blocks continuation and retries that checkpoint without repeating game events.

Loading during AI work cancels and retires that work, restores the last human End checkpoint, and creates fresh search/validation for the saved boundary. It never reuses a receipt. Same-live-turn retries retain the original clock. Older mid-AI saves remain readable. The unreleased in-flight save writer explored in candidate v2 was removed after this user correction; its earlier evidence is retained, not relabeled.

## Measured validation benefit

Fixed replies use eight development/held-out permitted roots, 30 measured repetitions after warm-up, counterbalanced execution order and identical action paths. There are **450 action measurements per arm**. These timings include the scheduler wait and final checks; they do not measure a different search algorithm.

| Median per action | L13 baseline | Background validation |
| --- | ---: | ---: |
| Snapshot preparation for detached validation | — | 11.461 ms |
| Background full validation | — | 33.033 ms |
| Main-thread validation/commit | 156.462 ms | 32.303 ms |
| Total delivery for the fixed action | 156.489 ms | 84.685 ms |
| Process CPU over that action | 156.250 ms | 78.125 ms |

The final uninstrumented real-search comparison contains **80 turns: 40 per arm**, five paired repetitions of eight deterministic Canal/Rail round-2/round-7 roots, with AB/BA ordering. The permitted root hashes match across each pair. Preparation, both actions, validation and dispatch remain visible in total delivery.

| Observed live-turn metric | L13 baseline | Background validation |
| --- | ---: | ---: |
| Median worst-frame interval | 238.620 ms | 122.187 ms |
| Median total delivery | 4,028.563 ms | 3,978.806 ms |
| Observed p95 total delivery | 4,176.359 ms | 4,198.471 ms |
| Maximum total delivery | 4,195.984 ms | 4,224.116 ms |
| Median Unity + native process CPU | 7,617.188 ms | 7,640.625 ms |

Median worst-frame time improves by **48.8%**; median total delivery improves by about **50 ms**. Aggregate CPU changes by about **+0.3%**. Tail delivery and some per-fixture delivery/CPU comparisons do not meet the original strict screens. One early-Canal fixture has variable early native completion, with baseline delivery ranging from roughly 2.75 to 4.16 seconds; its per-fixture CPU/delivery regression cannot be hidden by aggregate results. Five repetitions per fixture are an initial screen, not a population tail guarantee.

The user explicitly accepted retaining an optimization with good measured improvement without strictly meeting every numerical requirement. The [retention decision](../artifacts/rust-backend/20261026-l14-study-v2/optimization-retention-decision.json) records that exception. **The optimization remains enabled by default; failed numerical screens remain recorded as failed.** `build-rust-local-package.py --baseline-validation` retains the old call chain for future comparisons. Correctness requirements were not waived.

## Stage attribution and correctness

Diagnostic copies, separate from the timing builds, record correlated request/action identities, monotonic spans, execution threads, CPU, GC counts, request/reply bytes, serialization, pipe wait, deserialization, snapshot preparation, full validation and final mutation. Inclusive parent/child spans are not added together. The real traces show request serialization around 0.6–0.7 ms and reply parsing around 0.1 ms; the roughly 3.66-second pipe-read wait overlaps native work and is **not** attributed wholly to IPC.

Unity Mono does not expose `GetAllocatedBytesForCurrentThread`, and its process working-set properties returned zero. Those are missing measurements, not zero allocations or a memory pass. Heap deltas are explicitly not allocation counts. The external Win32 memory probe measures the owned native process separately.

The [Win32 memory run](../artifacts/rust-backend/20261026-l14-study-v2/native-memory/results.json) uses the same production native binary, four heterogeneous warm-up turns and 20 measured four-second turns. Peak working set was **328,921,088 bytes (313.68 MiB)** and post-retirement retained growth was **18,415,616 bytes (17.56 MiB)**, within the frozen 2 GiB / 32 MiB screens. This short repeated-turn test does not replace complete-game leak testing.

Recorded checks include:

- **1,436 fixed-reply checks**, including legal/invalid/incomplete/malformed paths, exact resulting-state parity after each action, queued cancellation, receipt reuse rejection and scheduler retirement.
- **845 Unity checks**, including all four 4/8/10/12-second settings, actual validation handoff cancellation, actor/revision changes, same-revision loading from the human checkpoint, action-two validation, history/menu replacement, worker failure and native liquidation without saves.
- **45 event/identity checks** comparing emitted event type, order, revision and complete semantic payload, plus all binding fields and rejection of worker-thread commit. Volatile event metadata is excluded explicitly; revision is retained.
- **65 checkpoint/liquidation checks**, including no save before human End, one checkpoint after End, duplicate-click protection, no AI/round/liquidation saves, failed-save retry, stale callbacks and compatibility with older partial-liquidation saves.
- **Four reporter tests** ensuring absent evidence cannot pass, tail maxima remain visible, and absolute/relative benefit conditions are both evaluated.
- Scoped binary/source preservation checks; the existing game assembly and all 48 embedded V4 profiles are preserved. The shared local save timing changes intentionally apply to V4 and NN gameplay.

## Original search controls

The benchmark tools retain raw failures and make missing gates explicit. A/B Masked search controls use the same Python search/evaluator with C# versus Rust simulation. Fixed-count Gumbel controls compare complete taped traces, and native component runs separate allocation instrumentation from ordinary timing. These measurements must not be relabeled as a matched timed Gumbel speedup.

All **12 fixed-trace Gumbel fixture/mode comparisons passed**, covering scripted replies and the pinned real ONNX model with two independent one-thread owners. Python's diagnostic wall interval includes simulator IPC and report construction; Rust's internal timer excludes file parsing and final trace serialization. Their diagnostic ratio cannot satisfy the matched implementation-speed gate.

The [native component run](../artifacts/rust-backend/20261026-l14-study-v2/native-components/results.json) completed **30 four-second cases**: 25 searched cases returned supported plans, five were forced-choice cases, and none used a fallback. Maximum native response-ready time was **3,638.27 ms**. The run completed 24,550 trajectories and 24,575 inference positions/batches. Recorded inference time was 83.60 seconds of 154.40 seconds of summed owner work across the runs. Time-to-leaf overlaps other component spans and is not added to that total. These are component measurements, not a strength result.

The legacy-control attempts also preserve setup failures. The L07 archive omitted a dependency directory needed by the frozen fast-host loader; the runner now checks and copies that directory against the archived runtime's exact host/dependency hashes. A subsequent attempt rejected the original `edge-shared-merchant-beer` root for link-tile conservation. L04 had already recorded this invalid synthetic root and its independently corrected C# capture. The corrected attempt uses those receipted captures for both arms, retains original inputs as rejection evidence, and does not relax native validation.

The [corrected legacy control](../artifacts/rust-backend/20261026-l14-study-v2/legacy-controls-v4/results.json) completed **80 valid turns, 40 per arm**, with the same Python Masked search, synchronous one-thread PyTorch evaluator, eight roots and counterbalanced five repetitions. Median completed Masked trajectories were 2 with C# simulation and 8 with Rust simulation under the four-second allowance. Those counts are comparable within this control; they are not compared with Gumbel trajectory counts or presented as a playing-strength gain.

## Bounded follow-up and further optimization

The user subsequently requested **short smoke tests with good coverage and no long sessions**. The [smoke protocol](../artifacts/rust-backend/20261026-l14-smoke/protocol.json) was recorded before the new measurements. It retains the g890 model, two one-thread sessions, K=8/c1=50/c2=0.1 and singleton inference. Matched timed turns still receive four seconds. There was no tournament or long full-game run. Historical thresholds, failed screens and original results remain intact.

The native profile identified repeated information construction as a substantial cost. `search/brass.rs` now borrows the immutable decision key already retained by the tree instead of copying its observation and every legal choice on each inspected node. Causal insolvency accounting uses the existing public observation in `ObservationMemory`, avoiding another full projection. That view has no private hand and is never sent to the policy. Rules, legal sets, feature tensors, NN weights, scheduling and validation guards are unchanged.

The [follow-up assessment](../artifacts/rust-backend/20261026-l14-smoke/assessment.json) records:

| Comparison | Observed result |
| --- | --- |
| Eight recorded full traces, including continuation/debt | Exactly equal before/after |
| Six fixed-work roots × two repetitions, AB/BA order | Plans, completed statistics and support snapshots exactly equal |
| Paired fixed-work time, searched cases | Median **6.9% faster** |
| Rust allocator bytes across six cases | **28.8% fewer** |
| Rust allocation calls across six cases | **36.4% fewer** |
| Six four-second cases before/after | All complete; five supported-search results and one forced result per arm |
| Current candidate Unity smoke suite | **818 checks passed** |
| Current candidate event/identity and human checkpoint checks | **45 + 65 passed** |
| Targeted native unit/search/session/cooperative/privacy/simulation tests | **45 tests passed**, test execution under three seconds total |

One of the ten searched fixed-work pairs was 0.2% slower; the other nine improved. No universal or statistical speedup is claimed. These fixed-work cases use 64 trajectories per owner and finish in fractions of a second; their separate 60-second failure watchdog is not a matched time allowance. In the four-second sample, response-ready times on the five searched roots changed from 2,435–2,744 ms to 2,047–2,728 ms. The frozen startup estimate can stop a completed quota before the deadline; these intervals are not a larger granted clock or proof of fully utilizing four seconds. Allocator counters are Rust allocation traffic, not retained memory, ORT's native allocator or Unity managed allocation totals.

The equivalent C#/Rust simulator-API smoke used six ordinary/staged roots and two counterbalanced repetitions. It compared the same sampled worlds, clones, legal choices and ordered complete-action steps, with exact four-seat views, context, chance and terminal results; only the transport UUID was excluded. Median sampled-world creation was **51.443 → 6.109 ms**, clone **18.982 → 10.213 ms**, legal-action API **0.608 → 0.564 ms**, and the complete staged step sequence **67.867 → 7.852 ms**. These boundaries include each adapter's IPC, projection and reconstruction; they are not bare language microbenchmarks. Raw traces and host pins are in [rules-api-01](../artifacts/rust-backend/20261026-l14-smoke/rules-api-01/results.json).

`tests/RustBackend/gumbel_session_reference.py` now supplies an independent scalar Python session/ensemble reference. Fixed-count root perturbations, completed-round snapshots, complete-prefix plans and pooled recommendations match native search using external ORT and PyTorch, within the pre-existing 2e-4 NN tolerance. Both sampled whole turns also replay action two without resetting their original deadline. The reference and executed source are pinned with the results. `tests/RustBackend/external_evaluator.py` uses the same checkpoint or export, one CPU thread, production Rust tensor encoding and normal result validation for G_ref/G_ref_ORT. External binary tensor framing belongs only to this research harness; deployed game framing is unchanged.

| Two-root timed smoke | Canal completed trajectories | Rail completed trajectories | Response/turn time |
| --- | ---: | ---: | --- |
| G_py, Python search + PyTorch | 43 | 36 | Whole turns 3,735 / 3,774 ms, including fresh action-two completion |
| G_ref, Rust search + external PyTorch | 785 | 936 | Complete plans ready at 3,622 / 3,627 ms |
| G_ref_ORT, Rust search + external ORT | 864 | 984 | Complete plans ready at 3,633 / 3,133 ms |
| G, Rust search + in-process ORT | 988 | 984 | Complete plans ready at 2,728 / 2,436 ms |

G_py includes Python feature encoding and simulator IPC. Its separately recorded startup estimate is measured from Python singleton trajectories; native controls retain their frozen L12 estimate. Dynamic remaining time changes admitted quotas. This small control sample supports implementation attribution and functional equivalence; it does not establish playing strength. The Python turns used legal bounded completion for their second actions after search stopped. They are not reported as fully search-supported two-action turns.

Two bounded NN/V4 comparisons used the admitted evaluation profile `bank-3dc65253a610191d66a5/pottery-flex-v3-masked-economy`, a Canal decision and a Rail round-8/turn-4 bonus root. All four returned legal complete actions; both near-terminal arms reached actual final settlement. NN responses took 2,831 / 353 ms; V4 caller times were 4,211 / 4,024 ms, so the V4 caller exceeded the requested four seconds and those overruns remain visible. This is a legality/settlement smoke, not a win-rate or noninferiority result. The existing 80-turn legacy-neural A/B control remains the historical legacy-search evidence. A first attempt correctly rejected a training-only V4 profile from the evaluation pool; the run then selected the already admitted evaluation profile, without changing admission policy.

The new native diagnostic executable and C# trace correlate wire request IDs. Native monotonic durations are subtracted from the containing C# frame round trip; clock origins are never subtracted across processes. On the short Unity trace, service time was 2,236.775 / 1,895.797 ms, with transport plus caller-dispatch residuals **0.215 / 0.383 ms**. Native queue, parsing, admission, pair setup, search/retirement, reply preparation, serialization and write durations are separate in the assessment. The write span can overlap the caller read, and the residual is not a one-way pipe latency. Snapshot preparation, background validation and main-thread guarded commit remain distinct C# spans.

The diagnostic Mono allocation callback was checked against [Mono's profiler API](https://github.com/mono/mono/blob/main/mono/metadata/profiler.h) and [object-size API](https://github.com/mono/mono/blob/main/mono/metadata/object.h). Its mandatory self-test allocated at least 100 objects / 102,400 payload bytes but observed zero callback events. The runtime has not enabled allocation instrumentation, so the counter is explicitly **unavailable**. The probe does not patch Mono or pretend heap deltas are allocation totals; it is not included in the deployable candidate.

The latest isolated package is [`20261026-l14-candidate-v5`](../artifacts/rust-backend/20261026-l14-candidate-v5/build-manifest.json). Protected IL/source scope and all V4 profiles pass preservation checks. The installed L13 package is unchanged. The test NN remains temporary and its utility-calibration receipt remains null.

## Remaining qualification

The bounded implementation follow-up is finished. **Full L14 qualification remains open** for two explicit limits:

1. Actual Unity managed allocation totals require a compatible allocation-enabled profiling runtime. The failed self-test is evidence of an unavailable measurement, not a zero-allocation result.
2. Statistical strength, rare failures and population latency tails are deferred under the user's short-test instruction. The historical tournament and failed numerical screens are preserved, not marked passed. No long study will be started automatically, and this smoke receipt does not authorize L15/lab promotion.

The plan, [frozen protocol](../artifacts/rust-backend/20261026-l14-study-v2/protocol.json), [save-policy amendment](../artifacts/rust-backend/20261026-l14-study-v2/save-policy-amendment.json) and raw receipts remain separate. The first protocol's mistaken K=16 was corrected in study v2 to the actual L13 K=8 before search measurements; its thresholds and earlier baseline validation evidence were preserved. No historical L13 qualification or final-model identity was rewritten.

The machine-readable [assessment](../artifacts/rust-backend/20261026-l14-study-v2/assessment.json) records `validation_optimization_retained: true`, original failed screens and pending gates. [Final build checks](../artifacts/rust-backend/20261026-l14-study-v2/retained-build-checks.json), [commands](../artifacts/rust-backend/20261026-l14-study-v2/commands.json) and [progress](../artifacts/rust-backend/20261026-l14-study-v2/progress.json) link the retained build to its verification and unchanged installed L13 files. Reproductions must use fresh output directories/labels to preserve historical evidence.

The latest [smoke assessment](../artifacts/rust-backend/20261026-l14-smoke/assessment.json) and [final checks](../artifacts/rust-backend/20261026-l14-smoke/final-checks.json) supersede that historical progress summary without altering it. `scripts/l14-smoke.py` runs separate bounded trace/count/allocation/timed/external/G_py/rules/quality groups; `scripts/report-l14-smoke.py` checks semantic equality and derives the measurements; `scripts/seal-l14-smoke.py` pins sources, candidate files, executed control sources and preservation checks. Use fresh labels and preserve failures. None of these scripts installs the candidate.
