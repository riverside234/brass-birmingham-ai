# Adaptive Rust search: local implementation and evaluation

**Deployment update:** the user subsequently authorized installation of cache-only `search_v2_focus_only`, keeping V205 disabled. The installed-player 4s/K8 smoke passed; see the [installation report](SEARCH_V2_INSTALLATION_REPORT.md). The measurements and production-control statements below retain their original pre-deployment scope.

Date: 2026-10-28. Candidate: **`search_v2_focus_only`**, policy `adaptive-focus/2.0.0`.

**Retain the cache-only candidate for further local qualification at all three settings. Keep installed search_v1 as the production control.** The candidate finds useful second-action search windows and returns legal whole-turn plans. It usually consumes more of the existing allowance, with more trajectory completions, higher CPU use and later native delivery. This is not a demonstrated engine speedup or win-rate improvement.

This implements the simpler option permitted by section 9 of the [plan](SEARCH_V2_IMPLEMENTATION_EVALUATION_PLAN.md). **The measurements in this report contain no statistical subtree reuse.** The later [V205 follow-up](SEARCH_V205_EVALUATION_REPORT.md) implements and tests that experiment at 4s/K8, then retains cache-only as the selected candidate. The final NN, lab training and production promotion remain outside this study. All inference uses the accepted temporary g890 export, ONNX Runtime CPU, two independent sessions and one compute thread per session.

## Implemented behavior

- The original root tournament remains on its owner thread, with resumable completed-round observations. No-switch execution keeps the same random draws, quota, tree, completed statistics and partial work.
- A switch needs the same **complete** first-action proof at two observations with both owner watermarks advancing, minimum support two, an ordinary two-action turn and an affordable successor tournament for both owners. The conservative estimate is the maximum of startup cost and the last eight trajectory costs; the preparation estimate is 20 ms. Actual remaining time is checked again after preparation.
- There is at most one preparation attempt per request and one accepted switch per turn. Both old owners retire before either begins projected-root search. A drained wave can publish the last completed-round watermark; later partial work never becomes eligible switch evidence.
- The permitted successor is reconstructed in Rust, without exporting sampled opponents' cards or a future deck. The supported paths exclude Scout, liquidation, chance transitions and turn/actor changes. Hidden-card allocation counts/pool and the reconstructed information state must match exactly.
- An independently validated two-action incumbent is retained. Missing supported action two uses explicitly labeled bounded legal completion. Failed projection resumes the original tournament; insufficient focused support retains the incumbent. A rejected composed plan retries the incumbent under the same delivery deadline. Runtime corruption, expired delivery and cancellation still fail closed.
- Focus uses a fresh root tournament and fresh counts, with a bounded exact-key NN cache moved to a temporary session on each existing compute owner. The original live session's revision/recall are not advanced by hypothetical search.
- `brass-composite-plan/2.0.0` records action ordinal, source phase, owner, watermark and conditional support. Joint support is null; changed composite actions cannot acquire fabricated joint-prefix visits. The owner-local session epoch and composite phase epoch are separate domains; original and projected revisions also remain distinct.
- The final local reply carries `brass-local-focus/2.0.0` under the original ticket. Unity compares the actual successor after action one with the entire projected permitted view/context/ledger/recall, then performs the existing fresh detached validation and guarded main-thread commit for action two. A mismatch clears the continuation and replans on the original clock.

Rust counts semantic choices and Unity counts game events. Only newly created revision/recall stamps are translated for the permitted-state comparison. The actual instance, generation, request, actor, event revision, cancellation and one-use validation receipt remain authoritative. A hypothetical revision never authorizes a live commit.

Code anchors: [core focus policy](../native/brass-ai-core/src/search/session/focus.rs), [projection](../native/brass-ai-core/src/search/brass.rs), [coordinator](../native/brass-ai-worker/src/focus.rs), [persistent owners](../native/brass-ai-worker/src/pair.rs), [actual-state guard](../src/BrassLocalAI/Native/NativeInput.cs), [Unity bridge](../src/BrassLocalAI/Unity/LocalTurnBridge.cs).

## Frozen experiment and deviations

The [control manifest](../artifacts/search-v2/20261028/control/manifest.json) pins the installed 1.4.38 package, managed assemblies, model and pre-edit sources. The control executable was rebuilt offline from those frozen sources, not from the candidate checkout. [Its build receipt](../artifacts/search-v2/20261028/control/rebuilt-control.json) records that provenance.

The [original protocol](../configs/search/search-v2-study.json) freezes eight roots, four development/four held-out partitions, three settings, seed pair, alternating A/B order and the candidate binary/source snapshot. The [development decision](../artifacts/search-v2/20261028/development-decision.json) precedes held-out execution. Gate/cost/model settings were not tuned on held-out results. Each A/B batch has its own 60-second watchdog; arms run sequentially with equal count-eight warmups and fresh per-turn caches. No long game or tournament was run.

The old L02 multi-sale fixture failed strict physical inventory admission. Correcting only link inventory exposed a second merchant-inventory defect. Both failed variants are preserved, and no search correctness check was weakened. The successful [fixture amendment](../configs/search/search-v2-fixture-correction-v2.json) uses the existing independent C# capture from `20261024-l04/regressions-v2/regressions.json`, retaining the two Gloucester cotton destinations. Its only additional study alteration is round 2/two remaining actions. Both binaries passed a short admission check before corrected timing. This root is explicitly an amended authored case, not an untouched held-out game trace.

The main successful matrix has **48 timed trials**, plus **two no-switch trials**: 328 seconds of admitted allowance, split into short batches. The six planned full-statistical-reuse ablations were omitted because that variant had not yet been built; the later V205 follow-up has its own 4s/K8-only protocol. The earlier three-trial gate pilot, rejected inputs, count-based correctness checks and isolated Unity smokes are recorded separately. First-case package preparation overlapped the first 4s development pair; it is retained and flagged, not silently replaced or used as evidence of negative no-switch overhead.

## Native results

Paired medians across all eight roots, including the no-switch result. These are trajectory completions, **not equivalent units of game work**: second-root trajectories start at a different state. Old visits contribute zero to the counts.

| Setting | Switched | Median change in new trajectory completions | Median paired native response change | Median process CPU change | Largest observed response |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4s / K8 | 8/8 | +29.9% | +403 ms | +19.3% | 3,043 ms |
| 8s / K8 | 7/8 | +27.4% | +873 ms | +19.4% | 6,062 ms |
| 8s / K16 | 8/8 | +22.5% | +435 ms | +11.1% | 5,934 ms |

Process CPU includes startup and the equal warmup; it is not per-session CPU or search-only CPU. The first snapshot and Unity costs are excluded from native response timing and measured separately below. The largest native process peak was **489.4 MiB**. No statistical visits/nodes were retained across roots. Exact cache hits exist, but these counters do not separately attribute every root-cache hit versus within-phase hits; no isolated cache-speedup claim is made.

Median new action-two-root completions were 625, 1,202 and 1,396 respectively (two owners combined; zero for the no-switch row). Requested K stayed fixed. Effective second-root K was 8 in the K8 settings and 8 or 16 in the K16 setting. Largest measured projection cost was 5.16 ms; the largest retirement/preparation handshake was 12.51 ms. These are observed maxima in a small sample, not tail guarantees.

All **50 successful native rows returned two complete actions**; the independent C# replay validated all 100 actions. Thus none of these native rows hides an unmeasured exceptional live replan. `rail-double/8s_k8` retired an owner before switching and correctly completed the original-root plan. No incumbent fallback was needed by the successful matrix; fallback itself is covered by deterministic checks.

The timed no-switch diagnostic on the Rail development root added 14.3 ms and 2.4% process CPU relative to its control. The first Canal diagnostic was faster than its flagged control, so it cannot establish negative overhead. More importantly, the fixed-count comparison against the frozen executable matched action proofs, perturbations, counts, returns, watermarks and eligible plans on both diagnostic roots. [Raw analysis and every paired row](../artifacts/search-v2/20261028/analysis.json) retain the inputs/outputs and receipts.

## Decision observations and limits

First actions matched in 23 of 24 paired cases. Seven whole-turn paths differed:

- Dense Canal, all settings: the same two industries were sold in the opposite order, with the same two beer supplies. No strength gain is inferred from ordering alone.
- Multi-sale, all settings: the candidate sold both cotton mills using the two merchant beers and their development bonuses; the control spent action two on paid development. Both full paths passed the independent C# rules. This is a useful immediate-economic alternative, while card retention and longer-term value still prevent a general optimality claim.
- Rail double-link, 8s/K16: the candidate reversed development/network action order and discarded a different card. The links and development choices remained legal. Its longer-term value is unresolved.

The [independent C# outcome capture](../artifacts/search-v2/20261028/tactical-outcomes/result.json) confirms the multi-sale candidate ends with £13 versus £7, income level +5 versus 0, and both cotton mills flipped versus neither, at all three settings. The remaining mat development is the same. Different retained cards and spent merchant beer remain relevant future tradeoffs. The Rail double-link alternatives have equal immediate money, income, VP and remaining link tiles; card choice is still different.

There is no independent win-rate or general noninferiority result, and no g890-value proxy is presented as one. The short evidence supports keeping the engineering candidate available; **production playing strength remains inconclusive**. Statistical reuse should be added only as a separately qualified experiment with a demonstrated benefit over this version.

## Correctness and local delivery evidence

- Native tests cover fake-clock affordability, full-path stability, both-owner watermark advancement, no-switch exact tournament preservation, deadline/cancellation publication rules, complete proofs, stale identities, cache/key equality, collisions and retired/foreign handles.
- Checks include 70 core tests in [the delivery log](../artifacts/search-v2/20261028/correctness-delivery.log), the worker fallback/coordination tests, and [12 Python policy/session-boundary tests](../artifacts/search-v2/20261028/python-checks.log). The first composite test fixture omitted admission's initial own-hand recall; its retained failing log was followed by a corrected fixture and passing test. No runtime check was relaxed.
- The projected Pass/Loan/Develop/Build transitions are checked against all 24 allocations of a tractable hidden-burn fixture, not a few random seeds. General applicability is restricted by the structural certificate and action allowlist.
- [Frozen-binary and cancellation integration](../artifacts/search-v2/20261028/native-integration-v2/result.json): 67 assertions, including cancellation immediately after original owners retire, blocked admission while retiring, no result delivery after cancellation and a successful subsequent new turn.
- [Independent C# replay](../artifacts/search-v2/20261028/boundary-all/result.json): 257 assertions across 50 plans. Changed cash, hand, actor, revision, history, cost ledger and recall reject projected continuations. Its imported-state helper adds only the confirmed Finish history marker that the real event adapter reconstructs; the Unity tests use actual events.
- The isolated Unity smoke runs the real native process, scheduler, detached validator and main-thread bridge. It checks one request for focused plans, both commits, no action-time save, exactly one save at completed AI End Turn and loading that next-turn boundary. Separate cancellation/load cases interrupt between actions. The installed saves and game files are not test targets.

Host timing and final lifecycle receipts are summarized in the generated [Unity analysis](../artifacts/search-v2/20261028/unity-analysis.json). Profiled builds are diagnostic copies only. Snapshot/event projection, IPC serialization/parse/wait, detached validation, successor comparison and main-thread guarded application have separate spans; nested spans must not be added as disjoint costs. The blocking native round trip includes search and is not mislabeled as IPC overhead. Mono did not expose reliable native peak memory here; null is unavailable, not zero.

One short matched Unity turn per setting/arm, using the same confirmed Canal fixture. Times below are milliseconds. Snapshot is the live turn's search-input captures; validation and commit are totals for both actions. Cold startup/warmup is separate in the raw receipt.

| Setting / arm | Delivery | Snapshot | Detached validation | Duplicate full validation | IPC serialize + parse | Main-thread guard + apply | Successor comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4s/K8 control | 3,678 | 33.7 | 144.3 | 0 | 2.2 | 57.8 | — |
| 4s/K8 focus | 3,169 | 22.2 | 77.6 | 0 | 18.1 | 66.4 | 1.6 |
| 8s/K8 control | 5,852 | 23.7 | 91.3 | 0 | 2.2 | 36.9 | — |
| 8s/K8 focus | 5,757 | 23.2 | 76.4 | 0 | 18.1 | 69.2 | 2.0 |
| 8s/K16 control | 5,384 | 24.9 | 88.5 | 0 | 1.6 | 35.0 | — |
| 8s/K16 focus | 5,625 | 23.4 | 76.8 | 0 | 18.2 | 68.6 | 1.8 |

All six used one request and committed both actions. The full permitted successor adds visible IPC payload/parse cost, while its equality check was about 2 ms. GC and presentation affect main-thread spans. These single observations do not establish a host-wide latency improvement; they demonstrate that the candidate still completes through the real delivery path within the requested allowance. Both additional between-action cancel/load runs actually switched to focus, committed action one only, retired work and retained the prior completed-turn checkpoint.

## Plan disposition and reproduction

| Step | Disposition |
| --- | --- |
| V201 | Frozen installed/control sources, model, policy and rebuilt executable; fixture amendment preserved separately |
| V202 | Scalar fake-clock/checkpoint hooks, resumable owners, recorded-count parity and timed no-switch diagnostic |
| V203 | Certified projected root, complete incumbent, fresh second-root statistics and exact NN cache |
| V204 | Two-owner coordination, affordability, bounded retirement, composite evidence and useful-window measurements |
| V205 | Deferred during this original study; subsequently [implemented and evaluated at 4s/K8](SEARCH_V205_EVALUATION_REPORT.md), with cache-only retained after comparison |
| V206 | Actual-successor guard, detached validation and bounded local delivery/lifecycle checks |
| V207 | 48 paired rows plus two no-switch rows; correctness/tactical/latency/resource evidence and explicit limitations |

Use `scripts/search-v2-tools.py build` for the pinned offline native build and `test --label <new-label>` for tests. `scripts/build-search-v2-managed.py --output artifacts/search-v2/<new-directory>` creates an isolated candidate from the sealed 1.4.38 runtime; it refuses to overwrite an existing candidate. `scripts/benchmark-search-v2.py run --batch <id>` replays one predeclared batch; use `--protocol configs/search/search-v2-fixture-correction-v2.json` for the corrected multi-sale batches. Outputs refuse overwrite. The scalar policy tests are `tests/RustBackend/test_search_v2.py`.

`scripts/verify-search-v2-native.py`, `scripts/verify-search-v2-delivery.py` and `scripts/smoke-search-v2-unity.py` retain independent correctness/delivery receipts. `scripts/summarize-search-v2.py` regenerates the derived native summary. Experiment artifacts are under `artifacts/search-v2/20261028/`; source and binary pins travel with each frozen build.

Remaining limits before any production promotion: broader independent decision-quality evidence; per-session CPU attribution and broader host latency sampling; qualification of any later model or retained-statistics variant; and a separate scoped installation/rollback decision. Installed 4/8/10/12-second settings, K mapping, saved AI choices, table-family restrictions, mixed per-seat difficulties and completed-turn saving remain on the production release.

[Installation verification](../artifacts/search-v2/20261028/installed-unchanged.json) confirms all 216 pinned installed package/managed files and the installation receipt are unchanged.

The final source audit also clears a retired coordinator on a rejected composition and makes successful incumbent delivery replace rejected focused evidence. These are exception-path bookkeeping changes after the timing freeze, not gate/search-policy tuning. The frozen timing build remains preserved; final-build parity, cancellation and a short Unity delivery check are recorded separately. Native final-build sources and package are in `artifacts/search-v2/20261028/final-candidate/`; do not substitute its binary into the frozen timing receipts.

Final receipts: [worker tests](../artifacts/search-v2/20261028/correctness-final-audit.log), [frozen-control parity and cancellation](../artifacts/search-v2/20261028/native-integration-final/result.json), [Unity delivery](../artifacts/search-v2/20261028/unity-final-audit/result.json), and [experimental qualification summary](../artifacts/search-v2/20261028/qualification.json).
