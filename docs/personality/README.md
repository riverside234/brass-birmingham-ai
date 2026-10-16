# Post-Step-5 personalities: P0-P6

Consolidated on 15 October 2026. This directory is the single maintained location for the P0-P6 design, guides, required follow-up work and historical reviews. Original paths are navigation stubs only.

| Document | Contents |
| --- | --- |
| [Design](DESIGN.md#design) | Shared model/intent design, personality discovery, difficulty objectives and stage contracts |
| [P0-P3 training and discovery](TRAINING_P0_P3.md) | Handoff, goal-head pilot, search readiness/teachers, distillation and repertoire discovery |
| [P4-P6 qualification and deployment](QUALIFICATION_P4_P6.md) | Roster checks, difficulty calibration, export, laptop runtime and installation |
| [Deployment performance](DEPLOYMENT_PERFORMANCE.md) | Implemented cache/native-runtime improvements, 12-second v9/W352 measurements, limits and proposed next experiments |
| [Audit history](AUDIT_HISTORY.md) | Dated correctness reviews, deployment performance evidence and continuation notes |
| [Encoder reuse background and follow-up](#encoder-reuse-follow-up) | Completed shared-session integration, Step 5 background and remaining stage-level validation |

## Reading order and search use

| Stage | Guide | Search use |
| --- | --- | --- |
| P0 | [Handoff](TRAINING_P0_P3.md#p0-p1) | No search; validates and freezes the eligible source |
| P1 | [Goal head and joint PPO](TRAINING_P0_P3.md#p0-p1) | Direct policy collection; searched behavior is excluded from ordinary PPO |
| P2 | [Readiness, search and teachers](TRAINING_P0_P3.md#p2-p3) | Direct readiness collection, then searched probes/teachers and policy distillation |
| P3 | [Repertoire discovery](TRAINING_P0_P3.md#p2-p3-p3-inexpensive-discovery-first) | Direct discovery followed by searched-controller confirmation; descriptor fitting itself needs no search |
| P4 | [Roster qualification](QUALIFICATION_P4_P6.md#p4) | Searched member/mixture evaluation of the frozen shared model |
| P5 | [Difficulty calibration](QUALIFICATION_P4_P6.md#p5) | Evaluation under the declared tier search budgets |
| P6 | [Deployment](QUALIFICATION_P4_P6.md#p6) | Export itself needs no search; runtime decisions and search benchmarks do |

## Status and use of historical material

The consolidation preserves existing commands, design requirements, audit records and their original dates. Historical model sizes, Step 5 settings, source hashes and fixture timings describe their recorded versions; they are not refreshed measurements of the current model or lab run. Check the current implementation, resolved run manifests and applicable versioned receipts before executing a continuation.

Step 5 encoder reuse and personality `SearchSession` cross-lane sharing are implemented. Deployment workers now use fused native snapshots, with public observation data constructed once per snapshot. The [deployment performance record](DEPLOYMENT_PERFORMANCE.md) documents the completed local headless comparison and remaining work. Broad P0-P5 collection measurements, trained-model qualification and interactive Unity acceptance remain pending. This documentation update records existing evidence; it does not rerun benchmarks or alter code, configurations or checkpoints.

The recorded P2 accounting discrepancy also remains open: its teacher implementation uses a 90% cumulative wall allowance while the design specifies 50%. Current Step 5 settings do not automatically select or migrate P2's policy. Preserve that distinction until an explicit implementation decision resolves it; this reorganization does not change either value.

<a id="encoder-reuse-follow-up"></a>

## P0-P6: encoder reuse status and remaining performance work

Status, including the `20261016` deployment evidence: **Step 5 and personality shared-session reuse implemented; local deployment throughput measured; broader stage and release validation pending.** Shared caches, fused native transport and snapshot-local public-view reuse are covered by the [implementation and measurement record](DEPLOYMENT_PERFORMANCE.md#implemented). Reuse these mechanisms before adding more caching. Experimental worker/batch settings and W352 are not new production defaults.

<a id="encoder-reuse-follow-up-motivation-and-measurement-scope"></a>

### Motivation and measurement scope

Start with the current [Step 5 encoder reuse report](../../ai-research/ENCODER_REUSE_REPORT.md). The earlier [grouped projection report](../../ai-research/GROUPED_PROJECTION_REPORT.md) recorded `new_encoder` / `persistent_miss` / `warm_hit` medians of 8.603 / 4.722 / 0.860 ms on v7. Those are historical motivation; current-v9 measurements appear below. The cases are not production cache-hit rates, and timings across different feature contracts/runtime conditions are not a controlled speed comparison.

The remaining follow-up is to verify encoder ownership and lifetime in each stage's actual collection path, reuse completed inputs only when their full identity matches, and measure the resulting work. Shared `SearchSession` integration is already present; do not duplicate it. Its local deployment results do not establish throughput for every personality stage.

<a id="step5-encoder-reference"></a>

### Implemented Step 5 techniques: background reference

The 15 October implementation and evidence below belong to the compact grouped v9 revision recorded as `e794a86e13b891d56fcb21a8a7580a8af1c754bc54065a8b636dd5b04f6792f8`. They are implementation examples for the personality follow-up, not a requirement to import that revision's teacher budgets.

| Existing optimization | Implementation reference | What to reuse or verify in personality paths |
| --- | --- | --- |
| One encoder per persistent reply endpoint | `endpoint_encoder`, `RemoteEvaluator` in [recovery/inference.py](../../src/brass_ai/recovery/inference.py) | Replacing a frozen-model evaluator on the same endpoint retains the encoder. Avoid constructing one per model/request; verify actual session and endpoint lifetimes. |
| Shared bounded cache of completed inputs | `EncodingCache`, `PreparedEncoder` in [recovery/encoding.py](../../src/brass_ai/recovery/encoding.py); `share_encoder_cache` in inference | Coordinator and private search lanes can reuse the same completed feature arrays. Step 5 binds the cache before evaluator construction; its capacity is 256 entries per bound reply lane, including 2,304 for a nine-channel teacher actor. Size against actual personality channels rather than copying that number. |
| Private builders with short cache locks | `EncodingCache.get/put`, lane-owned `PreparedEncoder` | Mutable categorical/derived builders remain private. Locks cover lookup/publication, not feature computation. Unrelated misses can run concurrently; simultaneous identical misses may still compute twice. Preserve deadline behavior rather than adding an unmeasured global lock or single-flight wait. |
| Content identity and read-only publication | `PreparedEncoder.encode` | The key binds derived schema, complete permitted observation/ordered choices, style and intent. Published state/action/seat arrays are read-only. Cache hits still check deadlines/cancellation. Input features can be shared across compatible models; predictions and remaining-utility budgets remain separately attributed. |
| Replay and game-level encoder reuse | `policy_encoder`, `run_game`, `_actor` in [recovery/execution.py](../../src/brass_ai/recovery/execution.py) | Resolve a search policy's `base.evaluator` as well as a direct policy's evaluator, so replay can reuse the encoded root. Reuse the lane encoder across games and avoid allocating an unused fallback encoder. Inspect equivalent personality replay paths before copying this integration. |
| Count shared work once | Per-game `search.encoding_cache` in execution records | Record `warm_hit`, `miss`, `evictions`, `retained` and `capacity` once for the shared actor lane, not once per seat/model. The existing detailed records do not add these fields to compact history; historical cached collections must not fabricate missing measurements. |

**What is implemented in personality search:** [PersonalityRemote](../../src/brass_ai/personality/search/inference.py) inherits endpoint-owned encoders, and [SearchSession](../../src/brass_ai/personality/search/engine.py) now calls `share_encoder_cache` before constructing coordinator, worker and frozen-opponent evaluators. Completed read-only features share one bounded LRU; builders, reply lanes, trees and predictions retain their own ownership. Capacity is 256 entries per acquired lane: with one frozen parent, two/three search workers use five/seven lanes, or 1,280/1,792 entries. Direct mode uses one lane. P3/P4 collection, P5 evaluation and [P6 runtime](../../src/brass_ai/personality/laptop/runtime.py) use the shared session code, but their separate collection and qualification workloads still need representative measurements. [The local deployment test](DEPLOYMENT_PERFORMANCE.md#measurements) measured approximately 5–7% within-turn warm hits with full-feature LRUs cleared at each root; it does not imply the 90% synthetic reuse rate below.

**Existing measurements, not rerun for this note:** the [verification receipt](../../artifacts/feature-engineering/encoder-reuse-v9-20261015/verification.json) records 41 distinct focused checks passing. They include content/seat/order/context identity, concurrent misses, bounded eviction, stopped encoding, replay reuse, separate model/budget predictions and exact compatible continuation. See [the benchmark receipt](../../artifacts/feature-engineering/encoder-reuse-v9-20261015/benchmark/report.json) and [benchmark driver](../../tests/FeatureEngineering/benchmark_encoder_reuse.py).

| Current-v9 encoding case | Samples | Median / p95 ms |
| --- | ---: | ---: |
| `new_encoder` | 168 | 18.549 / 44.167 |
| `persistent_miss` | 168 | 7.567 / 27.655 |
| `warm_hit` | 168 | 1.570 / 2.823 |

The controlled coordinator + eight worker roots + replay workload measured median encoding time **69.223 -> 21.426 ms** and p95 **325.222 -> 51.035 ms** (56 samples per arm). The shared-cache arm recorded **126 hits / 140 calls** per 14-position block, versus zero hits for emulated former local ownership. Both arms used the same current feature implementation and checked exact encoding parity. This Windows CPU benchmark used two tensor threads, eight worker threads and four alternating blocks; it intentionally repeated roots. The 3.23x component speedup and 90% hit rate do not establish whole-search/game speedup, production hit rates, label yield or playing strength. No full native search or GPU training was benchmarked by that comparison.

Use these mechanisms and focused checks as the starting point for the remaining work below. Keep the then-current model/features and each stage's own search recipe; the separate Step 5 clock/worker changes in the reference report are not part of this personality documentation request.

<a id="encoder-reuse-follow-up-required-coverage-by-stage"></a>

### Required coverage by stage

| Stage | Search role | Required follow-up |
| --- | --- | --- |
| P0 handoff | No search | Carry the correct feature/layout/source identities into downstream artifacts; this work must not mutate the completed source run. |
| P1 goal warm-up and joint PPO | Direct policy collection | Check encoder reuse in direct collection as well as fresh learner transitions. Preserve behavior-probability verification and the exclusion of searched actions from ordinary PPO. Do not introduce search into P1. |
| P2 intent search and teachers | Search probes, reserved teacher collection and distillation inputs; readiness collection is direct | Shared-session construction is implemented. Measure actual teacher/root/continuation reuse and label yield, plus direct readiness collection. Preserve independent mutable worker state and current acceptance rules. |
| P3 repertoire discovery | Direct discovery, then searched-controller confirmation | Cover both collection modes, especially repeated inference during searched confirmation. Descriptor fitting itself needs no search; preserve its split and provenance rules. |
| P4 roster validation | Frozen shared model with searched member/mixture panels | Verify reuse across repeated requests without mixing seats, controller inputs or panel jobs. Preserve tested member identities and qualification evidence. |
| P5 difficulty calibration | Search under each tier's declared budget | Measure actual collection throughput and completed work for each tier. Keep its calibrated clocks, worker settings and held-out evaluation rules intact. |
| P6 deployment | Export itself needs no search; runtime decisions and search benchmarks do | Shared caches, fused hosts, public-view reuse and a self-contained packaged native host are tested. Extend the two-position headless comparison to trained models, all tiers and interactive packaged Unity use; include startup, saves, cancellation and new games. |

The paths above are required implementation/review coverage, not a claim that each contains a defect. If a path already reuses encoders correctly, document that finding and validate it rather than adding redundant caching.

<a id="encoder-reuse-follow-up-correctness-requirements"></a>

### Correctness requirements

- Preserve the current feature semantics, legal-action order and masks, observer-relative encoding, hidden-information boundaries, and style/intent inputs. Cache identity must cover every input that affects the cached value, including observation/history, choices and applicable feature/layout versions.
- Distinguish feature caching from neural-output caching. Feature tensors may be reusable across compatible weight updates; cached predictions must remain bound to the exact model and all relevant controller/reward/context inputs. Verify invalidation at model, schema, session and game transitions as applicable.
- Keep mutable search trees, observations, private reply channels and per-seat intent isolated. Bound cache memory; avoid stale results, cross-game contamination and races when extending an encoder's lifetime.
- Preserve shared whole-turn deadlines, cancellation, worker retirement, fresh-source/model checks, teacher acceptance criteria, utility calibration and PPO attribution. A cache hit must still respect the request's cancellation/deadline checks.
- Treat P2-P6 search settings as their own versioned recipes. Do not silently copy Step 5 worker counts, phase clocks or cumulative wall-time settings into them. Encoder performance work must not relax evidence or qualification requirements.

<a id="encoder-reuse-follow-up-required-validation-and-reporting-when-implemented"></a>

### Validation already recorded and required for further changes

The [deployment record](DEPLOYMENT_PERFORMANCE.md#validation) includes 33 focused checks, bitwise feature parity, ONNX/batch parity and eight final whole-turn comparisons. The requirements below still apply to unmeasured stage paths and future optimizations; they do not imply that the completed shared-session work must be implemented again.

Use focused regressions and a small isolated benchmark; a long training or qualification campaign is not needed to verify the optimization.

1. Compare fresh and reused encoding on identical permitted observations and candidate ordering, including changed seats, history, choices, style/intent and feature versions. Verify tensor/output parity and correct misses/invalidation. Check cancellation and bounded memory.
2. Repeat the `new_encoder`, `persistent_miss` and `warm_hit` cases on the current target, then measure representative actual P2/P3/P4/P5/P6 request paths. Include P1 direct collection where affected. Report hardware, threads/workers, model/layout identity, corpus, warm-up, sample count and median/p95 timings.
3. Record encoder constructions, eligible cache lookups, hits/misses, encoding time and cache memory. Report production-like hit rates separately from the deliberately warm benchmark case. Include end-to-end collection/request time and completed searches/simulations, coverage, qualified labels, timeouts and overruns where applicable.
4. Compare search under the same model, permitted roots/seeds, worker counts and whole-turn budgets. Preserve search logic; faster timed execution may complete more work, so require encoding/inference parity rather than assuming identical timed-search choices. Keep simulation counts distinct from action/continuation support and label quality.
5. Report implemented paths, paths already correct, remaining limitations and any measured speedup. Do not infer stronger play or qualified personalities from cache hits, shorter encoding latency or a smoke test. Follow existing source/version rules for newly generated artifacts; preserve old immutable artifacts.

Completion requires evidence for the applicable personality paths and their measured results. Shared-session integration and the bounded deployment experiment are completed milestones; trained-model and stage-level validation remain open. Further performance hypotheses and their acceptance checks are recorded in [possible next improvements](DEPLOYMENT_PERFORMANCE.md#further-improvements).
