# V205: statistical descendant reuse at 4s/K8

**Deployment update:** after this study, the user requested installation of the selected cache-only implementation. It is now installed with V205 disabled; the short actual-player smoke passed. See the [installation report](SEARCH_V2_INSTALLATION_REPORT.md). Statements below about the unchanged installation describe the study's pre-deployment state.

Date: **2026-10-28**. Decision: **keep cache-only refinement as the selected candidate; do not enable V205**. V205 is implemented and passes the bounded correctness checks, but this smoke study does not show enough overall benefit to retain it in the selected search configuration. Its opt-in implementation and frozen candidate remain available for reproducibility. The installed game remains unchanged.

This is a separate follow-up to the [search-v2 plan](SEARCH_V2_IMPLEMENTATION_EVALUATION_PLAN.md) and [original cache-only report](SEARCH_V2_EVALUATION_REPORT.md). It does not replace or relabel their measurements. The temporary g890 model, ONNX Runtime CPU, two independent sessions and one compute thread per session are unchanged.

## Implementation and statistical contract

`search_v2` enables policy `descendant-reuse/1.0.0`. `search_v2_focus_only` remains the selected method and allocates no retention journal or parent map. The adaptive gate, affordability estimate, root K, NN cache policy, root draws, completion/fallback policy and Unity guards are the same in both arms.

- Each owner retains its existing `slotmap` arena. A bounded first-expansion parent forest proves ancestry through the entire selected first-action path. Semantic equality checks include legal ordering and actor; hashes alone never authorize reuse. Missing alternate ancestry produces a conservative miss.
- A journal records the first pre-update N/W of each touched edge since the last eligible completed round. On an accepted switch it restores those exact values, removes later expansions and prunes siblings. It does not clone the whole tree or undo sums by floating-point subtraction. Failed preparation before acceptance still resumes the original tournament untouched.
- Each owner independently repeats the permitted-transition certificate against its own original root. Projected view, own-hand recall, feature/reward context, utility ledger and conditional hidden-card allocation must match. Recorded-world tapes are excluded. New trajectories resample hidden worlds; no determinized hand is promoted to known information.
- The promoted root keeps compatible NN priors/value, with **N/W reset to zero**, fresh root perturbations/tournament, fresh plan-support ledger and fresh eligible watermark. Descendant N/W remain node-relative. Old edge contributions are reported separately and never count as newly completed trajectories or joint support.
- The original clock and live identity remain authoritative. Cancellation or pruning cutoff discards the moved arena and cannot publish a new root. A miss uses cache-only refinement on the remaining allowance. The diagnostic depth limit restarts at the new root; it is not a truncated-value horizon or discount.
- Ownership does not cross compute threads. Original live recall/revision do not advance during hypothetical search. Actual action-two state comparison, detached validation, final commit guards and completed-turn saves are unchanged.

Code: [retention/matching](../native/brass-ai-core/src/search/session/reuse.rs), [arena/journal](../native/brass-ai-core/src/search/tree.rs), [owner integration](../native/brass-ai-worker/src/pair.rs). The independent [scalar reference](../tests/RustBackend/gumbel_reuse_reference.py) reconstructs eligible statistics from completed trajectories, using full copies deliberately unlike the production journal. Its [pinned fixture](../tests/GumbelISMCTS/fixtures/reuse-v1.json) is checked by native tests.

## Frozen short protocol

Only **4,000 ms / requested K8** was timed. Effective second-root K was eight for both owners in every trial. The [protocol](../configs/search/search-v205-smoke.json) froze six coverage pairs plus two diagnostic second-seed pairs: **16 timed trials, 64 seconds of admitted search allowance**. Each pair ran independently with a 60-second watchdog, sequential arms, an identical count-eight warmup and fresh per-turn trees/caches. No long game, tournament, parameter sweep, 8s/10s/12s evaluation or lab work was run.

Both arms used the same release executable, selected by admitted method. Builds and package preparation finished before timing. Gate/reuse/model settings were not adjusted after results. Fixtures come from the earlier inspected corpus, including its already repaired multi-sale fixture; these are coverage cases, not newly blinded held-out examples. Main seeds were search 7/world 890; predeclared repeats used search 19/world 891 and reversed the initial diagnostic order.

All eight pairs switched at the same completed-round watermarks, selected the same complete first action and returned identical complete two-action paths. Later partial work was excluded from retention. Thus no observed plan or switch-point change is being presented as a reuse speedup. Deterministic fixed-count owner tests also cover both-hit, one-hit and neither-hit cases with fresh root counts under a common projected boundary.

## Results versus cache-only

Negative latency means V205 responded earlier. Counts include only newly completed trajectories from the two phases; inherited edge contributions are excluded.

| Pair | New completions | Native response difference | Process CPU difference | Peak memory difference |
| --- | ---: | ---: | ---: | ---: |
| Early Canal (`brass-003`) | -0.58% | -94.5 ms | -1.26% | +19.9 MiB |
| Loan/build | -0.16% | -5.6 ms | 0.00% | +16.7 MiB |
| Dense Canal | -0.23% | +21.2 ms | +0.32% | +12.3 MiB |
| Rail double-link | -0.21% | +81.2 ms | -0.80% | +1.2 MiB |
| Dense Rail | -0.47% | +17.1 ms | 0.00% | +13.5 MiB |
| Multi-sale | -0.31% | +75.1 ms | +3.09% | +14.3 MiB |
| **Six-case paired median** | **-0.27%** | **+19.2 ms** | **0.00%** | **+13.9 MiB** |
| Early Canal, second seed | -0.33% | -151.8 ms | -2.43% | +17.6 MiB |
| Dense Rail, second seed | -0.16% | -21.0 ms | -2.42% | +15.0 MiB |

Both owners hit compatible retained subtrees in all eight V205 trials: **16/16 hits**. Each retained 55–137 nodes and 58–519 old descendant edge contributions. Matching, certification, rollback and pruning together took **14.7–22.3 ms per owner**. These are actual reuse hits; the lack of overall benefit is not a failure to activate the feature.

Second-root new completions fell by a median **0.71%** in the primary cases. Across all eight pairs, cache-only performed 4,794 second-root trajectories and 4,778 NN forwards; V205 performed 4,755 trajectories and 4,754 forwards. Reuse bypasses retained nodes and reaches further leaves, but still normally evaluates one new leaf per trajectory. The smaller raw forward count is accompanied by fewer new trajectories, so it is not evidence of a useful per-trajectory inference saving. Deeper visits alone do not establish stronger play.

The repeat confirms a latency benefit in the early Canal example. Dense Rail changed from +17 ms to -21 ms, while its memory increase persisted. No returned decision changed in either repeat. This supports reporting a localized latency improvement, not enabling a more complex method for the whole setting. A rule selecting reuse only on observed winning fixtures would require a separate experiment and was not added.

Largest native responses were 3,070.6 ms for cache-only and 2,918.8 ms for V205. Largest process peaks were 245.9 and 264.4 MiB respectively. All completed inside the same 4s allowance; no fallback or extra request was needed by the native matrix. Process CPU and peak memory include startup, warmup and serialization; per-session CPU was not separately measured. These small observed samples are not population tail estimates or strength rankings.

[Raw rows, NN/stage counters and paired analysis](../artifacts/search-v2/20261028/v205/analysis.json) retain every case, including both repeat seeds.

## Correctness and actual delivery

- **88 native test executions passed**: 76 core, six worker-library, five worker-binary and one compile-fail documentation test. Coverage includes scalar reference parity, exact completed-watermark rollback, full semantic/hash-collision checks, foreign/stale handles, independent hit/miss owners, incompatible value/sampler/information, depth accounting, late backup rollback and cancellation during pruning. The existing exhaustive 24-hidden-allocation projection check remains in this suite. [Log](../artifacts/search-v2/20261028/v205-final-tests.log)
- Three Python gate tests and the scalar reuse fixture reproduction passed. Tracking with switching disabled preserves the baseline count/draw/return/plan trace.
- **67 frozen-control parity and cancellation checks** passed with the release executable, including cancellation after retirement and a subsequent fresh turn. [Receipt](../artifacts/search-v2/20261028/v205-native-checks/result.json)
- Independent C# replay passed **160 checks across all 16 plans / 32 actions**, including actual projected-successor comparison and mutated-state rejection. [Receipt](../artifacts/search-v2/20261028/v205-delivery/result.json)
- A further **304 envelope assertions** verified owner/thread counts, same deadlines, eligible watermark/root-count invariants, per-owner retained watermark, complete plans and null fabricated joint support. [Decision receipt](../artifacts/search-v2/20261028/v205/decision.json)

Four isolated Unity smokes used 4s only: one complete turn per arm, plus V205 cancel and load between actions. Each used its own isolated checkpoint and ended promptly.

| Isolated normal turn | Cache-only | V205 |
| --- | ---: | ---: |
| Final actual commit / delivery | 3,345.4 ms | 3,344.8 ms |
| Initial snapshot, outside search clock | 65.4 ms | 69.3 ms |
| Cold warmup, separately measured | 3,820.2 ms | 3,935.3 ms |
| Unity CPU | 484.4 ms | 500.0 ms |
| Native CPU | 5,718.8 ms | 5,687.5 ms |
| Search requests / completed actions | 1 / 2 | 1 / 2 |

The **0.6 ms delivery difference is effectively unchanged** in this single paired smoke. Both passed 745 Unity checks, saved exactly once after AI End Turn, and loaded the checkpoint including the next-turn boundary. V205 cancellation and load passed 742 and 743 checks respectively: each actually focused, committed only action one, retired work, and kept the prior completed-turn checkpoint. Native peak memory was unavailable through this Unity harness and is recorded as null. Existing stage-by-stage C# profiling is in the original report; these follow-up smokes measure total delivery and do not claim new per-stage profiling.

Receipts: [V205 delivery](../artifacts/search-v2/20261028/v205-unity-normal/result.json), [cache-only delivery](../artifacts/search-v2/20261028/v205-unity-cache-normal/result.json), [cancel](../artifacts/search-v2/20261028/v205-unity-cancel/result.json), [load](../artifacts/search-v2/20261028/v205-unity-load/result.json).

## Disposition and reproduction

V205 is **implemented and evaluated, rejected for selection at 4s/K8**. Keep `search_v2_focus_only` as the selected experimental candidate. Its [isolated package](../artifacts/search-v2/20261028/v205-cache-candidate/build-manifest.json) leaves retention disabled; the [V205 package](../artifacts/search-v2/20261028/v205-candidate/build-manifest.json) and [frozen source archive](../artifacts/search-v2/20261028/v205/source.zip) preserve the tested opt-in experiment. No default or installed policy enables V205.

The conclusion follows the plan's preference for the simplest useful implementation: correctness passed, but the primary cases and real Unity delivery show no overall improvement, the same decisions were returned, and peak memory increased consistently. This does not prove reuse can never improve playing strength or other time settings; those were not tested.

Build with `scripts/search-v2-tools.py build`, and run tests with a fresh `test --label <label>`. `scripts/benchmark-search-v205.py` freezes/runs the bounded protocol and regenerates its analysis. Existing trial folders refuse overwrite. `scripts/build-search-v2-managed.py --method search_v2 --output <fresh-artifact-directory>` creates the optional isolated experiment; its default method remains cache-only. `scripts/seal-search-v205.py` verifies frozen-source/binary receipts, evidence and the selected policy.

[Installation verification](../artifacts/search-v2/20261028/v205/installed-unchanged.json) confirms all **216 pinned installed files and the installation receipt are unchanged**. This follow-up makes no production promotion, final-model qualification or lab migration claim.
