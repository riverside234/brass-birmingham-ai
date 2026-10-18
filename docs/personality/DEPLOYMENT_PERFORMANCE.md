# Deployment search performance and further work

[Documentation index](README.md) | [P6 qualification and deployment](QUALIFICATION_P4_P6.md#p6) | [Audit record](AUDIT_HISTORY.md#audit-deployment-performance-20261016)

This page records the completed deployment performance revision and the `20261016` evidence runs. It separates implemented behavior, measured experimental settings and proposals. The documentation update does not run training, repeat benchmarks, switch architectures or qualify a game release.

The practical result is that **v9 and the larger W352×6 candidate completed the same number of simulations in the final short comparison**. W352 is a viable capacity experiment on this CPU, but these initialized-weight measurements do not demonstrate better play. The current main model and qualified laptop resource profiles remain unchanged.

A subsequent [isolated W448×6 experiment](../../ai-research/DEPLOY_W448_CAPACITY_REPORT_20261016.md) tested **3,671,409 utility-model parameters** against W352, including four legal late-game roots. All 24 turns stayed within the shared 12-second budget. W448's fixed-input B1 utility inference was 8–19% slower; aggregate completed simulations fell 10.5% / 12.3% with two / three workers, concentrated in one late-Canal root, while cumulative root coverage matched. These initialized-weight results do not establish a uniform search-throughput penalty or stronger play. Production defaults remain unchanged.

<a id="implemented"></a>

## Implemented runtime changes

| Component | Implemented change | Preserved boundary |
| --- | --- | --- |
| [SearchSession](../../src/brass_ai/personality/search/engine.py) and [shared encoding](../../src/brass_ai/recovery/encoding.py) | Bind coordinator, search-worker and frozen-opponent endpoints to one bounded LRU of completed read-only features before constructing evaluators. Reuse endpoint-owned encoder objects. | Builders, reply lanes, mutable trees and predictions remain private. Keys retain the complete permitted view, ordered legal choices, feature identity and policy context; hits still check cancellation/deadlines. |
| Search workers | Request `fast_host=True`, using the existing fused native search path. A rule step returns its checkpoint, chance state, permitted views and legal choices in one RPC. | Each simulation lane owns a persistent native host. Rules, rollout policy, search horizon, label-quality checks and completed-visit accounting are unchanged. Capability fallback keeps older runtimes usable. |
| [Native SearchSnapshot](../../src/BrassTrainingHost/FastHost.cs) | Materialize public observation fields once per snapshot and reuse their tokens in four reply dictionaries. | Replace **both `seat` and `own_hand`** for every seat before returning views. Tokens never persist across a rules mutation or sampled world. Pending private chance returns no views. |
| [Package builder](../../src/brass_ai/personality/laptop/package.py) | Select and verify the additive search executable, keep licensed dependency pins, and bundle explicit-chance metadata. | No training directory or compiler is required on the deployed laptop. Changed binaries/dependencies fail verification. Existing packages stay immutable. |
| [Laptop runtime](../../src/brass_ai/personality/laptop/runtime.py) | Use the capability-checked fused host for complete-plan revalidation as well as search workers. | Keep legal-plan validation, shared clocks, cancellation and the normal installation/qualification flow. |

The shared cache has **256 entries per acquired lane** in aggregate. With one frozen parent, two workers use five lanes and 1,280 entries; three workers use seven lanes and 1,792 entries. Direct mode retains one lane. These are bounded completed-input caches, not cross-model prediction caches.

The optimized search executable and its source manifests were rebuilt. The ordinary executable remained byte-identical; its manifest was refreshed to pin the updated shared source. The qualified R1/R4 binaries and installed game were not overwritten. Model weights, training state, reward coefficients and search acceptance criteria were preserved.

<a id="measurements"></a>

## Measured scope and results

Hardware/software: **Intel Core Ultra 5 135U, Windows, ONNX Runtime 1.22.1 CPUExecutionProvider, two tensor threads**. Both networks used initialized float32 weights and the existing factored scorer. The actor's context projection was zero-initialized and the test used `alpha=0`; nonzero personality bias, trained values and full-game behavior were not evaluated.

| Network | Six state projections | Fusion | Residual backbone | Base parameters | With utility head | With deployment goal extension |
| --- | --- | --- | --- | ---: | ---: | ---: |
| v9 | 32 / 64 / 48 / 192 / 224 / 48 | 620 → 304 | 304 × 6 | 2,005,917 | 2,015,937 | 2,081,770 |
| Experimental W352 | 32 / 64 / 48 / 256 / 288 / 48 | 748 → 352 | 352 × 6 | 2,636,205 | 2,647,761 | 2,721,898 |

Both retain the 12 context values and existing action/head widths. W352 was admitted only by the isolated benchmark loader; production config/checkpoint validation still rejects that experimental architecture. This is not a weight migration or a main-model selection.

The fixtures were Canal round 2 (`settled-next-round`, **387 root choices**) and Rail round 2 (`rail-action`, **119 choices**). Each turn had one **12-second budget shared across workers, both actions and continuations**. Existing finalization reserve left approximately 10.8 seconds for search. Persistent host construction was outside the clock, and full-feature LRUs were cleared before every measured turn. The test exercised the headless `SearchSession` and portable inference path, not a running Unity game.

| Final model / workers | Attempts, Canal / Rail | Completed simulations, Canal / Rail | Interrupted, Canal / Rail | Distinct visited root actions, Canal / Rail | Whole-turn seconds, Canal / Rail |
| --- | --- | --- | --- | --- | --- |
| v9 / 2 | 14 / 12 | **12 / 10** | 2 / 2 | 6 / 5 | 10.984 / 10.953 |
| W352 / 2 | 14 / 12 | **12 / 10** | 2 / 2 | 6 / 5 | 10.984 / 10.985 |
| v9 / 3 | 18 / 15 | **15 / 12** | 3 / 3 | 5 / 4 | 10.984 / 10.984 |
| W352 / 3 | 18 / 15 | **15 / 12** | 3 / 3 | 5 / 4 | 10.953 / 11.063 |

Counts are totals across all workers. All eight final turns completed their real legal actions, retained one deadline and stayed within budget. Each produced one structurally eligible diagnostic teacher label, with no insufficient-coverage result. Those labels were not used for training, and initialized utility predictions are not evidence of trained calibration.

The 18-turn transport/batching phase preceded these eight final turns. Matching-seed v9/two-worker probes improved from **8/8 to 10/8** simulations after enabling fused transport. Canal/Rail RPC counts fell from **3,078/3,445 to 555/589**, including the small real-turn driver. The subsequent public-view change improved matching-seed v9 totals from **18 to 22** with two workers and **21 to 27** with three, or **22.2% and 28.6%**. These are small-sample observations; do not combine percentages from different seed pairs into a general speedup guarantee.

Native snapshots remain the largest measured cost. Final pure ONNX forward work consumed approximately **0.26–0.49 seconds per turn**, across 194–242 evaluated positions including unfinished paths. The earlier isolated batch-1 neural benchmark found W352 roughly 20% slower at representative candidate counts. Equal completed-search counts therefore do not imply equal inference cost. See the [detailed attribution report](../../ai-research/DEPLOY_SEARCH_PERFORMANCE_REPORT_20261016.md).

### Batching and actual encoder reuse

The post-transport v9 pilot selected **one pending path and leaf batch size 1 per worker**, a central inference cap of **2 or 3** matching worker count, **two tensor threads**, and **1 ms flush**. Over two pilot roots, one versus two pending paths completed **18 versus 16** simulations with two workers and **21 versus 19** with three. Larger pending batches left more incomplete paths. Final forward calls were mostly batch 1, with some batch 2.

These settings belong to the experiment. Batch 2 was not retuned after public-view reuse, and the production P5/P6 qualification still requires its declared **two-worker laptop profile**. The three-worker harness result does not make three workers an accepted release setting. A resource/recipe change needs its own compatible configuration and qualification evidence.

**ONNX Runtime session note (two-worker CPU experiment, 8-second search):** Two ONNX Runtime sessions with 1 thread each performed better than one session with 2 threads. Small batches parallelize poorly inside a call, but two calls can run side by side. The [isolated Rust benchmark](../../artifacts/feature-engineering/rust-v11-two-sessions-2workers-8s-20261016/REPORT.md) measured **14.2% / 11.4% more completed simulations** for original v11 / v11 without v10 additions across matched complex-state trials. Each dedicated session used batch 1; the shared session batched up to 2, so the gains include batching and queue effects. This finding is a configuration to evaluate in personality deployment; it does not record an implemented runtime change.

**Export optimization and operation profiling (`20261016`):** Both current export paths simplify new graphs with pinned `onnxsim==0.8.1` and explicitly use `ORT_ENABLE_ALL`. A frozen export copy encodes all four seats with one grouped matrix operation, preserving the separate learned own/other weights, biases, LayerNorm affine parameters and relative-seat mapping. Training modules and checkpoint tensors are unchanged. Original-model versus ONNX parity is checked across dynamic batch/candidate shapes. Portable files retain standard opset-17 operators; the exporter inspects a temporary ORT execution graph and requires fused exact GELU and LayerNorm. This is execution-time fusion, not a switch to an ORT-specific portable model.

The [paired inference benchmark and operation profiles](../../artifacts/feature-engineering/onnx-graph-benchmark-20261016/REPORT.md) found little overall latency change: geometric-mean optimized/original ratios were **0.999 utility / 0.994 actor**, using one inference thread, batches 1/2 and real 119/387-candidate inputs. Dense matrix operations dominated the sampled kernel time. These measurements use diagnostic weights and do not establish a search or strength gain. **26 focused/regression tests passed**, including export parity, independent seat weights/affines, masks, seat mapping, checkpoint identity and trace accounting.

The subsequent [15-second Rust search comparison](../../artifacts/feature-engineering/rust-v11-optimized-2workers-15s-20261016/REPORT.md) used **two workers and two one-thread ORT sessions**, batch 1 per worker. Original/optimized v11 averaged **1,501.5 / 1,508.2** completed simulations in matched complex-state controls (**+0.44%**), and **1,712.7 / 1,693.7** under actual NN guidance (**−1.11%**). This run shows no meaningful throughput improvement from the export changes. The simplified no-v10 reference averaged **1,555.2** controlled completions; its guided counts depend on different search paths. All 33 trials passed the saved audits. Counted snapshots and worker work ended before 15 seconds; maximum report delivery including draining/cleanup was **15.0715 seconds**. Near-terminal Rail counts are reported separately.

To profile an export with its saved input sidecar, run `PYTHONPATH=src python -B scripts/profile-onnx.py path/to/model.onnx --output artifacts/onnx-profile-new --threads 1`. The opt-in tool writes per-node and per-operation timings, raw ORT traces and a host-specific execution graph to a fresh directory. Warmup and fence events are excluded; latency is measured separately with profiling disabled. Normal inference does not enable tracing. Fresh exports require the updated R2/P6 export dependencies.

Across the two final roots, cache counts were:

| Model / workers | `warm_hit` | Misses |
| --- | ---: | ---: |
| v9 / 2 | 20 | 369 |
| W352 / 2 | 20 | 374 |
| v9 / 3 | 32 | 447 |
| W352 / 3 | 32 | 446 |

These approximately **5–7%** hit rates measure real within-turn reuse with an empty full-feature LRU at each root. They should not be compared directly with the 90% hit rate in the deliberately repeated-root Step 5 component benchmark. Different private worlds and evolving legal choices legitimately need fresh features.

Three workers completed more simulations but covered fewer distinct root alternatives here. Root coverage remained only **4–6 of 119–387 actions**. Neither the worker comparison nor the additional W352 capacity demonstrates stronger play. Revisit architecture choice with completed IL weights and equal-budget game evaluation before making a strength claim.

<a id="validation"></a>

## Validation and known limits

The recorded implementation run passed **33 focused tests in 56.746 seconds**. They cover fast-host routing, a self-contained packaged native host without training files, tampered-binary rejection, legacy fallback, shared cache identity/ownership, cancellation and resource cleanup. Native comparisons cover 14 fixtures across both eras, all four observer views, legal choices, payment/continuation/bonus/debt behavior, unchanged checkpoint hashes and bitwise feature parity. Existing live-game restore, chance privacy and rollback checks also passed.

Separately, all four exported graphs passed Torch/ORT and row-versus-batch parity at `(batch, candidates)` shapes `(1,1)`, `(2,7)`, `(4,31)`, `(1,257)`. See the [validation receipt](../../artifacts/feature-engineering/deploy-public-views-20261016/validation.json), [test log](../../artifacts/feature-engineering/deploy-public-views-20261016/validation.log) and [export metadata](../../artifacts/feature-engineering/deploy-public-views-20261016/prepared.json). These checks were not rerun for this documentation edit.

**Existing imported-root restore limitation:** some states reconstructed from mid-game observations fail subsequent clone/replay restoration with `checkpoint_hash_mismatch` in both the legacy and optimized host. The opening fixture restores correctly. The measured search uses persistent private lanes and does not restore those imported forks. This issue remains open; hash validation was not relaxed. [The reproduction record](../../artifacts/feature-engineering/deploy-transport-20261016/existing-restore-limitation.json) is the starting point for investigating it before adding more checkpoint reuse.

The experiment does not cover trained-model strength, broad phase/position distributions, all personality settings, 2/5-second tier behavior, cold startup, full interactive Unity responsiveness or optional Intel GPU performance. Aggregate worker timings overlap; native RPC time is nested inside reconstruction/advancement. Also, `inference.queue_wait` includes pre-submission encoding/validation for personality requests, so it is not a measurement of pure transport delay.

<a id="further-improvements"></a>

## Possible further improvements — proposals, not implemented

Prioritize measured native work and trustworthy attribution before increasing NN batch sizes. No speedup estimate or automatic rollout is assigned to these proposals.

| Priority / area | Evidence or motivation | Bounded next experiment and acceptance checks |
| --- | --- | --- |
| First: imported-root restore correctness | Both old and optimized runtimes reproduce the mid-game hash failure. More aggressive state reuse would depend on correct restoration. | Isolate the first differing canonical field across reconstruction, checkpoint and replay, including cards, RNG/engine revision, history and chance. Verify both eras, payment/debt states, partial chance and branch isolation. Keep hash rejection active. |
| Native snapshot construction | Native advancement/snapshot work dominates; public-view reuse already improved completion counts. | Separate rule execution, choice generation, canonical hashing, journal copying, snapshot materialization and JSON serialization. Consider reusing immutable board metadata or duplicate canonical components only within a proven valid scope. Compare full wire output, legality, canonical hashes, chance privacy and rollback against the baseline. |
| Checkpoint and payload size | Every fused step still transports a checkpoint and four views, often with growing histories. | After resolving imported-root restoration, prototype bounded immutable journal prefixes or versioned delta snapshots in an isolated branch. Measure bytes, allocation and restore time. Verify exact full-state reconstruction, revision/world identity, reconnect/fallback behavior and memory limits; never omit private state from trusted hashes. |
| Encoding and cache misses | Full-feature reuse is only 5–7% in these sampled-world searches; a low hit rate is not necessarily a defect. | Attribute constructor, schema validation, digest, derived feature and allocation costs. Explore static public subcomponents or replay-root reuse where exact identity permits it. Preserve complete observation/choice/context keys and independent mutable builders. Measure miss cost, retained bytes and end-to-end completed work, not hit rate alone. |
| Queue and batch scheduling | Most final forwards are B1; B2 pending paths lost completions in the small pilot. Existing queue timing mixes preparation and waiting. | Record encoding completion, enqueue, dequeue, forward and reply times separately. Retest B1/B2 and flush behavior after native improvements, including draining already-ready work without extending deadlines. Report actual model-grouped batch histograms, candidate padding, expired requests, p95 latency and row/batch parity. Do not assume a larger batch is faster. |
| Worker coverage | Three private forests completed more paths but visited fewer root actions. | Compare two/three workers on a larger fixed root/seed set. Evaluate more balanced initial root allocation as a separate search-policy experiment, retaining exploration, legal masks, unbiased alternatives, private trees and supported continuations. Report root/plan coverage as well as visits; strength claims require equal-budget games. |
| Cold startup and hardware coexistence | This benchmark excludes persistent process construction and does not run Unity. | Measure package loading, graph warm-up, native host startup, first action, Play Now/cancel latency and process-tree memory with Unity active. Evaluate optional prewarming or another already-provisioned backend with output parity. Keep cold user-visible delay visible even when preparation occurs before the search clock. |

For every proposal, keep the same model, permitted roots/seeds, legal-action ordering, shared turn clock and search recipe unless the experiment explicitly changes search policy. Use a fresh output directory and alternate condition order. Record attempts, completed/interrupted paths, unique supported root actions and plans, label eligibility, elapsed/overrun time, actual inference batches, cache hits/misses and memory. Preserve source/provenance checks and leave existing runs/packages untouched. Increase test coverage only when a new change or unresolved finding warrants it; a long training campaign is not required for a transport or encoding check.

<a id="reproduction"></a>

## Reproduction, package provenance and cleanup

From the project root in the existing Python/dependency environment, use a fresh directory. This is an optional short developer benchmark, not a release qualification command:

```powershell
$Project = (Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = "$Project/src;$Project/tests/FeatureEngineering;$env:PYTHONPATH"
python "$Project/tests/FeatureEngineering/benchmark_deploy_transport.py" `
  --output "$Project/artifacts/feature-engineering/deploy-search-new-check" `
  --comparison-only --delete-models
```

The existing environment must already provide the repository's native-search and ONNX dependencies. `--comparison-only` runs the eight-turn v9/W352 comparison at the measured one-path setting; omitting it also runs the legacy/transport batching pilot. The harness regenerates initialized benchmark models, keeps production admission strict and removes its temporary ONNX files after a successful run. Review an interrupted run before cleanup rather than assuming its final receipt exists.

New export packages select the optimized source-checked host from `training/runtime/throughput/search` when available, install it into the lab's additive host cache, then bundle its verified executable/dependencies under `runtime/artifacts/r4-host`. A missing additive runtime falls back to the legacy host; a source or binary mismatch fails. If authored native sources change, rebuild using the existing `scripts/build-training-host.py` procedure before testing/packaging; do not edit hashes to admit an old binary. The current revision already includes the rebuilt runtime.

Follow [P6 export, qualification and installation](QUALIFICATION_P4_P6.md#p6-export-on-the-lab-server) for a new deployment candidate. Current source fingerprints require compatible evidence. Do not patch an existing immutable bundle or replace the runtime associated with an old neural save. The performance work did not install a package into the live game.

Evidence locations:

- [Original capacity report](../../ai-research/DEPLOY_CAPACITY_REPORT_20261016.md): encoder microbenchmarks, model dimensions, isolated inference and initial whole-turn results.
- [Performance revision report](../../ai-research/DEPLOY_SEARCH_PERFORMANCE_REPORT_20261016.md): transport/public-view attribution and validation details.
- [Transport pilot and comparisons](../../artifacts/feature-engineering/deploy-transport-20261016/search.json) and [final eight turns](../../artifacts/feature-engineering/deploy-public-views-20261016/search.json), with exact inputs and executed benchmark copies in their directories.
- [Final summary](../../artifacts/feature-engineering/deploy-public-views-20261016/summary.json) and cleanup receipts for [transport](../../artifacts/feature-engineering/deploy-transport-20261016/model-cleanup.json) and [public-view reuse](../../artifacts/feature-engineering/deploy-public-views-20261016/model-cleanup.json).

All eight temporary ONNX models from the two revision phases were deleted (**58,889,648 bytes**); both model directories were verified empty. Reports and receipts remain. No `.pt` checkpoint was created, and unrelated retained models were not removed by the revision. Raw `artifacts/` evidence may be local-only; the linked reports preserve its scope and measurements for other checkouts.

## 2026-10-18: independent CPU sessions and less discarded search work

Implemented in the current source, with isolated builds and benchmarks on an
Intel Core Ultra 5 135U. This implements part of the queue/scheduling proposals
above. It does not install a game update or change an immutable opponent bank,
existing training checkpoint, model weights, feature layout or training recipe.

ORT CPU now defaults to **two independent sessions per frozen model, one thread
per session**. Each session has its own inference owner and timing/counter state.
Both consumers take complete requests from the existing queue; replies and
shared slots retain their private owner. Requests are handled as B1 immediately,
without the former 1 ms batch flush. The thread allowance remains two inference
threads plus the existing search workers. Multi-model and personality routing,
legal masks, complete-action probabilities, feature-cache identities, model
fingerprints and export admission remain checked. OpenVINO CPU/GPU keep their
previous single-session defaults; GPU training inference is unchanged.

Independent sessions retain an extra copy of each model's packed weights and
session workspace. Native calls are synchronous: cancel/close waits for **both**
owners to retire before unmapping shared memory, then releases the graphs.
`report()` includes replica count, threads per session, per-replica metrics and
parity, and summed timing totals. Replica elapsed times can overlap; their sum
is not turn wall time. CPU session comparisons can explicitly construct
`PortableInference(directory, models, sessions=1, threads=2)` or `(sessions=2,
threads=1)`; larger total thread allowances are rejected.

The shared C# rule search now uses a turn-local single-flight cache: identical
permitted inputs still coalesce, while unrelated evaluations execute on their
requesting search workers. It no longer has a single serial drain owner or a
busy-yield flush. Completed arrays are copied, failures wake waiters and remain
retryable, cache capacity stays 512, and no extra evaluation threads are added.
The search also reuses the root choices it just validated and avoids generating
an unused next-player observation/choice list at an already reached leaf boundary.
Leaf horizons, opponent policies, matched scenario IDs and promotion/support
criteria are unchanged.

Both timed rule search and laptop NN search have optimistic late-launch
admission. After at least four completed simulations at the current root, the
estimate is the lower quartile of the latest eight completed durations. A worker
with less time than that estimate stops launching new paths and drains any
pending paths against the **same** deadline. It never invents an attempt, credits
a partial simulation, shortens a leaf horizon or extends the turn budget. Fixed
work C# runs bypass it. The follow-up below also enables it for timed NN
training/teacher searches, while fixed-count training retains its watchdog.
There is no cross-position timing history. Cold/slow searches with fewer than
four completions retain the old launch behavior.

Added telemetry includes `deadline_stops`, the duration estimate and remaining
worker time, `skipped_leaf_observations`, `reused_root_choices`, and heuristic
cache concurrency/wait time. Laptop game metrics also retain attempted paths and
`deadline_unused_worker_seconds` (a sum of worker clocks, not turn elapsed time).

### Measured scope and results

The initial rule comparison uses the **same pre-change working-source snapshot**
(which contained the inactive Network candidate; not the installed official v4),
four recorded roots (one per phase), two repeats, alternating order, two workers,
and a shared 2-second clock. Its lab profile uses the existing zero-opponent-turn
settled leaf; it is not a measurement of every installed game difficulty.

| Rule search, all eight decisions per version | Before | After |
| --- | ---: | ---: |
| Completed simulations | 99 | 129 |
| Interrupted simulations | 16 | 4 |
| Sum of distinct root alternatives per decision | 22 | 23 |
| Supported plans (at least two visits, summed per decision) | 18 | 21 |
| Mean request wall time, including native IPC | 2.053 s | 1.992 s |

| Phase, two decisions per version | Completed before → after | Interrupted before → after |
| --- | ---: | ---: |
| Early Canal | 18 → 26 | 4 → 0 |
| Late Canal | 61 → 74 | 4 → 0 |
| Early Rail | 16 → 18 | 4 → 2 |
| Late Rail | 4 → 11 | 4 → 2 |

An intermediate version with only cache concurrency and launch admission reduced
interruptions but slightly reduced completions (54 → 51). The redundant leaf
observation/choice work was then removed and the final paired test above rerun.
Timing varies between runs; compare paired controls, not totals across runs.

The NN test uses a deterministic, freshly initialized **current medium model**,
the same export in both modes, four permitted roots, two private search workers
and two pending paths per worker. It exercises the Python/native portable search
path; the earlier Rust experiment is separate. No trained IL artifact is modified.

| NN measurement | One session × two threads | Two sessions × one thread |
| --- | ---: | ---: |
| Warm queue processing, 24 requests, first run | 219 ms | 172 ms |
| Warm queue processing, 24 requests, second run | 390 ms | 266 ms |
| Completed simulations, four 8-second roots | 8 | 10 |
| Interrupted simulations, four 8-second roots | 16 | 15 |
| Distinct root alternatives, summed over those roots | 4 | 6 |
| Revisited complete plans | 0 | 0 |
| Mean search wall time | 8.004 s | 8.043 s |

The 8-second per-phase completion counts are Early Canal 4 → 4, Late Canal 0 → 0,
Early Rail 0 → 2, Late Rail 4 → 4. A preliminary 3-second test had only 1 → 3
completions over eight roots per version, so it was insufficient for assessing
full-search throughput. **The NN timing admission did not activate in these
small samples**; its launch/drain semantics are covered by controlled tests.
NN outputs match across queue configurations at `rtol=2e-5, atol=2e-6` for policy
and all value heads; each session also passes the export's Torch/ORT parity cases.

These measurements support reduced queue delay and less wasted rule-search
work on these fixtures. They do not establish improved playing strength, robust
teacher-label yield, or universal end-to-end speedup. NN native rules/encoding
and transport still dominate much of the elapsed time. In-flight synchronous
operations can retire after the deadline (the largest observed NN request was
8.109 s for an 8-second allowance); expired work is not backed up. Cold startup,
Unity responsiveness and peak memory are not qualified here. RSS was unavailable
from this environment's memory reporter; the additional session memory has not
been quantified.

### Validation and reproduction

- 43 focused Python tests passed, covering deadlines shared by workers/both
  actions, cancellation, partial thread/host creation, legal masks, batching,
  shared encoding parity, private replies and completed-evidence accounting.
  The final resource/backend-default cleanup passed 14 targeted tests, including
  one added session/thread startup failure case.
- C# passed 23 cache/concurrency/deadline checks, 93 real game runtime checks
  (two, three and four players), and 30 existing Network behavior checks.
- On all four benchmark phases, fixed-work runs of eight simulations had
  **exactly equal selected plans, root returns and complete-plan evidence**.
  Additional two-simulation checks preserve exact evidence for one/two opponent
  reply turns and the actual-next-root leaf policy.
- An expanded legacy `tests/MaskedISMCTS/test_core.py` run also found three
  unrelated stale expectations: archived configurations treated as current,
  the CPU gate report omitting its existing `aborted` field, and an old 1,840,109
  parameter count versus the current 5,910,916 base model. Those assertions were
  not rewritten as part of this search change; this is not a whole-repository
  green-test claim.

Reproduce from the repository root in the existing dependency environment, using
fresh output paths. The baseline argument is the saved **pre-change source**,
not a frozen opponent with a different strategy revision:

```powershell
python tests/MaskedISMCTS/benchmark_parallel.py `
  --output artifacts/search-parallel-new/rule `
  --baseline-source artifacts/search-parallel-20261018
python tests/Personality/benchmark_parallel_search.py `
  --output artifacts/search-parallel-new/nn `
  --baseline-inference artifacts/search-parallel-20261018/baseline_inference.py `
  --seconds 8 --repeats 1
```

The rule harness compiles both the standalone planner and the game helper into
its own directory; it does not run an installer. The NN harness creates one
diagnostic export without saving a training checkpoint. Normal inference
packaging takes the new ORT CPU default from source and still requires matching
package/source evidence. Frozen rule banks keep their old source and binaries.
New lab collections use the separate versioned runtime described below.

Implementation files: `src/BrassMaskedSearch/{HeuristicBatchQueue,DeadlineAdmission,
MaskedSearch,RootForest}.cs`, `src/brass_ai/personality/laptop/inference.py`,
`src/brass_ai/personality/search/engine.py`, and
`src/brass_ai/search/{ismcts,deadline}.py`. Focused tests and bounded harnesses are
under `tests/MaskedISMCTS/` and `tests/Personality/`.

Evidence: [final rule comparison](../../artifacts/search-parallel-20261018/rule-final/report.json),
[C# checks](../../artifacts/search-parallel-20261018/rule-final/checks.json),
[additional leaf parity](../../artifacts/search-parallel-20261018/rule-final/additional-leaf-parity.json),
[8-second NN comparison](../../artifacts/search-parallel-20261018/nn-8s/report.json),
[3-second NN pilot](../../artifacts/search-parallel-20261018/nn-benchmark/report.json),
[focused test log](../../artifacts/search-parallel-20261018/focused-tests.txt),
and [final resource tests](../../artifacts/search-parallel-20261018/final-resource-tests.txt).

## Shared official v4 search for game, IL and self-play

The selected strategy is **official v4: `bank-3dc65253a610191d66a5`**.
The `*-v3-masked-economy` profile filenames are historical aliases, not evidence
that Economic v3 is active. Network v4 (`bank-29b8fdbbd14ff00c7c88`) and the
Network/Scout experiment remain unselected. The installed game AI DLL hash
`366adb638e64a951db87eaa7f52222690ca62d866e64d938ddd0f0f214a69b42` matches
the archived 1.4.31 release; that archive has no NetworkOptions strategy calls.
The working source previously contained candidate strategy hooks. Those active
hooks are now aligned with the official v4 used by IL, retaining search
performance changes. Candidate source/history is preserved for reference.

Shared code boundaries:

- `BrassMaskedSearch/MaskedSearchExecution.cs` owns the search loop, completed
  backups, root-choice reuse and settled-leaf cutoff. `DeadlineAdmission.cs`,
  `HeuristicBatchQueue.cs` and `RootForest.cs` are shared by game and lab builds.
- `MaskedSearch.cs` retains official v4 priors, rollout decisions and incumbent
  construction. Lab builds link the exact game `Behavior`, `CashManagement`,
  `Conversion`, `Evaluator` and `Tactics` files. Platform hosts/coordinators
  remain separate; the game supports two to four players, the lab uses four.
- NN game/personality and training use `search/ismcts.py`, `search/deadline.py`
  and `recovery/parallel.py`. The common clock hook enables tail admission for
  timed searches, including teachers. Explicit simulation quotas disable that
  optimization and retain their watchdog. Pending owners still retire before
  return; incomplete work never becomes completed evidence.
- The ORT CPU backend retains two independent single-thread sessions. GPU
  training keeps its existing inference batching/service; sharing the search
  algorithm does not force an ORT CPU execution policy onto a training GPU.

Run `python scripts/build-shared-search.py --root .` after source changes and
before new IL generation or self-play. It creates a content-addressed worker
under `artifacts/legacy-ai/search/`, leaving frozen banks, models, configurations
and installed game binaries intact. Linux uses the existing Mono/compiler and
private pinned DLLs; `--mono`, `--compiler` and `--managed` can select their
locations. Missing or stale workers fail explicitly. Game builds likewise need
recompilation/deployment; editing a shared source does not modify an installed
DLL. Historical release reproduction scripts intentionally remain pinned.

New lab policy IDs bind the shared source/profile/runtime reference. Actor specs
retain it across dispatch, and source changes participate in reviewed resume
and cache freshness. IL manifests retain generation references; older audited
BC data remains admitted without requiring its old search executable or
relabelling its decisions. Self-play teacher search and v4 search stay **off**;
the search-free v4 path uses the same strategy/runtime with zero simulations.
Per-phase teacher reports retain admission stops and unused **worker-seconds**
(summed across owners, not a second turn allowance).

Validation: 52 focused Python tests passed, followed by 14 personality/search
identity tests and the final shared-entry-point tests. Game compilation passed
99 runtime checks (including two/three/four-player operation) and 23 parallel,
cache and deadline checks. The separate Network candidate test expects extra
double-rail construction and correctly does not describe the official v4;
it remains available only through `include_candidates=True` in the harness.
An expanded older training suite had nine stale fixture/expectation failures
(old board topology, a teacher fixture missing setup, and pre-descriptor config/
fingerprint assumptions); these unrelated tests were not rewritten here.

The official-v4 smoke completed in **42.8 seconds**. At four fixed roots, eight
simulations per root produced exactly the same selected plans, root returns and
complete-plan evidence as the original frozen v4. It also exercised the real
IL/self-play adapter both searched and search-free; all fast decisions ran zero
simulations, and protected bank/config/installed files were unchanged.

Warm native comparison, two workers, one shared second, one root per phase,
alternating version order (a small diagnostic, not a strength evaluation):

| Phase | Completed old → shared | Interrupted old → shared |
| --- | ---: | ---: |
| Early Canal | 6 → 8 | 2 → 0 |
| Late Canal | 9 → 10 | 2 → 0 |
| Early Rail | 4 → 4 | 2 → 2 |
| Late Rail | 1 → 0 | 2 → 1 |
| Total | **20 → 22** | **8 → 3** |

Mean native request wall time was 1.036 → 1.011 seconds. The shared version
withheld launches in four worker searches. Late Rail still had zero completed
simulations in this short sample; fewer discarded attempts does not prove
better coverage, labels or playing strength. Cold fresh-process adapter calls
were around 1.3–1.4 seconds for the one-second search setting: JIT/IPC and
in-flight operations can exceed the clock, and are reported rather than hidden.

Evidence: [official v4 smoke and paired timings](../../artifacts/search-shared-v4-20261018/official-v4-final/report.json),
[game runtime checks](../../artifacts/search-shared-v4-20261018/game-checks/RuntimeChecks.json).
Reproduce with `python tests/Recovery/smoke_shared_search.py --output artifacts/shared-v4-new`.
No long training or strength campaign was run.
