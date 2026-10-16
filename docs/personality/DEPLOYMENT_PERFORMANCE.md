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
