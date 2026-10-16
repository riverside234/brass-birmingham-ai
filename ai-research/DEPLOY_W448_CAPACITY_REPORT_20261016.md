# Isolated 3.67M deployment search experiment

The **W448×6 candidate is feasible within the tested 12-second deployment budget**, but the previous observation of unchanged simulation counts does not extend uniformly to this larger model. Across six roots, W352 completed **133 / 155** simulations with two / three workers; W448 completed **119 / 136**. The aggregate reductions are **10.5% / 12.3%**, concentrated in one late-Canal root. Cumulative distinct visited root actions were identical between models for each worker count.

Fixed-input ONNX measurements isolate a real neural cost: W448 utility inference at batch 1 was **8.3–19.2% slower**, with a **14.1% geometric mean slowdown** across six roots. Whole-search completion differences also include different initialized predictions, sampled paths, cache reuse and timing noise. They cannot be attributed entirely to parameter count. These results support considering W448 for a training experiment, not automatically replacing W352 or claiming stronger play.

## Architectures and isolation

| Component | W352 comparator | W448 candidate |
| --- | --- | --- |
| Six state projections | 32 / 64 / 48 / 256 / 288 / 48 | Same |
| Context passthrough | 12 | 12 |
| Fusion | Linear(748, 352), GELU | Linear(748, 448), GELU |
| Backbone | Six width-352 residual blocks, LayerNorm | Six width-448 residual blocks, LayerNorm |
| Action branch / policy and outcome hidden widths | 128 / 128 | Same |
| Base parameters, without utility | 2,636,205 | **3,656,781** |
| Parameters including utility | 2,647,761 | **3,671,409** |
| Including deployment personality/goal extension | 2,721,898 | **3,762,154** |
| Dense MACs per position, utility graph | 2,578,464 + N × 58,368 | 3,599,520 + N × 58,368 |

W448 copies W352's initialized group projections and action encoder, then initializes the resized fusion, backbone and width-dependent heads. Features, candidate encoding, factored scorer, head widths, search rules and utility specification are unchanged. Initializations use the preserved benchmark seeds; no trained checkpoint is loaded or migrated. The actor context projection remains zero and search uses `alpha=0`.

Both layouts use explicit benchmark-only identities. Production configuration validation rejects the candidate; the production 3M limit, main model, training configuration and existing checkpoints were not changed. Runtime/source hashes matched between preparation and search completion. Only the isolated harness, this report and a documentation link were added/updated for this experiment.

## Protocol and late-game coverage

Hardware/software: Intel Core Ultra 5 135U, Windows, Python 3.12.14, PyTorch 2.8.0+cpu, ONNX Runtime 1.22.1 CPUExecutionProvider. Two tensor threads, one ORT inter-op thread, sequential ORT execution and disabled spinning.

There were **24 measured whole turns**: two models × two worker counts × six roots. Each root/worker pair used the same seed, with model order alternated. There is only one paired repetition per condition. A turn's **12 seconds are shared across all workers, both actions and continuation decisions**; the existing finalization reserve leaves about 10.8 seconds for search. Native process startup and initial test-driver root reconstruction are outside this clock; reconstruction performed inside the search workers is included. Full-feature LRUs start empty for every turn; persistent encoders and action templates remain available.

The two earlier fixtures are retained. Four new roots come from one completed **331-decision legal game**, generated without neural search by a fixed, seeded, family-weighted policy. Before model timing, the largest legal branching factor at the start of a full turn was selected in Canal and Rail rounds 6 and 8. These are occupied boards reached by legal play, not early positions with changed round numbers. The generator is not a strong-play policy or a representative training distribution.

| Root | Root choices | Occupied industry slots | Origin |
| --- | ---: | ---: | --- |
| Rail round 2 | 119 | 4 | Existing authored `rail-action` fixture |
| Canal round 2 | 387 | 0 | Existing authored `settled-next-round` fixture |
| Canal round 6 | 122 | 20 | Legal fixture game |
| Canal round 8 | 30 | 23 | Legal fixture game |
| Rail round 6 | 110 | 18 | Legal fixture game |
| Rail round 8 | 40 | 18 | Legal fixture game |

Occupancy does not guarantee more expensive simulations. The exact late-Canal round-6 search paths were much shorter than the other roots' paths, so summing simulation counts heavily weights that root. The observation reconstruction of all six roots preserved their complete legal-choice lists.

## Whole-search results

Counts below are totals across search workers. A completion is a completed simulation; interrupted work is never counted as completed. Root coverage counts distinct root actions with positive completed visits, merging worker forests only within that root.

| Root | Workers | Completed W352 / W448 | Interrupted W352 / W448 | Distinct visited root actions W352 / W448 |
| --- | ---: | --- | --- | --- |
| Rail round 2 | 2 | 10 / 10 | 2 / 2 | 5 / 5 |
| Canal round 2 | 2 | 11 / 10 | 2 / 2 | 6 / 5 |
| Canal round 6 | 2 | **78 / 62** | 2 / 2 | 21 / 20 |
| Canal round 8 | 2 | 6 / 6 | 2 / 2 | 3 / 3 |
| Rail round 6 | 2 | 10 / 10 | 2 / 2 | 5 / 5 |
| Rail round 8 | 2 | 18 / 21 | 2 / 2 | 9 / 11 |
| Rail round 2 | 3 | 12 / 12 | 3 / 3 | 4 / 4 |
| Canal round 2 | 3 | 14 / 13 | 3 / 3 | 5 / 5 |
| Canal round 6 | 3 | **90 / 67** | 3 / 3 | 21 / 20 |
| Canal round 8 | 3 | 6 / 8 | 3 / 3 | 3 / 3 |
| Rail round 6 | 3 | 12 / 12 | 3 / 3 | 4 / 4 |
| Rail round 8 | 3 | 21 / 24 | 3 / 3 | 7 / 8 |

| Model / workers | Attempts | Completed | Interrupted | Sum of visited root actions / legal root choices | Qualified diagnostic labels | Insufficient coverage | Whole-turn seconds, min–max |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| W352 / 2 | 145 | 133 | 12 | 49 / 808 | 6 | 0 | 10.969–11.187 |
| W448 / 2 | 131 | 119 | 12 | 49 / 808 | 6 | 0 | 10.953–11.171 |
| W352 / 3 | 173 | 155 | 18 | 44 / 808 | 6 | 0 | 10.969–11.312 |
| W448 / 3 | 154 | 136 | 18 | 44 / 808 | 6 | 0 | 10.968–11.141 |

All 24 turns finished their legal real-turn actions, retained one shared deadline and stayed below 12 seconds. All produced one structurally qualified diagnostic label. **Zero labels were used for training.** Initialized utility values are not calibrated strength evidence, and passing the existing minimum coverage gate does not mean broad action coverage. The 49/44 totals sum per-root counts, not globally unique actions across unrelated boards.

W448 completed more simulations in the late-Rail round-8 root with both worker counts. Excluding the single high-completion Canal round-6 root, totals are W352 **55 / 65** versus W448 **57 / 69**. This is why the aggregate decrease should not be generalized as a uniform 11% capacity penalty.

The differing workloads are visible in the two-worker Canal round-6 profiles: W352 performed 424 state advances and 73 encoding misses for 78 completed simulations; W448 performed 401 advances and 117 misses for 62 completions. Both saw the same root and seed, but their models produced different traversal workloads.

## Fixed-input inference and batching

Each timing cell uses 30 warmed ONNX calls per model on identical encoded inputs, alternating model order. These measurements exclude encoding, queueing and native rules. Each batch repeats one root; it is not a mixed-size padding benchmark. Both export heads also passed Torch/ORT and row-alone versus batched-output parity on these inputs.

| Root | Utility B1 W352 ms | Utility B1 W448 ms | Increase |
| --- | ---: | ---: | ---: |
| Rail round 2, N=119 | 0.545 | 0.619 | 13.6% |
| Canal round 2, N=387 | 0.787 | 0.853 | 8.3% |
| Canal round 6, N=122 | 0.532 | 0.613 | 15.3% |
| Canal round 8, N=30 | 0.448 | 0.534 | 19.2% |
| Rail round 6, N=110 | 0.542 | 0.604 | 11.6% |
| Rail round 8, N=40 | 0.458 | 0.537 | 17.1% |

Across roots, the geometric mean W448/W352 latency ratios were:

| Batch size | Utility graph | Actor graph |
| --- | ---: | ---: |
| 1 | 1.141 | 1.133 |
| 2 | 1.188 | 1.118 |
| 3 | 1.134 | 1.104 |

Search retained the previously measured setting: **one pending path and batch size 1 per worker**, central batch cap equal to the worker count, **1 ms flush**. Actual forward batches were predominantly size 1, with some size 2 and no size 3. This experiment checks B1/B2/B3 inference parity and latency; it does not claim to have retuned end-to-end batching for W448.

Total ONNX forward time over six turns was W352/W448 **1.436/1.512 seconds** with two workers and **2.095/2.485 seconds** with three workers. Evaluated positions, including unfinished paths, were **936/945** and **1,121/1,131**. Thus the neural portion remains small relative to whole-turn elapsed time. The other profiled timings overlap across workers and must not be added together as wall time.

## Encoder reuse and validation

Encoder results on six roots, three repetitions per mode:

| Mode | Median ms | P95 ms |
| --- | ---: | ---: |
| New encoder each call | 9.692 | 16.631 |
| Persistent encoder, cache miss | 4.992 | 9.744 |
| Persistent encoder, warm hit | **0.804** | **1.191** |

Warm-hit outputs, persistent-miss outputs and new-encoder outputs matched the independent encoder exactly for state features, action features and seat mapping, with matching choice IDs and read-only arrays. Existing shared-cache behavior remains active without removing any correctness-sensitive key fields.

Search warm hits / misses were W352/W448 **136/800 versus 102/845** with two workers, and **219/905 versus 171/964** with three workers. Different traversal states legitimately produce different hit rates; a higher hit rate is not itself a strength metric.

Validation completed:

- Exact parameter counts, unchanged projection/action weights, six residual blocks and isolated configuration rejection.
- Four ONNX exports: all named-head outputs and legal masks passed Torch/ORT parity and batch invariance on synthetic shapes (1,1), (2,7), (4,31), (1,257).
- **72 real-fixture model/head/batch combinations** passed Torch/ORT and batch invariance checks across six roots, two models, two heads and B1/B2/B3.
- Reconstructed legality, exact encoder/cache parity, legal real actions, shared deadlines, complete 24-turn matrix, unchanged source hashes and empty temporary model directory.

Normal zero-initialized residual/utility branches and untrained predictions remain a limitation of this initialized-weight experiment. A trained export's weight working set and traversal distribution can differ. No trained-model tournament, playing-strength test, Unity UI qualification, training run or existing checkpoint modification was performed.

## Artifacts and reproduction

Evidence directory: `artifacts/feature-engineering/deploy-w352-w448-12s-20261016/`.

- `inputs.json`, `fixture-game.json`: exact roots, legal-game generation trace and fixed selection rule.
- `prepared.json`: model metadata, export hashes, source pins, parity, encoder and all fixed-input samples.
- `search.json`, `summary.json`: all 24 turns, decisions, completed root visits, timings, cache counts and summaries.
- `executed-benchmark.py`: the exact benchmark harness.
- `model-cleanup.json`: deletion receipt for **four ONNX files, 38,435,992 bytes**. The model directory is empty; no `.pt` files were created.

With the normal repository dependencies and `tests/FeatureEngineering` on `PYTHONPATH`, use a fresh output directory:

```text
python tests/FeatureEngineering/benchmark_deploy_w448.py --output artifacts/feature-engineering/deploy-w448-new-check --delete-models
```

The command generates isolated initialized models, performs the bounded test and deletes those model exports. It does not change deployment or training defaults.
