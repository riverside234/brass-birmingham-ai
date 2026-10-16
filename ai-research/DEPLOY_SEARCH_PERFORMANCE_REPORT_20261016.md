# Local deployment search: fused transport and public-view reuse

The optimized deployment path completed **12 / 10 simulations with two workers, and 15 / 12 with three workers**, on the Canal / Rail fixtures respectively. **v9 and W352×6 completed the same counts**, with the same root coverage in this short comparison. This supports W352 as an affordable candidate for a future training experiment on this CPU; it does not establish stronger play or justify replacing a trained v9 with initialized weights. Production model architecture and training defaults remain unchanged.

Measured on Intel Core Ultra 5 135U / Windows / ONNX Runtime 1.22.1 CPU, with two tensor threads. Both models use initialized weights, the current factored scorer, unchanged features and head widths. Utility parameter counts remain **2,015,937** for v9 and **2,647,761** for W352, an increase of 31.3%. The wider layout is 32 / 64 / 48 / 256 / 288 / 48 plus 12 context values, fusion 748 → 352 and six residual blocks. The experimental loader remains separate from production checkpoint admission.

## What changed

- `personality/search/engine.py`: persistent deployment workers now request the existing capability-checked fast native runtime. A rules step, checkpoint, four permitted views, chance state and choices are returned in one RPC. Each search lane still owns its private native host. Legacy bundles retain their capability fallback.
- `BrassTrainingHost/FastHost.cs`: one search snapshot materializes the public observation once. Four reply dictionaries share those immutable public tokens, while **both `seat` and `own_hand` are replaced independently for every seat**. This reuse ends with the reply; it is not a cache across revisions, sampled worlds or rules mutations. Pending chance still withholds all views.
- `personality/laptop/package.py`: bundles the verified optimized search executable and unchanged licensed dependencies, with explicit-chance metadata. The packaged runtime requires no training directory or local compiler. Binary hashes and dependency checks remain enforced.
- `personality/laptop/runtime.py`: complete-plan revalidation also uses the capability-checked fast path.
- Rebuilt the additive search executable and updated both throughput manifests' source pins. The ordinary executable stayed byte-identical. Qualified R1/R4 binaries, installed-game assemblies, weights, search policy, rewards, label-quality gates and training configurations were not modified.
- The prior shared immutable encoder LRU and private persistent builders remain active. No state, legal-action, context or private-hand fields were removed from encoding cache keys.

This change does not optimize away reconstruction validation, reduce rollout depth, weaken checkpoint hashes or count interrupted paths as completed work.

## Controlled measurements

The two native fixtures are `settled-next-round` (Canal round 2, 387 root choices) and `rail-action` (Rail round 2, 119 choices). This is a headless test of the production `SearchSession` and portable inference queue, not a Unity UI or full-game strength test.

Every turn has one **12-second allowance shared across all workers, both actions and continuations**. Existing finalization reserve leaves approximately 10.8 seconds for search. Native process construction is outside the turn clock. Full-feature LRUs are cleared before each measured turn; builders and categorical templates persist. Root/model ordering is reversed between the two fixtures. All reported simulation counts are totals across workers.

There were 18 transport/batching pilot and comparison turns, followed by eight final comparison turns. The final phase uses the same fixture seeds as the first phase's main model comparison. There are only two final turns per model/worker configuration; timing noise and different traversal workloads remain material.

| Final model | Workers | Attempts, Canal / Rail | Completed, Canal / Rail | Interrupted, Canal / Rail | Distinct visited root actions | Eligible diagnostic labels | Whole-turn seconds, Canal / Rail |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| v9 | 2 | 14 / 12 | **12 / 10** | 2 / 2 | 6 / 5 | 2 / 2 | 10.984 / 10.953 |
| W352×6 | 2 | 14 / 12 | **12 / 10** | 2 / 2 | 6 / 5 | 2 / 2 | 10.984 / 10.985 |
| v9 | 3 | 18 / 15 | **15 / 12** | 3 / 3 | 5 / 4 | 2 / 2 | 10.984 / 10.984 |
| W352×6 | 3 | 18 / 15 | **15 / 12** | 3 / 3 | 5 / 4 | 2 / 2 | 10.953 / 11.063 |

All eight turns completed their legal real-turn actions, retained one shared deadline, produced an eligible diagnostic label and had no insufficient-coverage result or budget overrun. Labels satisfy structural visit/plan rules only: initialized values have not acquired trained utility calibration, and no labels were used for training.

Three workers completed more simulations but visited fewer distinct root actions in these private forests. Neither worker count is demonstrated stronger. Even the improved search visits only 4–6 of 119–387 root actions here; it remains sparse NN-guided search.

## Attribution of the gains

The initial legacy-vs-fused v9 pilot uses matching seeds and batches: **8 / 8 → 10 / 8** completed simulations with two workers, a 12.5% increase in their total. On the Canal fixture, the counted native RPCs fell from 3,078 to 555 despite more simulated work. The Rail counts were 3,445 → 589. These totals include the small real-turn driver.

The subsequent public-view optimization uses matching seeds against the first phase's main comparison:

| Model / workers | Fused transport only | Fused transport + public-view reuse | Increase in completed total |
| --- | --- | --- | ---: |
| v9 / 2 | 10 / 8 | 12 / 10 | 22.2% |
| W352 / 2 | 9 / 8 | 12 / 10 | 29.4% |
| v9 / 3 | 12 / 9 | 15 / 12 | 28.6% |
| W352 / 3 | 12 / 9 | 15 / 12 | 28.6% |

Do not multiply these small-sample percentages into a general speedup guarantee. The two stages used different seed pairs, and discrete completion counts depend on rollout lengths and the cutoff.

Instrumentation sums overlapping worker time. For example, the final v9/two-worker Canal turn contains 598 simulated/real state advances, approximately 14.97 aggregate worker seconds inside advancement, 0.97 seconds of reconstruction and 1.92 seconds of encoding. Native request time is nested inside those totals. It is incorrect to add them and compare the result to the 10.984-second wall time. Native snapshots are still the largest measured component.

Pure ONNX forward time over each final turn was approximately **0.26–0.49 seconds**, with 194–242 evaluated positions including partial paths. W352's larger arithmetic did not reduce completed work in these eight turns. The earlier isolated B1 microbenchmark measured W352 about 20% slower for representative candidate counts; matching whole-search counts does not mean identical neural latency. Likewise, `inference.queue_wait` includes pre-submission preparation for personality requests and cannot be treated as pure IPC delay.

## Batching and encoder reuse

After enabling fused transport, the small v9 pilot compared one and two pending paths per worker:

| Workers | One pending path: completed total over two roots | Two pending paths: completed total | Interrupted totals, one / two |
| --- | ---: | ---: | --- |
| 2 | **18** | 16 | 4 / 8 |
| 3 | **21** | 19 | 6 / 11 |

The final comparison therefore used **`pending_paths=1`, `batch_size=1` per search worker**, central inference cap **2 or 3** matching the worker count, two tensor threads and the existing **1 ms flush**. Actual final forward calls were predominantly batch 1, with some batch 2. This is the measured experimental setting, not an alteration to existing qualified roster profiles. Batch 2 was not retuned after the public-view change; a larger deployment qualification could revisit it.

Across the two final turns, v9/two-worker encoding recorded 20 warm hits / 369 misses; W352 recorded 20 / 374. Three workers recorded 32 / 447 and 32 / 446 respectively. The approximately 5–7% hit rate is real within-turn reuse with cold full-feature caches at each root. Different sampled worlds and changing actions legitimately require fresh features; manufacturing more hits by dropping private state or legality from the key would be incorrect.

## Validation and limits

**33 focused tests passed in 56.746 seconds.** New checks cover fast-host selection, self-contained packaged execution without training files, binary tamper rejection, legacy capability fallback, 14 native fixtures across both eras, all four observer views, legal choices, unchanged checkpoint hashes and bitwise feature parity. Their action sequences include payment, continuation, merchant beer, bonus development and debt settlement. Live-game branch restoration and pending-chance privacy remain covered. Existing cache, cancellation, search, rollback and inference-resource cleanup tests also passed.

All four ONNX graphs passed Torch/ORT and row-vs-batch parity checks with shapes (1,1), (2,7), (4,31), (1,257). Production validation still rejects the benchmark-only W352 architecture. Scoped `git diff --check` passed.

Additional inspection exposed an **existing** limitation: cloning and restoring certain roots imported from mid-game observations raises `checkpoint_hash_mismatch` on both the legacy and optimized hosts. An opening root restores correctly. This is recorded in `deploy-transport-20261016/existing-restore-limitation.json`; hash validation was not bypassed. The measured search uses persistent private lanes and does not restore those imported forks. Passing the focused tests is not a claim that arbitrary imported-root restoration works.

## Evidence and reproduction

- `artifacts/feature-engineering/deploy-transport-20261016/`: initial legacy/fused baseline, batch pilot, paired model comparisons, exact initialized export metadata and executed benchmark source.
- `artifacts/feature-engineering/deploy-public-views-20261016/`: final eight turns, `summary.json`, exact native executable hash, model parity metadata, validation log and executed benchmark source.
- `tests/FeatureEngineering/benchmark_deploy_transport.py` and `tests/Personality/test_search_transport.py`: reusable bounded harness and focused regressions.

Run with the repository's normal Python dependencies and `tests/FeatureEngineering` on `PYTHONPATH`:

```text
python tests/FeatureEngineering/benchmark_deploy_transport.py --output artifacts/feature-engineering/deploy-search-new-check --comparison-only --delete-models
```

Use a fresh output directory. Omit `--comparison-only` to include the legacy/batch pilot. Neither command trains or changes existing checkpoints. New deployment bundles must be rebuilt through the existing packaging/qualification flow to include the faster binary; existing installed packages were not overwritten.

All **eight newly exported temporary ONNX models** from the two phases were deleted: **58,889,648 bytes**. Both model directories are empty, with hashes and sizes retained in cleanup receipts. No new `.pt` checkpoints were created. Earlier cleanup remains intact; unrelated retained models were not touched.
