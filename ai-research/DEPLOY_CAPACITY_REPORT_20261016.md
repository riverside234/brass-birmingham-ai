# 12-second local NN + search capacity experiment

Measured on Intel Core Ultra 5 135U, Windows, PyTorch 2.8.0+cpu and ONNX Runtime 1.22.1 CPUExecutionProvider, with two tensor threads. Both networks used initialized weights, as requested. These results measure runtime, completed search work and structural label eligibility; they do not establish playing strength or calibrated value quality. No production model, training recipe or checkpoint was replaced.

The conservative batching choice for this small experiment is **one pending path and leaf batch size 1 per search worker**, with a central inference batch cap of 2 or 3 for the corresponding worker count and the existing 1 ms flush. Three workers completed somewhat more simulations, while two workers covered more distinct root actions. Thus three workers are a throughput option, not a demonstrated stronger search configuration. The existing production deployment defaults were not changed by this experiment.

## Exact models and deployment path

| Model | Projection widths | Fusion | Backbone | Base parameters | Including utility | Including deployment goal extension |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Current v9 | 32 / 64 / 48 / 192 / 224 / 48 | 620 → 304 | 304 × 6 blocks | 2,005,917 | 2,015,937 | 2,081,770 |
| Experimental W352 | 32 / 64 / 48 / 256 / 288 / 48 | 748 → 352 | 352 × 6 blocks | 2,636,205 | 2,647,761 | 2,721,898 |

Both retain the 12 context values, current feature ordering, action branch, head hidden widths, factored policy scorer and float32 computation. The experimental model uses normal initialization for widened layers and shares the unchanged initialized action branch with v9. It is not a migration of v9 weights. Its distinct benchmark-only architecture identity is deliberately rejected by production config/checkpoint validation.

Whole-turn tests use the actual `SearchSession`, private Masked ISMCTS workers, native game hosts, and `PortableInference._evaluate` queue path. The test-only ONNX loader admits the experimental model without relaxing the production loader. The deployment actor extension is initialized with zero context projection and `alpha=0`; utility graphs supply rollout values. This is headless execution of the deployment search path, not a Unity UI test or a production deployment package.

Search recipe, rollout policies, action masking, utility reward, quality requirements and the whole-turn clock remain unchanged. Each 12-second allowance is shared across workers, both actions and continuations. The existing finalization reserve leaves about 10.8 seconds of search work; observed full-turn totals include applying the remaining legal actions.

## Search work within the clock

The main comparison uses two native fixtures (Canal `settled-next-round`, 387 root choices; Rail `rail-action`, 119 choices), two sampling seeds per fixture, and identical seeds across the two models. Model order is reversed on the second repeat. Each model/worker combination has four measured turns. Full-feature LRUs start empty on every turn, so later configurations cannot inherit the earlier fixture's full encoding results. Within-turn sharing stays enabled. Processes and encoder objects persist; native host initialization is outside the clock, but first-use execution costs and ordinary OS variation are not eliminated.

| Model / workers | Attempts, total | Completed simulations per turn | Interrupted, total | Mean completed / turn | Qualified diagnostic labels | Distinct visited root actions / turn | Whole-turn median / max seconds |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| v9 / 2 | 41 | 8, 8, 9, 8 | 8 | 8.25 | 4 / 4 | 4, 4, 5, 4 | 11.047 / 11.078 |
| v9 / 3 | 48 | 9, 9, 9, 9 | 12 | 9.00 | 4 / 4 | 3, 3, 3, 3 | 11.054 / 11.078 |
| W352 / 2 | 39 | 8, 7, 8, 8 | 8 | 7.75 | 4 / 4 | 4, 4, 4, 4 | 11.024 / 11.047 |
| W352 / 3 | 45 | 9, 9, 9, 6 | 12 | 8.25 | 4 / 4 | 3, 3, 3, 3 | 11.070 / 11.172 |

All 16 comparison turns completed their legal turn actions. There were no search deadline overruns or shared-clock resets. Zero `insufficient_coverage` rejections occurred in this comparison. A qualified label here means the existing visit/complete-plan support checks passed; untrained network outputs do not make it evidence of a good strategic decision. Root coverage is still narrow: only 3–5 of 119 or 387 legal choices received completed support.

The W352 model increases utility-model parameters by about 31.3%, but its observed completed work was only slightly lower in this small sample. Different initialized networks can choose different simulation trajectories, so these search totals do not isolate matrix compute cost. They cannot decide whether the additional capacity is worthwhile after training.

## Batch selection and numerical checks

The v9 pilot used two roots per configuration, before choosing the same batching configuration for both architectures. The batch setting below changes both `pending_paths` and the per-worker `batch_size`; it is not an isolated inference service batch-cap experiment.

| Workers | Pending paths / leaf batch | Central cap | Completed simulations, Canal / Rail | Diagnostic labels |
| ---: | ---: | ---: | --- | ---: |
| 2 | 1 | 2 | 4 / 4 | 2 |
| 2 | 2 | 4 | 4 / 4 | 2 |
| 2 | 4 | 8 | 0 / 0 | 0 |
| 3 | 1 | 3 | 9 / 9 | 2 |
| 3 | 2 | 6 | 8 / 6 | 2 |
| 3 | 4 | 12 | 0 / 0 | 0 |

Selection used completed simulations, then eligible-label count, then the smaller batch as a tie-break. Large batches launched 8 or 12 paths without finishing any before the deadline on these roots. Those four turns used the existing complete-plan fallback; they were not mislabeled as `insufficient_coverage` successes. Pilot and comparison counts differ, notably for two workers, so these are provisional choices from a bounded experiment, not a universal CPU optimum.

The selected central cap is an upper bound, not a request to wait for a full batch. In the main comparison, actual ORT batches were mostly 1, occasionally 2, and only two batches of 3 occurred. Across 2 and 3 workers, v9 used 1,067 B1 and 65 B2 forwards; W352 used 1,002 B1, 79 B2 and two B3 forwards.

Pure utility-graph medians below use fixed encoded fixtures, three warmups, three alternating Torch/ORT blocks and 21 measurements per backend/shape. Times are per entire batch, not per position.

| B × N | v9 ORT ms | W352 ORT ms | v9 Torch ms | W352 Torch ms |
| --- | ---: | ---: | ---: | ---: |
| 1 × 119 | 0.37 | 0.45 | 1.87 | 2.08 |
| 1 × 387 | 0.57 | 0.68 | 2.16 | 2.30 |
| 2 × 119 | 0.67 | 0.60 | 2.58 | 2.61 |
| 2 × 387 | 0.91 | 1.06 | 3.00 | 3.00 |
| 4 × 119 | 1.03 | 1.03 | 2.85 | 2.96 |
| 4 × 387 | 1.63 | 1.78 | 3.60 | 3.88 |
| 8 × 119 | 1.50 | 1.62 | 3.33 | 3.48 |
| 8 × 387 | 3.06 | 3.23 | 4.94 | 5.22 |

Raw results also retain batches 3, 6 and 16, actor-extension timings and all individual samples. Model microbenchmarks were run sequentially and are susceptible to thermal/background variation; the apparently faster W352 B2/N119 cell should not be generalized.

Four graph exports (utility and actor for each architecture) passed dynamic-shape Torch/ORT parity and legal-mask checks on B/N = 1/1, 2/7, 4/31 and 1/257. The maximum absolute difference across all named outputs was 0.000041962; same-row ORT outputs alone versus batched were identical in these checks. Thus the observed batching concern here is scheduling/latency, not a detected batch-dependent semantic error.

The factored scorer keeps per-candidate dense work at 58,368 MACs for both models. Fixed utility-model work is 1,948,064 MACs for v9 versus 2,578,464 for W352. These counts exclude encoding, native rules, queues and non-linear operations. Main-comparison inference-forward totals were only 0.69–1.03 seconds across each configuration's four roughly 11-second turns; the full workload includes substantial non-forward work.

## `warm_hit` fix and validation

The training actor path already shared immutable encodings, but the deployment `SearchSession` still created unrelated feature caches for its coordinator, workers and frozen opponent lanes. It now acquires its fresh endpoints first and binds one bounded shared cache before creating evaluators. Mutable feature builders, action caches and reply pipes remain private. The total bound remains 256 entries per lane: 1,280 entries with two workers and one parent, or 1,792 with three. Model predictions and utility budgets are not cached across models.

| Encoding mode, 14 positions × 3 blocks | Median / p95 ms |
| --- | --- |
| Construct encoder + encode (`new_encoder`) | 6.365 / 16.128 |
| Persistent encoder, full-feature cache disabled | 2.694 / 9.557 |
| Warm full-feature hit | 0.526 / 0.853 |
| Coordinator + two workers, separate caches | 7.528 / 33.837 |
| Coordinator + two workers, shared cache | 3.875 / 12.408 |
| Coordinator + three workers, separate caches | 10.603 / 45.308 |
| Coordinator + three workers, shared cache | 4.447 / 13.329 |

The controlled repeated-root probes verified bit-for-bit encoded features, choice order and read-only arrays. Each 14-position block changed 42 misses to 14 misses + 28 hits with two workers, and 56 misses to 14 misses + 42 hits with three. This is an encoding microbenchmark, not a measured 2× search speedup.

Actual search hit rates were much smaller: 31/577 and 43/621 accesses for v9 at two/three workers; 30/569 and 41/600 for W352. Most searched positions are new. The two-root v9 end-to-end ablation with former separate caches produced zero hits but still completed 8/8 simulations with two workers and 9/9 with three, matching the corresponding first-repeat shared-cache turns. An end-to-end completion gain from this cache fix was not demonstrated at this budget.

**19 focused tests passed** in 4.931 seconds: deployment cache ownership/isolation, all-seat native inputs, current feature parity, input/context/choice mutation, private builders, cache bounds, model replacement, remaining-budget separation, expiry/cancellation, and search/teacher behavior. One existing inference test used an obsolete authored slot fixture incompatible with compact features; it now uses a native fixture with seat rotation. Feature admission was not weakened. Scoped `git diff --check` passed.

The final bounded search run contains **32 turns**, **354.5 seconds** of measured whole-turn work, **32/32 completed legal turns**, **28 structurally eligible diagnostic labels**, no clock reset and no search overrun. An earlier incomplete run that reused cross-turn fixture encodings is retained as `initial-warm-cache-partial.json`; it is excluded from all final search tables and batch selection.

## Files, reproduction and cleanup

Production change: `src/brass_ai/personality/search/engine.py`. Regressions: `tests/Personality/test_search_encoder_reuse.py` and the native-fixture correction in `tests/Personality/test_search.py`. Experimental driver: `tests/FeatureEngineering/benchmark_deploy_capacity.py`. Architecture defaults, production exports and checkpoint acceptance rules remain unchanged.

From the checkout, with the local Python runtime and dependencies available:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src;tests\Recovery;tests\R2;tests\Reports;tests\FeatureEngineering;tests\Personality;artifacts\r2-pythondeps;artifacts\r0-pythonlibs;artifacts\r4-pythondeps'
python tests/FeatureEngineering/benchmark_deploy_capacity.py --output artifacts/feature-engineering/deploy-capacity-new-run --delete-models
```

Use a new output directory. The driver creates only benchmark models, runs the bounded experiment, and deletes its own ONNX files after successful search. It does not start training or install a Unity package.

Evidence is under `artifacts/feature-engineering/deploy-v9-w352-12s-20261016/`: `prepared.json`, `search.json`, `search.log`, `inputs.json`, `validation.json`, and the exact executed driver. Both models' four temporary ONNX graphs were deleted after the test, with hashes/sizes retained in `model-cleanup.json`. The separately requested cleanup deleted 581 older test/smoke/benchmark model files (4,717,111,606 bytes), recorded individually with SHA-256 in `cleanup-receipt.json`. Together: **585 model files, 4,746,556,430 bytes, approximately 4.42 GiB**.

Reports, demonstration datasets, dependency fixtures, tracked `training/models` and `results` checkpoints were retained. Two ambiguous standalone older probe files were retained. Historical test reports that refer to deleted temporary models now require those models to be regenerated before reproducing their old workflows; their recorded evidence was not rewritten.
