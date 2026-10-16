# V11 indexed selectors and batching verification

Measured on 2026-10-16. The selector change preserves the v11 model's parameters, feature inputs and learned computation for valid categorical references. It reduces dense arithmetic, but the measured CPU latency improvement is not consistent across backends or shapes.

## Implementation

`FourSeatNet.prepare_candidates` projects the target, merchant and industry banks once per position and prepends an exact all-zero row to each. `reference_context` converts the existing selector fields to indices inside the graph and gathers the corresponding projected embeddings. A zero entry prepended to each selector makes an absent, all-zero reference select row zero under first-maximum argmax semantics. A present one-hot selects its real bank row, shifted by one. Flattened banks and batch offsets prevent references from crossing between observations.

This removes the three candidate-by-bank matrix multiplications. Selector offsets are cached at model construction; forward execution does not look up feature names. There is one lookup per reference family, with no loop over candidates.

The native/reference and prepared encoders were checked on real game fixtures. The selector contract is binary one-hot **or all-zero**, not arbitrary real-valued weights. CPU admission checks reject multi-hot, soft, negative and nonfinite selectors in batching, oversized inference, dataset admission and replay/cache admission. Direct tensor/ONNX callers must also honor this categorical input contract; the graph does not run those CPU validation checks.

The existing 327-dimensional action representation is retained, including the features consumed by the action encoder. Indices are derived inside the graph rather than added as public inputs. Consequently, this change **does not shrink PPO's padded input tensors** or require regenerating Features-4 demonstrations.

### Entity and seat batching

These paths were already batched and have been preserved, with focused regression coverage:

- Each of the seven entity/player families uses one precomputed index buffer, one batched gather and one shared `Linear -> GELU -> LayerNorm` encoder call. There is no per-entity forward loop.
- The three other players are encoded together from `[B, 3, 287]`.
- The shared outcome head computes its global-state contribution once, broadcasts it over the four player representations, and processes the resulting `[B, 4, 128]` tensor together.
- Entity encoding, role projection and the policy scorer's state contribution remain reusable across candidate chunks.

## Parameters and arithmetic

| Model | Parameters, unchanged | Fixed MACs per position |
| --- | ---: | ---: |
| Base | 5,904,644 | 8,235,232 |
| Utility | 5,921,320 | 8,251,872 |
| Personality actor | 6,023,137 | 8,353,504 |

Per-candidate dense work changes from **106,112 to 83,232 MACs**, removing **22,880 MACs (21.56%)**. The removed work is `(128 + 9 + 6) * 160`. For the base model this is a 13.4% reduction in total counted MACs at 128 candidates and 18.1% at 400 candidates.

These are arithmetic counts, not latency predictions. They exclude argmax, gathers, memory movement, nonlinearities, normalization, feature encoding and input validation. One MAC counts as two FLOPs.

## Validation

The final current-source focused suite passed **21 tests in 32.213 seconds**. It covers:

- Every target/merchant/industry bank row, absent references and independent batch offsets.
- Real encoder/prepared-encoder agreement and malformed-selector rejection.
- Outputs, parameter gradients, state-input gradients and a bounded optimizer update against the preceding dense selector implementation. Role projections and residual/utility branches were deliberately made nonzero, so their initial zero initialization cannot hide errors.
- CPU BF16, utility and conditioned personality paths; shared family/four-seat batching.
- Parameter/state-dictionary identities, checkpoint loading, optimizer/RNG restoration and source admission.
- Native-fixture behavior-probability verification and a bounded PPO update.
- Dynamic-batch/dynamic-candidate ONNX export and output parity.

An earlier broader selection completed 42 tests with one CUDA-only skip. The final 21-test rerun is the current-source evidence; this was not a full repository test campaign. No long training, GPU benchmark or playing-strength evaluation was launched.

A separate read-only check loaded the actual preceding v11 smoke run at generation 1, with one committed game and **112 Adam state entries**. Model identity, RNG and persisted warmup progress were preserved, and its completed IL checkpoint was accepted. The original checkpoint and run-state files remained byte-identical. The feature encoder and active recipe configuration also remained byte-identical to their pre-change snapshots.

Evidence:

- [Final focused test log](../artifacts/v11-selector-current-source-tests.log)
- [Existing-checkpoint verification](../artifacts/feature-engineering/v11-selector-20261016/existing-v11-checkpoint-check.json)
- [Source review receipt](../artifacts/feature-engineering/v11-selector-20261016/source-review.json)
- [Paired benchmark data and samples](../artifacts/feature-engineering/v11-selector-20261016/paired-benchmark-final/report.json)

## Paired CPU inference benchmark

Both implementations ran with identical weights and inputs in the same process. The dense comparison reproduces the preceding three one-hot bank multiplications. Diagnostic weights activate local role, residual, utility and personality branches; they are not trained weights.

Environment: Windows 11, Intel Family 6 Model 170 CPU, PyTorch 2.8.0 CPU, ONNX Runtime 1.22.1 CPUExecutionProvider, FP32, two intra-op threads and one inter-op thread. ORT uses sequential execution with spinning disabled. Each variant/shape has three warmups and 21 timed samples in three alternating blocks. The bounded export/parity/benchmark run took 32.640 seconds.

Inputs are the saved v9 benchmark positions `rail-action` (119 candidates) and `settled-next-round` (387 candidates), reused for both v11 implementations. Timings measure tensor-ready inference, excluding encoding, admission, transport, action sampling and search. They are not end-to-end game or PPO throughput measurements.

All values below are median milliseconds. Positive ORT savings mean lower latency.

| Head | Batch | Candidates | ORT dense | ORT indexed | ORT savings | Torch dense | Torch indexed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Utility | 1 | 119 | 3.700 | 3.307 | +10.6% | 10.389 | 10.743 |
| Utility | 1 | 387 | 4.253 | 5.074 | -19.3% | 11.452 | 13.806 |
| Utility | 2 | 119 | 4.251 | 4.046 | +4.8% | 13.946 | 14.532 |
| Utility | 2 | 387 | 10.320 | 9.174 | +11.1% | 23.773 | 22.372 |
| Utility | 4 | 119 | 7.885 | 6.871 | +12.9% | 19.612 | 20.758 |
| Utility | 4 | 387 | 10.446 | 11.399 | -9.1% | 18.940 | 22.073 |
| Utility | 8 | 119 | 12.692 | 11.220 | +11.6% | 27.126 | 30.981 |
| Utility | 8 | 387 | 17.916 | 17.784 | +0.7% | 28.356 | 30.234 |
| Actor | 1 | 119 | 2.823 | 3.263 | -15.6% | 10.683 | 11.473 |
| Actor | 1 | 387 | 4.728 | 4.318 | +8.7% | 11.984 | 13.242 |
| Actor | 2 | 119 | 4.191 | 3.779 | +9.8% | 13.653 | 14.674 |
| Actor | 2 | 387 | 7.122 | 6.359 | +10.7% | 15.156 | 17.327 |
| Actor | 4 | 119 | 6.347 | 6.028 | +5.0% | 15.205 | 16.199 |
| Actor | 4 | 387 | 12.779 | 10.105 | +20.9% | 19.116 | 21.702 |
| Actor | 8 | 119 | 9.991 | 10.680 | -6.9% | 18.842 | 21.148 |
| Actor | 8 | 387 | 17.850 | 18.156 | -1.7% | 27.224 | 33.731 |

Across these equally weighted shapes, the geometric mean indexed/dense latency ratio is **0.9602 for ORT** (about 4.0% lower; 11 of 16 shapes faster) and **1.0986 for CPU PyTorch** (about 9.9% higher; only one shape faster). The reduced MAC count therefore does **not** establish a universal speedup; CPU PyTorch regressed in this small benchmark. Gathering, index construction and execution overhead are not represented in the MAC formula, and this benchmark does not isolate their individual costs. GPU and actual training throughput remain unmeasured.

The ORT dense/indexed outputs matched exactly on all 16 measured fixture shapes. The four exports also passed PyTorch/ORT parity for batch/candidate shapes `(1,1)`, `(2,7)`, `(4,31)` and `(1,257)`, with maximum absolute error across reported outputs of `4.38690185546875e-05`. Utility ONNX MatMul nodes decrease from 18 to 15; personality MatMul nodes decrease from 19 to 16. Each indexed graph adds three ArgMax operations. Temporary benchmark ONNX files were removed after successful checks; parity arrays, graph metadata and raw timings are retained.

## Compatibility and use

Architecture `brass-four-seat-mlp-resnet/11.0.0`, local layout `grouped-mlp-local-entities/2.0.0`, learned tensor names/shapes and feature/data contracts remain unchanged. A narrowly scoped [source-equivalence receipt](../configs/learning/v11-selector-source-equivalence.json) maps this exact optimized model source to the preceding v11 architecture fingerprint. Unknown model edits retain a different identity; old v9/v10 checkpoints are not admitted by this receipt.

Existing v11 weights and completed IL remain usable. For a committed self-play run from the immediately preceding reviewed v11 source, add `--resume-reviewed-source` to its usual continuation command on the first continuation with this source. Existing pending-collection and unrelated-configuration checks still apply. This is not a bypass for arbitrary source changes or cached old-source PPO data. Fresh runs need no new selector setting. Re-export an existing v11 checkpoint to obtain an ONNX graph containing the indexed selectors. See [v11 lab commands](../docs/MLP_V11_LAB.md).

The benchmark records full implementation `73ae0442...`. Concurrent shared-loader edits subsequently replaced the model factory dispatch with direct MLP dispatch, producing `77d472dc...`; the v11 model source remained identical. The final 21-test run and actual preceding-checkpoint check use that latter source. The exact hashes and MLP-only source transition are retained in the receipts; they do not approve Transformer checkpoint conversion.

## Changed files for this optimization

- `learning/model.py`, `learning/local_entities.py`: indexed selectors, admission helper and updated MAC accounting.
- `learning/batching.py`, `recovery/updates.py`, `recovery/replay.py`: selector admission at existing CPU data boundaries.
- `personality/laptop/export.py`: valid categorical export/parity sample inputs.
- `learning/contracts.py`, `step5/league.py`, `personality/laptop/package.py`, and the three learning source-review JSON files: exact architecture equivalence, completed-IL handoff, committed resume and bundle/source receipt inclusion.
- `test_v11_selectors.py`, `test_v11_selector_resume.py`, `benchmark_v11_selectors.py`: focused correctness, continuation and paired benchmark coverage. Existing v11/factored/grouped/compaction regressions were adjusted for zero bank rows, valid selectors and updated MACs.
- This report, `GROUPED_PROJECTION_CONTRACT.md`, `MLP_V11_REPORT.md` and `docs/MLP_V11_LAB.md`: behavior, compatibility and measured limitations.

This change is computationally equivalent for valid inputs within floating-point tolerances. These checks do not demonstrate improved playing strength or training stability.
