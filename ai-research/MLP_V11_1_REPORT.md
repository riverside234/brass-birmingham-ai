# MLP v11.1: local conditioning before action GELU

Implemented and measured on 2026-10-16. Architecture is now `brass-four-seat-mlp-resnet/11.1.0`. The local vectors enter the action nonlinearity and normalization:

```text
training: cand = LayerNorm(GELU(Linear(action) + target + merchant + industry))
export:   cand = LayerNorm(GELU(Linear(action) + selectors @ projected_banks))
```

The export follows the later requested refinement: **the action matrix stays constant**. It does not construct a position-dependent 327 x 160 effective matrix. ORT sees constant action weights, eligible for prepacking; the simplified graph expresses that operation as a constant-weight Gemm with reshapes. Packing internals were not separately instrumented.

## Execution and unchanged components

Training retains argmax-index gathers with an exact zero row for absent references. The three bias-free role projections remain zero-initialized. Tests confirm the initial policy exactly matches the same model without local action conditioning, and that nonzero roles produce a different policy from v11's post-LayerNorm addition.

Export concatenates the projected target, merchant and industry banks once into `[B,143,160]`. Static slices derived from the action feature manifest select `[B,A,143]`. Their single batched multiply supplies the local contribution. There are **22,880 dynamic bank values per position**, compared with 52,320 values for a full effective action matrix. No absent row, argmax or candidate-dependent gather is exported. State/entity gathers and stable-seat output gathers remain necessary.

The prepared bank and scorer state projection are reused across candidate chunks by `prepare_candidates` / `score_prepared`. A full ONNX forward prepares them once for that call; separate standalone ONNX session calls do not share an activation cache. No cross-position cache or mutable model-weight rewrite was introduced.

The existing ONNX simplifier and four-seat export batching were retained. Own and other player encoders use one grouped operation, preserving their separately learned weights, biases and LayerNorm affine parameters. Earlier profiles contained both player paths; the new profile also records their combined operation. The portable graph remains opset 17. ORT uses **ORT_ENABLE_ALL**, without approximate GELU. The medium utility execution graph has **25 fused exact GELU operations, 16 LayerNormalization operations and no unfused Erf**.

Banks, selectors, layouts, topology, trunk, factored scorer, heads and parameter counts are unchanged:

| Model | Parameters |
| --- | ---: |
| Base | 5,904,644 |
| Utility | 5,921,320 |
| Personality | 6,023,137 |

Training retains 83,232 dense MACs per candidate. Export's one selector multiply adds 22,880 MACs, for 106,112 per candidate; its dispatch/memory tradeoff must be measured rather than inferred from parameter count.

## Admission and training integration

Every selector block must contain strictly 0/1 values and at most one 1. Existing CPU batching, dataset/replay admission and oversized-inference checks are preserved. The named ONNX runtime also checks all input rows, including masked rows, before running. Direct raw ORT callers must obey the same input contract.

V11 weights are **incompatible**, even though their tensor dimensions match. Base/utility loaders and the personality model configuration reject the old architecture before loading weights. Existing source-review flags and historical equivalence receipts cannot bridge this change. The old receipts and all production checkpoints/runs remain untouched.

The audited **Features-4 heuristic IL dataset is reusable unchanged**. Feature, encoder, derived-feature, local layout, rules and target contracts were verified unchanged. The isolated smoke reused an earlier manifest and raw shards without rewriting them. Neural replay and source-bound caches retain their existing freshness checks.

`joint7-latest.json` changes only its architecture field. Config generation from v11 preserves its existing IL/self-play learning rates instead of applying the older width-upgrade factor a second time. Teacher settings, auxiliary value default, exploration, curriculum, reward and optimizer recipe are preserved. Fresh v11.1 IL and a new self-play directory are required; commands are in [the lab instructions](../docs/MLP_V11_LAB.md).

Architecture fingerprint: `fb085f74f1387258ba6fe8a66fdf40deaf6c6010e513f0c3c310e15129a75302`.
Full implementation used for smoke: `647d019877a38a6023926803956bf794186d23a81fd6b6a35124684ca86046e4`.

## Validation

**53 relevant tests passed** across the model/export and source-boundary selections. They cover exact zero initialization; nonzero role effects; parameter and state gradients against an explicit selected-row reference; strict selector admission; masks; absent and combined references; batch sizes 1 and 2; candidate counts 1, 2, 7, 127, 128, 129, 257, 771 and 772; chunk sizes 1, 127 and 256; base/utility/personality integration; dynamic ONNX parity; static action weights; fused execution; old checkpoint rejection; new checkpoint/Adam/RNG roundtrips; and unchanged data/layout semantics. CPU BF16 training-path regression coverage also passed.

The broader invocation ran 74 tests: 50 passed and **24 legacy R2 tests errored in fixture setup**, all at `V11 link topology differs: CanalBelperDerby`, before model construction. Those tests use an older fabricated board fixture. The topology/feature admission code is unchanged by this revision; it was not weakened to admit that fixture. A subsequent source-boundary selection passed all three tests. This is not a claim that the entire repository suite passes.

The isolated lifecycle smoke passed in **109.6 seconds**:

- Loaded and audited two existing heuristic game shards; used eight training and eight validation examples for one fresh width-512 IL optimizer update.
- Copied IL weights exactly into fresh self-play with an empty PPO optimizer.
- Completed one ordinary game, four learner trajectories, 303 decisions and 137 eligible actor decisions, followed by one PPO update with behavior-probability checks.
- Verified matching-source restart preserves Adam/RNG, progress, population and lineage, and exercised a teacher remote leaf inference boundary.
- Imported the personality extension and passed named ONNX export parity. Source dataset bytes stayed unchanged.

This smoke does not exercise a full teacher search collection or establish playing strength. Existing exporter deprecation messages and onnxsim/nanobind process-shutdown diagnostics were emitted; the successful processes returned zero and parity checks passed.

Evidence is under [v11.1 artifacts](../artifacts/feature-engineering/v11-1-20261016/verification.json): [model/export test log](../artifacts/feature-engineering/v11-1-20261016/focused-tests.log), [source-boundary tests](../artifacts/feature-engineering/v11-1-20261016/source-boundary-tests.log), and [lifecycle smoke](../artifacts/feature-engineering/v11-1-20261016/smoke/report.json).

## Bounded CPU benchmark

Both variants implement v11.1 with identical nonzero diagnostic weights, identical valid categorical inputs, seat batching and graph simplification. The reference retains indexed candidate gathers; the requested export uses static action weights plus the local multiply. Windows CPU, Torch 2.8.0+cpu, ORT 1.22.1 CPUExecutionProvider, one intra-op thread, three warmups and 21 alternating measurements per condition. These are synthetic tensor-ready whole-batch timings, excluding feature encoding, admission, transport and search.

| Batch | Candidates | Indexed median ms | Static action + local multiply ms |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 2.891 | 2.687 |
| 1 | 119 | 2.425 | 2.927 |
| 1 | 387 | 5.193 | 5.227 |
| 1 | 772 | 7.849 | 7.550 |
| 2 | 1 | 3.725 | 3.573 |
| 2 | 119 | 4.724 | 4.738 |
| 2 | 387 | 8.514 | 8.551 |
| 2 | 772 | 12.465 | 13.700 |

Results are mixed: for example, 1 x 772 was about 3.8% faster, while 2 x 772 was about 9.9% slower. No general speedup is established. Maximum absolute output error across these cases was **2.67e-5**; legal masks matched. No GPU, long training, full-search throughput or playing-strength evaluation was run.

[Raw timings and parity](../artifacts/feature-engineering/v11-1-20261016/benchmark/report.json) and [operation profiles](../artifacts/feature-engineering/v11-1-20261016/benchmark/profile/PROFILE.md) retain measurement scope and samples. Reproduce with `tests/FeatureEngineering/benchmark_v11_1.py --output <new artifact directory>` in the repository environment.

## Changed files

- `learning/model.py`, `utility_model.py`: architecture identity and training addition before GELU.
- `learning/onnx_tools.py`, `export.py`, `personality/laptop/export.py`: export-only selector multiply, metadata, categorical admission and extended parity shapes; preserve the prior simplification, fusion and player-batching work.
- `recovery/config.py`, `league.py`, `configs/step5/joint7-latest.json`: default identity, rate preservation and correct fresh-model admission message.
- `test_v11_1.py`, `benchmark_v11_1.py`, current architecture/export regressions and `test_v11_selector_resume.py`: correctness, timing and new compatibility boundary.
- The active [model contract](GROUPED_PROJECTION_CONTRACT.md), [lab commands](../docs/MLP_V11_LAB.md) and this report.
