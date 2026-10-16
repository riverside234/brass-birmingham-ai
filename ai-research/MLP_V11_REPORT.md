# MLP v11 implementation and validation

Subsequent update: the [v11 selector optimization](MLP_V11_SELECTOR_REPORT.md) replaces one-hot bank multiplications with indexed gathers while preserving the v11 checkpoint identity and parameters. Current dense work is **83,232 MACs per candidate**; arithmetic/timing figures below record the original v11 implementation unless explicitly marked otherwise.

Date: 2026-10-16. V11 is now the current MLP default. The change implements the requested wider grouped model and local neighborhoods, while keeping Features-4 state/action values and the existing training recipe except for the requested learning-rate/warmup/clipping adjustment. No existing training run was upgraded in place.

## Implemented model

| Component | V11 |
| --- | --- |
| Architecture / local entity layout | brass-four-seat-mlp-resnet/11.0.0 / grouped-mlp-local-entities/2.0.0 |
| Learned projection layout | grouped-state-layout/6.0.0 |
| Groups + context | 32 / 64 / 48 / 320 / 352 / 48 + 12 |
| Fusion / trunk | 876 → 512; eight width-512 residual blocks; final LayerNorm |
| Action encoder / policy scorer | 327 → 160; (512 + 160) → 192 → 1 |
| Slot / link / industry inputs | 72 / 148 / 219 |
| Merchant / market inputs | 7 / 2, unchanged |
| Entity banks / role projections | Family-shared width 64; bias-free 64 → 160 roles initialized to zero |
| Player encoders / outcome head | 284 / 287 → 64 with relative-seat embedding; shared 576 → 128 → 3 |
| Utility residual head | 516 → 32 → 4 |

State and candidate input dimensions remain **3,128 / 327**. The existing scorer factorization, once-per-position bank encoding, candidate chunk reuse, legal masks and stable-seat outputs remain supported. Local banks are assembled from existing state values; no new observations or hidden information are introduced.

The static topology is pinned to the native board definition. Slot neighborhoods put the selected slot first, then schema-ordered siblings, then zero rows. Link neighborhoods include every endpoint region, including the three-region Worcester/SouthWest/Kidderminster junctions. Their maximum neighborhood is seven slots, fitting the requested eight-slot padding. Industry inputs include own mat, three relative opponents' mats, then own derived facts. Admission rejects changed topology and checkpoint gather buffers.

| Model | Exact parameters | Ceiling |
| --- | ---: | ---: |
| Base | **5,904,644** | 6,500,000 |
| Utility | **5,921,320** | 6,500,000 |
| Personality | **6,023,137** | 6,600,000 |

The state encoder accounts for 5,550,656 parameters. The unchanged personality head's extension allowance is now 110,000 because width 512 makes that extension 101,817 parameters. The small diagnostic preset remains available (base 1,499,780), but lab IL and self-play use the medium width-512 preset.

## Data and training

**Reuse the existing audited Features-4 IL dataset as-is.** The model, encoder and admission paths were checked with an existing Features-4 manifest and raw shards; their hashes were unchanged after the smoke test. Only architecture identity changes in learning compatibility. Feature identities remain:

| Identity | SHA-256 |
| --- | --- |
| Features | 59facdd72a386b6046abbb875daf3e1051b92e580ff55d0ab37250dd0c1dfb6a |
| Encoder | 7d21b3052bf65c2d272a59859eebc6b3be1117b77e906a677402784b5ab90f06 |
| Derived features | 47afe02b5e98549081f5ab119a2dca2c90448c74fba8020fc7e04cb6bc16f260 |
| V11 architecture source | c7df5bf97fbe02f605aabeea89c9cc24d74981c5a4355a15f050f6b97f0cae39 |

The existing architecture-only exception applies to audited heuristic demonstrations. It does not relabel neural behavior, PPO trajectories or teacher targets. Old source-bound tensor caches are rebuilt through normal admission. The original feature layout v5 and the separately implemented Transformer's feature/token metadata remain intact; the new MLP projection widths live in layout v6.

Use **fresh v11 IL weights and a new self-play directory**. V9/v10 checkpoints fail architecture admission; source-review flags cannot bypass that. Fresh PPO copies completed v11 IL weights exactly, then starts its own optimizer. Same-run resume preserves Adam, RNG, counters, population and lineage.

Provisional defaults:

| Setting | IL | PPO / shared-network self-play updates |
| --- | ---: | ---: |
| Peak LR | 0.00006 | 0.000006 |
| Ratio to previous width-304 recipe | 0.6 | 0.6 |
| Linear warmup length | 500 updates | 100 updates |
| First update fraction of peak | 0.1 | 0.1 |
| Gradient norm clipping | 1.0 | 1.0 |

Warmup state is persisted in Adam's parameter group, checked against the selected configuration, restored on resume and retained in full/compact reporting. It advances on successful shared-network updates, including distillation when enabled. Existing head-only utility calibration keeps its temporary LR and does not advance warmup. There is no adaptive or post-warmup schedule. IL command-line defaults now read the selected configuration instead of retaining a hard-coded old LR.

The current teacher budgets (9/12/12/12 seconds), eight workers per teacher turn, five games per collection, cadence, eight distillation steps, null cumulative wall fraction, teacher auxiliary value behavior, PPO objectives, exploration, curriculum, reward and opponent search settings are preserved. The separate Transformer plan's no-teacher recipe is not applied to this MLP revision.

## Validation

The focused selection covers model/layout identities and exact dimensions/counts; old-checkpoint and altered-buffer rejection; neighborhood order, padding and junction links; reference/prepared encoder admission; zero-role initialization and learnable gradients; factored scorer forward/backward/chunk behavior; PPO probability reconstruction; data semantics; warmup validation and exact Adam/RNG continuation; and dynamic ONNX parity.

**52 distinct focused tests passed; one CUDA BF16 test was skipped because this runtime is CPU-only.** The initial 53-case run found one stale test assertion comparing the new learned layout v6 with the unchanged feature layout v5. It was corrected and passed in a focused export rerun. A final ten-test run also passed after correcting the layout object's default metadata version. These are focused results, not a claim that the entire repository suite was run.

Logs:

- `artifacts/v11-integration-tests.txt` — initial selection, including the obsolete assertion.
- `artifacts/v11-export-retest.txt` — corrected export check passed.
- `artifacts/v11-final-metadata-tests.txt` — ten final model/topology/warmup/export/Transformer-isolation checks passed.

The bounded lifecycle smoke passed in a fresh scratch directory, with an outer elapsed time of **201.1 seconds**, including interpreter startup. It reused one previously audited training game and one validation game, selected eight decisions from each for a single IL update, then collected one new ordinary self-play game and performed one PPO update.

| Smoke check | Result |
| --- | --- |
| Existing manifest/raw shards | Unchanged; no conversion or native re-audit |
| IL → PPO weights | Exact; PPO Adam initially empty |
| Actual ordinary collection | One table, four learner trajectories, 303 learner decisions |
| PPO sample usage | 137 distinct non-forced actor decisions; 64 distinct critic decisions |
| Sampling | Exploration 0.08, temperature 1.0; strict behavior validation passed |
| Warmup | IL counter 1/500, effective LR 6e-6; PPO counter 1/100, effective LR 6e-7 |
| Gradient clipping | PPO pre-clip norm 1.9658; configured cap 1.0 |
| Restore | Exact model and RNG; 112 Adam states; generation 1 / one game / population generations 0 and 1 preserved |
| Saved artifacts | State/checkpoint bytes unchanged by restore |
| Teacher integration | Remote neural leaf inference through one spawned service worker passed; no timed search campaign |
| Personality / export | Parent import and ORT CPU dynamic-shape parity passed |

Ordinary collection took 64.8 seconds; the PPO update took 3.7 seconds on this CPU fixture. Policy entropy was 3.0313 nats over the actor samples; this near-initialized, one-game measurement does not indicate strength. Personality export checked batch/candidate shapes (1,1) and (2,7); maximum absolute policy error was 2.16e-7 and maximum VP error was 3.63e-5. Existing tracer warnings were accompanied by successful dynamic-shape checks.

Receipt: [final smoke report](../artifacts/feature-engineering/mlp-v11-20261016/final-smoke/report.json). An earlier scratch smoke was invalidated when a metadata source fix landed during collection; the freshness guard rejected that collection before PPO. The final receipt comes from a separate clean run after source changes finished. No existing training checkpoints were modified.

The inspection/configuration CLIs were also exercised using the smoke's completed IL run. Generated self-play configuration differs from the latest recipe only in its IL checkpoint and rehearsal manifest bindings: [inspection](../artifacts/feature-engineering/mlp-v11-20261016/model-inspection.json), [generated config](../artifacts/feature-engineering/mlp-v11-20261016/verified-generated-config.json).

## Inference benchmark and limits

Windows 11, Intel Family 6 Model 170 CPU, PyTorch 2.8.0+cpu, FP32, two intra-op threads and one inter-op thread. Each shape used ten warmups and fifty measured forwards. Inputs were synthetic tensors with one-hot local references, and the full base policy/four-seat heads ran. Feature extraction, game rules, IPC and action sampling were excluded.

| Batch | Legal/padded candidates per position | Median | p95 |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 9.19 ms | 16.33 ms |
| 1 | 128 | **10.31 ms** | 25.14 ms |
| 1 | 512 | 12.97 ms | 27.29 ms |
| 8 | 128 | 17.62 ms | 36.68 ms |

The batch-eight value is per batch. The laptop timings have visible variance and are not an end-to-end deployment estimate. No concurrent v10 comparison or speedup claim is made. Base dense arithmetic is `B × (8,235,232 + A × 106,112)` MACs; the larger trunk and neighborhoods increase work and parameter/optimizer memory over v10. Existing inference worker/cache limits should be sized for the larger model when training at scale.

Receipt: [inference benchmark](../artifacts/feature-engineering/mlp-v11-20261016/inference-benchmark.json). No long training, playing-strength evaluation, GPU validation, timed teacher collection or Unity/OpenVINO deployment campaign was run. Improved playing strength and training stability remain unproven.

A subsequent [ONNX Runtime measurement](MLP_V11_ONNX_RUNTIME_REPORT.md) reused the exact prior-v9 fixtures and two-thread CPU settings. Across two bounded runs, v11 utility batch-one medians were **2.75–5.68 ms** for 119 candidates and **3.85–8.47 ms** for 387 candidates; personality medians were **2.86–2.91 ms** and **4.35–4.86 ms** respectively. ONNX Runtime was faster than PyTorch in every paired timing cell. The historical v9 numbers were lower, but the order/time variation prevents isolating an exact architecture slowdown. These measurements cover initialized graphs and exclude encoding, queues and search; they do not replace the earlier Torch-only benchmark's measurement scope.

## Files and commands

- Model/layout/topology: `src/brass_ai/learning/{model,state_layout,local_entities,board_topology,contracts,utility_model}.py`; new `configs/learning/{state-groups-6.0.0,local-entity-topology-2.0.0}.json`.
- Admission/provenance: `src/brass_ai/runtime_contracts.py`, `src/brass_ai/recovery/{encoding,replay}.py`, `src/brass_ai/step5/league.py`.
- Training: `src/brass_ai/recovery/{learning_rate,config,demonstrations,league,resume,updates,history}.py`, `src/brass_ai/step5/cli.py`, `configs/step5/joint7-latest.json`.
- Export/package and inspection: `src/brass_ai/personality/{model,laptop/package}.py`, `scripts/feature-engineering.py`.
- Focused checks: `tests/FeatureEngineering/{test_mlp_v11,test_v11_architecture,test_factored_scorer,test_grouped_projection,test_compact_widths,test_compaction,test_six_group_rollback,test_fe4,smoke_fe5,smoke_mlp_v11,benchmark_mlp_v11}.py`; `tests/Recovery/test_v11_learning_rate.py`; updated current-version assertions in `tests/R2/test_learning.py` and `tests/Recovery/test_training_performance_resume.py`.
- Documentation: this report, `GROUPED_PROJECTION_CONTRACT.md`, `docs/MLP_V11_LAB.md`, and current-target links in the prior lab documents.

The workspace already contained unrelated Transformer, teacher and deployment changes; those remain present. The file list above describes v11 integration touchpoints rather than attributing every existing dirty-file change to this revision.

Follow [the v11 lab instructions](../docs/MLP_V11_LAB.md) for exact fresh IL, fresh self-play and continuation commands. No conversion command is required for the existing compatible Features-4 dataset.
