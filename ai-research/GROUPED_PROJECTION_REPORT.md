# Main grouped target v7 — implementation and validation

> Historical v7 document. The active target is now [widened compact grouped v9](GROUPED_PROJECTION_CONTRACT.md). Its subsequent [encoder reuse benchmark](ENCODER_REUSE_REPORT.md) revisits new_encoder/persistent_miss/warm_hit on the current feature contract. Earlier evidence and timings below remain historical.

15 October 2026. **Implemented and validated for the requested bounded scope.** The main target uses **32/80/64/256/256/64**, the unchanged 12-value context tail, **764 -> 304 joint fusion**, and **six width-304 residual blocks**. Only the current target was trained, exported and profiled. This is an experimental capacity choice; playing-strength improvement is unproven.

## Identity and preserved scope

- Architecture: `brass-four-seat-mlp-resnet/7.0.0`.
- Layout: `grouped-state-layout/3.0.0`, `configs/learning/state-groups-3.0.0.json`.
- Layout SHA-256: `6e0285af769eccafbfece19955bb0bec4566e615f9ed59d49bfa12752c446355`.
- Validated implementation ID: `34a62056d352342da5d33662b34e4c35eeb11146c5d1a5be4e6ee3cc6a7141f7`.
- Features remain `permitted-features/2.0.0`, with 5,489 state / 548 action components.

The [layout contract](GROUPED_PROJECTION_CONTRACT.md) contains the full network diagram, semantic group composition and all exact budgets. The existing index mapping, feature values, masks, normalization, ownership, action branch and heads are preserved. Existing manifests, runs and source-review receipts were retained. No installed-game files were changed.

The inspection inventory, source/configuration hashes and all new receipts are in [grouped-v7-20261015](../artifacts/feature-engineering/grouped-v7-20261015/). The final verifier checked unchanged source/configuration files against the inspection inventory and bound the IL, smoke, exports and profile to the final implementation.

## Files and integration

- `learning/state_layout.py`, `learning/model.py` and the new layout JSON define the exact widths, fusion and fresh identities.
- `learning/utility_model.py`, `recovery/league.py`, and `configs/step5/joint7-latest.json` admit the target and report actionable fresh-weight errors.
- Existing partial checkpoint/export admission changes in `learning/checkpoint.py`, `learning/export.py`, `personality/checkpoint.py` and `personality/laptop/export.py` were retained and tested. Architecture checks precede generic source checks; strict tensor/index validation remains active.
- `personality/handoff.py` and `personality/laptop/package.py` include layout 3.0.0. The implementation fingerprint follows the active layout through `step5/league.py`.
- `scripts/feature-engineering.py` inspects the target, binds completed IL to initialization/rehearsal, and now exports the exact utility checkpoint with optional diagnostic personality integration.
- `tests/FeatureEngineering/test_grouped_projection.py` covers the allocation, semantics, gradients, residual initialization, all budgets, one state pass, masking, synthetic incompatible metadata, stale layouts, resume and named exports. Current-count expectations were updated in `test_fe4.py`, `test_six_group_rollback.py`, `tests/R2/test_learning.py` and `tests/Personality/test_personality.py`.
- `tests/FeatureEngineering/smoke_fe5.py` adds explicit isolated current-model table selection, four functional games and teacher label-use accounting. `profile_grouped.py` profiles only the target checkpoint and matching exports.
- This report, the layout contract, revision-plan status and English lab guide describe the completed target.

The checked-in production recipe changed **only `architecture`**. The actual generated target config changed exactly **`architecture`, `imitation_checkpoint`, and `rehearsal_manifest`** relative to inspection. [resolved-config-diff.json](../artifacts/feature-engineering/grouped-v7-20261015/resolved-config-diff.json) records those bindings. PPO, exploration 0.08, temperature 1.0, entropy/value weights 0.005/0.5, curriculum, four-seat sampling, disabled champion gate, insolvency 0.005/0.075, teacher five games/eight steps, cadence/reservations/quality checks, 12/12/16/24 clocks, null cumulative wall fraction, opponent budgets, monitoring, retention and population rules were preserved.

## Absolute parameter and work budgets

| Preset | State encoder | Base | Utility | Personality |
| --- | ---: | ---: | ---: | ---: |
| small, width 128 | 1,534,240 | 1,655,853 | 1,660,241 | 1,695,626 |
| medium, width 304 | 2,585,840 | 2,752,509 | 2,762,529 | 2,828,362 |

The main state encoder contains 1,236,384 projection parameters, 232,560 fusion parameters, 1,116,288 residual-block parameters and 608 final-LayerNorm parameters. All counts match the plan. Existing 3M / 3.1M limits and the personality extension limit remain active.

| Preset | Base fixed MACs | Utility fixed MACs | Personality fixed MACs | Per padded candidate |
| --- | ---: | ---: | ---: | ---: |
| small | 1,547,952 | 1,552,304 | 1,587,504 | 103,040 |
| medium | 2,617,328 | 2,627,312 | 2,692,960 | 125,568 |

Dense work is `B * (fixed + N * per_candidate)`; one MAC is two FLOPs. Gather, activations, normalization, masks, encoding, queues and search are excluded. The [architecture-budget receipt](../artifacts/feature-engineering/grouped-v7-20261015/architecture-budget.json) includes tensor byte counts.

## Tests, data and bounded smoke

**291 focused tests passed in 139.535 seconds.** The suites include PPO behavior-probability reconstruction, forced-action filtering, source/cache admission, teacher clocks, retention, personality import, package/runtime contracts and dynamic ONNX parity. Target rejection fixtures alter current metadata/tensors; no previous model is constructed for admission tests. ONNX emitted its existing exporter/tracing warnings; checked dynamic batches/candidates and all named outputs passed.

Two complete qualified heuristic games were re-audited with two native workers in **25.734 seconds**: train seed **17100000**, 345 positions / 13,595 candidate rows; validation seed **17100001**, 346 positions / 14,271 candidate rows. All 691 raw records, legal lists, labels, game identities and splits remained exactly equal to the source games. No demonstration games were regenerated.

The isolated smoke took **229.672 seconds**: one fresh width-304 IL epoch, normal completion/handoff, two one-game PPO iterations, four additional functional games and four teacher turns. IL initialization seed was **812771**; PPO setup seeds were **18510000/18510001**, sampling seeds **129570000/129570001**, with normal rotations 0/1. Each ordinary table used four current-model seats at exploration 0.08.

IL completion resumed without changing its checkpoint/progress. PPO inherited the exact selected IL weights with an empty PPO optimizer, then restored **68 Adam states and exact RNG state** at the second-iteration boundary. Committed progress reached iteration 2 / two ordinary games; the previous checkpoint, population generations 0/1/2, curriculum origin and IL lineage were preserved.

| PPO iteration | Eligible actor decisions | Distinct actor rows / presentations | Unused eligible actor rows | Distinct critic rows / presentations |
| --- | ---: | ---: | ---: | ---: |
| 1 | 139 | 139 / 139 | 0 | 192 / 192 |
| 2 | 138 | 138 / 138 | 0 | 192 / 192 |

Four additional games used the same frozen target, validation setup **54121**, all seat rotations and sampling seeds **378847–378850**. **All four completed; 1,203 choices were legal; zero failures.** They provide functional coverage only.

Smoke-only settings: one ordinary game per iteration, one CPU actor with two host lanes, one heavy actor, two tensor threads, one audit reader, update batch 64 / microbatch 32 / critic batch 64 / KL probe 128 / padded-candidate cap 8192 / one epoch. Rehearsal, scheduled evaluation and retention were disabled in this isolated output. A test-local selector forced current-model self-play; production curriculum validation was unchanged. Historical/legacy collection, full scheduled distillation and scheduled evaluations were not exercised by this smoke.

## Teacher phase probes

Each phase received one reserved turn using the same calibrated target checkpoint, four training search workers and the unchanged shared clock. Probe world seed was **65413**; reservation/sampling seeds are in the receipts. No quality, coverage, freshness or timeout requirement was relaxed.

| Phase / budget | Attempts / completed / interrupted | Visited / legal roots (revisited) | Qualified labels | Insufficient coverage | Timeouts / overrun seconds | Search / whole-turn seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| canal_early / 12s | 16 / 8 / 8 | 4 / 387 (2) | 1 | 0 | 0 / 0.000 | 11.875 / 12.031 |
| canal_late / 12s | 17 / 9 / 8 | 4 / 29 (2) | 1 | 0 | 0 / 0.000 | 11.922 / 12.204 |
| rail_early / 16s | 16 / 8 / 8 | 4 / 184 (2) | 1 | 0 | 0 / 0.000 | 15.875 / 16.047 |
| rail_late / 24s | 46 / 39 / 7 | 3 / 3 (3) | 1 | 0 | 0 / 0.000 | 23.922 / 24.109 |

**Four qualified labels generated, zero distinct labels trained, zero presentations/reuse, four unused labels, and zero teacher-induced policy drift.** Distillation training was unexercised. These are diagnostic partial-turn labels, not a complete five-game collection. Whole-turn elapsed includes preparation/finalization and legal continuations outside the enforced search interval. The inference service recorded six expired requests at deadlines and no service error. Limited root coverage and one observation per phase do not establish general label yield or sufficient search.

## Same-checkpoint export parity

The final utility checkpoint and its initialized personality extension were exported as float32, opset-17 named graphs. The extension imports the exact utility parent; its context injection is zero-initialized and goals are untrained. Both graphs checked every named output, stable-seat mapping and legal mask at **(B,N) = (1,1), (2,7), (4,31), (1,257)**. The profile loaded the matching actor checkpoint and rechecked the stored backend parity arrays.

- utility: passed; maximum absolute output error **3.0517578e-05**.
- actor: passed; maximum absolute output error **4.196167e-05**.

Utility checkpoint SHA-256: `bb74c5356d1292e9e1289adb50b38293338447486778c0f8ce62a1dfed29e08e`. Metadata binds architecture, layout hash, exact widths, reward, source compatibility and model identity. These diagnostic exports do not qualify a personality roster or game installation.

## Target-only CPU profile

Hardware: **Intel(R) Core(TM) Ultra 5 135U**, Windows-11-10.0.26200-SP0; Python 3.12.14, PyTorch **2.8.0+cpu**, ONNX Runtime **1.22.1**, **CPUExecutionProvider, two tensor threads, two laptop search workers**. The fixed permitted corpus contains 14 positions spanning both eras and 1–387 legal choices, constructed with fixture seeds 1201/99. Candidate order and encoded-input hash are retained in the receipt.

Four microbenchmark blocks used three warmups, then 15 pure forwards or seven complete requests per shape/block. The table reports pooled **median / empirical p95 milliseconds**, 60 samples per pure-forward cell and 28 per complete-request cell. Complete requests include validation, encoding/cache lookup, queueing, transport, batching and the personality graph. Pure utility and actor timings include gather/projection/fusion.

| B × padded N | Torch utility | ORT utility | Torch actor | ORT actor | Complete actor request |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 × 1 | 3.259 / 4.543 | 0.801 / 1.276 | 3.780 / 4.729 | 0.792 / 0.970 | 16.608 / 18.806 |
| 1 × 333 | 4.656 / 6.057 | 1.613 / 2.481 | 4.956 / 5.955 | 2.131 / 2.659 | 17.469 / 32.629 |
| 1 × 119 | 3.787 / 4.791 | 1.138 / 1.797 | 3.950 / 5.299 | 1.114 / 1.859 | 16.546 / 18.724 |
| 2 × 6 | 3.707 / 4.942 | 0.870 / 1.539 | 4.271 / 5.948 | 1.007 / 1.798 | 16.709 / 32.546 |
| 2 × 387 | 6.261 / 9.061 | 2.550 / 4.118 | 7.141 / 9.000 | 3.932 / 6.776 | 32.920 / 37.982 |
| 2 × 119 | 4.311 / 7.147 | 1.509 / 2.473 | 4.497 / 6.727 | 1.844 / 3.320 | 18.159 / 33.073 |
| 8 × 6 | 3.304 / 6.194 | 1.030 / 1.937 | 4.273 / 6.032 | 1.318 / 2.702 | 49.769 / 67.052 |
| 8 × 387 | 16.706 / 24.857 | 12.706 / 15.856 | 16.671 / 20.132 | 12.771 / 17.593 | 78.698 / 116.551 |
| 8 × 119 | 7.905 / 11.086 | 3.439 / 5.191 | 7.999 / 9.749 | 5.033 / 6.368 | 51.112 / 67.770 |

Block-to-block variation is material. ORT utility block medians ranged **0.753–0.828 ms** for B1/N1 and **11.460–13.958 ms** for B8/N387; complete B8/N387 request block medians ranged **65.045–86.777 ms**. All raw samples and cold encoder-cache request timings are retained. These CPU observations do not measure GPU training throughput or establish a speedup.

| Feature encoding path | Samples | Median / p95 ms | Min–max ms |
| --- | ---: | ---: | ---: |
| new_encoder | 168 | 8.603 / 31.777 | 4.219–55.824 |
| persistent_miss | 168 | 4.722 / 22.129 | 2.572–33.933 |
| warm_hit | 168 | 0.860 / 1.680 | 0.489–2.927 |

One fresh-worker observation: **16.778s** from worker startup through imports, models, graph/parity service, native workers and corpus encoding (**17.391s** including process launch). Checkpoint load was **0.170s**; portable service startup/parity was **0.341s**. OS file cache was uncontrolled; one startup sample is not a tail-latency estimate.

Storage/memory scopes: utility float32 parameters **11,050,116 bytes**, grouping buffer **43,912 bytes**, full optimizer-bearing checkpoint **33,282,915 bytes**, utility ONNX **11,148,331 bytes**, actor ONNX **11,416,215 bytes**. A separate, untimed encoding pass retained **3,765,730 Python-tracked bytes**, peaking at **4,863,531 bytes**. A separate untimed target inference process reported **388,501,504 bytes RSS**, **388,501,504 bytes peak working set**, and **898,125,824 bytes private commit** through the native Windows reader. That process loaded both utility/actor Torch models and both ORT graphs and executed B8/N387 forwards; it excludes search child processes and Unity. The timing run originally lacked the optional psutil reader; its samples were retained unchanged, with the exact executed driver saved alongside them. The maintained profiler now has a native Windows fallback. Python allocation tracking excludes most Torch/ORT native allocations and is not total process memory.

## Whole-turn deployment search work

Two roots (Canal and Rail), two repeats each, world/sampling seeds **57744, 57738, 57845, 57839**, under each existing 2/5/12-second clock. The target utility model supplies rollout values; the target actor has an initialized goal branch with bias disabled (`alpha=0`). Processes and inference caches stay warm; each turn starts a new search. The deadline is shared across both actions/subchoices.

| Budget | Turns | Completed simulations per turn | Interrupted total | Fallbacks | Whole-turn median / max s | Completed / elapsed s | Search overruns |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2s | 4 | 0, 0, 0, 0 | 16 | 4 | 2.118 / 2.172 | 0.000 | 0.000s |
| 5s | 4 | 0, 0, 0, 0 | 16 | 4 | 4.821 / 4.954 | 0.000 | 0.000s |
| 12s | 4 | 4, 4, 4, 4 | 16 | 0 | 11.188 / 11.250 | 0.357 | 0.000s |

**All 12 turns completed both legal actions with no shared-clock reset.** Zero completed work at the tested 2/5-second settings is a measured limitation; the controller used its existing legal fallback. At 12 seconds each tested turn completed four simulations. Whole-turn totals include finalization and follow-up choice application; search-overrun telemetry refers to the enforced search interval. No budgets or fallback/quality gates were changed. Search labels from these probes are diagnostic and are not used for training.

## Reproduction and limits

Use [the English lab guide](../docs/FEATURE_ENGINEERING_LAB.md) for data audit/re-cache, fresh IL, config generation, new self-play, named export, target-only profiling and same-architecture continuation. [commands.json](../artifacts/feature-engineering/grouped-v7-20261015/commands.json) records the local environment and executed commands. Primary receipts are `architecture-budget.json`, `data/conversion.json`, `tests.log`, `smoke/report.json`, `exports/receipt.json`, `profile/report.json`, `resolved-config-diff.json` and `review.json` under the evidence directory.

Functional correctness and the requested local profile are verified. Full scheduled distillation, production historical/legacy table execution in this smoke, GPU throughput, aggregate search-process memory and full personality deployment qualification remain unmeasured here. The existing focused safeguards passed. No playing-strength or training-stability improvement is claimed.
