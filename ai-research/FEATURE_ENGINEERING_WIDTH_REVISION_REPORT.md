# Compact v9 width and teacher revision

> Follow-up: [encoder reuse and 9/12/12/12 teacher clocks](ENCODER_REUSE_REPORT.md) supersede this report's 9/9/12/16 timings. The width revision and its original evidence below are unchanged.

Implemented 15 October 2026. This follows the completed v8 compaction without changing its feature encoding. Use [the updated lab commands](../docs/FEATURE_ENGINEERING_LAB.md) and [active contract](GROUPED_PROJECTION_CONTRACT.md).

## Effective changes

| Setting | Active value |
| --- | --- |
| Architecture | brass-four-seat-mlp-resnet/9.0.0 |
| Grouped layout | grouped-state-layout/5.0.0 |
| State / action dimensions | 3,128 / 327, unchanged |
| Group inputs | 31 / 218 / 109 / 1,314 / 1,247 / 197 |
| Six projection widths | 32 / 64 / 48 / 192 / 224 / 48 |
| Context passthrough | 12, unchanged |
| Fusion | Linear(620, 304) |
| Backbone / candidate branch | Six width-304 residual blocks / 327 -> 128, unchanged |
| Base / utility / personality parameters | 2,005,917 / 2,015,937 / 2,081,770 |
| Teacher workers per game | execution.teacher_search_workers_per_turn = 8 |
| Parallel teacher games / search worker capacity | 5 / 40 |
| Early Canal / late Canal / early Rail / late Rail clocks | 9 / 9 / 12 / 16 seconds per reserved turn |

New layout SHA-256: f7bd96900365293a314e167256142104ede1f390a9a7820dce95cff91307476b.
The old layout file remains immutable. Layout membership, ordering and feature-manifest hash match v8; only learned widths, fusion size and the layout identity change. Checkpoints and ONNX metadata reject prior architecture/layout identities before tensor loading. No existing run or checkpoint was modified.

## Existing joint7 IL dataset compatibility

The previous lab conversion output **artifacts/il-replay/features4-compact-v8-batch7/manifest.json remains compatible**. There is no need to rerun conversion or regenerate heuristic games. Its directory name need not change. Train fresh v9 IL into a new directory using that same manifest.

The Features-4 manifest remains byte-identical, with semantic digest 59facdd72a386b6046abbb875daf3e1051b92e580ff55d0ab37250dd0c1dfb6a. Both encoders and all feature_engineering source files remain unchanged. The only changed learning-compatibility field is architecture_sha256. The existing audited-heuristic loader already permits that difference while requiring exact rules, encoder, derived-source, target and reference identities and rejecting neural behavior/search targets.

The immutable feature manifest contains the original v8 projection proposal as historical metadata. Active projection widths now belong solely to the separately versioned model/layout contract, which still hashes the complete feature manifest. This avoids invalidating data to change a model capacity setting.

Architecture-bound cache receipts become stale and are safely re-encoded in memory during normal IL loading. Original manifests, raw shards and cache files are not relabeled or overwritten. Initial loading may be slower than a cache hit; this does not repeat native transition auditing. Incompatible older feature versions remain rejected. Previously trained v8 model weights remain incompatible and require fresh v9 IL and self-play directories.

The remote joint7 dataset was not accessible from this local checkout. Compatibility was verified from the unchanged contract and by loading an existing locally converted v8 Features-4 dataset through the production IL loader, without converting it again.

## Teacher timing and allocation

Phase definitions are unchanged: Canal rounds 1-4 / 5-8, Rail rounds 1-6 / 7-8. All eight private search workers receive the same absolute turn deadline. Both actions and continuation choices share it. Cancellation, coverage, label-quality, freshness and calibration guards remain active.

A dedicated optional teacher worker key preserves generic/evaluation search_workers_per_turn=4 and v4 search_workers=4. Configs without the new key retain their existing worker count. Heavy actors reserve enough inference channels for either role; each policy factory creates only its requested role's forest. Five active teacher games use 40 search-worker channels plus five coordinator channels. The shared global CPU admission allowance remains 60; 40 is capacity, not a continuously measured utilization count.

The latest checked-in recipe and configuration generator apply the new settings. Explicit archived recipes and their migration receipts remain unchanged. Teacher reports and compact history retain search_workers_per_game, parallel_game_capacity and total_search_worker_capacity. Existing per-phase attempts, completions, coverage, elapsed-time and distinct-label/distillation-use reporting remain intact.

search_wall_fraction stays Python None / JSON null: only the cumulative allowance is disabled. Per-turn deadlines and failure bounds remain active. Collection estimates resolve the new phase clocks, and policy initialization/failure handling resolves the maximum phase budget (16 seconds).

Other recipe values remain unchanged: five teacher games, cadence 160, two collections per configured iteration, eight distillation steps, minibatch/reuse limits, exploration 0.08, temperature 1, entropy/value weights 0.005/0.5, latest-model four-seat collection, curriculum, disabled champion gate and insolvency event cost 0.005 capped at 0.075. Source-review allowlists were not changed.

## Validation

32 distinct focused regressions passed across the bounded validation commands:

- Immutable Features-4 contract and all input/group indices; new layout identity and rejection of old widths/checkpoint metadata.
- Base, utility and personality parameter/MAC counts; one small forward/backward batch, all group/fusion gradients, legal masks and unchanged six-block/action/head shapes.
- Same-target checkpoint/Adam/RNG/progress restoration and rejection of a same-shaped altered index buffer.
- Existing heuristic dataset admission and identical re-encoded arrays, with unchanged source/cache hashes; incompatible features and neural replay still rejected.
- Teacher-only eight-worker routing, generic/evaluation four-worker routing, five-game/40-worker channel allocation, CPU-limit validation and compact reporting.
- All four shared phase deadlines across eight workers, shared action/continuation budgets, cancellation/overrun disqualification, collection estimates and disabled cumulative limits.
- Latest configuration generation and archived/custom configuration behavior.

One new dataset regression initially used an obsolete synthetic board fixture, which the static-board guard correctly rejected. Its repair then had an incorrect fixture helper call. Both test-fixture issues were fixed without weakening production admission; the affected test passed. Failed logs remain available. Passed checks were not repeated just for documentation.

The isolated smoke **passed in 96.781 seconds**, using the existing v8 conversion at artifacts/feature-engineering/compaction-v8-20261015/smoke/data/manifest.json. This is a complete workflow duration, not an inference benchmark:

- Production loader admitted 345 training and 346 validation examples; the fixture selected eight from each partition.
- One IL epoch and exactly one IL optimizer update completed.
- The actual configure CLI bound both initialization and rehearsal to the new completed IL model and the existing dataset; all other recipe fields matched.
- Self-play utility handoff preserved all IL weights and began with an empty PPO optimizer; the utility checkpoint roundtrip passed.
- Diagnostic personality import preserved the utility core. Named ONNX parity passed for all ten outputs at shapes (1,1) and (2,7), including legal padding and seat remapping. Maximum absolute error was 2.0981e-5, on VP output, within the existing tolerance. Runtime: ONNX Runtime CPU 1.22.1.
- Existing dataset/cache files remained byte-identical. No conversion, new game, PPO iteration or timed teacher collection ran.

Existing ONNX tracer warnings for Python shape guards were observed; both tested dynamic shapes passed. Larger shapes, GPU/Unity deployment, timed teacher yield/coverage, long training and playing strength were not measured. The diagnostic utility/personality heads are not release-qualified.

## Runtime implications

The base gains 101,472 parameters (about 5.33%). Fixed dense state work increases by 101,408 MACs per position to 1,899,168; candidate work remains 97,280 MACs per padded candidate. These are computed counts, not measured latency.

Four seats times the sum of the phase clocks gives 184 reserved seconds per teacher game, or **920 summed seconds per five-game collection**, compared with 256 / 1,280 previously: a 28.125% decrease in nominal allocated turn time. This is neither a wall-time prediction nor a cumulative time cap. Games run concurrently and add ordinary moves, startup, encoding, inference and distillation overhead. Eight workers increase potential CPU demand, so lower wall budgets do not imply lower CPU work or faster collection. The existing capacity/phase/quality reports should be used to assess actual yield.

No improvement in stability, label yield or playing strength is claimed.

## Changed files and receipts

Production changes: learning/model.py, learning/state_layout.py, learning/utility_model.py; new configs/learning/state-groups-5.0.0.json; recovery/config.py, recovery/execution.py, recovery/teacher_collection.py and recovery/league.py; personality/handoff.py and personality/laptop/package.py; configs/step5/joint7-latest.json. The active contract/lab instructions and historical compaction notices were updated. Focused regressions and a reusable small smoke were added; existing target dimension/count assertions were synchronized.

Evidence directory: artifacts/feature-engineering/compact-widths-v9-20261015/.
It contains before.json (preservation inventory), inspect.json, focused.log, admission.log, dataset-test.log, recipe.log, verification.json, smoke.log and smoke/report.json.

Tested implementation identity: 63c9da11e4aa582af8ad41ea32c85781728ff6418c953a2df1170f7718d5d9b4.
