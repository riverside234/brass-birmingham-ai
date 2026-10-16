# Grouped MLP v10 implementation and validation

Date: 2026-10-16. Architecture: `brass-four-seat-mlp-resnet/10.0.0`.

This implements the two requested MLP changes: direct action references to local entities, and a shared player-conditioned official outcome head. It does not implement the Transformer architecture or its different self-play recipe. Existing unrelated workspace changes were retained.

## Model changes

The external input contract remains **state 3,128 / action 327**, Features-4 and grouped layout 5. The six projection widths remain **32 / 64 / 48 / 192 / 224 / 48**, with a 12-value context passthrough, fusion **620 → 304**, and **six width-304 residual blocks**. The action encoder stays width 128.

Local views are gathered by feature name from those existing state values. Each entity family has a shared `Linear → GELU → LayerNorm` encoder producing 64 values:

| Family | Entities | Raw values per entity |
| --- | ---: | ---: |
| Building slot | 49 | 18 |
| Link | 69 | 4 |
| Merchant | 9 | 7 |
| Observer industry/development | 6 | 111 |
| Coal/iron market | 1 | 2 |
| Observer public player facts | 1 | 284 |
| Other public player facts | 3 | 287 |

The existing action target, merchant and industry selectors select their corresponding local banks. Separate bias-free 64 → 128 role projections are added to the existing action embedding before scoring. All-zero references contribute exactly zero. Existing action identities, card information and derived consequences remain in the action encoder; no new feature calculation, private observation, search or rules call is added. Empty building slots remain represented.

Global encoding, local encoding, reference-bank projections and the factored policy state contribution are computed once per position, including when candidates are processed in chunks. Prepared representations live only for that forward/chunk call and are not retained across model updates.

The official outcome head now shares `(global 304 + player 64) → 128 → 3` across four players. Player local encoders use public economy/history, finance/order and industry mats, plus relative-player identity embeddings. Raw outputs have shape `[B,4,3]`; the outcome column is selected before applying win softmax, rank/VP transformations and the existing stable-seat remapping. Values remain independent of candidate count/order. The combined-return utility residual head, zero initialization, calibration rules and target semantics are retained; its prior uses the new official outcomes.

The local layout is bound into model configuration, architecture fingerprints, checked checkpoint index buffers and ONNX metadata:

| Identity | SHA-256 |
| --- | --- |
| Unchanged Features-4 manifest | `59facdd72a386b6046abbb875daf3e1051b92e580ff55d0ab37250dd0c1dfb6a` |
| Unchanged grouped layout | `f7bd96900365293a314e167256142104ede1f390a9a7820dce95cff91307476b` |
| New local entity layout | `e12a686152366a55817f9fa7465b8265e1d476915ff8d63efa061343d4b55c4b` |
| Validated architecture source | `7328ca65ec087183a4c92c7606e48b982d127003121ede84da01388dabf8181e` |

## Exact counts and work

| Preset/model | Parameters | Fixed dense MACs per position |
| --- | ---: | ---: |
| Medium base | **2,084,756** | 3,336,480 |
| Medium utility | **2,094,776** | 3,346,464 |
| Medium personality | **2,160,609** | 3,412,112 |
| Small base | 1,013,444 | 2,269,920 |
| Small utility | 1,017,832 | 2,274,272 |
| Small personality | 1,053,217 | 2,309,472 |

Each adds **76,672 MACs per padded candidate**, including explicit reference selection matmuls. The medium base gains **78,839 parameters (3.93%)** over v9's 2,005,917. Its unchanged grouped state encoder contains 1,867,536 parameters. The scorer remains `[128,432]` with its state projection shared once per position.

This feature addition increases compute: v9's factored base formula was `B × (1,938,080 + A × 58,368)`; v10 is `B × (3,336,480 + A × 76,672)`. These are dense arithmetic counts, excluding feature extraction, gathers, normalization, nonlinearities and transport. They are not wall-time predictions. Existing 3M base/utility and 3.1M personality ceilings remain unchanged.

## Data, initialization and continuation

Existing audited Features-4 heuristic demonstrations can be reused through the normal architecture-only data admission path. Feature/encoder/derived/rules/target identities still have to match. Labels and train/validation assignments are unchanged; incompatible cached tensors are rebuilt under the existing freshness rules. Old neural trajectories are not admitted as current PPO behavior.

V9 checkpoints are intentionally incompatible. V10 requires fresh IL weights and a new self-play directory; source-review flags do not reinterpret old weights. Matching v10 checkpoint continuation preserves weights, optimizer, RNG, progress, population and lineage. The checked-in latest recipe selects v10. The config-generation CLI was exercised against the completed isolated IL run and changed only its imitation-checkpoint and rehearsal-manifest bindings; all other current recipe values matched.

See [MLP_V10_LAB.md](../docs/MLP_V10_LAB.md) for exact IL, configuration, fresh self-play and continuation commands. Existing training checkpoints and production datasets were not modified.

## Validation

| Command/scope | Result |
| --- | --- |
| `test_mlp_v10` | **8 passed** |
| `test_factored_scorer test_grouped_projection test_six_group_rollback test_learning` | 25 passed, 1 CUDA-BF16 skip, 24 legacy fixture errors before model execution |
| `test_compact_widths test_compaction test_latest_model_recipe test_transformer_checkpoint` | 18 passed, 2 legacy fixture errors, 1 obsolete v9 fingerprint assertion |
| Updated Transformer factory-isolation assertion, rerun alone | **1 passed**; now checks that Transformer config resolution leaves current MLP compatibility unchanged |
| Current medium-count and real-feature ONNX/personality-budget checks from `test_fe4` | **2 passed** |
| `feature-engineering.py inspect` and isolated `configure` | Passed; correct v10 identity/counts and unchanged recipe values |
| Whitespace check on touched tracked files, allowing Windows CRLF | Passed |

The 8 new regressions cover named local gathers, all 128 target / 9 merchant / 6 industry references and absent-reference zeros, local information reaching only the selected action branch, shared player-head algebra and seat mapping, finite/nonzero new-branch gradients, masking, one preparation across candidate chunks, utility/personality integration, exact checkpoint/Adam/RNG round trips, old-model rejection and static-gather tamper detection.

The existing native-fixture PPO regression verifies collected behavior probability reconstruction before updating and rejects deliberately stale behavior. Base, utility and personality ONNX parity checks passed, including dynamic batch/candidate shapes `(1,1)`, `(2,7)`, `(4,31)` and `(1,257)`. CPU BF16 factorization checks passed; CUDA BF16 was unavailable.

**Broad-suite limitation:** the 26 remaining errors use old R2 game fixtures whose `Belper:0` slot types violate the current compact-feature static profile. They fail in feature admission before v10 forward execution. The same feature checks remain strict; neither their acceptance rules nor those unrelated fixtures were changed. This is not a claim that the entire repository test suite passes. The one architecture-specific test failure was corrected and its rerun passed.

## Isolated lifecycle smoke

Receipt: [smoke/report.json](../artifacts/feature-engineering/mlp-v10-20261016/smoke/report.json). Log: [lifecycle-smoke.log](../artifacts/feature-engineering/mlp-v10-20261016/lifecycle-smoke.log).

The existing lifecycle harness ran under a **480-second process-tree deadline** in a new output directory. It passed in **268.7 seconds including startup**, with **258.6 seconds** recorded inside the smoke. It used one tiny audited source game per split and re-audited only those test copies, then selected 8 train and 8 validation positions for a single IL optimizer step. This smoke-only audit does not mean the production Features-4 dataset must be regenerated.

The smoke then forced one latest-model self-play table and verified:

- Exact completed-IL → self-play initial weight transfer and a fresh PPO optimizer.
- Four learner trajectories, **301 decisions**, **134 actor-eligible** and 167 forced decisions; exploration 0.08 and temperature 1.0.
- Exactly one PPO optimizer step plus the existing ordinary utility calibration; no timed teacher collection/distillation or evaluation games.
- Matching-generation resume preserved weights, all 100 initialized Adam parameter states, RNG, committed iteration 1, game count 1, population generations 0/1 and checkpoint bytes.
- Rejection of a foreign tensor-cache attestation.
- A spawned remote teacher-leaf inference request with correct model/reward identity and finite utility, without launching a search tree.
- Exact parent import into the personality model and ONNX CPU parity at `(1,1)` and `(2,7)`. Largest reported errors were policy `2.09e-7`, win `4.47e-8`, rank `2.38e-7`, VP `7.63e-6` and utility `2.98e-8`.
- Source identity and production configuration values unchanged during the run.

Ordinary collection took 80.2 seconds; PPO took 3.2 seconds in this fixture. Rehearsal, scheduled evaluation and retention were disabled only in the scratch fixture. Historical/legacy tables, long training, timed teacher search/distillation, GPU/OpenVINO/Unity deployment and playing strength were not exercised.

## Small inference benchmark

Receipt: [inference-benchmark.json](../artifacts/feature-engineering/mlp-v10-20261016/inference-benchmark.json). Reproduce with `tests/FeatureEngineering/benchmark_mlp_v10.py --output <new-path>` and the repository Python dependencies.

Windows 11, Intel Family 6 Model 170 CPU, PyTorch 2.8.0 CPU, FP32, 2 intra-op threads and 1 inter-op thread. Each case used 10 warmups and 50 timed complete base-model forwards. Inputs were tensor-ready synthetic values with one-hot entity references. Times exclude rules, feature encoding, IPC and sampling. No task-owned test or smoke process was running concurrently with this benchmark; other system load was not controlled.

| Batch | Candidates per position | Median ms per batch | p95 ms per batch |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 7.660 | 9.805 |
| 1 | 128 | 7.962 | 10.857 |
| 1 | 512 | 11.863 | 14.704 |
| 8 | 128 | 17.752 | 21.586 |

These measurements establish that the implementation runs on the current CPU environment. They do not establish a speedup, full-game throughput, improved stability or stronger play.

## Files changed for this revision

- `src/brass_ai/learning/local_entities.py`: new versioned local gathers and shared entity encoders.
- `src/brass_ai/learning/model.py`: v10 config/identity, policy reference fusion, shared player head and corrected MAC accounting.
- `src/brass_ai/learning/batching.py`, `utility_model.py`, `src/brass_ai/personality/model.py`: common prepared-state/head path for chunked, utility and personality inference.
- `src/brass_ai/learning/contracts.py`: local-entity source participates in the MLP architecture fingerprint.
- `src/brass_ai/recovery/league.py`, `configs/step5/joint7-latest.json`: current v10 admission messaging and architecture selection.
- `scripts/feature-engineering.py`: local layout and role reporting.
- `tests/FeatureEngineering/test_mlp_v10.py`, `benchmark_mlp_v10.py`: new focused checks and bounded timing diagnostic.
- Current-model assertions in `test_factored_scorer.py`, `test_grouped_projection.py`, `test_compact_widths.py`, `test_compaction.py`, `test_six_group_rollback.py`, `test_fe4.py`, `smoke_fe5.py`, `tests/R2/test_learning.py`, `tests/Recovery/test_training_performance_resume.py` and `tests/Transformer/test_transformer_checkpoint.py`.
- `ai-research/GROUPED_PROJECTION_CONTRACT.md`, this report, `docs/MLP_V10_LAB.md` and an archival banner in `docs/FEATURE_ENGINEERING_LAB.md`.

Preexisting Transformer, teacher, deployment, source-review and runtime changes in the shared working tree are not attributed to this revision.
