# Step 5 evaluation scaling checks

Executed 29 September 2026 on Windows CPU with PyTorch 2.8.0+cpu. These checks verify evaluation execution and integrity; they do not qualify playing strength or establish lab GPU throughput.

| Check | Result |
| --- | --- |
| Step 5 test suite | **35 passed**, including six evaluation tests covering config/CLI overrides, the complete held-out seat matrix, stable sampling/seat assignment, cached resumption, failure handling and independent search-policy ownership. |
| Serial versus batched routing | Explicit rotations and per-game RNG reproduce the original individual-game collector's decisions on an authored four-seat fixture. |
| Evaluation isolation | Evaluation-only shards are rejected by the training reader; changed execution specs cannot reuse an existing evaluation directory. |
| Real rules-host smoke | **Four completed four-player games**, one held-out setup with the candidate in every seat once, four concurrent host processes. No failed games; checkpoint files unchanged. |
| Promotion guard | The untrained one-setup smoke correctly remains ineligible. |
| Frozen bank | No changes to the R3-pinned collector; existing bank/admission tests pass within the Step 5 suite. |

The real-host report is [stored locally here](../artifacts/step5/evaluation-checks/4f1117b52f23/result/report.json). It is ignored generated evidence, not required in a Git clone. The tested Step 5 implementation ID is `9be910b103d9fd744555b2203d073e7bbf4be4fd1b6d36745fe4591fb12fe841`.

The four-game evaluation took **341.52 seconds**, including replay validation and serialization. This has no matched serial timing baseline and uses small untrained networks on a laptop CPU. It does not establish a speedup. Legacy/search lane ownership was checked with test doubles; no additional legacy or search tournament was run for this config update.

The separate [training scaling checks](STEP5_SCALING_CHECKS.md) record the earlier implementation's real serial/parallel and short-search evidence. This update keeps those results as historical evidence, with their original source identity.

Run the commands in [the evaluation scaling guide](STEP5_EVALUATION_SCALING.md#focused-checks) on the lab server, then measure a fresh evaluation directory. CUDA memory, 32-host throughput and eight-worker timed-policy throughput remain lab measurements. The full 1,024-game evaluation has not been run locally.
