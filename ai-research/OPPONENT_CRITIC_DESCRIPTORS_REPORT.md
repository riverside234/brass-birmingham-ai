# Opponent critic rows and seat descriptors — v11.2

Implemented and validated on 2026-10-18. This extends the existing gated GAE and two-iteration PPG pipeline. It is an experimental critic change; no playing-strength improvement has been demonstrated.

## Effective behavior

Architecture is `brass-four-seat-mlp-resnet/11.2.0`. Features-4, the grouped projections, 512-wide/eight-block trunk, local action conditioning and actor/scorer are unchanged. Four relative-seat descriptors contain five roles, a clipped log generation lag and 28 style values. A shared `34 → 32 → 32` encoder supplies a seat embedding to each outcome head and its mean to the utility head. Only critic heads read these embeddings; new head columns start at zero.

| Model | v11.1 parameters | v11.2 parameters | Added |
| --- | ---: | ---: | ---: |
| Medium base / IL | 5,904,644 | **5,910,916** | 6,272 |
| Medium utility / self-play | 5,921,320 | **5,928,616** | 7,296 |
| Existing optional goal wrapper | 6,023,137 | 6,030,433 | 7,296 |

The medium utility path adds 25,856 dense MACs per position; per-candidate MACs remain 83,232. These counts exclude nonlinearity, memory traffic and normalization and are not a latency benchmark.

Only current learner nonforced rows enter the actor and GAE. Critic training includes all learner rows and, once enabled, scored snapshot rows. IL/gen0 and v4 decisions are excluded. A frozen opponent can share the incoming learner's weights: admission therefore uses the behavior-policy identity, not model identity alone. Existing exact behavior checks, legal masking and forced-action filtering remain active.

The 5% game-id holdout excludes every seat of those games from PPO critic fitting, utility calibration and PPG, including the disk buffer. Actor/cloning may still use learner decisions in those games. This is a critic holdout, not an independent policy evaluation set. Snapshot targets use their own causal insolvency return/remaining budget and all four official outcome targets with the existing seat mapping.

Descriptors are recorded in each game's sealed job provenance and retained in prepared/disk rows. Incoming learner fingerprints use one raw-temperature-1 probe pass per iteration; pool fingerprints are cached on admission; IL once; v4 uses recorded probe choices. Each observed era's family group and each nonempty build-conditional industry group sums to one. Missing eras/builds produce explicitly reported zero groups. No new v4 games are needed. Critic-only dropout selects 25% of games per epoch using a deterministic hash. Advantages, calibration and reported metrics use true descriptors.

## Automatic gates

`critic_data.rollout.mode=auto` starts learner-only. An accepted value phase with nonnegative learner held-out descriptor EV gain and passing existing health/KL checks enables snapshots on the next iteration. Their initial weight is 1.0. Learner EV at activation is saved as the Phase-1 baseline; a later decrease exceeding 0.02 reduces the weight to 0.5. These are sequential diagnostics, not a controlled measurement of snapshot benefit. An explicit `learner_only` mode is available in a new config.

The existing self-play lambda gate remains **1 → 0.98 → 0.95**, with its accepted-value, health and PL-slope checks. History and v4 each stay at **1** until their **learner-only** held-out EV is at least self-play learner EV minus 0.05 for **ten consecutive healthy iterations with accepted value fitting**. Their streaks and activation generations are independent. Missing/nonfinite EV or unhealthy evidence resets the relevant streak. Snapshot EV cannot qualify this gate. Activation affects the next collection, persists on restart and follows later self-play lambda changes; there is no automatic rollback. Sparse v4 holdouts may delay qualification.

Thirty iterations after each extension, reporting flags a review of the common PL rating and by-table advantage standard deviation. No additional evaluation campaign is scheduled. Unknown is the default descriptor for search/deployment. The plan's own-copy unknown-versus-learner search choice remains a later PL experiment; it cannot be selected from this smoke test.

## Compatibility and state

Use `feature-engineering.py upgrade-imitation` on completed v11.1 IL. It writes an immutable new handoff directory, copies every old tensor, initializes only the descriptor extension and records parent checkpoint/source/dataset hashes. The normal loader rejects old architectures; only the explicit completed-IL path accepts the pinned v11.1 contract. Handoff re-audits completion and verifies every resulting tensor against the deterministic upgrade. No IL training is required. Production IL and existing self-play checkpoints were not modified.

New configs sign the descriptor, critic-row and per-table GAE settings. The 50/45/5 table mix, 24-member quality pool, four/two/two learner seats, 0.08 learner exploration, rewards, PPO/LR/IL-KL controls, rehearsal, PL ratings and opponent policies remain intact. Teacher search, all self-play v4 search, Monitor and Broader Evaluation remain disabled. The two-iteration PPG cache persists descriptors/roles/table types and is removed only after the next checkpoint/state commit. Missing caches reconstruct from verified replay in the same learner-then-snapshot order.

ONNX accepts `[B,4,34]` descriptors, with a broadcastable all-unknown initializer when omitted. Explicit known/unknown inputs and omitted inputs work at batch sizes 1 and 2; dynamic candidate checks cover 1–772 candidates. Existing exact GELU/LayerNorm fusions and highest ORT optimization remain enabled. Exported policy logits are independent of descriptor values.

## Validation

**149 distinct tests passed; one CUDA-only test was skipped on this CPU host.** The suites overlap; these are deduplicated test IDs. Coverage includes zero-extension output equality, critic gradients/policy independence, normalization and relative seats, deterministic dropout, ingress role exclusion (including equal-weight frozen opponents), snapshot debt/utility targets, holdout exclusion, independent GAE streak boundaries, two-optimizer restart, disk rebuild/cleanup, original encoder/scorer behavior, config generation and ONNX parity. CPU output equality at upgrade was checked exactly for small/medium models and batches 1/2. GPU bitwise equality and production throughput were not measured.

Evidence: [test inventory and source identities](../artifacts/critic-descriptors-20261018/validation.json), [99-regression run](../artifacts/critic-descriptors-20261018/regressions-final.log), [23 admission/gate checks](../artifacts/critic-descriptors-20261018/row-admission-final.log), [ONNX checks](../artifacts/critic-descriptors-20261018/export-final.log), [architecture/export regressions](../artifacts/critic-descriptors-20261018/architecture-verified.log), [final cache-fixture correction](../artifacts/critic-descriptors-20261018/cache-fixture-final.log), [independent table gates](../artifacts/critic-descriptors-20261018/table-gate-final.log). The architecture run exposed an outdated authored fixture; its corrected native-topology/revision version passed the final focused rerun. Existing error-message/dimension fixtures were updated for 11.2.

Two bounded CPU smokes used a separate diagnostic output tree:

| Check | Result |
| --- | --- |
| Medium mixed-table compute | **10.93 s**; two updates plus exact restored-next-update comparison; both Adam states, RNG-dependent update results and gate states matched |
| Native fresh start then resume | **152.12 s**, two one-game iterations from explicitly upgraded diagnostic IL; no ratings/teacher games |
| Cache recovery | Native smoke deliberately removed its first cache; resume rebuilt it, then deleted it after the next commit |
| Source preservation | Original IL/data and previously committed diagnostic checkpoint hashes unchanged |

The authored compute fixture has 96 learner, 12 snapshot and 36 excluded IL/v4 rows per collection. Holdout identities deliberately cover all three table types; 64 learner + 8 snapshot rows are critic-train eligible. Actual usage:

| Iteration | PPO distinct learner / snapshot critic rows | Accepted PPG current / previous rows | PPG result |
| --- | ---: | ---: | --- |
| 1 | 57 / 7 | 72 / 0 | Two accepted epochs; 128 learner + 16 snapshot presentations |
| 2 | 56 / 8 | 0 / 0 | Current + previous 72-row sets available; attempted epoch rejected by held-out loss guard and restored |

The second update also matched after full checkpoint restoration. Toy learner descriptor EV gains were approximately `1.7e-9` and `-2.4e-8`, which provide no evidence of benefit. The native collections had 312/296 learner decisions and no hash-held-out games. PPG correctly skipped with `insufficient_critic_split`; snapshot and mixed-table GAE gates stayed closed. Native fingerprints used 60 Canal and 64 Rail action positions. Full scope and metrics are in the [compute report](../artifacts/critic-descriptors-20261018/compute-final/report.json) and [native report](../artifacts/critic-descriptors-20261018/native-final/report.json).

A real `configure` invocation produced and validated the new recipe bound to the diagnostic upgraded IL. Report export copied the upgrade receipt and original IL reports along with both iterations and fixed-probe metadata. The diagnostic run disables retention, so the exporter correctly lists its absent `retention-history.json`; no report was silently fabricated. [Export manifest](../results/critic-descriptors-20261018-smoke/export-manifest.json).

## Runtime and disk implications

There are no additional collection games. The plan's approximate 21k snapshot rows add about 25% to learner-only critic volume once enabled. Runtime also includes one incoming fingerprint probe pass and a second held-out evaluation with unknown descriptors. Guard stops and actual table/row counts determine realized work.

Each cached critic row is **14,503 bytes** plus an array header. For an illustrative 107k total learner/snapshot rows with a 95% train split, this is **1.47 GB per cache**, about **2.95 GB** while both generations coexist before commit. Two similarly sized iterations give about 50 value batches per epoch at batch size 4,096, up to 150 across three epochs. These are estimates, not measured joint7 GPU throughput.

## Files and commands

Core changes: `learning/{model,utility_model,seat_descriptors,checkpoint,batching,contracts,onnx_tools,export}.py`; `recovery/{critic_data,critic_upgrade,actor_value,gae,value_phase,value_buffer,updates,utility,ordinary,ingress,replay,league,quality_pool,il_kl_probe,config,demonstrations,history}.py`; existing personality wrapper/export descriptor plumbing; `configs/step5/joint7-latest.json`; the completed-IL source review; `scripts/{feature-engineering,export-training-reports}.py`. Focused critic/export/smoke tests were added and affected architecture/config/report fixtures updated. Earlier GAE/PPG/rehearsal work and unrelated personality documentation were retained.

Use the exact [upgrade, config and fresh-start commands](../docs/MLP_V11_LAB.md#fresh-self-play-from-completed-v111-il), then the [continuation command](../docs/MLP_V11_LAB.md#continue-v112-self-play). The original IL name remains `features4-v11-1-batch7-il`; the new handoff is `features4-v11-2-batch7-il`, and the new config/run name is `joint7-v11-2-grouped-lr-critic`.
