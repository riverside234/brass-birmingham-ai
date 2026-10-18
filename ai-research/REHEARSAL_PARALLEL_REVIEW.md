# Rehearsal loading and generation-one review

Date: 2026-10-18. Source bundle: [joint7 v11.2 generation 1](../results/joint7-v11-2-grouped-lr-critic-g0001-20261018T215854542444Z/iteration-0000/report.json).

## Findings from the actual run

- Rehearsal prepared **31 shards / 16,384 sampled rows in 614.50 seconds**, 57.4% of the 1,070.70-second iteration. It ignored the already configured `audit_workers=16` and loaded serially. The window contained 14,900 distinct source rows and 6,797 nonforced presentations; PPO actually used 6,128 distinct rehearsal rows in 32,382 presentations. Forced filtering and repeated draws are expected, and remain unchanged.
- The console's `PPO=832.7s` combined rehearsal, PPO/controller work and the **118.03-second value phase**. The underlying `ppo_seconds=714.70` also included rehearsal. The remaining 100.20 seconds includes head calibration, PPO and post-PPO diagnostics; it is not just 38 gradient steps.
- `teacher=21.7s` included final probes, bookkeeping and data release. The recorded search report confirms **zero teacher collections, labels and optimizer steps**.
- PPG attempted **16 steps**, reached raw-policy KL **0.0021471** above its **0.002** cap, and restored the epoch's weights and value Adam state. Accepted steps/rows are therefore zero. Attempted held-out loss improved from **1.48559 to 1.44860**, but that cannot waive the policy guard. Final probe KL after rollback was approximately **2.59e-9**. The existing clone-beta response after three KL trips in five iterations remains active.
- PPO completed **38/38** steps, with categorical KL **0.0028131**, below its 0.01 guard. LR **2.618e-6** and multiplier 1.0 are consistent with the 100-step warmup. The fixed validation probe has all 2,048 positions; overall IL KL **0.0017399** is below 0.15 and every group is below 0.30.
- The composition is exactly **160 self-play / 144 history / 16 v4 tables**, yielding **960 learner trajectories**. The initial quality pool contains only the IL anchor; new admission is not due. No snapshot training is expected yet.
- Eleven whole games (2,663 learner rows) were selected by the 5% hash holdout. Self-play held-out EV is **0.28171**, history **0.03120**, and v4 has no held-out game this iteration. Warmup/health history and the rejected value fit keep the gates closed. Sparse v4 evidence can delay its ten-consecutive-iteration gate; missing evidence continues to reset its streak, as specified. Holdout selection and gate thresholds were not changed.

## Implemented changes

Rehearsal now uses the existing persistent replay reader pool and `audit_workers`, so the current config uses **16 CPU workers**. At most one shard per worker is submitted at a time. Full raw hashes, membership, labels, legal choices and feature admission still run for every record. Only distinct selected rows need neutral-context BC preparation and compact metadata; repeated draws are reconstructed in their original order.

Workers write selected rows directly to disposable disk files and return small receipts. Dense features do not cross IPC. An OS-locked scratch ledger reserves bytes before writes, enforcing one shared 4 GiB output cap. Writers finish or fail before cleanup, including corrupted sources, cap rejection and process death. No sampled row is silently dropped. Temporary windows are still deleted after each update. Memory can now include up to 16 decoded source shards; the disk cap is not a RAM cap.

Reporting separates rehearsal preparation/cleanup, PPO, post-PPO diagnostics, value phase, teacher work and final diagnostics. The console shows `value_steps=accepted/attempted` and `value_stop`. Value reports include attempted value/clone presentations alongside accepted usage. Compact history keeps these diagnostics and loader worker/timing fields.

A fully rejected value phase reuses its already measured train metrics. When neither the value phase nor teaching changed weights, final IL-KL and categorical probes reuse their exact post-PPO results. Accepted fits still remeasure final weights. Optimizer math, RNG, sampling, safeguards, model identity and the training recipe are unchanged.

## Bounded benchmark

[Machine-readable benchmark](../artifacts/rehearsal-parallel-20261018/benchmark/report.json): one real audited diagnostic game replicated into 16 distinct raw shard files, **5,520 source rows / 2,048 sampled rows**, no feature sidecars, Windows CPU (14 logical processors), two learner tensor threads, one measured pass per mode.

| Preparation path | Seconds |
| --- | ---: |
| Previous serial full-shard preparation | 56.61 |
| Serial preparation of selected rows | 44.80 |
| Selected rows with 16 readers | 14.92 |

Combined speedup: **3.79×**. Cache contents, sampled identities/order and training RNG matched exactly; each path wrote about 138.5 MiB. Starting the 16 readers took another 10.04 seconds, reported separately; normal training already starts/reuses those readers during collection. This small benchmark does not predict full joint7 loading speed, GPU throughput or playing strength.

## Validation

**88 focused tests passed**, with final per-test outcomes in the [validation inventory](../artifacts/rehearsal-parallel-20261018/validation.json). The tested source is `e2b6bf879c8b9da5b043c8f4c8aedf76b45bec9095867e4246b7decd1c316144`; the model/learning compatibility digest remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.

Focused tests cover serial/parallel byte and row-order equality, selective preparation equivalence, rejection of invalid unselected records, unchanged PPO losses/weights/RNG, shared byte caps, interruption/process-death cleanup, prepared IL caches, two-iteration value buffers, exact two-optimizer continuation, value rollback and accounting, automatic LR/IL-KL gates, and the reported run's explicit source transition.

The [native smoke](../artifacts/rehearsal-parallel-20261018/native/report.json) passed two one-game CPU iterations, fresh start then resume, in **106.78 seconds including startup**. It used two persistent readers (one selected shard per iteration), 64-row windows, and reused the fixed validation probe. Rehearsal took 4.40/4.24 seconds; both temporary windows were deleted. Missing PPG cache reconstruction and old-cache cleanup passed, original IL/dataset and the committed checkpoint stayed unchanged, and final post-PPO probes were reused exactly. Its empty critic holdout correctly kept the value phase/gates inactive. Teacher time now prints 0.0s, with diagnostics separately reported.

The production GPU/checkpoint is on the lab machine; it was not run or modified here. No long training or rating campaign was launched. Production speed and stability still need subsequent normal-run reports.

## Continue the existing run

Keep the current config, IL folder, self-play directory and fixed validation cache. This is an explicit reviewed transition from `de2aac3edd606da9c23fad732c6bc4d8dc6118ad2d801af8b9ae6ffb9326f314`; unrelated source/config edits and pending old collections remain rejected. Model weights, both Adam states, RNG, progress, population, gates, probe identity and checkpoint lineage are preserved. No fresh start, config regeneration or IL upgrade is needed for this fix.

```bash
python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --expected-generation 1 \
  --iterations 20
```

Use `--iterations 1` for a short timing check. If more generations have already committed, use the actual generation or omit the optional expected-generation guard. Stop at a committed iteration before changing source; do not delete pending replay to bypass freshness checks.

Changed production files: `recovery/rehearsal_window.py`, `demonstrations.py`, `league.py`, `value_phase.py`, `history.py`; the two source-review JSON files; lab instructions. Focused tests, a rehearsal benchmark and the native smoke were updated/added. Model/config/probe files and archived results were not edited.
