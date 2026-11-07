# PPO diagnostic scan and probe cadence revision

For the latest g1241 run, use the [g1241 continuation](TRAINING_G1241.md), which
preserves this update. The [combined g1041 continuation](TRAINING_G1041.md)
included this update, the throughput update, 70/30 tables and early PPG beta
feedback in one reviewed config. The g921 example below is historical.

The new versioned `update.kl_control` disables logging-only full-dataset scans
after PPO epochs. The full pre-update behavior-probability reconstruction still
runs on every row, including forced rows. Legacy configurations keep their prior
behavior until explicitly migrated; newly generated latest-model configs use
the new block.

```json
{
  "contract": "ppo-sparse-probe-control/1.0.0",
  "probe_rows": 2048,
  "sampled_kl": "diagnostic-only",
  "probe_check_steps": 4,
  "full_dataset_kl": false
}
```

The same fixed probe checks raw T1 and actual behavior-policy exact categorical
KL against the unchanged `target_kl`. Checks run after optimizer step 1, every
four steps (counted across epochs), and each epoch's last step. A normal
48-step, two-24-step-epoch update has 13 guard checks instead of 48. The separate
final probe report remains active and supplies post-PPO/pre-teaching LR feedback.
No PPG settings, data, objectives, checkpoint tensors, warmup or LR state change.

There can be three unchecked optimizer steps between checks. A crossing is
detected at the next scheduled check; transient excursions between checks are
not measured. PPO keeps its existing stop-after-update behavior, without adding
rollback. Nonfinite gradients are still rejected every optimizer step.

`actor.epochs[].post_update_kl` is null when the full logging scan is disabled.
The console uses `KLsample_probe` and `KLfull=disabled`; it never relabels the
bounded sampled estimate as a full-dataset scan. Compact history retains the
exact checks, checked step indices, cadence, measured maximum and stop reason.

## Reviewed continuation

For an interrupted run whose committed generation is still 921, run from the
repository root on the lab machine. If it has advanced, use the actual committed
generation in all three commands and a new output filename. The helper checks
the full checkpoint hash, rejects pending collections, and writes only a config.
It preserves the currently committed automatic reward stage. No checkpoint or
PPG cache refresh is needed for this change.

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 921 --apply && \
python scripts/configure-ppo-probes.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 921 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-ppo-probes-g921.json && \
CUDA_VISIBLE_DEVICES=7 python scripts/step5.py \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-ppo-probes-g921.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 921 --iterations 20
```

## Bounded validation

Run with the normal project dependencies and `src`, `tests/Recovery`, `tests/R2`,
`tests/FeatureEngineering`, `tests/Personality`, and `tests/Reports` on PYTHONPATH:

```bash
python -B -m unittest test_ppo_sparse_probes.SparseProbes test_ppo_exact_only.ExactOnly \
  test_categorical_kl_guard.ProbeTests test_categorical_kl_guard.GuardTests \
  test_compact_il_resume -v
python -B tests/Recovery/smoke_ppo_sparse_probes.py \
  --output artifacts/ppo-sparse-probe-smoke --repeats 3
```

The g921 timing estimate is 7.6 seconds from removing two logging scans, plus
`7.0 * (1 - 13/48) = 5.1` seconds from fewer guard passes: approximately 12.7
seconds per otherwise equivalent update. This is an estimate, not a CUDA
benchmark; early stopping and epoch-tail checks can change the actual savings.

Validation on 2026-11-06: all 34 focused tests passed. The isolated CPU smoke
used two PyTorch threads, 48 authored rows, 48 optimizer steps, one untimed
warmup per path, and three timed repetitions per path. Median wall time was
38.902s before and 17.472s after. Guard checks fell from 48 to 13 and logging
scans from two to zero; the mandatory initial behavior verification and final
probe report remained. Weights, Adam state and RNG state matched exactly for
the no-early-stop fixture. Final exact probe KL was 0.000686214 on both paths.
The fixture used LR 6e-7 to stay below the unchanged KL cap; production LR was
not changed. This small CPU probe contains 48 rows, not the production 2,048,
and tests ran concurrently, so these timings are a functional smoke result,
not a controlled production-throughput claim. No production checkpoint, long
training run or evaluation campaign was used.

The full smoke measurements are in
`artifacts/ppo-sparse-probes-20261106/report.json`.
