# Step 5 evaluation on the 64-CPU lab server

**Archived pre-2.2 instructions:** use [Fresh Masked ISMCTS training](TRAINING_RECOVERY_LAB.md) for current commands. This document records earlier recipes and checkpoint-import/search workflows; do not run those commands for the new random residual-model run.

**30 September execution revision:** the current preset retains the full 64-setup panel and adds persistent private actors/shared inference. [TRAINING_RECOVERY_LAB.md](TRAINING_RECOVERY_LAB.md) also defines an eight-setup development panel that excludes held-out Pottery Flex. This guide's original threaded implementation is historical; its config is preserved as `previous-evaluation.json`.

The dedicated [evaluation preset](../configs/step5/evaluation.json) scales evaluation alongside Step 5 training. `evaluate` selects this file automatically unless you supply `--config`. Use the existing conda environment `ai-training`. This evaluates frozen Step 5 checkpoints; the later personality/difficulty stage remains separate.

## Default workload

| Setting | Value | Meaning |
| --- | --- | --- |
| `setups` | **64** | Held-out board/deal setups, each played with the candidate in all four seats. |
| `workers` | **32** | Maximum concurrent rules simulators for neural-only lineups. |
| `batch_size` | **64** | Maximum neural inference batch; actual batches depend on which policies need decisions. |
| `max_candidate_cells` | **131072** | Splits inference batches with many legal choices without dropping choices. |
| `legacy_workers` | **8** | Independent games when a lineup contains legacy opponents. |
| `search_workers` | **8** | Independent games when the candidate uses search. |
| `threads` | **4** | PyTorch CPU threads. |
| `parallel_hosts` | **true** | Concurrent calls to independent rules processes. |
| `include_legacy` | **true** | Includes mixed admitted v3 and held-out Pottery Flex lineups. |
| `use_search` | **false** | First measure the trained network; search is a separate comparison. |
| `partition`, `seed` | `validation`, **910000** | Reproducible held-out setup selection. |

With two distinct neural reference identities, there are four lineups: recent neural, mixed neural history, mixed v3, and held-out Pottery Flex. Each has **256 games**, giving **1,024 games total**. Lineups run in sequence, so their worker counts do not add together. With only one unique neural reference, the duplicate mixed-neural lineup is omitted: three lineups, 768 games with legacy enabled.

Evaluation has no `minibatch_size`, gradient updates or optimizer steps. Increasing its batch size changes inference scheduling, not learning. Larger batches do not guarantee high GPU utilization: host execution, Python feature encoding and serialization can still dominate.

## Run a full evaluation

Replace the example checkpoint names with generations that exist. Use two distinct **trained** reference checkpoints and keep that reference panel fixed when comparing candidates.

```bash
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/evaluation.json \
  evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled/generation-0050.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled/generation-0025.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled/generation-0010.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-scaled-g50
```

Put global options such as `--config` and `--device` before `evaluate`. The explicit config path above is optional because it is the new default. Passing an older training config explicitly still works, but retains the older five-setup/no-legacy evaluation defaults; use the dedicated file for this larger workload.

For faster progress checks, append `--setups 4 --no-include-legacy` and use a new output directory: **32 games** with two distinct references. For a substantial neural-only comparison, append just `--no-include-legacy`: **512 games**. These neural comparisons should guide training progress; beating the weak legacy opponents alone is insufficient.

To test the learned Pottery context, add `--focus Pottery` with a separate output directory and the same reference panel/setups. This uses a soft learned context; it does not force Pottery actions. Use `--partition test --seed 1910000` for a final held-out test after tuning, again with a separate output directory. Do not use that test result to tune the model or choose promotion.

## Optional search comparison

Add `--search` and choose a new output directory to evaluate the existing **S2** candidate against the same frozen opponents. The preset retains a 30-second search allowance on every fourth own turn, with no per-seat attempt cap. This is not S3, and it does not enable search on every turn. `--no-search` explicitly disables it.

Search and legacy policies have wall-clock turn budgets and private context. They run in eight independent game loops with separate policy/worker instances. Neural-only games use shared batched inference across up to 32 simulators. This prevents a timed policy's allowance being spent waiting behind unrelated games in the coordinator. CPU/GPU contention can still affect timed search; record concurrency and use matched execution settings when comparing search variants. The larger evaluation config alone does not establish an equal-budget search advantage.

## Integrity, resuming and interpreting results

Each `(setup, candidate seat)` job has a stable sampling seed independent of its batch position. All four rotations must complete exactly once. Cached shards include the job list, policy identities and evaluation spec identity. Failures abort completion instead of counting as losses or wins; successfully written batches can be reused by rerunning the exact command.

The output spec pins checkpoints, resolved config, host build, source implementation and the legacy pool when used. Use a **new output directory** when changing these, including worker counts, search settings, focus or setup count. Previous-format evaluation outputs also need a new directory. Evaluation shards remain marked evaluation-only and cannot enter the training reader.

Inspect `report.json` for separate lineup results, seat exposure, Pottery usage, elapsed time, execution mode/worker count and peak allocated CUDA memory. Throughput on a resumed evaluation includes cached games; measure performance on a fresh output directory. Compare confidence intervals clustered by **64 setups**, not by treating 256 seat rotations as independent samples.

Promotion still requires at least 30 validation setups, two distinct trained neural reference identities, a lower clustered win-share bound above 25% in each neural lineup, and no Pottery-collapse warning. The new default exceeds the minimum setup count; it does not automatically qualify the candidate. Prespecify the evaluation size instead of adding games until an interval passes.

## Applying this update to an existing training run

This update changes Step 5 source files covered by the trainer's implementation signature. Checkpoints remain loadable. If a training run was already started with the previous source, finish/stop that process before updating the checkout, then continue into a new directory using `--from-run`; do not edit its stored signature. For example:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/lab.json \
  train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled-v2 \
  --from-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled \
  --iterations 2
```

The source directory must be your actual existing run, and the destination must be new. This preserves committed weights, Adam/RNG state, history references and the setup cursor, as described in [the training scaling guide](STEP5_SCALING.md). Keep the referenced source checkpoints. If both scaling updates are applied before starting the new training run, no additional fork is needed.

## Focused checks

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/verify-step5.py
python /data/not_backed_up/yxu209/ai-training/scripts/check-step5-evaluation.py --device cuda
```

The second command plays one real setup in all four seats against one untrained reference, with legacy disabled. It verifies completion and rotation, not playing strength. Its generated report is under `/data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-checks/`. See [local check results](STEP5_EVALUATION_CHECKS.md) for executed evidence and remaining lab measurements.
