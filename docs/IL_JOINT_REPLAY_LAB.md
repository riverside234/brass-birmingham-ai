# Joint imitation rehearsal: v4-batch4 through v4-batch7

This experiment continues **batch7's selected weights** on the full audited union of batches **4, 5, 6 and 7**. Batches 2 and 3 are no longer included in the default dataset source list. It does not initialize a random model. With sufficient compute, all unique training games are retained rather than imposing a 50% latest / 50% historical sample cap. The existing trainer globally shuffles decisions each epoch and preserves its teacher/era/action balancing. This is full-data supervised rehearsal inspired by [Experience Replay for Continual Learning (CLEAR, NeurIPS 2019)](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html); it is **not** the paper's complete off-policy RL algorithm, and its strength advantage is not guaranteed.

The helper is isolated in `scripts/il_replay.py`; it does not change the existing IL/PPO source or presets, source fingerprints, installed game, original manifests or checkpoints. All actions use the conda environment and project root below. No training starts until you run the `train` action.

## 1. Prepare the union

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/il_replay.py prepare
```

Default source directories are `/data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch7` and the corresponding batch6, batch5 and batch4 directories. The helper reads each completed run's `state.json` and `report.json` to locate its **actual** manifest and verify its recorded hash; it does not assume that the dataset file has the same name as the run. Keep these source runs and their manifests/replay shards available. If the runs live elsewhere, use `--latest-run` and `--history-runs` with their paths inside this project.

This changes the dataset range, not the model's parent-lineage checks. If the retained checkpoints themselves name deleted batch2/3 runs as ancestors, the existing `completed_imitation` check still requires those recorded ancestors during training/handoff. Excluding their datasets does not recreate missing provenance. The helper does not silently bypass that separate check. A failed preparation at the missing batch3 `state.json` happens before the merged manifest is written; update the helper and rerun `prepare`. If an older pooled manifest was already successfully created, choose a new `--experiment` and `--run` for the changed source pool.

The output is `/data/not_backed_up/yxu209/ai-training/artifacts/il-replay/v4-batch7-joint/manifest.json`, an ordinary `bc-dataset/1.0.0` manifest accepted by the existing loader. It retains original replay hashes and feature receipts, rejects incompatible rule/feature/teacher-calibration metadata, and deduplicates audited semantic game identities. Identical overlapping games belong to the newest source in the per-batch diagnostic report. A conflicting game ID or relabeled split fails instead of being silently merged. Existing **train / validation / test** setup partitions are preserved; test data are not used by training or validation.

Preparation only inspects manifest receipts and file existence. Actual replay bytes, admitted teachers and features are checked by the unchanged loader when training/validating, avoiding another full rules replay audit. A preparation result is not a substitute for these checks. Repeating the same command is idempotent; changing the sources or recipe requires a new `--experiment` and `--run`, not editing existing receipts.

## 2. Continue IL on the full pool

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/il_replay.py train --device cuda
```

This calls the existing `step5.py imitate` entry point with `--from-imitation-run` pointing to batch7. Defaults for this separate experiment are **LR 3e-5**, **at most 6 epochs**, **minimum 2 epochs**, and **patience 2**. These are a conservative starting recipe, not claimed optimal hyperparameters. All unique training examples are seen each completed epoch, so this is more work than a single-batch epoch. Batch size, precision, CPU loading and GPU batching inherit `configs/step5/lab.json`; `--workers` optionally overrides loader processes. The existing 20/40/60/80/100% logs remain available. The trainer creates a new optimizer for this new IL run, just as ordinary new-dataset continuation does.

The candidate is saved separately to `/data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch7-joint/best.pt`. Original 6/7 files stay intact. After training completes, the helper automatically runs the baseline comparison in step 3, so a finished `train` action includes the original-model safeguard. Repeating `train` with exactly the same arguments resumes this experiment through the normal source/config/checkpoint guards. To change LR, epochs, source pool or other signature settings, choose a new `--run`; use a matching new `--experiment` when its candidate/source receipt changes.

## 3. Check old and new held-out data

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/il_replay.py validate --device cuda
```

This is also run automatically after `train`; the separate command is useful if validation was interrupted or needs to be repeated. It compares **original 7, original 6, and the joint candidate**, without optimizer updates, on the same union validation partition. It reports each batch separately, including policy loss and action accuracy, plus the aggregate using the existing BC loss denominators. An overlapping game is counted once under its retained source, not once per copy. Inspect old-batch loss/accuracy as well as batch7's metrics; this makes a forgetting hypothesis testable rather than assuming that lower game VP proves forgetting.

`artifacts/il-replay/v4-batch7-joint/validation-comparison.json` retains all three models and selects a validation reference including the unchanged originals. A new candidate must improve loss by more than `1e-4` to replace the best original **in this report**. This does not rewrite a checkpoint or claim gameplay promotion. The existing IL trainer's `best.pt` is still the lowest-validation-loss epoch within the child run; if the report selects original 6/7, use that original run for handoff instead of renaming/copying its checkpoint into the child.

## 4. Compare actual gameplay separately

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/il_replay.py evaluate --device cuda
```

Default evaluation uses **40 fresh validation setups × four seat rotations = 160 games per panel** with seed cursor `1947000`, shared across candidates. All three models play the same v4 panel. The joint candidate also faces three copies of its own `initial.pt`, which contains the original batch7 weights with the correct child-run identity. This avoids the existing evaluator's prohibition on arbitrary cross-run references. The total is **640 evaluation games** by default, kept separate from training. `--setups` and `--seed` configure this independent evaluation; changing them requires a fresh `--experiment` output location.

The active evaluation preset controls the same v4 search budget for every model; NN candidates play without search. Cached reports are reused only under the existing exact-spec guards. Results go to `artifacts/il-replay/v4-batch7-joint/evaluations/comparison.json` and the component reports. Compare official win share and placement first, then VP and failure/low-score diagnostics; a score-only change does not establish a strength change. No automatic model replacement, installation, RL launch, or guarantee of stronger play is made.

After reviewing results, use the **chosen completed run directory**, not an arbitrary copied checkpoint, as the ordinary Step 5 `--imitation-run`. For example, if the joint candidate is chosen:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-joint-il-v1 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch7-joint \
  --iterations 20
```

Keep the original batch lineage, source manifests, pooled manifest and selected run intact for the existing IL completion and PPO handoff checks. This helper deliberately does not delete historical datasets or alter model-retention rules.

## Local verification

Focused checks cover source union and unchanged loader compatibility; semantic deduplication; conflicting identities, partitions and tampered receipts; idempotent preparation and protected sources; separate policy/value aggregation; baseline selection; common evaluation seeds and same-run references; automatic post-training validation; and a tiny CPU BC continuation/validation/handoff smoke using synthetic replay fixtures. The smoke preserved original checkpoint hashes and produced a child accepted by the existing completed-imitation checks. The default-source regression specifically prepares batches 4--7 while batch2/3 directories are absent. No lab batch4--7 data, CUDA throughput, native gameplay tournament or playing-strength gain was tested locally.
