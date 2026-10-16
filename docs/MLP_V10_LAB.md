# Grouped MLP v10: archived IL and self-play instructions

The current model is v11. Use [MLP_V11_LAB.md](MLP_V11_LAB.md); the commands and counts below describe the previous v10 revision.

V10 adds direct target/merchant/industry references to the policy scorer and a shared per-player win/rank/VP head. It keeps Features-4 inputs (state 3,128 / action 327), the six projections 32/64/48/192/224/48 plus 12 context values, fusion 620 → 304, six width-304 residual blocks and the existing training recipe. Base/utility/personality parameter counts are **2,084,756 / 2,094,776 / 2,160,609**.

See [the model contract](../ai-research/GROUPED_PROJECTION_CONTRACT.md) and [validation report](../ai-research/MLP_V10_REPORT.md). This is an experimental model change; playing-strength gains have not been established.

## Reuse the existing joint7 dataset

The already converted, audited **Features-4 heuristic IL dataset** remains usable. Keep its existing manifest, split assignments, labels and raw shards. A directory name containing v8 does not make its Features-4 data incompatible. Do not rerun dataset generation solely for v10.

The normal loader still checks the rules, feature, encoder, derived and target contracts. Its existing architecture-only admission exception applies to audited heuristic demonstrations, not old neural PPO trajectories or search targets. Architecture-bound tensor caches may be rebuilt in the new IL run; source data is not rewritten. Older feature versions still require their existing conversion procedure.

**V9 model weights and optimizer checkpoints are not v10 checkpoints.** Start fresh v10 IL weights in a new directory. Do not use `--from-imitation-run` with a v9 run, reuse a v9 self-play directory, or expect `--resume-reviewed-source` to convert architectures.

## Fresh v10 IL

Use the actual converted manifest path if yours differs. The commands below use the existing joint7 conversion path and create separate v10 run directories.

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/

python scripts/feature-engineering.py inspect

python scripts/step5.py --config configs/step5/joint7-latest.json imitate \
  --manifest artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run artifacts/step5/features4-v10-batch7-il \
  --epochs 10 --min-epochs 3 --patience 2 --workers 8
```

These epoch/worker arguments are an example lab invocation, not changed model defaults. The initial manifest load can be expensive if no current-source cache exists. The inspected model architecture must be `brass-four-seat-mlp-resnet/10.0.0`.

## Configure and start self-play from completed v10 IL

After IL completes successfully, generate a configuration pinned to that run's accepted checkpoint. Use a new output filename; the helper does not overwrite an existing config.

```bash
python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-v10-batch7-il \
  --output configs/step5/joint7-features4-v10-batch7.json

python scripts/step5.py --config configs/step5/joint7-features4-v10-batch7.json train \
  --fresh \
  --run artifacts/step5/joint7-features4-v10 \
  --imitation-run artifacts/step5/features4-v10-batch7-il \
  --iterations 1
```

Fresh PPO starts from the completed v10 IL weights with its own empty Adam state. The current curriculum, exploration 0.08, reward, utility calibration, teacher auxiliary supervision and teacher collection/distillation settings remain unchanged. In particular, this MLP revision does not adopt the separate Transformer plan's no-teacher recipe. Teacher phase clocks remain 9/12/12/12 seconds, five teacher games, eight search workers per teacher turn, eight distillation steps and `search_wall_fraction=null`.

## Continue the same v10 run

After the one-iteration command finishes at generation 1:

```bash
python scripts/step5.py --config configs/step5/joint7-features4-v10-batch7.json train \
  --run artifacts/step5/joint7-features4-v10 \
  --imitation-run artifacts/step5/features4-v10-batch7-il \
  --expected-generation 1 \
  --iterations 20
```

Set the optional `--expected-generation` guard to the actual committed generation, or omit it when intentionally resuming whatever generation is current. Matching v10 continuation preserves model weights, Adam, RNG, counters, population and lineage. Existing source-review and pending-collection checks still apply to later source changes.

## Small local validation

With the repository Python dependencies available:

```bash
PYTHONPATH=src:tests/FeatureEngineering:tests/R2:tests/Recovery \
  python -B -m unittest test_mlp_v10 test_factored_scorer test_grouped_projection -v

PYTHONPATH=src python -B tests/FeatureEngineering/benchmark_mlp_v10.py \
  --output artifacts/feature-engineering/v10-local-benchmark.json
```

The benchmark measures CPU tensor-ready inference only. Use a new output path. Training strength, full game throughput and GPU/deployment behavior require separate measurement; unit tests and one-game smoke validation do not establish them.
