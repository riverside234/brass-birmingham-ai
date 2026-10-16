# Grouped MLP v11.1: fresh IL and self-play

V11.1 is the current MLP default: projections **32/64/48/320/352/48 + 12**, fusion **876 → 512**, **eight width-512 residual blocks**, a width-160 action encoder, and a 192-wide policy scorer. Local slot/link/industry banks include neighboring slots and opponents' industry mats; their bias-free role projections start at zero. The shared per-player outcome head remains width 128. See the [model contract](../ai-research/GROUPED_PROJECTION_CONTRACT.md) and [validation report](../ai-research/MLP_V11_REPORT.md).

## Existing joint7 IL data

Reuse the already converted, audited **Features-4** dataset as-is. The state/action dimensions remain **3,128 / 327**, and feature, encoder, derived-feature, rules and target contracts are unchanged. The new model checks the static board topology as part of admission. Existing native-board data satisfies those checks, including the three-region Worcester/SouthWest/Kidderminster links.

Keep the manifest, labels, split assignments and raw shards. The normal loader may rebuild source-bound tensor caches in the new IL run's `.imitation-cache/`; do not regenerate demonstrations solely for this upgrade. A path containing v8 is still valid when its contents are Features-4.

Use **fresh v11.1 IL weights and a new self-play directory**. V11 and all earlier checkpoints cannot be resumed or reinterpreted as v11.1, including with source-review flags. Old neural PPO trajectories are not current behavior data.

## Training defaults

| Setting | IL | Self-play shared-network updates |
| --- | ---: | ---: |
| Peak learning rate | **6e-5** | **6e-6** |
| Previous width-304 rate | 1e-4 | 1e-5 |
| Warmup updates | **500** | **100** |
| First-update multiplier | **0.1** | **0.1** |
| Gradient norm clipping | **1.0** | **1.0** |

Warmup is linear from 10% on the first update to 100% on update 500/100, then stays constant. These are provisional settings, not a proven optimum. The counter is stored in Adam's parameter group and restored with the checkpoint; restarting does not restart warmup. Self-play counts successful PPO and teacher/distillation shared-network updates. Its existing temporary head-only utility calibration rate is retained and calibration steps do not advance warmup.

The selected JSON config contains:

```json
"learning_rate": 0.000006,
"learning_rate_warmup": {
  "contract": "linear-update-warmup/1.0.0",
  "imitation_steps": 500,
  "selfplay_steps": 100,
  "start_factor": 0.1
}
```

It also sets `imitation.learning_rate=0.00006` and `update.grad_clip=1.0`. To try clipping 0.5, change that value in a new explicitly selected training configuration. `imitate --learning-rate` and `--warmup-steps` override the IL rate and warmup length. Zero warmup steps explicitly disables the ramp. Reported effective rates can be lower than the peak during warmup; full and compact self-play histories retain the schedule counter.

## Fresh v11.1 IL

Use the actual converted manifest path if yours differs. New run paths avoid modifying previous checkpoints.

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/

python scripts/feature-engineering.py inspect

python scripts/step5.py --config configs/step5/joint7-latest.json imitate \
  --manifest artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run artifacts/step5/features4-v11-1-batch7-il \
  --epochs 10 --min-epochs 3 --patience 2 --workers 8
```

The epoch/loader-worker arguments are example lab settings. Inspection should show architecture `brass-four-seat-mlp-resnet/11.1.0`, local layout `grouped-mlp-local-entities/2.0.0` and **5,904,644 base parameters**. The IL CLI now takes its default learning rate from the selected config, so it does not silently retain the old hard-coded 1e-4.

To resume an interrupted v11.1 IL run at its committed epoch, repeat the same command with identical settings and source. Warmup, Adam and RNG restore from its checkpoint. Do not use `--from-imitation-run` to load an older architecture.

## Fresh self-play from completed v11.1 IL

After IL completes, generate a config bound to its selected checkpoint. The helper requires a new output filename.

```bash
python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --output configs/step5/joint7-features4-v11-1-batch7.json

python scripts/step5.py --config configs/step5/joint7-features4-v11-1-batch7.json train \
  --fresh \
  --run artifacts/step5/joint7-features4-v11-1 \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --iterations 1
```

The initial PPO model copies the completed IL weights exactly. PPO receives a fresh Adam optimizer and its own warmup counter. The utility extension is initialized/calibrated through the existing guarded path.

The current curriculum, four-seat collection, exploration 0.08, temperature, reward and teacher settings are preserved. Teacher clocks remain **9/12/12/12 seconds**, five games per collection, eight search workers per teacher turn, eight distillation steps and `search_wall_fraction=null`. This upgrade does not adopt the separate Transformer plan's no-teacher recipe.

## Continue v11.1 self-play

After the one-iteration run commits generation 1:

```bash
python scripts/step5.py --config configs/step5/joint7-features4-v11-1-batch7.json train \
  --run artifacts/step5/joint7-features4-v11-1 \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --expected-generation 1 \
  --iterations 20
```

Use the actual generation for the optional guard, or omit it to resume the current committed generation. Matching-source continuation preserves weights, Adam/RNG, warmup, counters, population and checkpoint lineage. Existing freshness/source-review rules remain active for later source changes and pending collections.

V11.1 adds the selected local vectors **before action GELU and LayerNorm**. Training keeps indexed selectors; new ONNX exports use the fixed action linear weights plus one `143 × 160` per-position projected-bank multiply. The exporter simplifies the graph, checks exact GELU/LayerNorm fusion, and runs ORT at `ORT_ENABLE_ALL`. Role projections still initialize at zero. The Features-4 data is unchanged, but this is a different learned function from v11: **fresh IL is required**, and no source-review receipt bridges v11 weights. See the [v11.1 report](../ai-research/MLP_V11_1_REPORT.md).

Existing signed runs and checkpoints are left untouched. Do not point these commands at an old v11 run directory. The rates remain IL `6e-5` / self-play `6e-6`; config generation from v11 does not apply the earlier 0.6 rate adjustment again.

## Small local checks

With the repository Python dependencies available:

```bash
PYTHONPATH=src:tests/FeatureEngineering:tests/R2:tests/Recovery \
  python -B -m unittest test_v11_1 test_mlp_v11 test_v11_architecture test_v11_learning_rate -v

PYTHONPATH=src:tests/FeatureEngineering python -B tests/FeatureEngineering/benchmark_v11_1.py \
  --output artifacts/feature-engineering/v11-1-local-benchmark
```

The benchmark compares the indexed and selector-multiply execution forms of the same v11.1 model. It measures CPU tensor-ready inference only. Neither smoke tests nor these timings establish improved playing strength.
