# Features 2: fresh imitation learning and self-play

Use the `ai-training` conda environment and project root
`/data/not_backed_up/yxu209/ai-training/`. This guide is for the new feature/model
contract. Keep existing datasets, runs and exported packages in their own paths.

## Contract and preserved recipe

The active contract is `permitted-features/2.0.0`, with 5,489 state inputs and
548 action inputs. All six groups A–F are required. State summaries feed the
shared policy/value/utility embedding; candidate effects feed the action encoder.
The final 12 state entries remain style[4]/intent[8].

The medium model keeps width **304** and four residual blocks. Its versioned
architecture is `brass-four-seat-mlp-resnet/2.0.0`. The explicit base/utility cap
is **3,000,000**; the personality cap is **3,100,000** with its existing 100,000
extension allowance. Counts: 2,413,760 state encoder; 2,580,429 base including
policy/official heads; 2,590,449 with utility; 2,656,282 with goals.

Fresh config generation retains the current latest-model recipe: ordinary learner
exploration 0.08, temperature 1.0, entropy weight 0.005, value weight 0.5, four-seat
learner collection on current-model tables, existing curriculum and v4 window,
no champion gate, reward/insolvency 0.005 per event capped at 0.075, search and
resource settings. Teacher collection remains five games, cadence 160 ordinary
games, eight distillation steps, and the existing quality/freshness/reuse checks.
The latest-model recipe's pinned phase clocks remain **12/12/16/24 seconds** for
early Canal / late Canal / early Rail / late Rail. This recipe deliberately
preserves its explicit clocks instead of applying archived gated-run migrations.
The generated config retains them. Each clock is shared across workers, both
actions and subchoices.
`search_wall_fraction=null` disables only cumulative wall-fraction admission;
per-turn deadlines, cancellation and failure timeouts remain active.

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py inspect
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-latest.json \
  preflight
```

## Audit and re-encode existing raw demonstrations

V1 encoded caches and checkpoints are incompatible. Raw **heuristic** demonstrations
can be reused only through this explicit conversion and full native transition
audit. The converter checks the old contract, source hashes, every permitted
observation/legal list/transition and terminal result, then writes a new manifest
and `.features-v2.*` caches in a new directory. It preserves the complete raw game
objects, choice IDs/order, chosen labels, setup groups, train/validation/test split,
behavior records and teacher admission metadata. The source files are not changed.
Normal training still checks the qualified heuristic bank/admission evidence.

The explicit converter also accepts earlier Features-2 **heuristic raw** recordings
after an encoder/source repair, provided the frozen raw layout, rules, wire and
target contracts still match. It performs the full native audit again and records
the original compatibility in the new shard header. Source changes invalidate
encoded-cache receipts; never edit fingerprints to reuse old tensors. Neural
behavior, search targets and checkpoints do not gain a migration exception.
If `features2-batch7` was converted before the review fixes, use a new output:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py convert \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-batch7/manifest.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-reviewed-batch7 \
  --workers 48
```

Use that new manifest for subsequent fresh IL. These fixes change exact encoder
and source fingerprints despite the verified identical numeric tensors. Previously
created neural checkpoints remain usable with their pinned runtime; the current
loader rejects them, and `--resume-reviewed-source` does not override a learning
compatibility mismatch. This patch supplies no checkpoint migration or rewrite.
See the [post-implementation review](../ai-research/FEATURE_ENGINEERING_REVIEW.md)
for the fixes, feature-quality checks and their measurement limits.

The following example uses the existing batch-7 manifest path. Substitute the
actual audited manifest if that dataset has a different location. The output
directory must not exist.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py convert \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/v4-batch7-joint/manifest.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-batch7 \
  --workers 48
```

Conversion uses separate CPU processes and private rules hosts per replay shard.
The default is up to 48 workers, capped by available CPUs (including process
affinity and `SLURM_CPUS_PER_TASK`) and the number of shards. `--workers 1` runs
serially. Each worker uses one tensor thread; reduce `--workers` if memory or
storage throughput limits concurrency. A single replay shard remains one job.
Completion progress is reported immediately; the final manifest retains source
order and is committed only after all audits and duplicate checks pass.
`conversion.json` records the effective worker count, process IDs and elapsed time.
If an earlier conversion created the output directory, use a new output name;
the converter refuses to overwrite it. This scheduling change leaves feature
encoding and model compatibility unchanged.

Read `conversion.json` and the new `manifest.json`. Missing required source facts
produce a failure, never zero-filled feature groups. Neural behavior and search
teacher targets cannot be relabeled through this converter. If the raw observations
or declared rules do not pass, generate fresh demonstrations with the existing
admitted v4 generator and audit them in separate output paths:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-latest.json \
  demo-generate --games 1000 \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/features2-v4
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-latest.json \
  demo-audit \
  --replays '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/features2-v4/*.jsonl.gz' \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-v4-manifest.json
```

Generation keeps the current v4 search settings and can be expensive. These are
lab commands for the user to start, not an implementation smoke. Verify that the
audited manifest has nonempty training and validation partitions before IL.

## Focused checks and tiny fresh-IL smoke

Run the focused implementation suites before starting a lab campaign:

```bash
PYTHONPATH=/data/not_backed_up/yxu209/ai-training/src:/data/not_backed_up/yxu209/ai-training/tests/R2:/data/not_backed_up/yxu209/ai-training/tests/Recovery:/data/not_backed_up/yxu209/ai-training/tests/FeatureEngineering \
python -m unittest test_fe1 test_fe2 test_fe3 test_fe4 test_fe_review test_conversion_workers test_learning test_recovery \
  test_exploration08 test_latest_model_recipe test_teacher_phase_revision \
  test_time_budgets test_inference_deadline_race test_teacher_readiness_review \
  test_inference_performance test_followup_inference_review -q
```

For a small native integration smoke, create a separate conversion manifest with
one complete training and one complete validation heuristic game. Pass it to the
following helper; do not pass a full dataset to a smoke intended to be small.
It trains a real width-304 model for one epoch, checks same-version IL completion,
plays six ordinary games over two PPO iterations with Adam/RNG resume, and runs
one four-seat-rotated diagnostic evaluation setup. It uses reduced CPU actors,
minibatches and PPO epochs in its isolated output. `--teacher` adds four real
reserved-turn probes with the resolved production clocks and unchanged quality
gates. Native partial-turn labels are diagnostic only and do not enter training.

```bash
PYTHONPATH=/data/not_backed_up/yxu209/ai-training/src:/data/not_backed_up/yxu209/ai-training/tests/R2:/data/not_backed_up/yxu209/ai-training/tests/Recovery:/data/not_backed_up/yxu209/ai-training/tests/FeatureEngineering \
python /data/not_backed_up/yxu209/ai-training/tests/FeatureEngineering/smoke_fe5.py \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-tiny/manifest.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/feature-engineering/lab-smoke \
  --teacher
```

## Fresh full IL, selection and new self-play

Use new output directories. The full IL command keeps the existing supervised
minibatch/precision settings and starts random **new-schema** weights. Omit
`--initial` and `--from-imitation-run` when starting this feature revision.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-latest.json \
  imitate \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-batch7/manifest.json \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-batch7-il \
  --epochs 10 --min-epochs 3 --patience 2
```

The selected `best.pt` is an imitation-loss selection, not evidence of strength.
A small validation panel can compare it with that same IL run's random initial
checkpoint. Keep held-out test setups separate from selection:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/evaluation.json \
  evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-batch7-il/best.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-batch7-il/initial.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/evaluation/features2-il-validation \
  --setups 8 --partition validation --no-search --no-include-legacy \
  --lineups recent-neural
```

Bind the completed IL checkpoint into a new config. This preserves the current
recipe and records the new IL path/hash/model ID/manifest/lineage plus the explicit
model version/limit. It does not copy the old IL weights or old PPO state. The
checked-in base config retains the historical IL identity as reference metadata;
it cannot be used to import that old-schema checkpoint into features 2.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py configure \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-batch7-il \
  --output /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features2-batch7.json
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features2-batch7.json \
  train --fresh \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/joint7-features2 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-batch7-il \
  --iterations 1
```

Resume the new run with the same config and IL identity. For example, after that
first committed iteration, run 20 additional iterations:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features2-batch7.json \
  train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/joint7-features2 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-batch7-il \
  --expected-generation 1 --iterations 20
```

`--expected-generation` is an optional guard; set it to the actual committed
generation, or omit it to continue the current checkpoint. Same-version resume
keeps this new run's model, Adam/RNG, games, curriculum, population and lineage.
Old-schema PPO/IL checkpoints and old encoded/teacher caches remain incompatible,
including when `--resume-reviewed-source` is supplied. Retain the old pinned
runtime/export with old models; this revision supplies no old-weight migration.

## Interpreting results

Inspect per-phase search attempts/completions, visited/revisited action coverage,
qualified and insufficient-coverage labels, elapsed time, distinct labels used,
unused qualified labels, presentations/reuse and teacher policy drift. Preserve
the utility calibration and fresh-model/source checks. A zero-label short search
is a measured zero; do not relax its gates or duplicate rows to force an update.

Compare development progress, tier mix, higher-tier flips, unflipped era-end
exposure, insolvency, final VP, placement and wins on independent setups. More
development, entropy, qualified labels or a passed functional smoke does not
establish improved playing strength. Use the performance report's CPU-only
measurements as diagnostics; measure GPU utilization, completed search work and
qualified-label throughput again on the actual lab hardware.
