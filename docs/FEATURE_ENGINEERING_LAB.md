# Six-group features and five residual blocks: IL and self-play

Use the `ai-training` conda environment and project root
`/data/not_backed_up/yxu209/ai-training/`. This guide is for the new feature/model
contract. Keep existing datasets, runs and exported packages in their own paths.

## Contract and preserved recipe

The active contract is the original six-group `permitted-features/2.0.0`,
with **5,489 state inputs / 548 action inputs**. All groups A–F remain required.
Hand coverage is disabled: its 28 state facts, four action facts and runtime
coverage pass have been removed. Existing raw hand counts and action/card
categorical inputs remain. State summaries feed the shared policy/value/utility
embedding; candidate effects feed the action encoder. The final 12 state entries
remain style[4]/intent[8].

The medium model keeps width **304** and now uses **five residual blocks**.
Its architecture is `brass-four-seat-mlp-resnet/4.0.0`; input schema 2 is reused
unchanged. The explicit base/utility cap remains **3,000,000**; personality remains
**3,100,000** with its 100,000 extension allowance. Counts: 2,599,808 state encoder;
2,766,477 base; 2,776,497 with utility; 2,842,330 with goals. The preceding four-block
models require their pinned runtime. This revision provides no weight/optimizer
migration; use fresh five-block IL and resume its own new checkpoints.

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

Old checkpoint architectures and source-bound encoded caches are incompatible.
Raw **heuristic** demonstrations
can be reused only through this explicit conversion and full native transition
audit. The converter checks the old contract, source hashes, every permitted
observation/legal list/transition and terminal result, then writes a new manifest
and `.features-v2.*` caches in a new directory. It preserves the complete raw game
objects, choice IDs/order, chosen labels, setup groups, train/validation/test split,
behavior records and teacher admission metadata. The source files are not changed.
Normal training still checks the qualified heuristic bank/admission evidence.

The converter accepts known v1/v2/v3 **heuristic raw** layouts when their rules,
wire and target contracts match, including the archived hand-coverage recordings.
The active output has only A–F. No new teacher games are required solely for this
rollback or the fifth residual block. The native audit regenerates tensors and
records original compatibility.
Neural behavior, search targets and old encoded tensors cannot use this exception.

To reuse an already converted Features-2 dataset, this is an **alternative** to
converting the original batch-7 manifest below. Choose one source, and use a new
output directory:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py convert \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-batch7/manifest.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-five-block-batch7 \
  --workers 48
```

Use the resulting manifest for fresh IL. Previous-architecture neural checkpoints remain
usable with their pinned runtime. The new loader rejects them, including with
`--resume-reviewed-source`; there is no old weight/optimizer migration or run reset.
The new checkpoint's own weights, Adam/RNG, counters and scheduling state resume
normally after training starts with the five-block architecture.
See [the rollback report](../ai-research/SIX_GROUP_FIVE_BLOCK_REPORT.md) for
validation. The earlier hand contract/report are archived experiment evidence;
they do not describe the active encoder.

The following example uses the existing batch-7 manifest path. Substitute the
actual audited manifest if that dataset has a different location. The output
directory must not exist.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py convert \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/v4-batch7-joint/manifest.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-five-block-batch7 \
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
the converter refuses to overwrite it. The worker count changes scheduling only; the new feature/model contract is
recorded independently in every manifest and cache receipt.

Read `conversion.json` and the new `manifest.json`. Missing required source facts
produce a failure, never zero-filled feature groups. Neural behavior and search
teacher targets cannot be relabeled through this converter. If the raw observations
or declared rules do not pass, generate fresh demonstrations with the existing
admitted v4 generator and audit them in separate output paths:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-latest.json \
  demo-generate --games 1000 \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/features2-five-block-v4
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-latest.json \
  demo-audit \
  --replays '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/features2-five-block-v4/*.jsonl.gz' \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-five-block-v4-manifest.json
```

Generation keeps the current v4 search settings and can be expensive. These are
lab commands for the user to start, not an implementation smoke. Verify that the
audited manifest has nonempty training and validation partitions before IL.

## Focused checks and tiny fresh-IL smoke

Run the focused implementation suites before starting a lab campaign:

```bash
PYTHONPATH=/data/not_backed_up/yxu209/ai-training/src:/data/not_backed_up/yxu209/ai-training/tests/R2:/data/not_backed_up/yxu209/ai-training/tests/Recovery:/data/not_backed_up/yxu209/ai-training/tests/FeatureEngineering \
python -m unittest test_fe1 test_fe2 test_fe3 test_fe4 test_fe_review \
  test_six_group_rollback test_hm_semantics test_hand_retention test_audit_retention \
  test_conversion_workers test_learning test_recovery \
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
For the smaller two-game variant, add `--ordinary-games-per-iteration 1`.

```bash
PYTHONPATH=/data/not_backed_up/yxu209/ai-training/src:/data/not_backed_up/yxu209/ai-training/tests/R2:/data/not_backed_up/yxu209/ai-training/tests/Recovery:/data/not_backed_up/yxu209/ai-training/tests/FeatureEngineering \
python /data/not_backed_up/yxu209/ai-training/tests/FeatureEngineering/smoke_fe5.py \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-five-block-tiny/manifest.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/feature-engineering/lab-smoke \
  --teacher
```

## Fresh full IL, selection and new self-play

Use new output directories. The full IL command keeps the existing supervised
minibatch/precision settings and starts random **five-block** weights. Omit
`--initial` and `--from-imitation-run` when starting this feature revision.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-latest.json \
  imitate \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features2-five-block-batch7/manifest.json \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-five-block-batch7-il \
  --epochs 10 --min-epochs 3 --patience 2
```

The selected `best.pt` is an imitation-loss selection, not evidence of strength.
A small validation panel can compare it with that same IL run's random initial
checkpoint. Keep held-out test setups separate from selection:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/evaluation.json \
  evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-five-block-batch7-il/best.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-five-block-batch7-il/initial.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/evaluation/features2-five-block-il-validation \
  --setups 8 --partition validation --no-search --no-include-legacy \
  --lineups recent-neural
```

Bind the completed IL checkpoint into a new config. This preserves the current
recipe and records the new IL path/hash/model ID/manifest/lineage plus the explicit
model version/limit. Rehearsal is bound to this same re-encoded IL manifest;
its weight and duration are preserved. It does not copy old PPO state. The
checked-in base config retains the historical IL identity as reference metadata;
it cannot import that previous four-block checkpoint into the five-block architecture.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py configure \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-five-block-batch7-il \
  --output /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features2-five-block-batch7.json
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features2-five-block-batch7.json \
  train --fresh \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/joint7-features2-five-block \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-five-block-batch7-il \
  --iterations 1
```

Resume the new run with the same config and IL identity. For example, after that
first committed iteration, run 20 additional iterations:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features2-five-block-batch7.json \
  train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/joint7-features2-five-block \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features2-five-block-batch7-il \
  --expected-generation 1 --iterations 20
```

`--expected-generation` is an optional guard; set it to the actual committed
generation, or omit it to continue the current checkpoint. Same-version resume
keeps this new run's model, Adam/RNG, games, curriculum, population and lineage.
Previous four-block PPO/IL checkpoints and incompatible encoded/teacher caches are rejected,
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
establish improved playing strength. Treat local CPU smoke timings as diagnostics;
measure GPU utilization, completed search work and
qualified-label throughput again on the actual lab hardware.
