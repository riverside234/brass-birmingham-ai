# Compact grouped v8: lab instructions

Use the ai-training environment at /data/not_backed_up/yxu209/ai-training/.
The main target is **3,128 state / 327 action inputs**, projections **32/64/48/160/192/48 + 12**, fusion **556 → 304**, and **six width-304 residual blocks**. The candidate encoder remains width 128. Base/utility/personality counts are **1,904,445 / 1,914,465 / 1,980,298**.

See [the active model contract](../ai-research/GROUPED_PROJECTION_CONTRACT.md) and [the implementation receipt](../ai-research/FEATURE_ENGINEERING_COMPACTION_REPORT.md). Feature/layout/model identities are respectively 4.0.0 / 4.0.0 / 8.0.0. Old runs are not converted or reset.

## Preserved recipe

Generate from configs/step5/joint7-latest.json. Exploration stays 0.08, temperature 1, entropy/value weights 0.005/0.5, with the existing family/industry exploration distribution, curriculum, latest-model four-seat collection and learner-only collection against frozen opponents. The champion gate stays disabled. Insolvency remains 0.005 per event capped at 0.075.

Teacher collection remains five games, cadence 160, eight distillation steps and the existing minibatch/reuse/quality checks. Effective phase clocks remain **12/12/16/24 seconds** (early Canal / late Canal / early Rail / late Rail), shared across workers, actions and continuations. search_wall_fraction=null disables only the cumulative allowance. Deadlines, cancellation, failure timeouts, source freshness and utility calibration remain active. Deployment search settings are unchanged.

~~~bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py inspect
~~~

## Re-encode compatible raw demonstrations

Use a new output directory and an actual qualified heuristic source manifest. The converter verifies source hashes, rules/wire/target pins, external behavior, absence of search teachers, splits/labels and every native transition, then creates new encoded caches. Existing files remain untouched. Do not relabel old neural PPO probabilities or teacher targets.

~~~bash
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py convert \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/v4-batch7-joint/manifest.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features4-compact-v8-batch7 \
  --workers 48
~~~

Check conversion.json and nonempty train/validation partitions. Worker count is capped by CPU allocation and shard count; lower it for memory/storage constraints. Unsupported static board/catalogue variants fail admission and require a new contract.

## Fresh IL and separate self-play

Existing layouts/checkpoints cannot continue into this target. These are full lab commands; implementation validation used only the tiny smoke below. Do not use an existing output directory or the smoke checkpoint for production.

~~~bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-latest.json \
  imitate \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features4-compact-v8-batch7-il \
  --epochs 10 --min-epochs 3 --patience 2

python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py configure \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features4-compact-v8-batch7-il \
  --output /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features4-compact-v8-batch7.json

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features4-compact-v8-batch7.json \
  train --fresh \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/joint7-features4-compact-v8 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features4-compact-v8-batch7-il \
  --iterations 1
~~~

The generator binds initialization **and rehearsal** to completed target IL and its re-encoded dataset. The checked-in base config's older IL reference is not target-compatible. Use the generated config. PPO inherits those IL weights but starts with a new optimizer. Device selection follows the existing environment/configuration.

After the first committed target iteration, continue for 20 additional iterations:

~~~bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/joint7-features4-compact-v8-batch7.json \
  train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/joint7-features4-compact-v8 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/features4-compact-v8-batch7-il \
  --expected-generation 1 --iterations 20
~~~

--expected-generation is optional: use the actual committed target generation or omit it. Same-target continuation preserves weights, Adam/RNG, counters, completed games, population, curriculum and lineage. Source-review flags cannot bypass incompatible features/layouts.

## Named CPU export

Choose a retained utility checkpoint from this new run and a new destination:

~~~bash
python /data/not_backed_up/yxu209/ai-training/scripts/feature-engineering.py export \
  --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/joint7-features4-compact-v8/generation-0001.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/exports/compact-v8-g0001 \
  --include-personality
~~~

This checks named CPU ONNX outputs. The optional personality flag imports the exact utility parent into an initialized goal extension; its goals are untrained and it is diagnostic, not release-qualified. Omit it for utility-only export. Qualified personality/package workflows retain their gates and carry the current feature/layout manifests.

## Minimal debugging workflow

~~~bash
export PYTHONPATH=/data/not_backed_up/yxu209/ai-training/src:/data/not_backed_up/yxu209/ai-training/tests/R2:/data/not_backed_up/yxu209/ai-training/tests/Recovery:/data/not_backed_up/yxu209/ai-training/tests/FeatureEngineering:/data/not_backed_up/yxu209/ai-training/tests/Personality
python -m unittest test_compaction -v
~~~

Use a tiny existing qualified heuristic manifest with exactly one complete training and one complete validation game. The smoke performs re-audit/conversion itself, selects eight IL examples per partition as a test fixture, and collects one current-model game:

~~~bash
python /data/not_backed_up/yxu209/ai-training/tests/FeatureEngineering/smoke_compaction.py \
  --manifest /data/not_backed_up/yxu209/ai-training/artifacts/il-replay/tiny-qualified-heuristic/manifest.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/feature-engineering/compact-v8-smoke-lab
~~~

The smoke checks fresh IL handoff, actual behavior probabilities, utility calibration, checkpoint/Adam/RNG restoration, population/lineage, a spawned teacher leaf request and actor ONNX parity on two small shapes. It uses one IL update and one PPO optimizer step, with three head-only calibration steps reported separately. Resume is checked without another training iteration.

Only the fixture uses CPU two threads, one actor/host, reduced minibatches, no rehearsal/evaluation/retention and explicit current-model table selection. Production curriculum/teacher settings are preserved. --resume-debug can finish an incomplete current-source fixture after a harness fix; it refuses another game after generation 1 is committed. Completed smoke directories are immutable.

Stop after these checks pass. Broaden only to diagnose a concrete failure. No full suite, seed sweeps, timed teacher collection, benchmark, old-model comparison or strength evaluation is required. The smoke proves neither strength preservation nor deployment speedup.
