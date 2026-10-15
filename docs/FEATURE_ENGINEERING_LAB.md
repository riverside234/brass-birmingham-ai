# Widened compact grouped v9: lab instructions

Use the ai-training environment at /data/not_backed_up/yxu209/ai-training/.
The main target is **3,128 state / 327 action inputs**, projections **32/64/48/192/224/48 + 12**, fusion **620 -> 304**, and **six width-304 residual blocks**. The candidate encoder remains width 128. Base/utility/personality counts are **2,005,917 / 2,015,937 / 2,081,770**.

See [the active model contract](../ai-research/GROUPED_PROJECTION_CONTRACT.md) and [revision receipt](../ai-research/FEATURE_ENGINEERING_WIDTH_REVISION_REPORT.md). Feature/layout/model identities are **4.0.0 / 5.0.0 / 9.0.0**.

## Reuse the dataset already generated from joint7

**Do not rerun conversion** if you already completed the previous lab command producing:

~~~text
artifacts/il-replay/features4-compact-v8-batch7/manifest.json
~~~

That Features-4 dataset remains compatible. Its directory name can still contain v8: this is a data path, not a checkpoint architecture. The feature manifest, reference encoder, derived semantics, input order and dimensions are unchanged. Normal IL loading preserves labels/splits, verifies all existing admission rules, and re-encodes stale architecture/source-bound caches one shard at a time in workers. Standalone IL keeps the prepared results under the run's `.imitation-cache/` and reads dense features per minibatch. It does not overwrite the dataset or repeat the native transition audit. Initial loading can therefore take longer than a cache hit.

Use your actual converted manifest path if different. The unconverted original v4-batch7-joint dataset still needs the original conversion step; this width revision does not make older feature versions compatible. Conversion is needed only when no Features-4 conversion exists:

~~~bash
python scripts/feature-engineering.py convert \
  --manifest artifacts/il-replay/v4-batch7-joint/manifest.json \
  --output artifacts/il-replay/features4-compact-v8-batch7 \
  --workers 48
~~~

## Effective self-play settings

Generate from configs/step5/joint7-latest.json. Teacher collection uses five games, cadence 160, eight distillation steps and the existing minibatch/reuse/quality checks.

| Phase | Reserved rounds | Shared turn budget |
| --- | --- | ---: |
| Early Canal | Canal 1-4 | 9 s |
| Late Canal | Canal 5-8 | 12 s |
| Early Rail | Rail 1-6 | 12 s |
| Late Rail | Rail 7-8 | 12 s |

The new execution.teacher_search_workers_per_turn is **8**: five concurrent teacher games provide **40 private search workers**, plus coordinators and inference. The existing global CPU admission queue remains active. Each clock is shared across workers, both actions and continuations. Worker count never multiplies the turn's seconds. Generic execution.search_workers_per_turn and v4 search_workers remain **4**, preserving evaluation/opponent search.

search_wall_fraction=null disables only the cumulative allowance. Deadlines, cancellation, failure timeouts, source freshness and utility calibration remain active. Startup/collection logs and compact history report teacher workers per game and total capacity; these are capacity figures, not a promise that every worker is busy throughout collection.

The four clocks allocate 180 reserved seconds per teacher game, or 900 summed seconds per five-game collection before ordinary moves/overhead. Concurrent games mean this is not elapsed wall time. The maximum phase timeout initialization is now 12 seconds.

Each actor keeps one encoder per reply lane across frozen-model replacements. Its coordinator and private search workers share a bounded cache of completed, read-only feature arrays; mutable feature builders remain private. Per-game execution records expose search.encoding_cache (warm_hit, miss, evictions and retained entries), counting the shared cache once rather than once per seat/model. See [the encoder benchmark and its scope](../ai-research/ENCODER_REUSE_REPORT.md).

Exploration stays 0.08, temperature 1, entropy/value weights 0.005/0.5, with the existing family/industry distribution, curriculum, latest-model four-seat collection and learner-only collection against frozen opponents. The champion gate stays disabled. Insolvency remains 0.005 per event capped at 0.075. Deployment search settings are unchanged.

## Fresh v9 IL, using the existing dataset

The width change requires fresh model weights. Existing v8 IL/self-play checkpoints cannot resume into v9, even though their heuristic dataset is reusable. Use new model/run directories and leave prior runs intact.

~~~bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/

python scripts/feature-engineering.py inspect

python scripts/step5.py --config configs/step5/joint7-latest.json imitate \
  --manifest artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run artifacts/step5/features4-compact-v9-batch7-il \
  --epochs 10 --min-epochs 3 --patience 2
~~~

### Retry a worker failure during IL loading

After updating the checkout, rerun the **same** `imitate` command, manifest and run directory. For the documented run, a conservative initial worker cap is:

~~~bash
python scripts/step5.py --config configs/step5/joint7-latest.json imitate \
  --manifest artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run artifacts/step5/features4-compact-v9-batch7-il \
  --epochs 10 --min-epochs 3 --patience 2 --workers 8
~~~

Keep your original epoch, patience, learning-rate, batch and device options if they differ. `--workers` controls CPU preparation only; it does not change training batches or labels. The previous 16-worker default remains available. An abrupt child-process failure now halves preparation concurrency and retries unfinished shards, down to one worker. Persistent failure at one worker stops with source context; no shard is skipped.

A failure while loading the first training split occurs before this new run's initial model/state is saved. It can be retried in the same directory without resetting or regenerating anything. The older loader did not save its completed preparation, so the first retry after this repair must prepare those shards again. The repaired loader commits each prepared shard and reuses it on later retries after verifying the raw SHA, preparation identity and output checksums. Corrupt derived cache is rebuilt; changed raw data or failed admission is still fatal.

Prepared features are uncompressed and require additional disk space. The startup report includes `prepared_bytes`, `resident_index_bytes`, `cache_hits`, elapsed preparation time and worker retries. `resident_index_bytes` counts numeric indexes only, not total process/GPU memory. RAM still includes one raw/prepared shard per live worker, small dataset-wide label/index columns and bounded minibatches. Disk reads replace the previous all-dataset RAM retention; epoch speed depends on storage. Prefer sufficient fast local storage and inspect OS/scheduler logs if a worker is still killed. `BrokenProcessPool` alone does not establish OOM.

See [the loader repair and validation report](IL_LOADER_RECOVERY_2026-10-15.md). Existing committed unfinished IL remains source-bound; this repair does not silently reset or re-sign it. Completed v9 IL from the immediately preceding cache revision remains accepted for handoff. Existing self-play from that exact source can use its normal command with `--resume-reviewed-source` once at a committed boundary, without changing its configuration. The previously reviewed v9 clock transition remains available with the same restrictions.

After IL completes, generate a new configuration. This binds initialization and rehearsal to that completed v9 IL checkpoint and the same existing dataset, and applies the new teacher settings:

~~~bash
python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-compact-v9-batch7-il \
  --output configs/step5/joint7-features4-compact-v9-batch7.json

python scripts/step5.py \
  --config configs/step5/joint7-features4-compact-v9-batch7.json train --fresh \
  --run artifacts/step5/joint7-features4-compact-v9 \
  --imitation-run artifacts/step5/features4-compact-v9-batch7-il \
  --iterations 1
~~~

The checked-in base config has an older IL reference; use the generated config for self-play. PPO inherits the new IL weights and starts with an empty optimizer. These are production commands, not commands run during implementation validation.

## Continue the same v9 run

After its first committed iteration, continue for 20 additional iterations:

~~~bash
python scripts/step5.py \
  --config configs/step5/joint7-features4-compact-v9-batch7.json train \
  --run artifacts/step5/joint7-features4-compact-v9 \
  --imitation-run artifacts/step5/features4-compact-v9-batch7-il \
  --expected-generation 1 --iterations 20
~~~

--expected-generation is optional: use the actual committed generation or omit it. Same-target continuation preserves weights, Adam/RNG, progress, population, curriculum and lineage. Do not regenerate or edit a signed run's configuration casually; source-review flags cannot bypass incompatible weights or feature contracts.

### Adopt the cache revision in an existing preceding-v9 run

The preceding v9 implementation (63c9da11e4aa582af8ad41ea32c85781728ff6418c953a2df1170f7718d5d9b4) has an exact reviewed transition. Completed IL from that source remains usable; no IL retraining is needed for this cache/timing change. An unfinished IL run remains bound to its original source.

For an already-generated self-play config, change only teacher_work_budget.seconds_per_phase to:

~~~json
{"canal_early": 9.0, "canal_late": 12.0, "rail_early": 12.0, "rail_late": 12.0}
~~~

At a committed iteration boundary, run the same continuation command with --resume-reviewed-source once:

~~~bash
python scripts/step5.py \
  --config configs/step5/joint7-features4-compact-v9-batch7.json train \
  --run artifacts/step5/joint7-features4-compact-v9 \
  --imitation-run artifacts/step5/features4-compact-v9-batch7-il \
  --resume-reviewed-source --iterations 20
~~~

This exact transition preserves weights, Adam/RNG, counters, population and lineage; the existing source-change safeguard requests fresh utility calibration. A pending old-source collection still blocks the transition. Other source identities or unrelated config changes are not approved by this entry. Fresh configuration generation automatically uses 9/12/12/12; explicitly configured historical clocks are not silently rewritten.

## Named CPU export

Choose a retained v9 utility checkpoint and a new destination:

~~~bash
python scripts/feature-engineering.py export \
  --checkpoint artifacts/step5/joint7-features4-compact-v9/generation-0001.pt \
  --output artifacts/exports/compact-v9-g0001 \
  --include-personality
~~~

The optional personality export imports the exact utility parent into an initialized goal extension. Its goals are untrained and diagnostic. Omit the flag for utility-only export. Qualified package workflows retain their gates and carry Features-4 and layout-5 metadata.

## Bounded debugging only

The width revision was checked with focused model, admission, timing and worker-allocation regressions, then one tiny IL/handoff/export smoke. It collected no games and ran no timed teacher search. For a concrete regression, run only the relevant tests; a full suite, long training, comparison campaign or benchmark is not needed.

~~~bash
export PYTHONPATH=src:tests/R2:tests/Recovery:tests/FeatureEngineering:tests/Personality
python -m unittest test_compact_widths test_compaction \
  test_teacher_phase_revision.PhaseClockTests -v

# Requires an existing tiny Features-4 manifest: one train + one validation game.
python tests/FeatureEngineering/smoke_compact_widths.py \
  --manifest artifacts/il-replay/tiny-qualified-features4/manifest.json \
  --output artifacts/feature-engineering/compact-v9-smoke-lab
~~~

The smoke admits the existing dataset through the production loader, uses eight examples per partition for one IL optimizer step, checks completed-IL handoff and configuration generation, then verifies named ONNX parity on two small shapes. Source dataset/cache files stay unchanged. It does not establish label yield, training stability or playing strength.
