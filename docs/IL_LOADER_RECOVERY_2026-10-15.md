# IL loader recovery, 2026-10-15

## Finding

The reported Linux run lost a worker after 2,799 of 3,498 training shards. `BrokenProcessPool` establishes abrupt worker termination, but does not identify OOM, a scheduler kill or a native crash without the OS/job logs.

Inspection found an independently actionable memory problem: the parent retained each shard's complete prepared dense arrays in `parts`, then flattened every shard before training. Bounding pending futures did not bound this completed-result memory. Sixteen simultaneous raw readers/encoders and dense result serialization added peak memory. Reducing worker count alone would not remove the dataset-wide retention.

## Repair

Standalone `step5.py imitate` now prepares one shard per worker into the owned run's `.imitation-cache/{train,validation}/`. Dense arrays do not cross process IPC. Workers return small receipts; the parent commits each completed shard independently. Training keeps numeric labels/indexes and reads float32 state/action arrays for each bounded minibatch. Eight file handles per split and the existing two-item CUDA prefetch queue bound open files and queued batches.

The binary records preserve metadata, targets, identity, state, all legal action features and stable-seat mapping. The numeric index binds balancing keys and game identities as well as offsets/counts. Receipts bind the raw manifest entries, partition, allowed teachers, preparation/feature identity and output checksums. Loading still verifies the authoritative raw SHA on cache hits. Invalid derived cache is rebuilt; changed raw data, unauthorized teachers, invalid splits or malformed replay remain errors. No labels or shards are skipped.

Abrupt worker termination retains all committed receipts and halves concurrency, with bounded retries down to one worker. A further failure at one worker stops with the unfinished source path and diagnostic guidance. Ordinary data-validation errors are not hidden behind worker retries. Interrupted partial writes have no committed receipt and are rebuilt.

The standalone IL trainer closes its prepared stores on success and failure. A directory containing only its preparation cache can be retried after failure before epoch zero. Existing checkpoint/source/signature guards remain intact. The ordinary in-memory loader API for other consumers is unchanged; this repair targets the reported standalone IL ingestion path.

## Preserved behavior

Manifest/shard/row order, all legal choices, exact float32 inputs, teacher/era/action balancing with the threefold cap, forced-action handling, candidate-cell bucketing, natural and balanced validation, optimizer step order and early-stop selection remain unchanged. No architecture, feature semantics, model width, teacher timing, worker allocation for search, PPO, exploration, reward or curriculum settings changed.

The current code identity is `014ac0ca4f920053eef265cb58b65f58901b7d35b2f6b1c1a1eaf8dbe92a5809`. The model/feature compatibility digest remains `747216b5ab873126a6e048ade198883593113f70a815b9012ce40b8f66d443cb`.

Completed IL from preceding source `e794a86e13b891d56fcb21a8a7580a8af1c754bc54065a8b636dd5b04f6792f8` retains explicit handoff admission. Self-play from that source has an exact reviewed continuation with unchanged configuration. The earlier `63c9da11...` v9 source retains the previously reviewed unchanged-or-exact-clock-transition permission. All older approval entries are preserved. These entries do not admit unknown source, incompatible architecture, unrelated configuration changes or pending old-source collections. Unfinished committed IL is still source-bound; no state reset or signature rewriting was added.

## Validation

Focused regressions cover exact features, metadata and labels; minibatch and candidate-cell boundaries; balancing; natural/balanced metrics; identical trained parameters, Adam state and RNG after one update; bounded feature access/file handles/IPC receipts; corrupted, uncommitted and relabeled cache; changed raw input; admission rejection; same-directory retry before any initial checkpoint; completed-run cache reuse; and two actual spawned-worker exits. One injected failure verifies a committed shard survives a 2-to-1-worker retry. Persistent failure at one worker verifies a clear error without data omission.

All **14 focused regressions passed** (114.54 seconds). Continuation regressions check exact weights/Adam/RNG, progress/population/lineage, source/configuration checks, pending-work rejection and preservation of previous source reviews. Commands and evidence:

```bash
export PYTHONPATH=src:tests/R2:tests/Recovery:tests/FeatureEngineering:tests/Personality
python -m unittest test_bounded_imitation test_bounded_imitation_resume test_encoder_reuse_resume -v
```

Evidence: `artifacts/recovery/il-loader-bounded-20261015/verified-tests.txt`.

The isolated smoke uses an existing qualified Features-4 manifest with one complete training game (345 decisions) and one validation game (346 decisions), one CPU optimizer update, then a repeated invocation of the completed run. It checks cache hits, unchanged checkpoint/state hashes, unchanged source replay/cache files and completed-IL handoff. No gameplay, teacher search or long training is launched.

```bash
python tests/Recovery/smoke_bounded_imitation.py \
  --manifest artifacts/feature-engineering/compaction-v8-20261015/smoke/data/manifest.json \
  --output artifacts/recovery/il-loader-bounded-20261015/smoke-final
```

The smoke **passed** in 53.06 seconds including initial preparation, training, verified cache reuse, checkpoint/hash checks and handoff. Evidence: `artifacts/recovery/il-loader-bounded-20261015/smoke-final/report.json`; combined verification: `artifacts/recovery/il-loader-bounded-20261015/verification.json`. Use a new output directory when repeating it. This is local CPU validation, not a replay of the user's entire Linux dataset or a GPU memory benchmark. It does not establish a playing-strength improvement.

## Runtime and retry

Prepared records are uncompressed, so this trades additional disk storage and minibatch I/O for removal of dataset-wide dense-feature RAM retention. One raw/prepared shard per active worker still consumes memory; labels, indexes, epoch permutations and diagnostic vectors scale with row count. Reported `resident_index_bytes` is the numeric index size only, not process RSS. Source/output hash verification still reads data on reuse.

The tiny smoke's prepared files total 47,826,719 bytes and its resident numeric indexes total 58,060 bytes. These sizes are specific to its 691 decisions and legal-choice counts; they are not an estimate for all 3,498 shards. Fast local storage is preferable. First preparation may add write time; subsequent runs can avoid repeated feature construction, with throughput depending on source/cache reads and worker startup. No production throughput claim is made.

After syncing these changes to Linux, rerun the original `imitate` command and keep its manifest/run/hyperparameters. Add `--workers 8` for a lower initial memory footprint; the default 16 is still accepted and the new fallback is automatic. See the [exact lab retry command](FEATURE_ENGINEERING_LAB.md#retry-a-worker-failure-during-il-loading).

The old failed loader did not persist its completed preparation, so the first retry must prepare those shards again. The new receipts preserve subsequent progress. This traceback occurs before the first initial checkpoint/state is created; it does not require dataset regeneration or deleting the run. If the OS still kills workers, inspect its logs and job memory/disk limits rather than treating this patch as proof that every possible worker-death cause has been eliminated.

## Changed files

- `src/brass_ai/recovery/demonstrations.py`: standalone IL integration, numeric balancing and resource lifetime.
- `src/brass_ai/recovery/prepared_store.py`: verified prepared shards, bounded retry/IPC and lazy supervised dataset.
- `configs/learning/imitation-handoff-sources.json`, `configs/learning/selfplay-resume-sources.json`: exact compatible-source receipts.
- `tests/Recovery/test_bounded_imitation.py`, `test_bounded_imitation_resume.py`, `smoke_bounded_imitation.py`: focused correctness/failure checks and bounded real-data smoke.
- This report and `docs/FEATURE_ENGINEERING_LAB.md`: retry instructions, scope and storage implications.
