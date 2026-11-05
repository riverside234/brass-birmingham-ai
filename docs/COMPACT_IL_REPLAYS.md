# Compact raw IL replays

New `step5.py demo-generate` collections automatically use
`four-seat-il-compact-replay/1.0.0`. Keep using the existing generation, audit
and imitation commands, with a new generation output directory. No config flag
or new dependency is required. The filenames remain `*.jsonl.gz`.

The generation spec and each job's provenance pin the storage format. The
generation report includes `replay_storage.format`, unique shard count and
compressed bytes. Retry under the same spec reuses published shards; an
incompatible existing destination is rejected. Existing datasets, manifests,
checkpoints and self-play replay files are not converted or overwritten.

## What is stored

Each game stores its longest observed history once. Individual observations
reference the exact prefix they originally saw; a history that is not a prefix
stays inline. `players`, `slots`, `links` and `merchants` use an initial snapshot
followed by structural deltas. Shape changes use full replacements. All other
fields remain intact, including private hands, legal choices/order, chosen
indices, teacher identity, behavior probabilities, setup split and outcome.
There is no quantization or tensor packing.

The reader reconstructs independent full observations before normal admission,
native transition auditing, feature encoding or training. No future history is
exposed to the encoder. It rejects malformed deltas/prefixes and unknown formats.
This format accepts heuristic IL games only, with no neural behavior, search
labels or branches. Legacy full-JSON replays remain supported. The feature
conversion/re-audit tool also accepts compact sources.

## Training and caches

Prepared IL arrays remain dense and unchanged. Compaction is expanded once per
raw shard load/preparation, not per optimizer minibatch. A valid disk-backed
prepared-cache hit verifies hashes and uses the existing dense files without
decoding the raw replay. In-memory rehearsal expands a shard when loading or
refreshing its samples, then trains on its prepared arrays.

The decoder participates in source-bound cache identity. Upgrading this source
can trigger a one-time rebuild of older derived/prepared caches through existing
freshness checks. It does not require regenerating or relabeling existing games.
The implementation also fixes missing `InputAudit`/`BatchCache` initialization
in the disk-backed `PreparedDataset`, required by inherited minibatch collation.

Self-play/PPG storage formats, training settings and active run state are
unchanged.

## Reviewed g921 continuation

The g921 trained-restart run can use the current source without checking out an
older commit. The source-review registry permits the exact transition from its
saved `4c514df4...` implementation with unchanged effective configuration. It
still requires `--resume-reviewed-source`; arbitrary source or config changes
remain rejected.

`configs/learning/derived-source-equivalence.json` also records the exact
transport-only change to `feature_engineering/data.py` as feature-equivalent.
This preserves compatibility with the existing model and completed IL dataset:
the decoder reconstructs the original observations and the encoded tensors are
unchanged. Unknown feature-source edits retain their own fingerprint. The
equivalence receipt itself participates in the training source fingerprint.

After syncing the updated source, run from the repository root on the lab host:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 921 --apply && \
CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/step5.py \
  --cuda-visible-devices 7 --device cuda \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-g760.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 921 --iterations 120
```

Recovery archives any unfinished old-source collection left by Ctrl+C; it does
not roll back the committed g921 checkpoint. Training then produces g922–g1041.
The existing g760 config follows the already committed automatic reward stage.
This source-only transition preserves weights, Adam/RNG state, utility
calibration, population, progress and controller state. Focused tests use the
exported g921 signature and an isolated real CPU checkpoint to check restoration,
retry before the first new commit, strict rejection and partial-collection
archival. No production training checkpoint is modified by these tests.

## Local verification

Two existing full heuristic games (691 decisions) were copied into an isolated
test directory, reconstructed exactly, and passed native transition audits.
Their dense prepared `.bin` contents have identical SHA-256 hashes between
legacy and compact variants.

| Measurement | Legacy | Compact |
| --- | ---: | ---: |
| Raw compressed bytes, both games | 5,770,980 | 450,633 |
| Storage reduction | | 92.19% |
| Raw read/expand, game 1 | 1.026 s | 0.418 s |
| Raw read/expand, game 2 | 0.614 s | 0.432 s |
| IL preparation, game 1 | 1.231 s | 1.090 s |
| IL preparation, game 2 | 1.362 s | 1.131 s |

Both formats use the current writer's gzip level 1. Earlier archival estimates
used older IL shards compressed at level 9; those sizes are not directly
comparable. Read timings are medians of five warm local-file reads; preparation
timings are medians of two reads with validated feature receipts. They are a
small Windows CPU smoke test, not production Linux storage or GPU throughput
measurements. The writer timings include validation/cache work and warmup effects
and are not used to claim a speedup. No new teacher searches or long training ran.

The reproducible smoke report is
`artifacts/compact-il-storage-smoke-20261105/report.json`. With the normal project
dependencies installed, rerun on one or two completed heuristic shards using a
new output directory:

```bash
PYTHONPATH=src python -B tests/Recovery/smoke_il_replay_storage.py \
  --sources <first-existing-shard.jsonl.gz> <second-existing-shard.jsonl.gz> \
  --output artifacts/compact-il-storage-smoke-new --native-audit
```

Focused regressions cover exact reconstruction, signed zero/type/shape changes,
non-prefix histories, observer isolation, malformed data, provenance, unchanged
features/targets, parallel preparation/cache hits, atomic failure, generation
defaults/retries, re-audit ingress and existing ordinary actor replay reuse.
