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
unchanged. This change does not add a self-play source-review override.

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
