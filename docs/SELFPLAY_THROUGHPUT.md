# Collection, rating and cleanup throughput revision

For the current g1041 run, use the [combined g1041 continuation](TRAINING_G1041.md).
It includes this update, sparse PPO probes, 70/30 tables and early PPG beta
feedback in one reviewed config. The g921 example below is historical.

This execution-only revision preserves game seeds, seat/model identities,
sampling/RNG semantics, complete legal-action checks, features, PPO/PPG settings,
rating history, priors, seat correction and paired promotion criteria. It does
not modify existing run/checkpoint files during configuration preparation.

- **Rating validation:** complete replay/feature/hash/seat/setup validation now
  streams through up to `audit_workers` readers while games finish. Readers use
  the actor service's shared CPU admission budget. They return compact outcomes;
  publication remains in job order. Context evaluation retains its existing
  complete-game callback path rather than losing context evidence.
- **Rating timings:** each completed slot writes `timing-report.json`, separately
  from deterministic PL evidence. It reports registration, games, reader join,
  worker totals, history reading, full fit, bootstrap, interval calculation,
  report writing, plots and state commit. Worker totals overlap; they are not
  additive wall time. The existing report exporter includes this sidecar.
- **Collection inputs:** non-forced ordinary neural decisions get an immutable
  snapshot after full admission. Boundary/encoder lookups reuse its input
  digest; request submit/response checks reuse the exact original canonical
  request digest only while model/context and immutable input identities match.
  Mutable and search inputs retain rehashing. IPC strips input admission receipts
  and readers revalidate. Forced decisions avoid unnecessary snapshot copying.
- **Bootstrap:** 16 CPU processes, one BLAS/OpenMP thread each, still 200 refits,
  full-fit warm starts, main-process draws and deterministic ordered results.
  CPU affinity can lower the effective worker count. Validation requires enough
  configured audit and CPU slots; increasing workers does not relax convergence.
- **Cleanup:** unchanged-path digest reuse is local to one prune operation on
  POSIX, bound to device/inode/size/mtime/ctime/mode/link count. Changed files
  rehash; final quarantine deletion always rehashes, including after restart.
  Windows always rehashes because its creation-time-based ctime is insufficient
  to detect some rapid same-size writes. Empty-parent cleanup is deduplicated per
  journal batch. Pins, durable intents, restore and the five-iteration cadence
  remain intact. Reports expose reused bytes and final-delete verification.

The first input-only experiment showed that freezing alone was insufficient;
the final implementation also reuses immutable request digests. Benchmarks are
CPU microbenchmarks, not claims about collection wall time on the lab GPU.

## Continuation

Use the actual committed generation. The example below assumes it is still 921.
The helper derives settings from committed state, preserving automatic reward
progress, and also adopts the previously requested four-step PPO probes with
logging-only full scans disabled. It writes a new config; it does not reset
weights, optimizers, RNG, warmup, PPG buffers or counters. Interrupted old-source
collections must be archived by the existing recovery command first.

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 921 --apply && \
python scripts/configure-selfplay-throughput.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 921 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-throughput-g921.json && \
CUDA_VISIBLE_DEVICES=7 python scripts/step5.py \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-throughput-g921.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 921 --iterations 20
```

Do not regenerate the fresh-run recipe for continuation. An existing explicit
config with four bootstrap workers retains four until explicitly migrated.

## Validation

Focused regressions cover immutable input identity and tensors, mutable-input
rejection, IPC re-admission, complete replay receipt/seat checks and ordering,
restart-safe rating evidence, hash mutation, interrupted retention and restore,
serial/parallel PL samples and paired confidence intervals, narrow configuration
migration and checkpoint/Adam/RNG/calibration continuity.

The bounded smoke script is `tests/Recovery/smoke_collection_throughput.py`.
It uses authored tensors/replays and synthetic rating outcomes; no live games,
long training or evaluation campaign. Process startup is included in the small
reader and PL timings, so more workers need not be faster for these fixtures.

Validated on 2026-11-06:

- The final focused run passed **31 tests** covering actor input/behavior/RNG
  equivalence, deadline handling, replay ordering, retention hashes, config
  migration, and checkpoint/Adam/RNG/calibration continuity. The new continuation
  tests live in `test_collection_throughput_resume.py`; the pre-existing
  `test_throughput_resume.py` is preserved separately.
- The existing PL, rating restart/retention and interrupted-cleanup regressions
  also passed, including cold/warm/serial/parallel agreement and paired
  confidence-interval promotion decisions. Six CPU preparation regressions
  passed after updating their authored fixture's static board topology.
- Final immutable-request input smoke: 40 authored decisions per repetition,
  three paired repetitions, median **0.5304 s -> 0.3766 s** (about **29% less
  CPU input-processing time**). Request identities and encoded tensors matched
  exactly. The behavior regression also matched sampled actions, recorded
  probabilities, opening counters and RNG state. Results are in
  `artifacts/collection-input-digest-smoke-20261106.json`.
- Parallel replay smoke: eight tiny authored shards produced identical outcomes
  in identical job order. Serial took **1.08 s**, four processes including
  startup took **10.05 s**. This verifies correctness, not a speedup; production
  replay readers overlap game execution, and the new reader-join timing measures
  the remaining tail.
- Bootstrap smoke: 200 refits, 128 synthetic games, four agents. Four workers
  took **56.52 s**; requesting 16 (limited to **14** available CPUs locally)
  took **146.93 s**. Samples, ratings, paired intervals and promotion decisions
  matched exactly (maximum sample difference **0**). More workers were slower
  on this small Windows fixture. The lab's much larger historical fit needs the
  new bootstrap timing before claiming a throughput gain.
- Cleanup smoke preserved the final deletion hash check. Local Windows uses
  the conservative no-reuse path; POSIX metadata reuse and invalidation were
  covered by focused tests. No Linux filesystem throughput claim is made.

The replay/bootstrap/cleanup smoke report is
`artifacts/collection-throughput-20261106/report.json`. Its earlier input-only
measurements predate request-digest memoization; use the separate final input
report above for that component. No production run, checkpoint or evaluation
evidence was changed, and no training or live evaluation campaign was launched.
