# Teacher shadow verification revision

**Current adapter policy `/3`: missing root-value fallback.** The following pasted lab traceback
ended in `Native teacher closed output: teacher: invalid request: evaluation deadline`.
After a forced root or an expired first search evaluation, the Session can legitimately have no
root evidence or cached NN statistics. The value-record fallback incorrectly reused the expired
search deadline. It now uses the existing non-search NN delivery watchdog (10 seconds), restoring
the original search deadline on success or error. The prediction is solely for the root training
value: it adds no trajectories/policy labels, does not reselect the action or enter the Session cache.
Stale/cancelled/expired NN replies and wrong root identity remain errors; no substitute values are
invented. A successful fallback reuses its exact root encoding for the value row, avoiding a second
feature walk. Cached root statistics still avoid another NN request.

Native debug tests cover forced and multi-choice roots after search expiry, exact root feature/value
parity, restored deadlines, cache reuse and rejected wrong-identity/expired replies. Current evidence:
`distribution/rust-collector-performance/shadow-root-value-fix.json`. The same preparation command
detects and rebuilds `/2` into `/3`, preserving checkpoint, qualification and completed-table caches.

## Investigation: why a completed search call can lack root value

The action-selection engine and the teacher recorder have different requirements. The engine may
return a legal forced/fallback action without evaluating the root NN. The all-seat teacher still
requires a root utility record for that turn. Returning from `Session::decide` is not proof that
the root NN ran or that an acceptable reply was obtained.

| Path | Root NN/evidence | Teacher value source |
| --- | --- | --- |
| Valid root prediction/search tree | Root statistics/evidence available | Reuse that exact result |
| One legal action | Core intentionally skips root NN | Independent value-only prediction |
| Work/delivery cutoff before root request | No root NN result generated | Independent value-only prediction |
| First root inference expires/is discarded | No acceptable matching-root result | Independent value-only prediction |

Each new turn creates a new Session with an empty private NN cache. The wrapper also resets its
dedicated `root_stats` at the new root and sets it only after a matching NN reply passes validation.
This dedicated snapshot is separate from the Session's 128-entry LRU; LRU eviction does not clear it.
The present traceback proves that the recorder entered the missing-value fallback after search expiry,
but contains no evidence distinguishing a forced root from a root request that never completed.
GPU/CPU queue pressure is a plausible contributor for nonforced misses, not a measured conclusion.

For the next normal lab run, `root_value_sources` reports five integer counters separately for fast/full:
search evidence, cached root NN, forced root with NN skipped, root NN not issued, and root NN without
an accepted result. Counters are emitted once per completed table in `metrics`, then retained in
`iteration-G/teacher-shadow/collection.json` under `collection_metrics`. No extra NN calls, per-turn
log records, hashing or profiling clocks are introduced. Old recovered tables may lack these counters;
their coverage is unknown, not zero, and they are not recollected just to obtain diagnostics.

A high forced count is expected optimization behavior. High nonforced not-issued/no-result counts
would motivate profiling root preparation and inference queueing rather than enlarging the cache.

**Earlier follow-up: adapter policy `/2`.** The next run failed with
`Native teacher closed output: teacher: teacher post-search allowance exceeded`
(export `20261108T035830347906Z-20261108T035713286446Z-shadow-training-9c6cf95d-33e7c981`).
The first repair removed Python/ACK/sealing deadlines but missed Rust's pre-verification handoff
check. That check is now removed for both profiles. Handoff metadata may report zero remaining
reserve and native post time beyond the old reserve; timing types and actual search limits remain
checked. Preparation detects/rebuilds the old adapter, retaining its selector as history.

2026-11-07. The latest pulled result is
`results/rust-lab-qualification/20261108T032034123141Z-20261108T032009416212Z-shadow-training-a9162cac-7bb81128`.
It contained only the child's exit status. The supplied traceback identifies
`Post-search verification deadline exceeded` as the initiating failure; sibling cancellation followed.

## Current timing policy

User override: **both 0.8s and 4s searches have no independent verification hard deadline**.
Remove Python verification, Rust pre-handoff/late-ACK and deferred policy/value/cache sealing limits. Keep actual
search/K/quota/session settings and core selection behavior unchanged. The wrapper no longer aborts
when serialization/label handoff outlasts the old reserve. Active-search/stuck-worker
watchdogs, cancellation, IPC bounds, job/model/revision/actor bindings, target/feature/logit checks and
strict complete-table/checkpoint admission remain. Exclude Python verification wait from the active
search watchdog; it grants no new search time. The installed local game is unchanged.

Detailed Python timings are sampled at each profile's first turn per table and every 64th turn
(at most three entries per 64-turn table). Samples separate CPU preparation, shared-model queue,
model/handoff and receive-to-ready time. They are not a total. Native ACK timing is sampled at turn 1
and every 64th turn; untimed row counts preserve offline coverage/replay. Cheap coarse per-table/barrier
publication timing remains, without a latency pass/fail cap. Profiling consumes no RNG or extra forwards.

## Reduced work and preserved integrity

The old path copied all model weights from GPU to CPU for hashing, scanned the source tree and
constructed an encoder under the shared verifier lock on every labeled turn. It then deep-copied
and froze duplicate row trees. These costs serialized across up to 32 tables.

Full model/source identity is now established once per frozen collection. Each turn checks parameter/
buffer identity, storage/version/layout/config bindings without copying weights; the completed-table batch
hashes the model once, retaining mutation checks per handoff. Each lane reuses an encoder. Rows detach once into immutable containers shared
by admitted LabelRows. CPU row/target/schema/feature admission runs outside the shared model lock;
only RNG/mode-preserving independent logit parity is serialized, with cancellable lock acquisition.
Plain/cold rows retain full content admission. Policy targets, features, sampling, utility/reward,
optimizer/RNG, search_v1 and recovery semantics are preserved. `weights_only=True`/strict loading remain.

The exact source-equivalence review preserves the semantic teacher/search producer; it does not claim
that the adapter binary is unchanged. The new binary and original qualification are pinned separately.

The `/2` follow-up also establishes source identity once for the job batch, rather than rescanning
each job/calibration binding. Rust moves owned label/root-value trees into the outgoing message and
serializes into one frame buffer, avoiding a second full JSON payload copy. A golden framing check
asserts byte-for-byte compatibility. These are structural work reductions; no measured throughput
speedup is claimed without a lab run.

## Lab command

Synchronize this code and stop the old trainer. From the lab code repository root:

```bash
python scripts/prepare-rust-teacher-verification.py --expected-generation 1257 --offline && \
python scripts/prepare-rust-collector-performance.py --expected-generation 1257 --offline && \
python scripts/continue-rust-shadow.py --expected-generation 1257 --iterations 2 --share-target 0.25
```

The first command builds the teacher verification adapter against its original sealed search core,
runs Rust debug/wire regressions and pins the selector under `artifacts/rust-backend/teacher-verification/`.
A matching prepared adapter is verified/reused. Missing or altered runtime pins fail before collection.
The second prepares/reuses the ordinary adapter; the third is real g1257→g1259 shadow training.
Original qualification/checkpoint/backend state remain. No passed B01–B07 qualification is repeated.
If dependencies are uncached, omit `--offline`. If training has advanced, use the actual committed
starting generation in all three commands. Runtime options stay in the training JSON; no exports.

The launcher now tees child stdout/stderr to `training.log`, records its exact pin plus a bounded failure
tail, and exports the complete log with `qualification.json` to `results/`. Push that exported folder
from `results/`; it now contains the actual child traceback.

## Debug verification

Only affected local debug regressions run; no local training/benchmarks. Coverage includes both profiles'
slow verification, sparse timings and untimed row coverage, cancellation while waiting for the model,
stale replies, bad targets/logits, changed parameters, live/cold output parity, cache/checkpoint recovery,
adapter pin/reuse and failure-log export. Native tests cover late ACKs, wrong job/index/status and unchanged
absolute search/second-action cutoffs. Exact source/counts are recorded in
`distribution/rust-collector-performance/shadow-verification-fix.json`.
The `/2` handoff fix and additional work reductions are recorded separately in
`distribution/rust-collector-performance/shadow-handoff-fix.json`; the first repair's report remains historical.
Actual Linux/CUDA continuation/throughput remains pending on lab; debug tests establish no speedup.
