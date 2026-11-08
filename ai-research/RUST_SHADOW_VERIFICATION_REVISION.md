# Teacher shadow verification revision

2026-11-07. The latest pulled result is
`results/rust-lab-qualification/20261108T032034123141Z-20261108T032009416212Z-shadow-training-a9162cac-7bb81128`.
It contained only the child's exit status. The supplied traceback identifies
`Post-search verification deadline exceeded` as the initiating failure; sibling cancellation followed.

## Current timing policy

User override: **both 0.8s and 4s searches have no independent verification hard deadline**.
Remove Python verification, Rust late-ACK and deferred policy/value/cache sealing limits. Keep actual
search/K/quota/session settings and native selection cutoffs unchanged. Active-search/stuck-worker
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
buffer identity, storage/version/layout/config bindings without copying weights; completed-table handoff
still hashes the model. Each lane reuses an encoder. Rows detach once into immutable containers shared
by admitted LabelRows. CPU row/target/schema/feature admission runs outside the shared model lock;
only RNG/mode-preserving independent logit parity is serialized, with cancellable lock acquisition.
Plain/cold rows retain full content admission. Policy targets, features, sampling, utility/reward,
optimizer/RNG, search_v1 and recovery semantics are preserved. `weights_only=True`/strict loading remain.

The exact source-equivalence review preserves the semantic teacher/search producer; it does not claim
that the adapter binary is unchanged. The new binary and original qualification are pinned separately.

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
Actual Linux/CUDA continuation/throughput remains pending on lab; debug tests establish no speedup.
