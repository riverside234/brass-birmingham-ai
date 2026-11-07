# B07: cheaper native collection and bounded adoption

Implemented on 2026-11-07. **Lab performance/canary, production adoption, and
the next normal production commit/restart remain Pending.** Run the
[new g1250 bundle](../distribution/rust-lab-b07-g1250/README.md); do not rerun
passing B01–B06 qualifications. The installed game is unchanged.

## What the completed B06 record shows

Pulled from the results repository at `f06c0a1`:
`results/rust-lab-qualification/20261107T080012822445Z-20261107T075601452661Z-g1250-b06-c1cd94b2-2c29d5e3/evidence/qualification.json`.
B06 passed correctness and bounded measurements on g1250. Both ordinary arms
produced three identical parent families, 1,161 decisions and 675 forced rows.
No natural branch occurred in this small sample.

| Work, seconds | Reference | B06 native adapter |
| --- | ---: | ---: |
| Collection, including startup/publication | 25.485 | 36.226 |
| First admission/read | 2.507 | 18.457 |
| First preparation | 0.604 | 0.582 |
| Accepted work total | 28.596 | 55.265 |
| Warm replay read | 3.008 | 8.179 |

The native transition, projection and encoding components together consumed
less than one second per native family. Python lanes and repeated full
admission were the dominant avoidable costs. Native lanes previously shared
the learner's process/GIL; inference-forward totals were 19.677s versus 8.657s
on the same 446 positions. Contention is a hypothesis to test with the new
process layout, not a measured isolated GIL speedup. One historical table used
the qualified whole-table reference fallback; its cost remains in comparisons.

The eight matched 4s/K16 teacher turns passed with maximum post-work 0.297s.
Frozen-row batch verification took 0.514s versus 1.638s for streamed checks;
one-use reuse took another 0.331s. Streaming reduces the final pause but does
not establish lower total verification cost. B07 preserves the existing bounded
teacher verifier; it does not weaken checks to force a passing timing result.
B06 therefore passed measurement/correctness, **not useful performance adoption**.

## Collection and validation changes

`lab_pool.NativePool` retains the existing frozen planner, ready-branch scheduler,
CPU admission, model cohorts, inference service and reference fallback. Actors
now run in spawned processes with the configured lanes per process; a separate
writer owns publication. Processes match the reference pool's active width.
The saved 60-slot cap is shared by actors, writer and preparation readers.
Board histories pass as bounded file/hash handles, not learner-process IPC
payloads. Completed families continue streaming into existing preparation.
Branch checkpoints remain immediate, bounded messages; no sibling waits in a lane.

Publication calls the complete native family/admission validator once. It no
longer writes and validates an intermediate `FamilyStore` archive before
repeating the same admission. Input frames also reuse the reference actor's
immutable `BOUNDARY.admit` path for sampling/encoding. No policy, RNG, feature,
reward, auxiliary-target or Rust engine equation changes occur.

The independent writer adds a validation receipt bound to current implementation,
replay hash, full admission/job binding, family seal and feature-cache hashes.
Normal ingestion verifies those exact bytes and bindings, then reconstructs
checked games without replaying behavioral/RNG validation. All in-memory row
and metadata seals remain enforced. No extra duplicate board cache is written.
The loaded implementation fingerprint is computed once per process; deployments
start fresh workers/readers. Replay and feature contents are hashed on every
read, never accepted through an mtime/size shortcut.
Old-source receipts or missing derived features use full validation. Changed
bytes/receipts fail. A restart's on-disk `.record.json` is **not** independent
authority: `Publisher.load` performs full validation against the scheduler's
frozen job before returning it. Diagnostic data remains excluded from training.

Failures preserve the original error and abort shared CPU admission before
joining workers. Duplicate/stale branch messages also cancel before executor
shutdown. Partial/unsealed parts never train; a replay published immediately
before a receipt-write crash can be revalidated and recovered. Retention handles
the nested native replay directory and pins current v4 feature sidecars plus
backend rollback dependencies.

The local fixture comparison is reproducible with
`python scripts/rust_lab_b07_publication.py --out artifacts/b07-publication.json`.
It compares three alternating batches of three authored families, including
historical and V4 fallback, and checks equal prepared tensors/targets. Its
receipt in the bundle is a CPU component measurement, not a GPU throughput claim.
The first check exposed repeated source hashing; caching only the process's
loaded-source identity removed that overhead. The final small-fixture medians
were 0.778s baseline versus 0.821s for publication plus first read, and 0.274s
versus 0.231s for warm reads. Full validation passes fell from three to one per
family (warm reads: one to zero), with equal outputs. This tiny fixture shows
the reduced validation work and a faster warm read, but **does not establish
end-to-end gain**. The matched lab comparison remains the adoption gate.

## Bounded lab qualification

The bundle pins the original g1250 checkpoint
`de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`
and exact config `895d866502425dad30f2277b822dba2a15247f9a701d2dd5dd10b7e396de3ca8`.
Its source review records every changed learning dependency relative to B06.
Existing restricted config/source transitions are retained, not broadened.
The original successful ordinary/teacher native binaries are reused by hash.

1. Short CPU regressions cover admission equality, branch targets, cache/source
   invalidation, worker/writer/reader errors, stale/duplicate/canceled teacher
   replies, full-state adoption failure/recovery and rollback integrity.
2. One matched ordinary batch per backend: at most three frozen parents plus
   natural branches, current execution configuration and neutral recipe, no
   forced fork. Measure startup, publication, reads, preparation, GPU queues,
   process CPU/memory and accepted work separately. Watchdog: 360s.
3. Two independent shadow passes, at most two all-seat tables per pass (train
   and held-out), frozen g1250 and w=0. Full turns use search_v1, 4s/K16, two
   independent one-thread sessions; the existing all-seat fast turns remain
   0.8s/K4. Original 1s/0.2s post caps include verification and sealing. The
   slower table's policy-sealing cost is conservatively charged to each final
   turn before its value sealing. Reload both buffers and a shadow checkpoint.
   No teacher, search-head, PPO or PPG update in these passes. Watchdog: 210s each.
4. Full diagnostic checkpoint/Adam/RNG/dependency restore and the existing short
   authored-input current-recipe PPO/PPG component commit/reload. Zero new games
   here; this is not a production iteration. Watchdog: 180s.

Each attempt exports partial failures or success under
`results/rust-lab-qualification/`. The two shadow passes have fresh verifiers,
processes and epochs; frozen ordinary data is reused. The exact g1250 null
utility receipt is allowed only by the user's existing shadow authorization.
It stays null in production; other missing readiness defers the teacher scope.

`passed` records bounded correctness. `adoption_ready` additionally requires
equal accepted work and measured improvement versus the current reference arm.
No whole-iteration speedup, saturated-worker scaling, strength or full 28+4
teacher schedule is inferred. Unobserved natural branches stay a performance
coverage gap; earlier native branch correctness remains valid.

## Explicit backend adoption and rollback

`manage-rust-lab-b07.py adopt` operates under the existing run lock. It requires
the passing current-source B07 receipt, useful matched gain, and exact currently
committed generation/checkpoint/config. It copies the complete rollback state
and dependencies and restores PPO, PPG, companion Adams and RNG before touching
production state. The existing reviewed source-transition mechanism records the
source change. A separate operational `native_backend` selection routes ordinary
training only; champion/evaluation jobs remain on their existing reference pool.
Teacher mode must remain off. Defaults stay reference until this command runs.

The next normally authorized iteration writes backend identity into its full
checkpoint and report. `verify-next` checks actual native-family use, exactly
G+1, the state/checkpoint binding, and full isolated restart. This command does
not launch another training campaign. Interrupted collection can reuse fully
revalidated families; a cached family still counts as native work.

`reference` switches at the latest clean committed boundary and retains every
ordinary checkpoint, counter and compatible cache. It records the switch so a
pre-next-checkpoint mismatch cannot silently select a backend. A missing native
binary cannot block this rollback. `verify-rollback` separately restores the
saved full pre-adoption point and its exact source/config/dependencies without
rewinding current training. That byte-preserved point and source-transition
backups remain available for disaster recovery; it is not a weights-only reset.
Backend-only selection is stored in state and subsequent checkpoints, never
smuggled into a learning-config exception. Later teacher enablement remains a
separate explicit readiness/beta/PPO/PPG decision. DOMiNO/AZ_DB remain off.
