# g6120: search step, PPO, PPG buffer and data hashing speed

Code-only. No training change: every search, PPO update and cache file is the same, checked against the
replaced code on fixtures and real CPU-teacher tables. Lab numbers below are from the g6120 export
(iteration 6118; g6119 ran beside other lab jobs, so it is used for proportions only).

## Request (2026-11-26)

1. Finish the open parts of g5781: the search step still copied the whole game state and rebuilt its indexes
   (about 26 µs per simulation), and the value phase kept its fixed costs (the buffer write, about 4.2 s, and the
   ownership-stop setup).
2. Investigate the PPO phase and improve it in similar ways.
3. Consider cutting the auxiliary heads from collection and search inference.
4. During the work: remove the critic writer's block copies (zero-copy), then drop SHA-256 for the PPG cache,
   and then for all PPO/PPG data and buffers ("just trust it").

## 1. Search step (Rust core)

`Brass::step` called `after_legal_choice`: clone the whole `NativeState` (history included), apply the move,
rebuild every index into new allocations, then drop the old state. Measured with the stub-network harness
(`tmp/step-perf`, seeds 3 and 4, two interleaved rounds; reference digests `ae912578…`, `448eb8bf…` unchanged):

| µs per simulation | before | after |
| --- | --- | --- |
| whole simulation | 111.6-127.9 | 95.4-109.2 |
| search step | 25.2-28.6 | 6.4-7.2 |
| of which: state copy | 9.1-10.2 | 0 |
| of which: old state dropped | 3.5-4.0 | 0 |
| of which: index rebuild | 3.9-4.4 | about 1 (inside the move) |
| of which: recall refresh | 2.4-2.8 | 1.9-2.1 |

Changes (all in `lab_cpu_overlay.TEACHER_CORE_FILES`, so the next continuation rebuilds the CPU teacher):

- `state/phases.rs`: `apply_legal_in_place` replaces `after_legal_choice` (its only caller was the search step).
  The checks and the transition are shared with the transactional path (`admit_transition`,
  `commit_transition`), which keeps copy-then-commit for every other caller.
- `state/chance.rs`: `resolve_chance_in_place`, sharing `chance_pick`/`commit_chance` with `after_chance`.
- `search/brass.rs`: the step moves the world's own state to its successor. A failed step leaves the world
  partly updated; every caller (traversal, projection, plan replay) returns the error and drops the world.
- `state/indexes.rs`, `state/mod.rs`: `rebuild_indexes` refills the indexes in their storage (a reset
  union-find and cleared bit sets are exactly `empty()`).
- `state/observation.rs`: the recall refresh sorts each hand into its existing buffer.

Real CPU teacher, built as on the lab from HEAD's and the new core files, on the same lab-budget tables
(`tmp/step-perf/compare_teacher.py`, head and new interleaved twice; outputs identical):

| | before | after |
| --- | --- | --- |
| standard table: non-network µs per simulation | 186.8-197.9 | 162.7-165.4 |
| standard table: seconds | 24.7-26.2 | 24.2-24.3 |
| cotton table: non-network µs per simulation | 186.8-191.4 | 156.4-157.2 |
| cotton table: seconds | 20.1-20.2 | 19.1-19.2 |

The network forward is about 72% of a table's search time, so a 13-18% cut in the rest is 2-7% of a table.
At g6118 the teacher tables finished at 44 s of a 79 s collection that publication bounded, so this frees CPU
for the publishers rather than shortening the collection directly.

## 2. Value phase fixed costs

### Buffer write (`Buffer.stage`, 4.0 s at g6118)

At g6118: serialization 2.30 s, flush 0.36 s, SHA-256 of the 1.8 GB critic file 0.83 s, sidecars about 0.5 s.

- Contexts: each row's context was keyed by a SHA-256 of its canonical JSON and deep-copied when new.
  `ppg_cache_writer.context_key` uses the exact typed sorted items for the flat scalar contexts of a view
  (the same equality as the canonical JSON; anything else keys by the JSON itself), and a new dict as the copy.
- Zero-copy writer (`write_critic_file`): a record is its state, its first action, then 683 bytes of small
  fields, with no padding. Each block goes to the file in one gather write (`os.writev`) of every row's own
  state and action arrays plus a small buffer of the other fields. There is no temporary array, no strided
  copy and no page fault per page of the mapping; the kernel's copy into the page cache is the only copy. The
  finite check stays on the destination values: the small fields in their buffer, and the state and action
  are the source bytes themselves, which a sealed row had checked when it was sealed. A block with any other
  array layout or type takes the original structured fill. The file is synced before its receipt is written.
  Where the system has no `writev` (Windows), the block's bytes are joined once.
- No content hash: the PPG cache files are the trainer's own, synced before the receipt, and read back within
  the window. Receipts now record sizes; every reopening checks the sizes and the JSON receipt, and the
  sidecars validate their own contents. A SHA-256 in an older receipt is no longer recomputed.

On the laptop (`tmp/g6120-perf/bench_writer.py`, 32,768 sealed lab-shaped rows, interleaved, identical file
bytes): 1.64-1.75 s (old: copy into the mapping, flush, SHA-256) against 1.08-1.13 s (new, Windows' joined-write
fallback, synced), about a third less. Linux takes the gather write, which also skips the join copy. For the
lab's 124k rows this is an estimated 4.0 s → under 2 s.

### Ownership-stop setup (3.7 s at g6118; not changed)

Split from the export: search rows built from the selected tables 1.13 s, their era-head λ targets 1.86 s,
the stop's own precompute and first measurements about 0.7 s. The λ targets are a Python pass over 35,877 rows
(`search_rows.search_corrections`); vectorizing it would save an estimated 0.7-1 s. Caching the search rows'
device states across iterations (most tables return in the next pool) would save an estimated 0.5-0.8 s.
Neither is done.

## 3. PPO phase

At g6118 PPO took 35.5 s: 46 steps (16.3 s, the first 4.0 s), behavior check 3.8 s, freeze values 3.4 s,
critic prepare 3.3 s, prepare 3.1 s, preservation report 1.6 s. A step took 0.26 s for 4 actor and 4 critic
512-row microbatches, a few ms of GPU work: host-bound, as the value phase was. Changes:

- Teacher staging (most of the first step's 4.0 s: `staged_rows.seconds` 3.74 s): every iteration hashed and
  re-checked all 8,096 staged label rows, which stay in the buffer for eight iterations. `teacher_buffer.admitted`
  seals each row once when it enters the buffer (the same layout, finite and selector checks), and
  `StagedRows` skips the audit for sealed rows. Laptop, 8,096 lab-sized rows: audit 1.87 s → 0.03 s.
- Device rows (`updates.DeviceRows`): every PPO pass collated its batches on the CPU workers and the main
  thread pinned and uploaded each one (28 GB per iteration). A plain `Dataset` on CUDA now keeps its collated
  inputs on the device (about 6.5 GB for the actor rows and 1.7 GB for the critic rows at g6118) and gathers
  each batch there, equal to the collated batch; over 12 GiB, with mixed array types, or out of device memory,
  the dataset keeps the original path.
- Frozen values (`value_preservation.FrozenValues`, 1.6 GB): kept on the device. Freezing no longer copies each
  output back per batch (one finite read-back at the end), and each critic batch gathers its rows there instead
  of a device read-back, a host gather and an upload of about 26 MB per step. PPG receives host copies.
- Preservation outputs read the descriptors with the batch's own device index (no read-back and re-upload).
- The entropy term's per-row group lookup is done once per dataset.

The fixture PPO update with these changes (device rows forced on the CPU) is bit-identical to the replaced
code: weights, Adam state and every report field except two counters of the replaced path.

## 4. Cutting auxiliary heads from inference

- The era heads and the soft-policy head are separate companion modules: neither is in the inference graph.
- The search head is already left out of the CPU bundle wherever no table blends it (g4551 head-less bundles).
- What remains is the main model's value outputs (outcomes, descriptor encoder, utility head): 222k of 6.05M
  parameters (3.7% of the 24.2 MB of weights) and about 3% of the multiply-adds per position. Ordinary autoplay
  reads only the logits; the teacher's search needs the values at its leaves.
- At g6118 the collection was bound by publication (`peak_unfinished_families` at its limit of 128; the teacher
  tables finished at 44 s of 79 s). A policy-only bundle would save about 3% of ordinary-game inference CPU and
  no collection time, and cost a second ONNX export and parity check per model per iteration. Not done.

## 5. SHA-256 dropped from PPO/PPG data and buffers

| Where | Before | Now |
| --- | --- | --- |
| PPG cache files | SHA-256 at stage (0.83 s) and at every restart | sizes |
| PPG sidecars at each reopening | SHA-256 | sizes (contents validated) |
| Teacher shards | every shard of the window re-hashed each iteration | not hashed (gzip CRC and row checks on a cold read) |
| Search-value shards | re-read and re-hashed on every read | cached rows; cold reads parse without a hash |
| Lane → publisher shared parts | hashed by the lane, re-hashed by the publisher | sizes (format 3) |
| Publisher decode | every row re-hashed against the lane's seals | row counts and page bindings |
| Transient file parts | SHA-256 of the part and its record | sizes |
| PPO input audit of sealed rows | SHA-256 of the actions | the sealed object |
| Local-scoring envelopes | seal recomputed at stage and at every reopening | checked once, when the labels are bound |

Kept: a shard's one SHA-256 at publication (its identity key in receipts and caches), the durable replay's
family and replay hashes (crash-recovery admission binds to them; about 20 ms per game in publishers), the
lane's record seals in sealed `lab-native`, and the per-row input digests that key the feature rows.

## Checks

- Core suite (81 tests), including the new `in_place_transitions_equal_the_transactional_ones`: every state of
  four played games, every legal choice of every third state, in-place against transactional exports and
  indexes, refilled indexes against a fresh build.
- Harness digests unchanged, and real CPU-teacher tables identical (both above). The new teacher passes its 44
  `cpu_` tests and the cotton, decision-cache and in-place core tests.
- Native workspace type-check with all targets; the full worker library (25 tests).
- `tests/Recovery/test_ppo_speed_g6120.py` (9 tests): device-row batches equal collated ones; the whole PPO
  update equals the replaced path; entropy weights; teacher staging; context keys; partial gather writes; the
  file writer's bytes equal the array writer's (sealed, plain and fallback rows); the staged cache's files and
  receipt equal the pre-g6120 writer's and reopen by size; value rows read without hashing.
- `test_publisher_decode`, `test_value_work_optimization` and the updated part and buffer tests pass (one listed
  known failure in `test_value_work_optimization`).
- Not finished: the last run of the focused suites (23 modules) was stopped on request before its summary, with
  3 errors and 5 failures counted by then and not yet identified. The known HEAD failures in that set account for
  four (`test_gae_value_phase` three, `test_value_work_optimization` one); the run before the identity was
  re-recorded also failed every signature-transition check, as it must until then.

## Cost

No extra CPU, games or held-out data. GPU memory: about 8 GB of device rows and 1.6 GB of frozen values during
PPO (16.9 GB reserved at g6118); host memory falls by the same 1.6 GB. Integrity: a same-size corruption of a
PPG cache file between its write and a restart would no longer be detected (truncation, missing files and
mismatched receipts still are); teacher and search shards rely on their gzip CRC and row checks, and the
lane-to-publisher parts on the parent owning their memory and files.

## Execution identity

`0fb8d4ce3be2c9c781f1ef22fb7c5d79930dc82c610382be0d19a5d6ef17a05f`, with predecessor `cb7721f7…` (the g6120
checkpoint's). The entry is `g6120_step_and_ppo_speed` in `configs/learning/b07-replay-location-review.json`.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 6120 --iterations 10 --share-target 0.065
```

## What remains (not done)

- Value phase: the ownership-stop setup (section 2) and the per-step host time (`step_profile.value_rows`,
  183 ms per step at g6118).
- Publication: JSON decoding of each game's part (515 ms per game), the feature-cache compression (128 ms) and
  the replay gzip (97 ms) are larger than any hash left.
- Engine: decision materialization (about 39 µs per simulation in the harness).
