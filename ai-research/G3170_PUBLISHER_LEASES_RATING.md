# g3170: publisher decode, pre-issued shared leases, PL rating every 20 iterations

This finishes the two parts of build step 9 that [STEP9_ORDINARY_AUTOPLAY.md](STEP9_ORDINARY_AUTOPLAY.md) left open, and moves the PL strength rating from every 10 to every 20 iterations. The run stays at 320 ordinary games.

The publisher and lease changes are code-only. They need no flag, apply from the first iteration that runs the code, and change no recipe, reward, search or PPO setting. Jobs, RNG draws, rows, features, seals and replay bytes are unchanged. The rating cadence is a recipe setting adopted with `--rating-every 20`. No Rust source changed, so the CPU workers are not rebuilt.

## g3141–g3170 on the lab (first 30 iterations with step 9 and `time,validation`)

Another project shared the lab machine during this run, so wall times are not comparable with earlier iterations. Per-game CPU and the structure of the critical path are comparable.

**Step 9 worked as designed.**

| | g3138 / g3139 (Python loop) | g3168 / g3169 (autoplay) |
| --- | ---: | ---: |
| Games played inside the worker | 0 / 0 | 319 / 320 |
| IPC calls per game | 507 | 21–23 |
| Lane-seconds per iteration | 1,318 / 1,731 | 232 / 239 |
| Worker seconds per iteration (`autoplay_worker_seconds`) | – | 179 / 192 |
| RNG-replay audits executed / failed | 6 / 0 | 7 / 0 and 6 / 0 |

The one game on the Python loop in g3168 had a `lab-legacy` search seat, which is the intended fallback.

**Collection is now bound by the teacher tables, not by ordinary games.** In g3169 the last teacher search ended at 76.9 s of a 78.1 s collection. The 28 ordinary lanes were busy for 239 of about 2,200 available lane-seconds. They waited because unfinished families reached the 128-family limit and the shared buffer peaked at 5.5 of its 6 GB. Publishers could not keep up, at about 2.6 s of CPU per game (`part_read` 1.28 s median, publication 0.68 s, replay preparation about 0.5 s). Collection averaged 72.9 s over g3141–g3155 and 81.8 s over g3156–g3170, against 85.0 s over g3101–g3140. The background load makes those differences unreliable.

**Shared-memory writes were mostly waiting.** A write took a median of 0.28–0.63 s per game. Copying and hashing took about 0.07 s of that. The rest was the parent's allocation reply, which ran to 9–17 s at the 90th percentile in g3168, plus a second CPU-ticket wait just for the copy.

**Value-phase revision (`time,validation`).** Baseline explained variance on new games, the well-powered critic metric, rose from 0.307 ± 0.003 (g3101–g3140) to 0.314 ± 0.004 (g3141–g3155) and then 0.320 ± 0.004 (g3156–g3167). Held-out explained variance moved from 0.300 to 0.316 and then 0.326. The first two iterations after adoption flagged `mixedEV below_tolerance` (0.251 against 0.302), then recovered and stayed above the reference. The early stop kept at least one block in 19 of 30 iterations.

**PL rating (128 games per slot, noisy).**

| Generation | 3110 | 3120 | 3130 | 3140 | 3150 | 3160 | 3170 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PL points vs g1290 | 191.5 | 128.5 | 152.2 | 124.9 | 183.9 | 203.0 | 159.7 |

g3160 is the best point estimate. The 95% interval for g3170 is 123–206. No regression alert fired.

**Log.**
- The first start at 18:34 failed with `NameError: BackgroundPurge`; that was fixed in commit 306423a2.
- After that, all 30 iterations ran without an error.
- The ONNX `failed to run "Concat"/"Reshape"/"ScatterND"` warnings were already in earlier logs (303 times in the g2880 export).
- Two teacher worker exits took 7.8 s and 9.1 s, within the 120 s grace period.

## What changed

### Publisher decode: unchanged subtrees are parsed once

The step-9 note attributed the publisher's cost to every row repeating the full history. Measured on a real 299-row game, the history is only about 5 MB of the game's 20 MB of row JSON. Each row's board is the bulk, `players` alone being about 35 KB per row, and most of it is unchanged from the previous row.

`lab_delta.SharedRows` parses a delta-wire row field by field. A history event, a top-level view field, a player or the `choices` list reuses the previous row's object when its JSON text is byte-identical to it. A JSON value's text delimits itself, so identical text followed by `,`, `}` or `]` is an identical value, type included. `1`, `1.0` and `true` are never confused, as Python's `==` would confuse them. Only changed text is parsed.

- **What a row contains.** Each row equals `json.loads` of its bytes: the same keys in the same order, the same values and the same types. `view` and `choices` are immutable (`FrozenDict`/`FrozenList`) and shared with earlier rows where unchanged. The rest of the row stays a plain dict. Anything unexpected parses that row the old way and starts sharing again.
- **Canonical receipts.** Each frozen view carries its exact canonical bytes and input digest (`canonical_inputs.CanonicalInput`). The two binds (decode and the structural audit) read the digest instead of encoding the view again. The replay encoding (`dumps_family`) splices the bytes. Results are byte-identical to `json.dumps(sort_keys=True)` and `contracts.digest`.
- **Audits still validate in full.** A receipt is not an actor `Admission`, so on audited games the input boundary still validates every view. A view edited after decoding has no receipt and is hashed by content, so a mutation still fails its seal.

### Pre-issued shared leases (`brass-ordinary-shared-part/2`)

- **The lease comes with the job.** The parent sizes it at 64 MiB at first and at 1.25 × the largest export so far after that. The lane writes its export inside the CPU ticket it played the game under, with no allocation round trip and no second ticket wait.
- **The lease is trimmed when the result arrives.** The parent shrinks it to the chunks actually written and unlinks the rest. The budget counts the full lease while the game plays, so memory stays bounded.
- **Larger exports.** An export that does not fit takes the original exact-allocation path, and the offer is released.
- **One hash pass.** Each chunk is hashed once on each side. The part's seal binds the ordered chunk sizes and hashes, where before the whole payload was hashed a second time.

The Rust worker still sends its pages to the lane over the pipe. Writing them from Rust into `/dev/shm` would save about 0.05 s per game more. The crate forbids `unsafe` (so no memory mapping), and that path could only be tested on Linux, so it is not done.

### PL rating every 20 iterations (`--rating-every 20`)

`rating_cadence` (contract `rating-cadence/1.0.0`) skips due slots after the generation it is adopted at. From g3170, ratings run at g3190, g3210 and so on, and g3180 is skipped.

- **What stays the same.** The rating block, its state, every committed slot and the gate's slope units (PL points per committed iteration) are unchanged. The reference setups still rotate one position per rated slot.
- **Recipe review.** Recipe recognition peels the key, and the key is left out of the collection signature, so the adoption preserves pending collection.
- **Cost and side effect.** The saving is about 85 s every 20 iterations. Generations that are no longer rated no longer leave rating snapshots, so restart weights exist every 20 iterations from now on.

## Evidence (Windows laptop, fixture weights)

**One captured production-like game (299 rows), publisher work per game.**

| Part | Before | After |
| --- | ---: | ---: |
| Read and decode (`part_read`) | 0.96 s | 0.27 s |
| Structural audit's re-bind | 0.33 s | 0.01 s |
| Replay encoding | 0.38 s | 0.03 s |
| Total | 1.6 s | 0.31 s |

On that game:
- Rows, digests and replay bytes were checked equal to the plain formulas.
- 4,134 subtrees were reused and 1,588 parsed, with no fallbacks.
- The full semantic audit passed, with all 299 views validated by the input boundary.

**Tests.**
- `tests/RustBackendLab/test_publisher_decode.py` (new) covers rows equal to `json.loads` (growing and rewritten histories, non-ASCII text, `1`/`1.0`/`true` flips), receipts, fallbacks and leases. A fixture family published through the plain wire, the delta wire and the delta wire with sharing switched off gives identical replay bytes.
- `tests/Recovery/test_rating_cadence.py` (new) covers due slots, slot ordinals and recipe recognition.
- `test_cpu_inference.RealLane` plays a third game with a pre-issued lease: no allocation round trip, settled, decoded.
- The 131 focused tests pass, as do `RealLane` and `NativeParity` against the local collector.
- The 10 known `test_b02`/`test_b06`/`test_b07` failures and the 6 `test_panel_gates` errors (a missing old results export) are unchanged. All of them also fail on the tree without these edits.

**Lab estimate.** Publisher CPU from about 2.0 to about 0.8 s per game, roughly 350–400 core-seconds less per iteration. At 320 games the collection wall time stays teacher-bound, so the saving shows as CPU and backlog, not as seconds of collection. It is what makes 480–640 games affordable. Read the actual numbers from the next report.

## What to check in the next report (`iteration-N/report.json`)

- **Decode.** In `execution[*].publication_phases`, `part_read_seconds` should drop from about 1.3 s to about 0.4 s per game, and `decode_sharing.fallbacks` should be 0.
- **Publication.** `publication_seconds` should drop, mostly `full_validation_and_assembly_seconds` and `json_encoding_seconds`.
- **Leases.** In `actor_part_transfer[0]`, `shared_preissued` should be true and `shared_allocate_wait_seconds` 0 for nearly every game. `native_backend.collection.shared_buffers.preissued_leases` should show almost all offers as `used` and few as `too_small`. Also watch `shared_buffers.peak_bytes` and `dispatch.shared_pressure_seconds`.
- **Rating.** The g3180 report has no `strength_rating` slot, and g3190 has one.

**Switches.** If publication fails with a seal or "row binding" error, rerun with `BRASS_SHARED_ROWS=0` in front of the command, which parses every row in full. If dispatch stalls on shared memory, use `BRASS_PREISSUED_LEASES=0`, which asks the parent for every lease as before. Each switch changes only that path.

## Code

| File | Change |
| --- | --- |
| `src/brass_ai/recovery/canonical_inputs.py` (new) | Exact canonical bytes and digests: receipts, frozen-event cache, `dumps_family` |
| `src/brass_ai/rust_backend/lab_delta.py` | `split_records`; `SharedRows` subtree-sharing parser |
| `src/brass_ai/rust_backend/lab_shared.py` | Decode via `SharedRows`; part format 2 (`reserve`/`settle`, one hash pass); switches |
| `src/brass_ai/rust_backend/lab_pool.py` | The lease is issued with the job and written in the game's ticket; `decode_sharing` in publication timings |
| `src/brass_ai/rust_backend/lab_admission.py` | The replay encoding uses `dumps_family` |
| `src/brass_ai/recovery/replay.py`, `inputs.py`, `rust_backend/lab_features.py` | Input digests via `canonical_inputs.input_digest` |
| `src/brass_ai/recovery/rating_cadence.py` (new), `ratings.py`, `rating_coverage.py`, `config.py` | Rating cadence |
| `src/brass_ai/rust_backend/lab_shadow_config.py` | Recipe recognition and transition for the cadence; left out of the collection signature |
| `scripts/continue-rust-g1491.py`, `continue-rust-shadow.py`, `continue-rust-g1190-restart-2.py` | `--rating-every 20` |
| `configs/learning/b07-replay-location-review.json` | Execution identity re-recorded; g3170 checkpoint identity added as a predecessor; review notes |
