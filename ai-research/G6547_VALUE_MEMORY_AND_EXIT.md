# g6547: value-phase steps, GPU memory, trainer exit and publication decode

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, newest export g6547
(`joint7-v11-2-g4970-rail025-soft8-beta0.02-g6547-20261127T203905330022Z`, iterations 6543-6546, one trainer process
3280409). Code-only: no recipe or config change. The value phase's float rounding changes (joined batches); every other
change gives the same numbers, bytes and digests.

## Request (2026-11-27)

1. Finish what g6120 left open: the ownership-stop setup, the value phase's per-step CPU time, and publication's JSON
   decoding and compression, the costliest or most promising first.
2. Mid-task: PPO peaking at 39.5 GB allocated and 61 GB reserved on the GPU is too much; and a stopped trainer kept its
   GPU memory until `kill -KILL`. Fix both.

## Where an iteration's 145 s go (g6415-g6546, retention summaries)

| part | seconds |
| --- | --- |
| collection (teacher tables and ordinary games together) | 76 |
| PPO | 21 (35.5 at g6118, before g6120) |
| value phase | 13-20 |
| full GC pass after the commit | 15 |
| replay load, checkpoint, evaluation, commit, cleanup (every 5th) | about 10-17 |

## 1. Value phase: one device batch per step

### Where the time went (g6543-g6546)

A step is 4,096 value rows in 8 microbatches of 512. Host time per joint step: value rows 142-180 ms, search rows
19-35 ms, clone rows 17-19 ms, clip and device wait 0.7-1.1 ms (the GPU is idle most of the step). The era-head
companion's own timers put its forward at 7.1 ms per joint microbatch and 6.7 ms per head-only microbatch; a head-only
step costs about 131 ms (8 microbatches), and there were 0-38 of them per iteration (`ownership_stop.seconds.head_only`
2.1-5.1 s). The CPU workers are not the limit (`cpu_preparation.wait_seconds` 0.3-0.5 s per phase). On the fixture a
joint step dispatches about 1,870 ops for its value rows, 1,530 for the search rows and 770 for the clone rows, so the
step is set by per-op host cost (Python and kernel launches) multiplied by 8 microbatches, not by the rows.

### Change

- `value_phase.joined_value_batches`: for state-only models, consecutive parts are joined on the device into batches of
  `JOINED_ROWS` (4096) rows: one forward and one backward per step instead of eight. The CPU workers still prepare
  512-row parts in parallel; the critic replay's prepared columns (targets, utility, remaining, descriptors, seats)
  are joined too and attached for the joined batch, so every column lookup still hits them. Used by the joint steps,
  the head-only steps and every held-out scan (`metrics`).
- The ownership stop measures 4096-row parts (`joined_config`), and its search validation rows likewise.
- The search rows run as one chunk per step (`search_auxiliary.chunk_rows`, a step's share is about 668 rows), and their
  lambda-target forward in 4096-row chunks.

The same rows in the same order, the same loss sums and normalization. Float rounding differs (different matmul
shapes and gradient accumulation), so it is not bit-identical: on the g4770 fixture phase with two-row microbatches
(several parts per step) every reported number is within 1e-4 relative of the unjoined phase (largest: 0.2% on a policy
KL of 3e-6, 5e-9 absolute). The era heads' target-cache hit counter counts the joined batches (diagnostic only).
`JOINED_ROWS = 0` restores the old batching.

### Expected (an estimate; the laptop has no CUDA GPU)

Assuming the per-microbatch host cost (about 17 ms joint, 13 ms head-only) is paid once per step instead of eight
times: about 0.12 s per joint step and 0.09 s per head-only step, about 5 s per iteration at g6546 (16 joint, 32
head-only steps), plus about 0.1-0.2 s per held-out scan (61 to 8 forwards). Each step now waits for its 8 worker parts
before its one forward (about 16 ms of preparation, parallel on the 4 workers). `value_phase.step_profile` in the next
export measures it.

Cost: a 4096-row training batch holds roughly 1-3 GB of activations on the GPU during the value phase (the trunk saves
about 250 KB per row for backward in fp32, less under bf16). The PPO device rows (8.2 GB) are now freed before the
value phase (section 3), so the phase's peak is lower than before.

## 2. Ownership-stop setup (1.8 s)

At g6546: search-row loading 0.33 s, their lambda targets 0.9 s, the stop's target precompute 0.23 s, its first
measures about 0.3 s. `search_rows.search_corrections` was the largest part: about 24 µs of NumPy calls per row.

Change: corrections per table as arrays (a table's 64 roots and about 167 rows stay in cache): seat frames are channel
gathers per seat rotation, a chain step is one array operation per root, the rows are written per rotation into a
table block, and the block is copied once. Each element sees the same operations in the same dtypes (float32 frames,
float64 chains, cast on write), so the result is bitwise the same; `_search_corrections_rows` keeps the per-row form
as the reference. At the lab's size (216 tables, 36,072 rows, Rail-end on) on the laptop: 0.86-0.92 s → 0.62-0.69 s.
A whole-array version over all tables was slower (1.4-1.8 s): its temporaries (hundreds of MB) were memory-bound.

Not changed: the search-row loading (8 µs per row) and the stop's target precompute (per-row metadata) are small.

## 3. GPU memory

### What the reports show

| export | allocated after PPO | peak allocated | reserved | device rows |
| --- | --- | --- | --- | --- |
| g6118 (before g6120) | 1.7 GB | 5.2 GB | 16.9 GB | none |
| g6228-g6229 | 11.6 GB | 39.2 GB | 20.3 GB | 6.5 + 1.7 GB |
| g6543-g6546 | 11.5 GB | 39.5 GB | 61.0 GB | 6.5 + 1.7 GB |

The peak was the process's peak since it started (never reset), so no report said which phase set it. The 8.2 GB of
device rows and 1.6 GB of frozen values stayed allocated after PPO until the next iteration. `DeviceRows` was used by
every plain CUDA `Dataset`, each with its own 12 GiB limit: besides PPO's actor and critic datasets, the teacher-drift
probe, the post-teaching drift check over all learner rows and the teacher utility recalibration would each stage a
store, possibly while PPO's was still alive. The reserved 61 GB is the caching allocator keeping freed blocks.

### What the device rows buy

PPO sub-timings, g6118 (CPU-worker collation) against g6546 (device rows):

| part | g6118 | g6546 |
| --- | --- | --- |
| forward_backward (46 calls) | 16.28 s (3.74 s of it teacher staging) | 6.47 s (0.52 s) |
| policy_measure | 3.79 s | 2.36 s |
| freeze_values | 3.40 s | 1.96 s |
| preservation_final | 1.60 s | 0.48 s |
| probe_prepare | 0.79 s | 0.21 s |
| PPO total | 35.5 s | 20.1 s |

About 10 s per iteration comes from the device rows (the rest from teacher staging, g6120). g6228-g6229 had PPO at
43.5 s with device rows (policy_measure 15.6 s), so other GPU work on the lab at that time is likely; compare within one
quiet process. They stay, but bounded.

### Change

- Device rows only for PPO's actor and critic datasets (`Dataset.device_rows_allowed`, set by `_ppo_nonforced`); every
  other dataset collates as before g6120.
- One 12 GiB budget for all live stores together (`DeviceRows.live`), not 12 GiB per store.
- Freed when PPO returns, also on an error (`release_device_rows` in `ppo_nonforced`); a released dataset takes the
  original path if read again. The frozen values were already freed with PPO's locals.
- `gpu_memory.Phases`: at four boundaries per iteration (before PPO, after PPO, after the value phase, before the report)
  each phase's peak allocated and reserved bytes are recorded (`report['gpu_memory']`), the allocator's free blocks go
  back to the driver (`empty_cache`, so another process on the GPU can use them), and the peak counters restart. The
  existing `memory` fields' peaks are then peaks since the previous boundary.

Expected: about 8-10 GB less held between PPO and the next iteration, reserved memory returned after every phase, and
no stores outside PPO. The 39.5 GB peak itself is not explained by the stored rows (8.2 GB); the next export's
`gpu_memory.phases` names the phase that sets it.

## 4. A stopped trainer kept its GPU

The launch chain is `continue-rust-g4970.py` → `subprocess.call` → `continue-rust-g1190-restart-2.py` (runs
`continue-rust-g1491.py` → `continue-rust-shadow.run_training`) → `Popen` → `scripts/step5.py train` (the GPU process).
On Ctrl-C every process gets SIGINT. `run_training` sends the trainer SIGINT, waits 15 s and then kills its whole tree;
but `subprocess.call` in the top script SIGKILLs its own child a quarter second after Ctrl-C, so `run_training` died
first and the trainer was orphaned. A trainer whose exit then waited on a blocked worker join (non-daemon pool threads
are joined at interpreter exit) never ended and kept its CUDA context until `kill -KILL`. A plain `kill` (SIGTERM) of
a wrapper killed only that wrapper and left the trainer training.

Change:
- `scripts/step5.py` runs under `brass_ai.step5.exit_guard.guarded`: SIGTERM and SIGHUP stop it like Ctrl-C (a signal
  already set to ignored, e.g. by nohup, stays ignored). From the first stop signal, and again when the command returns
  or raises, a watchdog allows 60 s; after that it writes the thread list and every thread's stack to stderr (the
  training log), kills the process's own children and ends the process with `os._exit` (the driver frees the GPU). A
  second stop signal does that at once.
- `continue-rust-g4970.py` uses `exit_guard.call`: on the first Ctrl-C it waits for its child (which stops the trainer
  within run_training's 15 s); a SIGTERM to it alone reaches the child as Ctrl-C; a second stop kills the child's tree.
- `run_training`: SIGTERM stops the trainer like Ctrl-C, and further Ctrl-C/SIGTERM are ignored during its 15 s stop so
  they cannot cut the cleanup short; children started during the stop are killed too.

Tests: `tests/Step5/test_exit_guard_g6547.py` (a finished and a failed command with a never-ending thread exit within
the grace with their own codes, the stacks naming the blocked thread). In WSL (Linux signals, `tmp/g6547-perf/wsl_exit.py`):
SIGTERM with a blocked exit ends in 2 s (code 143) with the stacks; a second SIGTERM during a stuck cleanup ends it at
once; SIGHUP ends it (129); the launcher waits for its child's cleanup after a group Ctrl-C and after a SIGTERM to the
launcher alone; the old `subprocess.call` killed the child before its cleanup.

## 5. Publication decode and compression

Per game at g6546 (320 games, 24 publishers): part read (decode) 0.49 s, feature cache 0.13 s, replay write 0.10 s,
validation 0.06 s, JSON encoding 0.027 s, family seal 0.017 s; in the publishers' ingress preparation another 0.22 s.

The decode is the publisher's shared-subtree JSON parse (`lab_delta.SharedRows`): on a smoke game's 930 real rows
rebuilt into delta-wire pages (`tmp/g6547-perf/bench_decode.py`) it costs 681 µs per row against 1,201 µs for plain
`json.loads`. About half was freezing parsed subtrees into immutable FrozenDict/FrozenList trees in Python
(`admitted_inputs._freeze`, generator per item), and about 40% the canonical encoding of changed fields for the per-row
input digest (`canonical_inputs.dumps`, which built a new JSON encoder on each of about 70,000 calls).

Change: `_freeze` copies each dict in C and replaces only its container values in place (same keys, order, values and
types); `dumps` reuses one C encoder built with json.dumps's own options, checked against the plain formula at import
(plain formula without the C accelerator). Decode 681 → 531 µs per row (−22%). On the real rows: rows equal
`json.loads` with frozen types, receipt digests equal `contracts.digest`, canonical and family bytes equal sorted
`json.dumps`.

The compression parts (feature cache 0.13 s, replay gzip level 1 plus its identity hash and fsync 0.10 s) were not
changed: the replay hash is an identity, and the feature cache is small next to the decode.

What bounds collection is still open. With 54 teacher tables holding 54 of the 60 CPU tickets, ordinary games and
publications share the rest until the tables finish (28-53 s); peak unfinished families sat at the limit (128). Over
130 iterations collection time follows the teacher finalization that runs in the trainer during collection (deferred
sealing r = +0.84, label check r = +0.70) and falls when the last search ends later (r = −0.56): finalization overlapping
the ordinary tail slows it. The trainer's own per-game ingress is small (unpickling a publisher's prepared rows 0.04 s,
re-sealing 0.06 s). So the decode saving (about 0.1 s of publisher CPU per game, 30-50 s per iteration) shortens
collection only as far as the tickets bind; a per-game timeline (dispatch, native end, publish start and end) in the
report would settle it.

## Checks

- `tests/Recovery/test_value_speed_g6547.py`: joined worker parts equal the per-part batches and columns (real PPG replay,
  real CPU workers); the joined fixture phase reports the unjoined numbers within rounding; corrections bitwise equal
  on varied tables; device rows only for opted-in datasets, one budget, released after PPO; canonical dumps and freeze
  equal their plain formulas.
- `tests/Step5/test_exit_guard_g6547.py` and the WSL signal smoke test above.
- Focused suites, after the identity was re-recorded (137 tests: this change's two files, the g6120 PPO speed, g5781
  value-phase speed, g4770 era heads, g3423 search auxiliary, g3471 search rows, g3318 ownership stop, g3528 Rail-end,
  g5466 auxiliary revision, PPG eight/dense/g210 and publisher decode suites): 135 pass. The 2 errors are the
  known `test_ppg_g210.Restored` g660 cases, whose export is not on this laptop; they fail at HEAD too
  (`docs/KNOWN_TEST_FAILURES`).
- The new paths' GPU behavior (joined batches, device-row release, per-phase peaks) is not measured: the laptop has no
  CUDA GPU. The next export shows it in `value_phase.step_profile`, `actor.device_rows` and `gpu_memory`.

## Cost

- Value phase: about 1-3 GB more transient GPU memory during the phase (less than the 8.2 GB freed before it); float
  results not bit-identical.
- Device rows: rebuilt every PPO as before (about 3 s, `device_rows.*.seconds`); nothing new.
- `empty_cache` four times per iteration (`gpu_memory.phases.*.release_seconds`), and the next phase allocates again.
- Exit guard: a stopped trainer is force-ended after 60 s of a stuck exit; commits are atomic, so this is the same as
  the `kill -KILL` used now, but automatic and with the stacks logged.

## Execution identity

`310796d78385ea572902734ed5dbbf9fbdcf8394a9bed79318ad7b6731348f60`, predecessor `4aaa5fba…`; entry
`g6547_value_memory_and_exit` in `configs/learning/b07-replay-location-review.json`. No Rust or teacher-core change:
no CPU teacher rebuild.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 6547 --iterations 10 --share-target 0.065
```

## Not done

- A per-game collection timeline, to settle what bounds the ordinary tail.
- The full GC pass after each commit (15 s per iteration, walking about 14M objects).
- The teacher finalization's trainer-side work during collection (check 18-25 s, deferred sealing 5-8 s).
- PPO microbatches (the same per-op host cost as the value phase; widths vary, so not joined here).
