# Build step 9: ordinary games played inside the CPU worker

Step 9 of [TRAINING_IMPROVEMENT_PLAN_G2910.md](TRAINING_IMPROVEMENT_PLAN_G2910.md) (section 5, design point 5) moves action sampling for ordinary PPO games into the Rust CPU collector. The run keeps 320 ordinary games per iteration. Whether to raise that to 480–640 is decided after the lab shows the saving.

It is a code-only change with no recipe flag. It applies from the first iteration that runs the new code, once the CPU workers are rebuilt. The continuation rebuilds them by itself.

## Where an ordinary game's cost went (g3139, 320 games)

| Part | Core-seconds per iteration | What it is |
| --- | ---: | --- |
| Python decision loop, appends | 796 | 507 Python↔Rust round trips per game; each reply is the next view as JSON (12.4 MB per game), delta-decoded and frozen in Python. Rules and features in Rust: about 85 s. |
| Python decision loop, rest | about 660 | Per-decision canonical JSON, SHA-256 and deep freeze of the view (`lab_perf.admit`), row and evidence dicts. |
| Policy calls | 277 | Of which ONNX forward 66 s; the rest is the `infer` round trip. |
| Shared-memory write | 644 | About 336 s waiting for the parent to allocate the lease, CPU-ticket waits, and copying and hashing 45 MB. |
| Publisher read and decode | 405 | Rebuilds and parses every row with its full history; seals each row. |
| Publication | 225 | Validation, family JSON, replay file, feature cache. |
| Replay preparation | 150 | Fresh PPO shards. |

The lane, meaning the first three rows, accounted for 1,731 of about 3,150 core-seconds.

## What changes

- **One call per game.** For an eligible game, `collect_lane` sends the four seats' settings once (`ordinary-autoplay/1.0.0`). The worker then plays the game to its end in `lab-native-cpu/src/cpu_autoplay.rs`. For each decision it runs the CPU forward, the float32 softmax, opening temperature, v3 family exploration and the ordinary cotton recipes (1.0.0 to 4.0.0, including the Rail-era build level). Every row is appended through the unchanged `append_compact` path. Python then exports the game exactly as before.
- **The same RNG draws.** Each seat's `random.Random` state, its 624 Mersenne Twister words and index, goes to the worker and comes back at the end. The worker draws in Python's order: `random()`, `choices` by cumulative weights and bisection, `_randbelow` for the cotton pick. It sums like NumPy, with pairwise float64 and float32 sums. The games are the ones the Python loop would have played.
- **The same audits.** The publisher still replays the draws with Python's RNG: cotton seats on every game, and all four seats on the audited games (first per stratum and every 64th). A rejection fails publication; nothing is accepted with a relaxed tolerance. Autoplayed views skip the actor-side `BOUNDARY` admission, so on audited games the publisher checks every view in full instead.
- **Fallback.** Branching, non-neural or non-v3 seats, GPU inference, strict validation, continuations and diagnostic hooks keep the Python loop. All 320 production games qualify today: four neural v3 seats on CPU inference, no branching. `BRASS_ORDINARY_AUTOPLAY=0` in the environment keeps every game on the Python loop.

## Evidence (Windows laptop, fixture weights, 4 games)

Two frozen jobs ran through both paths with the same seeds. The first is a production-like cotton table (exploration 0, mixed focus, opening T1.3 for three actions, cotton 4.0.0 at 0.2 on one seat). The second has exploration 0.1, a pottery focus, opening T2 and a rotated table.

- **Rows.** All 618 decisions had identical views, choices, chosen actions, contexts and feature tensors, including the 4 cotton picks.
- **Probabilities.** Behavior fields were identical apart from the probability. Probabilities differed by at most 4e-8 across paths, because NumPy's and Rust's float32 `exp` differ in the last bit. Within each path, the publisher's full four-seat replay audit passed at its 1e-12 tolerance.
- **Rust unit tests.** These check the CPython RNG stream, the v3 exploration distribution and NumPy's summation order. They run on the lab whenever the CPU workers rebuild (`cargo test --lib cpu_`).

| Lane time per game | Python loop | Autoplay |
| --- | ---: | ---: |
| Cotton table, 299 decisions (includes lane start) | 2.31 s | 0.66 s |
| Exploration table, 319 decisions | 3.18 s | 0.53 s |

The worker itself takes about 0.40 s per game. The rest is the export and the hand-off to shared memory.

## What to expect on the lab (*estimates*)

- **Lane time.** Lane-seconds per iteration fall from about 1,730 to about 400. Ordinary CPU falls from about 3,150 to about 1,800 core-seconds, roughly 40%.
- **Collection time.** The ordinary pool should finish before the last teacher table instead of about 10 s after it, so collection would shorten by about 10–15 s. With fewer ordinary processes competing, teacher tables may also run faster.
- **Reliability of these numbers.** The laptop runs one game at a time. The lab runs 64 lanes, 24 publishers and the teacher on 64 CPUs, so its per-game times are about twice the laptop's. Read the actual saving from the first iterations.

## Not done in this step, and why

- **Workers writing features straight into shared memory.** The copy is small: writing a game's 45 MB took 0.05 s here. On the lab the 2 s per write is mostly waiting, about 1 s for the parent's allocation round trip plus CPU tickets. Removing that wait needs a new lease-ownership design, not a faster copy. Decide this after the first autoplay iterations show whether the wait limits throughput.
- **The publisher's decode, now the largest ordinary cost.** Every row carries the game's full history, so parsing and sealing a game is quadratic in its length: 0.8–1.2 s per game here (`json.loads` 0.33 s, row seals 0.2 s). Sharing the history between rows would roughly halve the remaining cost. It touches the replay seal format, so it is its own change. Do it before raising the game count to 480–640.

## First iteration: what to check in `iteration-N/report.json`

- **Coverage.** Each game's `execution[*].native_metrics[0].timings.autoplay_games` should be 1 for every game. `ipc_calls` should be about 20 per game instead of 507.
- **Lane time.** The sum of `native_metrics[0].timings.total_seconds` was 1,731 s at g3139. `autoplay_worker_seconds` is the worker's share.
- **Audits.** `native_backend.collection.validation_policy.publication_checks.sampling_rng_probability` should show executed 6 and failed 0. Cotton tables should publish.
- **Collection time.** Compare `timings.collection_seconds` and `native_backend.collection.wall_seconds` with about 86–88 s at g3138/g3139.
- **Remaining costs.** Watch `publication_phases.part_read_seconds` and `actor_part_transfer[0].shared_allocate_wait_seconds`.

If publication fails with "Frozen sampling RNG differs" or "Behavior denominator differs", rerun the same command with `BRASS_ORDINARY_AUTOPLAY=0` in front of it. That restores the Python loop, and nothing else changes.

## Code

| File | Change |
| --- | --- |
| `lab-native-cpu/src/cpu_autoplay.rs` (new) | CPython RNG, NumPy sums, v3 exploration, cotton recipes, `Lane::autoplay`; Rust unit tests |
| `src/brass_ai/rust_backend/lab_cpu_overlay.py` | Registers the module, adds the `autoplay` operation and advertises it in `hello` |
| `src/brass_ai/rust_backend/lab_autoplay.py` (new) | Eligibility, seat hand-off, reply checks, RNG restore, timings |
| `src/brass_ai/rust_backend/lab_transport.py` | `Collector.autoplay` |
| `src/brass_ai/rust_backend/lab_collection.py` | Uses autoplay for eligible games; `actor_validated` is false for them |
| `scripts/prepare-rust-cpu-inference.py` | Adds `test_ordinary_autoplay` to the build regressions |
| `tests/RustBackendLab/test_ordinary_autoplay.py` (new) | Routing, hand-off and overlay tests; native parity when `BRASS_CPU_COLLECTOR` and `BRASS_CPU_BOOTSTRAP` are set |
| `configs/learning/b07-replay-location-review.json` | Execution identity re-recorded; review note `g3140_ordinary_autoplay` |
