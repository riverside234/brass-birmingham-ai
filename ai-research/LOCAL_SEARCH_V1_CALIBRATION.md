# Local search_v1 re-calibration: release 1.4.62 (installed)

The user's request of 2026-11-21: re-calibrate the local search_v1 for the faster core engine of release 1.4.61
(`G4475_TEACHER_TRAINING.md`, item 4), and install it.

Result: each package's startup estimate went from 7.26 ms to 6.36 ms per simulation. A turn's search now runs about
14% more simulations in the same thinking time, for example 470 instead of 411 per lane at 4 s. Every search still
finishes its quota inside its time. The patch is installed as `1.4.62-rust-local`.

Scripts are in `scripts/local-game/search-calibration/`:
- `calibrate-search-v1.py` runs the calibration: `measure`, `full` and `verify`;
- `build-search-calibration.py` builds the release;
- `qualify-search-calibration.py` checks it: `qualify` before installation, `audit` after;
- `install-search-calibration.ps1` installs it and also supports `-Action Rollback` and `-Action Recover`.

Evidence is in `artifacts/search-calibration/`.

## What the estimate does

A timed local search sizes itself before it starts. Each of its two lanes gets this many simulations:

> quota = available time / estimate

The search answers as soon as the quota is done.
- **Where the estimate comes from.** It is the package's `startup_ns`, in its six policy files. After four or more
  simulations in a turn, later searches in that turn use their own measured cost instead (the lower quartile of
  the last eight).
- **Why it governs most of the time.** The game asks for the whole turn's plan in one request, so a turn's main
  search is sized by `startup_ns`.
- **The old value.** 7.26 ms was the L12 calibration maximum on the pre-decision-reuse engine. With the faster
  engine, release 1.4.61 finished its quota in about half to three quarters of its time and left the rest unused.

## Calibration

Measured on this laptop with the installed 1.4.61 packages. Both stages cover:
- all four models;
- K 16 and K 8;
- the warm-up root and every captured position (`artifacts/rust-backend/20261025-l10-capture`), with each model's
  own context and reward and the insolvency ledger rescaled to its event cost. Forced, debt and bypassed roots
  search nothing and are left out.

The cost of a lane is the time from receiving the request to its last completed simulation, divided by the
simulations completed. It therefore includes root preparation.

| stage | what | lanes | median | maximum | full quota |
| --- | --- | --- | --- | --- | --- |
| 1 | the L12 method: about 64 simulations per lane, 2 repetitions | 736 | 2.98 ms | 5.63 ms | — |
| 2 | real quotas, at stage 1's maximum (about 530 per lane at 4 s) | 368 | 4.19 ms | 6.36 ms | 164 of 184 |
| verification | real quotas at 6.36 ms: every position at 4 s, plus K 16 at 12 s | 400 | 3.49 ms | 5.76 ms | 200 of 200 |

Findings:
- **Real quotas cost more than 64 simulations suggest.** Stage 1 alone (5.63 ms) would have overrun: 20 of 184
  stage-2 searches missed their quota at that estimate. So the estimate is stage 2's maximum, rounded up to a
  microsecond, which is the L12 rule applied at the quota a turn actually runs. Stage 2 ran under about 15 minutes
  of sustained load, so it covers a hot laptop.
- **One position type sets the ceiling.** The cost per simulation depends on the position far more than the engine
  change does. The wide Canal round-1 roots (about 400 legal moves; the warm-up root and positions 0, 3 and 4) cost
  4.2–6.4 ms. The network scores every legal move at every leaf, and that forward pass is untouched by the engine
  change. Rail and later positions cost 2.4–4.8 ms. One estimate must be safe for the widest roots, which limits the
  gain to about 14%.
- **Time used in verification:** at most 90.5% of the available time on the slowest search.

## Release 1.4.62

The builder took the installed 1.4.61 and changed only `startup_ns` (7262519 → 6360000) and `startup_source` in all
six policy files of each package (K 8, 16 and 32, at T 1.0 and 1.5), plus their pins in `build.json`. Unchanged:
- binaries, native source and models;
- warm-up roots, K mappings and temperatures;
- the managed patch.

The game hashes the policy files against `build.json` at run time, so no managed rebuild is needed.

**Qualification** compared 1.4.62 with the installed 1.4.61 on 4 models × K 16 and K 8 × 4 positions, interleaved,
at 4 s:
- Simulations rose from 814–822 to 930–938 per search; the median ratio is 1.14.
- Every candidate search completed its quota before the work deadline, with replies at 1.2–2.8 s.

**After installation** the audit started each package from the game folder. Each passed its warm-up and completed
its quota (938 simulations on the warm-up root), and the receipt matches. The receipt records `searchCalibration`,
with the recovery backup `artifacts/patch-recovery/search-calibration-e526…`. The game itself was not opened.

To undo it:

```
& "scripts/local-game/search-calibration/install-search-calibration.ps1" -Action Rollback
```

## Further gain (not done)

Most positions could run about 1.5–2× the simulations they now get. Rail and later roots finish their quota in
1.2–1.9 s of the 3 s of search time. Getting that needs a position-aware startup estimate: for example by era and
round, or by the root's legal width, measured the same way per bucket.

That is a worker change, not a policy edit:
- a new policy field and its check in `brass-ai-worker/src/local.rs`;
- the estimate chosen per request;
- four rebuilt binaries;
- calibration per bucket;
- a new qualification.
