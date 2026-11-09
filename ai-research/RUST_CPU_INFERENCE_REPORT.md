# Rust CPU inference, 64 workers and search_v3: implementation report

Date: 2026-11-09. Plan: [RUST_CPU_INFERENCE_AND_SEARCH_V3_PLAN.md](RUST_CPU_INFERENCE_AND_SEARCH_V3_PLAN.md).

## 1. Continue from g1290

On the lab, with the trainer stopped and this commit pulled, run these in order:

```bash
python scripts/prepare-rust-cpu-inference.py --expected-generation 1290
```

```bash
python scripts/qualify-rust-cpu-inference.py --expected-generation 1290
```

```bash
python scripts/continue-rust-teacher.py --expected-generation 1290 --iterations 10 --share-target 0.25 --inference cpu
```

The prepare step rebuilds the two workers, because their source changed. The qualification only reads the run and takes roughly 15–20 minutes; see section 3. Push its evidence with the commands it prints.

A retry repeats only what has not passed. It skips stages that passed on the same code, build, checkpoint and settings, and it does not rerun the main test suite that the prepare step already passed on this build. `--fresh` repeats everything.

To return to the GPU path at any committed generation, use `--inference gpu`.

**The first iteration keeps its old teacher tables.** The crashed run had already finished and published the 32 teacher tables for g1290 at N=256/48. The next run reuses them, which is the normal crash recovery. Recollecting them at the new budget would stop on `Uncommitted teacher shard differs`, because those shards are bound to their bytes. N=640/96 starts with the tables for g1291.

## 2. What changed

| Area | Change |
| --- | --- |
| Crash fix | `export_bundle` assumed every model has a utility head. The imitation-learning anchor (generation 0, `FourSeatNet`) does not. Its bundle now carries an all-zero utility output, and evaluators omit `utility` and `reward_id` for it, as the GPU service does. |
| Teacher budget | Fixed count: full searches N=640 with K=16, quick searches N=96 with K=8. Set with `--teacher-full`, `--teacher-fast` and `--teacher-fast-k`. Full K stays 16 because labels require it. |
| Search limits | A fixed-count search now runs to its quota. The cap is N × 60 ms (38 s full, 5.8 s quick), which only stops a stuck worker. The Python verifier, the 180 s table watchdog (now 1800 s) and the 30 s turn watchdog (now 120 s) use the same limits on this path only. The sealed job, and its 4 s and 0.8 s fields, are unchanged. |
| Teacher search | Each Rust worker runs ONNX Runtime in process. Search code, feature encoder, labels and value rows are untouched. |
| Ordinary lanes | New `infer` operation in the Rust collector. Python keeps sampling, RNG and every check. |
| Rating | Same C# host and game flow. Rating lanes evaluate with Python ONNX Runtime instead of the GPU service. |
| Workers | 64 lanes × 1, CPU budget 64, reserve 0, one lane per CPU, spread across sockets. Applied at run time only. |
| Lane startup | Each lane process now receives only the two paths it reads, which removes the serial wait described below. CPU lanes also start work as soon as they are ready and no longer lease GPU reply channels. Each lane warms its Mono bootstrap before its first game. |
| Startup measurement | Every collection records `startup_breakdown` per lane, and teacher tables record `mono_bootstrap_seconds`. See section 5. |
| Training entry | The ordinary pool now receives the signed recipe and the 64-lane worker settings separately. The teacher on-mode check compares a digest of the recipe, including its worker counts, so it must see the signed one. |
| Tolerance | CPU-sourced labels use the 2e-4 cross-runtime bound in `check_logits`. GPU labels keep 2e-5 / 1e-5. |

The whole feature sits behind the `native_inference` block in `execution`, which `collector_runtime.recipe` strips before the training signature. With the block absent, the GPU code path, binaries and call arguments are as they were, apart from the extra startup measurement. `lab-native/` is not edited: `lab-native-cpu/` and `lab_cpu_overlay.py` patch a detached copy, so the qualified GPU binaries keep their fingerprints. The review in `b07-replay-location-review.json` records the new execution identity.

### What the first lab run showed

The results in `rust-lab-qualification/20261109T071049919614Z-…` say `passed: false`. The prepare step and the CPU teacher phase did work on Linux: 32 tables finished in about two minutes at N=256/48. The run then stopped in ordinary collection on the anchor export above, before any game or checkpoint.

### Lane startup: the cause, found by the new measurement

An earlier version of this report guessed that the per-lane Mono bootstrap caused the 16.6 s lane startup. The stored reports show that guess was wrong.

- Across all 37 stored collections at 24 processes and 48 lanes, actor startup was 16.3–16.9 s, and the 24 publisher processes took about 0.85 s.
- A lane reports ready before it has used Mono, so Mono time is not part of that wait.
- The first Mono call on each lane took 0.14 s (median) and at most 0.22 s, including its share of JIT.

The first lab qualification measured the real cause. Starting each lane process took 0.71 s, one after another: 11.4 s for 16 processes and 45.5 s for 64, against 24 × 0.69 s = 16.6 s before. Each process was handed the whole build receipt, which pickles to 170 KB. That is more than a pipe holds (64 KB), so `Process.start()` blocked until that child had finished importing. Lanes only read two paths from it, so they now receive just those. The next run measures the result; `processes_start_without_waiting` requires under 0.2 s per process.

## 3. Lab qualification

`scripts/qualify-rust-cpu-inference.py` holds the run lock, reads the run and never writes to it. It keeps going after a failure, so one run shows every problem. The unfinished `iteration-1290` from the crashed run stays where it is for the normal recovery. The qualification records its files beforehand and fails if any change.

| Stage | What it proves |
| --- | --- |
| `environment` | Linux, GPU, enough CPUs, Mono, ONNX Runtime version, pinned checkpoint. |
| `regressions` | Both CPU test suites against the prepared Linux workers, including all-seat blending, an anchor-and-learner game and a real spawned lane. Only the two expected skips are allowed. |
| `bundles` | Every model the league can seat has a bundle with parity under 2e-4, including an anchor with no utility head. |
| `parity` | Rust collector against PyTorch on real positions, for the learner, a utility-only model and the anchor. |
| `teacher` | 16 real tables through the production collector at N=640/K=16 and N=96/K=8, at blends 0, 0.25 and 1. Checks quotas, K, labels and that the root values match PyTorch's `utility + w·delta`. |
| `ordinary` | 128 real frozen jobs at 64 lanes with strict validation, seating anchors first. 16 of them are matched against the GPU path. |
| `rating` | Real rated games with the learner and anchor, on CPU and on the GPU service. |
| `startup` | Per-lane startup phases, Mono bootstrap and peak memory from the 64-lane run. |

A diverged game is counted, not ignored. Probabilities must agree within 3e-3 relative or 3e-5 absolute up to the first different action, and at most 15% of matched games may diverge. No more than 2% of searches may hit the safety cap, and no more than 5% of labels may be rejected.

It does not run a full training iteration: PPO, PPG and checkpointing are not exercised, because a full iteration cannot run on a copy of the run. The continue command is the first real training use.

## 4. Verification on the laptop

- All 65 CPU tests pass with real compiled workers, apart from the two expected skips: the timed-budget one-turn run, which you asked to skip, and the local-host rating game, whose real version is the `rating` stage. Covered: all-seat blending at zero and non-zero weights, every anchor path, a full mixed-model game, a spawned lane through to its export, and the qualification's own checks.
- One all-seat table at the full production budget, with the fixture model: every search reached its quota, none hit the cap, the table took 124 s, and a full search took about 3.5 s (5.5 ms per trajectory).
- The existing lab suites show 7 errors out of 183 tests: `Admitted actor inputs are immutable` (4), `TeacherPhase.config` (1) and the rollback pin fixture (2). These come from old fixtures, and none is in code changed here.

## 4a. First results from the lab qualification

Measured on the lab at g1290, before the startup fix:

| Measure | Result |
| --- | --- |
| Teacher, 16 tables on 16 lanes at N=640/96 | 45 s; full search median 1.75 s, quick search median 0.24 s; none hit the cap |
| Blended root values against PyTorch | worst relative error 6e-7 |
| Ordinary, 128 real jobs at 64 lanes | 17 s of live collection, after 46 s of startup |
| 16 ordinary games matched against GPU | none diverged; worst probability difference 1.4e-5 relative |
| Mono bootstrap per lane | median 0.126 s |
| Peak memory, whole process tree | 71 GB |

**search_v3 was not adopted.** The lab and the game stay on search_v1, and nothing was installed. `scripts/search_v3_patch.py`, `build-search-v3.py` and `run-search-v3-smoke.py` remain for later.

## 5. What to read after the first lab iteration

- `report.exploration_metrics.branching.scheduling.startup_breakdown`: `phases` holds `mono_bootstrap_seconds`, `python_import_seconds`, `lease_seconds`, `session_import_seconds`, `lane_wait_seconds` and `total_seconds`, each with min, p50, p95, max and mean. `per_lane` holds each lane. The log prints a one-line summary.
- `report.timings`: `collection_seconds`, `teacher_collection_seconds`, `evaluation_seconds`, `iteration_seconds`.
- `teacher_phase` metrics: `mono_bootstrap_seconds`, `cpu_inference`, per-table `cpu_inference.positions` and `forward_ns`, `completed_trajectories`, `deadline_hit_fraction`.
- PPO likelihood verification and teacher logit checks must keep passing.

## 6. Known limits

- The teacher phase took 45 s for 16 tables on the lab at N=640/96, so a 32-table phase should take about the same on 32 lanes. A full iteration has not yet run at this budget.
- Teacher and ordinary collection still run one after the other, so the teacher phase uses only 32 of the 64 lanes.
- The lab usable CPU count is exactly 64, so 64 pinned lanes leave none spare for the publishers and the parent process; they share those CPUs.
