# Rust CPU inference, 64-worker collection and search_v3 plan

Date: 2026-11-08. Status: **implemented**; see [the implementation report](RUST_CPU_INFERENCE_REPORT.md) for what was built, what was verified and the command to continue from g1290. The rest of this document is the original plan, and it was written before any benchmark ran.

Evidence comes from the latest pulled export, [`joint7-v11-2-g1290-trained-restart-g1290-20261109T041246972332Z`](../results/joint7-v11-2-g1290-trained-restart-g1290-20261109T041246972332Z/state.json), plus the code and reports linked below. Anything marked *estimate* is a projection that the measurements in section 8 will replace. A "lane" is one game or search worker; "lane-seconds" sum time across lanes.

## 1. Summary

Collection is the largest cost in training. In g1288 it took 249 of 322 seconds (teacher 114 s, ordinary 135 s). Most of that time is workers waiting for inference, and the wait is not mainly GPU compute:

- Teacher tables spend **91%** of their lane-seconds waiting. Each 4 s search completes a median of **58** trajectories against a frozen quota of 247, and the 0.8 s fast searches complete **11**.
- Ordinary lanes spend **55%** of their lane-seconds in the inference round trip.
- During the teacher window the GPU forward is busy only about half the time. Each leaf request crosses a Python process shared by 32 lane threads and the GPU service thread.

**Recommendation:** move policy and value inference inside the workers using the game's existing stack (ONNX Runtime 1.22.1 `CPUExecutionProvider`, `ort` 2.0.0-rc.10, one compute thread per session), behind an opt-in execution setting, and run 64 pinned single-lane workers. PPO, PPG and head training stay on the GPU. The current GPU inference path and its binaries stay available for comparison and rollback.

Estimated effect, assuming about 3 ms per position (section 3.3):

| | Today (g1288) | CPU, same search budgets | CPU, count-based teacher budgets (decision D1) |
| --- | ---: | ---: | ---: |
| Ordinary collection (including about 13 s of planning) | 135 s | 35–50 s | 35–50 s |
| Teacher collection | 114 s (time-boxed) | 105–115 s (time-boxed) | 35–45 s |
| Total collection | 249 s | about 110 s with overlap, 150–165 s without | 60–70 s |
| Full iteration | 322 s | about 185 s with overlap | 135–145 s |
| Trajectories per full / fast search | 58 / 11 | roughly 500–1,200 / 100–250 | fixed count |

Search is time-boxed, so at unchanged budgets the teacher gains quality rather than wall-clock time. To save wall-clock time as well, the budgets must change (D1).

`search_v3` is a separate, gated step after the CPU path works. It changes timed scheduling only and keeps count-mode math identical (section 6). The local game is patched only if the lab adopts it.

## 2. Where training stands

- Run `joint7-v11-2-g1290-trained-restart`, committed **g1290**, model `501ac43b…`. It runs on the transferred machine: Linux 6.8, 2 × EPYC 9335 (`cpu_count` 64, affinity 0–63), RTX PRO 6000 Blackwell, CUDA 12.8.
- Launcher: `scripts/continue-rust-teacher.py` (teacher **on**, A10/B20/C10 stage schedule). Reward `brass-combined-return/3.0.0`. Fixed teacher β `0.16110139609854138`. PPG β floor 0.25. Branching 0%. search_v1, V205 off.
- Teacher shape per iteration: 28 train tables (+4 held-out every tenth iteration), all-seat search. Each table has 16 full turns (4 s, K16) and 48 fast turns (0.8 s, K4), so about 102 s of search time per table.
- Search head: Stage A, `w=0`, `scale=0`. The counter stays at 4/10 while eight fresh return ages rebuild after transfer (3 of 8 at g1290). Stage B will set `w` from 0.05 to 1.0 over 20 collections, so the CPU path must support `w>0` before Stage B collections start.

## 3. What the logs show

### 3.1 Measured cost

Teacher detail is from iteration 1289 (32 tables, `teacher-shadow/collection.json`, `report.json`).

| Measure | g1288 | g1289 |
| --- | ---: | ---: |
| Iteration total | 321.8 s | 591.8 s (includes 135.6 s rating, 46.8 s cleanup) |
| Teacher collection wall | 113.9 s | 116.5 s |
| Ordinary collection wall (native) | 122.4 s | 111.8 s |
| — of which actor startup | 17.3 s | 17.5 s |
| Ordinary GPU batches / positions | 15,051 / 53,716 (mean 3.6) | 14,142 / 53,850 (mean 3.8) |
| Ordinary forward time | 82.6 s | 70.9 s |
| Ordinary queue wait, mean per request | 32.7 ms | 30.7 ms |

| Ordinary lanes, g1289 (320 games, 48 lanes) | Lane-seconds | Share |
| --- | ---: | ---: |
| Total | 4,184 | 100% |
| Inference round trip (`inference_envelope_seconds`) | 2,318 | 55% |
| Rust step calls (`native_step_seconds`, 98,835 calls, about 10 ms each) | 979 | 23% |
| Policy work outside inference | 62 | 1.5% |
| Everything else (features, export) | 825 | 20% |

| Teacher, g1289 (searches with a time budget) | Full (4 s, K16) | Fast (0.8 s, K4) |
| --- | ---: | ---: |
| Searches | 511 | 1,530 |
| Frozen quota N (median) | 247 | 46 |
| Completed trajectories (median) | 58 | 11 |
| Completed halving rounds (median) | 1 | 0 |
| Work-deadline hit | 97.8% | 99.6% |
| `search_supported` / `bounded_policy_completion` / `gumbel_prior_fallback` | 268 / 159 / 73 | 3 / 1 / 1,520 |
| Second-action searches with no time left (frozen N=0, all prior fallback) | 150 | 319 |

Across all tables: 48,929 leaf requests, 3,160 of 3,468 lane-seconds waiting (91%), 64.6 ms mean wait per request. Feature encoding is only 7.5 s in total (0.15 ms per leaf).

Labels: 511 root labels and 0 continuation labels in g1289. Earlier iterations had 0–20 continuation labels. The teacher plan expected about 31 labels per table; the actual count is about 16.

### 3.2 Diagnosis

1. **The teacher bottleneck is the per-leaf path through Python.** [`teacher.rs`](../lab-native/src/teacher.rs) builds a JSON binding for every leaf, including a SHA-256 over the serialized information state. [`lab_teacher_inference.py`](../src/brass_ai/rust_backend/lab_teacher_inference.py) then decodes, checks and hashes it inside the trainer process. Thirty-two lane threads and the GPU service thread share one GIL. In the teacher window the service forwarded 48,302 positions in 4,245 batches (mean 11.4). Forward totalled about 58.6 s of a 116 s window, so the GPU was busy about half the time, yet each request still waited about 65 ms.
2. **The quota calibration ignores load.** `calibrate()` measures one idle lane: 15.9 ms per trajectory, which freezes N≈247 for 4 s. Under real load a trajectory takes about 70 ms, so the search completes only about a quarter of N. Halving then stops after round one, or before finishing a round for fast searches, and nearly every search runs into its deadline.
3. **Fast searches are effectively prior samples.** 1,520 of the 1,530 budgeted fast searches end as `gumbel_prior_fallback`.
4. **Second actions often get no search.** 469 of 2,510 searches are second-action roots that start with N=0 because the first action used the whole allowance and no supported continuation exists.
5. **Ordinary lanes wait on small batches.** Mean batch size is 3.8 and queue wait is about 31 ms per request. Rust step calls also cost about 10 ms each, far above the Rust work itself (about 0.46 s per game in `native_cost`). That points at Python/GIL and pipe handoff with two lanes per process. This should improve with one lane per process.
6. **Actor startup costs 17.5 s per collection** (24 spawned Python processes importing torch).

### 3.3 Why CPU inference should win here

- The game's ONNX path measured 2.8–5.1 ms per position on the development laptop ([L09](RUST_BACKEND_L09_REPORT.md)). L14 recorded 83.6 s of inference for 24,575 positions, or 3.4 ms each ([L14](RUST_BACKEND_L14_REPORT.md)).
- Those runs used the same 5.9 M-parameter architecture as this run. A planning figure of 3 ms per position on a Zen 5 EPYC core is an *estimate*; phase C0 measures it.
- Current demand is about 102,000 positions per iteration, or about 300 CPU-seconds. Against 64 cores that is about 5 s of wall time.
- With the deeper searches CPU inference allows (about 530,000 positions per iteration at the budgets in section 1), inference needs about 1,600 CPU-seconds, which fits inside the 3,360 lane-seconds of a 105 s teacher window on 32 lanes.

Ordinary estimate: 4,184 lane-seconds, minus the 2,318 s inference wait, plus about 160 s of CPU inference, is about 2,030 lane-seconds. Over 64 lanes that is about 32 s, plus publication and any remaining startup.

Sources for the hardware: 32 cores, 128 MB L3 and 3.0 GHz base / 4.4 GHz boost come from retailer and spec listings. A 512-bit AVX-512 datapath is documented for the EPYC 9005 family. The CCD layout (four 32 MB L3 slices per socket) comes from one retailer listing. Phase C0 confirms all of this with `lscpu` and `numactl -H`.

## 4. Target design

### 4.1 Data flow

```text
Teacher (GPU today):  Rust search → JSON + tensors over pipe → Python → shared slot → GPU service → back
Teacher (CPU target): Rust search → in-process ORT session → Rust search

Ordinary (GPU today):  Rust state → features over pipe → Python policy → GPU service → Python sampling → Rust step
Ordinary (CPU target): Rust state → features and ORT forward inside Rust → logits to Python → unchanged Python sampling → Rust step
```

Both paths keep their current surrounding structure: the job planner, frozen jobs, publishers, verifier and recovery.

### 4.2 Model bundles

- Export with the existing pipeline ([`export.py`](../src/brass_ai/personality/laptop/export.py), [`onnx_tools.py`](../src/brass_ai/learning/onnx_tools.py)): opset 17, FP32, exact GELU, dynamic batch and candidate axes, parity at the six existing shapes.
- **Add one output, `search_delta_4`**, when the model has `search_value_head`. It computes [`correction()`](../src/brass_ai/learning/search_value.py) with unknown descriptors on the same encoded state. A model without the head keeps the five-output graph and may only run with `w=0`.
- Bundle contents per model id: `model.onnx`, `model.json`, parity arrays, and a lab manifest binding `model_id`, `checkpoint_sha256`, `export_sha256`, `reward_id`, `max_cost`, architecture, feature schema and layouts, runtime hash, session options, and the output list.
- Cache by `model_id` under the run directory, immutable. Historical pool models export once, a few at a time; the learner exports once per iteration. A beta export took about 10 s on the laptop, so start the learner export as soon as the checkpoint commits. Pin cached bundles through the existing retention rules.
- Runtime: pin `libonnxruntime.so.1.22.1`. Prefer the official Linux release archive, because the PyPI wheel may not ship the shared library; C0 checks this and records the hash.

### 4.3 Rust evaluator

New feature-gated module in `lab-native` (`cpu-inference`), adapted from [`brass-ai-inference`](../native/brass-ai-inference/src/runtime.rs). The sealed crate's `Manifest::validate` and `CpuEvaluator::load` hard-code five outputs and the game's manifest contract, and any edit under `native/` changes the sealed source fingerprint. So the adaptation lives in `lab-native` and leaves `native/` untouched.

Kept exactly from the game:
- `ort` `load-dynamic` with `init_from`, one intra-op and one inter-op thread, sequential execution, no spinning, Level3 optimization, exact GELU, two warm-up forwards.
- The `!Send` session-per-owner rule.
- Borrowed singleton tensors, the `Packed` batch layout, and finite-output, padding and shape checks.

Added for the lab:
- The sixth output and an output list read from the manifest.
- A per-process model cache: one session in a teacher worker, least-recently-used with at most 8 sessions in an ordinary lane.
- Per-worker counters (load, encode, forward, positions, cache hits, RSS).
- Per-position blending: `if w == 0 { utility } else { utility + (w as f32) * delta }`, matching the PyTorch path's float32 arithmetic and exact zero-blend bytes.

### 4.4 Teacher search

- The seam already exists: `Peer` implements `LeafEvaluator<Key, String>`. Add `CpuPeer`, which keeps the same deadline check, the same `FeatureEncoder` call, `search_value`/`root_watch`/`root_stats` handling and the same `Statistics::new(logits, value)` construction. f32 logits and values widen with `f64::from`, exactly as today's JSON round trip does.
- `request.job` is unchanged. A new optional `inference` block carries the bundle manifest path and hash. Load-time checks compare `model_id`, `reward_id` and `max_cost` against the frozen `expected` block.
- `Worker.collect` is unchanged apart from the handshake reporting the backend. No `evaluate` events arrive in CPU mode. Existing `turn`, `calibration` and `finished` events, the turn verifier and the table cache are untouched.
- **Calibration under load (CPU path):** run the calibration workers concurrently across the lanes that will search, and keep the existing rule (exclude warm-up, median of three, maximum root median). This is a new procedure, recorded as such; the search equations do not change.
- The verifier's [`check_logits`](../src/brass_ai/recovery/teacher_labels.py) compares against the GPU PyTorch model at `atol=2e-5, rtol=1e-5`. ONNX Runtime differs from PyTorch by up to about 5e-5, so CPU-sourced labels need the established cross-runtime tolerance (`2e-4`). The tolerance is selected by the recorded backend; the GPU path keeps today's. This keeps an independent GPU check on every labelled turn.

### 4.5 Ordinary self-play

- Add an `infer` operation to the Rust collector. It encodes features with the existing `FeatureEncoder`, runs the session for the seat's model, and returns f32 logits and heads as raw arrays. The Rust side keeps its feature tensors, so training-array export needs no Python feature reuse.
- `remaining` is `[cap; 4]` from the bundle, matching `utility_budget(model, [None, …])`. The ordinary feature context keeps its `ordinary-policy-only` identity and zero budget, and is not validated against the model manifest.
- A new `NativeCpuEvaluator` exposes the `RemoteEvaluator` interface used by `NeuralPolicy`. It rebuilds the response dictionary, computing `probabilities` as a float32 softmax of the returned logits with the same formula as `torch.softmax`, so `accept_response`, `select_batch`, `sample_training`, evidence and the publication audit run unchanged.
- `DecisionEncoder` becomes a no-op in CPU mode (no `compare`). C# V4 seats keep their existing route. Rating and evaluation jobs keep the GPU `ActorPool` route.

### 4.6 64-worker topology

| Setting | Today | CPU path |
| --- | --- | --- |
| `processes` × `hosts_per_process` | 24 × 2 = 48 lanes | 64 × 1 = 64 lanes (one Python process, one Rust child per lane) |
| `cpu_budget` / `cpu_reserve` | 64 / 4, capacity 60 | 64 / 0, capacity 64 |
| Teacher lanes | `min(32 tables, capacity, lanes)` = 32 | 32 |
| ORT threads per worker | — | 1 intra, 1 inter, sequential, no spinning |
| Publishers, readers | ≤ 24 and 16, admitted | unchanged, shared with lanes through FIFO admission |

Rules:

- **One lane per logical CPU** (64 logical CPUs with affinity 0–63), so a worker and its Rust child share one core (the protocol is strictly alternating).
- Every **timed search lane owns its core exclusively**. While teacher tables run, pin teacher lanes and mask ordinary lanes, publishers, readers and the trainer off those cores. A descheduled worker silently loses search time.
- **Spread lanes evenly across the two sockets.** Allocate session memory after pinning so first-touch places it on the local NUMA node.
- Physical overlap of teacher and ordinary jobs is a C5 option. The trainer currently runs them sequentially (`shadow_collector.collect` then `ordinary.collect`). They are independent frozen jobs, so they can overlap physically while admission and reporting keep today's order.
- The first measured limit is expected to be L3 and memory bandwidth with 64 separate weight copies. C0 measures scaling at 1/8/16/32/64 workers. Only if it falls short does the plan add shared initializers (ONNX Runtime has `session.use_ort_model_bytes_for_initializers`; whether 1.22.1 supports the memory-mapped variant is unverified).
- 64 spawned Python processes raise the 17 s startup concern. C5 measures it and adds a persistent worker pool or a `forkserver` preload only if needed.

### 4.7 Configuration, identity and rollback

Opt-in through a new operational block in `execution`, stripped from the training signature in the same way as `collector_validation` and `replay_bindings` ([`collector_runtime.py`](../src/brass_ai/recovery/collector_runtime.py)):

```json
"native_inference": {
  "contract": "brass-lab-cpu-inference/1.0.0",
  "ordinary": "cpu-ort",
  "teacher": "cpu-ort",
  "pinning": "one-lane-per-cpu"
}
```

- **Absent means GPU, byte-identical to today.** The GPU binaries, their receipts and their code path are not rebuilt. New binaries (`brass-lab-collector-cpu`, `brass-lab-teacher-cpu`) are additive: the teacher builds on its reward-patched core (`e6bedd50…`), the collector on its unchanged sealed core, and both are pinned through a new prepare script in the style of `prepare-rust-collector-performance.py`.
- Lane count and CPU budget changes go through a reviewed execution-only transition, in the style of [`throughput_config.py`](../src/brass_ai/recovery/throughput_config.py). No weights, Adam state, RNG, counters, β or reward change.
- Frozen job digests do not change. Each table and family records its backend, runtime hash and export hash as provenance. Cached work stays valid across a backend switch, using the existing `previous_producer` mechanism.
- Rollback is removing the block and restarting from the same committed generation.

## 5. What must not change

| Contract | How the CPU path keeps it | Proof (section 8) |
| --- | --- | --- |
| Features (state 3128, action 327, seat mapping, remaining, descriptors) | Same Rust `FeatureEncoder`, no Python re-encoding | R2 |
| Legal-action scoring | All legal candidates, canonical order, singleton without padding; `Packed` with false masks for batches; 8,192-candidate bound retained | R1, R3 |
| Raw logits | ORT f32 logits with exact `f64` widening; within the existing cross-runtime tolerance of the PyTorch model; `Peer` semantics otherwise unchanged | R1, R3, R9 |
| Utility, search-head correction, blending | `search_delta_4`, float32 `utility + w·delta`, exact bytes at `w=0`, per-seat `w` for head-to-head | R1, R3 |
| Sampling probabilities | Python `select_batch` / `sample_training` unchanged; float32 softmax recomputed by formula; behavior probability, temperature, exploration unchanged | R4, PPO likelihood check |
| RNG semantics | Python per-seat `random.Random` draws unchanged; Rust `Draws` and sample seeds unchanged; NN evaluation consumes no RNG | R4, R5 |
| Model and reward bindings | Manifest binds model, checkpoint, export, reward, `max_cost`, runtime; checked against frozen jobs at load; per-call identity checks retained | R6 |
| Replay and label bindings | Records, seals, teacher labels and value rows unchanged; backend recorded as provenance | R2, R8, R9 |
| Cancellation | Process kill on cancel retained; one ORT call is a few milliseconds and cannot be preempted; deadlines checked before and after; no partial publication | R7 |
| Recovery | Frozen job digests, table cache and family publication unchanged; resume after crash; backend switch across restart | R8 |
| GPU training | PPO, PPG and head training code paths unchanged | R12 |

## 6. search_v1 improvements and search_v3

### 6.1 Observed search_v1 behaviour

| Behaviour | Cause | Fixed by |
| --- | --- | --- |
| N frozen too high under load (247 vs about 58 completed) | Idle-lane calibration | CPU path plus load-matched calibration |
| Fast searches fall back to the prior (1,520 of 1,530) | Same | CPU path (about 100+ trajectories in 0.8 s) |
| Whole-turn searches end early on the local game (L14: ready at 2.4–2.7 s of 4 s) | `FrozenQuota::timed` freezes N from a conservative startup estimate for the whole root | search_v3 rolling rounds |
| Deadline-cut rounds waste their partial work for move choice | Halving on a frozen N | search_v3 rolling rounds |
| Second-action searches often get no time (469 of 2,510 searches had frozen N=0) | The first action used the whole allowance and left no supported continuation | Re-measure after CPU; reserve only if still present |
| Continuation labels nearly absent | Too few trajectories for two-action plan support | Expected from the CPU path; measure |

### 6.2 search_v3 definition (gated)

Timed-mode scheduling only. Count-mode math (Gumbel draws, K selection, completed Q, selection, backup, `σ`, plan ledger, watermarks, `c1=50`, `c2=0.1`) stays identical, and existing golden traces must pass unchanged.

1. **Rolling round quotas.** At each round boundary with `m` survivors and `r` rounds left, set `q = max(1, floor(remaining_sims / (m·r)))`, where `remaining_sims = floor(remaining_work / cost)` and `cost` is a rolling median of completed trajectories at the current root, falling back to the session estimate and then the calibrated startup value. Keep the existing K reduction when the budget is small. The deadline still stops work; the last completed round still drives the recommendation.
2. **Hot-path cleanup.** Reuse scratch buffers in `Statistics::select`/`completed_q` (currently several `Vec` allocations per visit) and reduce repeated key hashing, with bit-exact trace tests. This requires a core edit, which is why it belongs here.
3. **Optional root-diverse waves.** Use the existing cooperative scheduler with `pending ≤ 4`, each trajectory under a different root edge, and ORT batches of up to 4. L09 laptop figures showed about 20–60% less inference time per position at cap 4, depending on the workload. This is a separately flagged variant with its own measurement.
4. **Optional action-two reserve**, only if starvation persists in CPU-path data.

`search_v2` adaptive focus and V205 reuse are out of scope; both were evaluated and not selected.

Identity: `method: search_v3`, `session: brass-gumbel-is-mcts/session-v3`, algorithm `count-v1` unchanged. Adopting it invalidates the teacher policy-label window, per the existing rule, because the search profile changes.

### 6.3 Local game patch (only if the lab adopts search_v3)

Follow the search_v1 restoration precedent ([SEARCH_V1_DEFAULT_UPDATE](SEARCH_V1_DEFAULT_UPDATE.md)):

- `native/brass-ai-core` scheduler and `Method::SearchV3` routing in `brass-ai-worker`.
- Admit `search_v3` in `NativePackage.cs` (currently rejects anything but `search_v1`).
- Set `method` in the `policy*.json` of each `BrassNative` package.
- New build, qualify, install and smoke scripts derived from `build-search-v1-default.py` and its siblings.
- Verify 4/8/10/12 s allowances, K8/K16/K32, temperature 1.0/1.5, cancellation and load between actions, completed-turn saves, and exact rollback and interrupted-install recovery.

## 7. Work plan

### C0 — Baselines and machine facts (lab, read-only)

- Record `lscpu -e`, `numactl -H`, `free -g`, AVX-512 flags, SMT state.
- Freeze a checkpoint and job list from the current committed generation (`scripts/freeze-cpu-inference-benchmark.py`, new) using a diagnostic fork; never touch the training run.
- Record GPU-arm baselines for benchmarks B2–B4.
- Exit: frozen inputs and GPU baseline exported to `results/rust-lab-qualification/`.

### C1 — Model bundles

- New `src/brass_ai/rust_backend/lab_cpu_models.py`, `search_delta_4` export wrapper, cache and retention pins, background learner export, runtime pin.
- Exit: R1, R10.

### C2 — Rust evaluator

- `lab-native/src/cpu_eval.rs`, Cargo feature with the existing `ort` lock entries (these already satisfy `check_shared_dependencies`).
- Exit: R3, R6.

### C3 — Teacher integration

- `CpuPeer` in `teacher.rs` behind the feature; `lab_shadow.py` and `lab_teacher.py` wiring; load-matched calibration; backend-selected `check_logits` tolerance; metrics; source-review entry.
- Exit: R5, R7, R9, B2, B3.

### C4 — Ordinary integration

- `infer` op in the Rust collector; `NativeCpuEvaluator`; `OrdinarySession`/`DecisionEncoder` wiring; source-review entry.
- Exit: R2, R4, R8, B4.

### C5 — 64-worker topology

- Reviewed execution transition, per-lane pinning and masks, session-cache limit, startup measurement, optional overlap of teacher and ordinary jobs.
- Exit: B1, B5, R11.

### C6 — Adoption

- `scripts/configure-cpu-inference.py`, then one normal iteration on a diagnostic fork, then one real iteration with explicit go-ahead.
- Check report fields, rehearse rollback to GPU at the same generation.
- Exit: section 8.3 criteria.

### V — search_v3 (after C6)

- Implement behind a flag on a detached core copy, run gates in 8.2, then adopt and patch the game.

## 8. Verification

### 8.1 Focused regressions

Authored fixtures locally; lab-only items run on the lab. No long sessions.

| ID | Check |
| --- | --- |
| R1 | Export parity: PyTorch vs Python ORT vs Rust ORT at (1,1), (2,7), (4,31), (1,257), (1,772), (2,772), including `search_delta_4`, tolerance 2e-4 |
| R2 | Feature identity: `feature_arrays` bytes equal in-process encoder bytes on recorded frozen decisions |
| R3 | Evaluator parity on recorded leaves (GPU `predict` vs Rust ORT), all heads, `w` in {0, .05, .5, 1}, exact bytes at `w=0`, singleton vs batched |
| R4 | Scripted-logit tape through GPU and CPU code paths: byte-identical rows, behavior probabilities and per-seat RNG end states; strict publication audit passes |
| R5 | Count-mode traces with a scripted evaluator through `CpuPeer` match the existing golden traces |
| R6 | Rejection: wrong model, export, reward or `max_cost`; tampered ONNX or runtime; stale binary; invalid `w` binding |
| R7 | Cancel during load, during a forward and between turns; no partial table or family; first error preserved; resources released |
| R8 | Crash mid-table and mid-family, resume; GPU↔CPU switch across restart; cached bundle reuse re-verifies hashes |
| R9 | `check_logits` passes CPU labels at 2e-4 and rejects beyond it; GPU tolerance unchanged |
| R10 | Bundle cache reuse, concurrent creation, retention pins |
| R11 | Two short rounds at 64 lanes: descriptors, RSS and shared memory return to baseline |
| R12 | With the block absent: GPU binaries, wire frames and config hash unchanged; existing `teacher_wire`, `lanes`, B06/B07 suites pass |

### 8.2 Matched smoke benchmarks

Same checkpoint and the same frozen jobs for every arm: **G48** (today's path), **C48** (CPU, 48 lanes, to isolate inference), **C64** (CPU, 64 pinned lanes). AB/BA order, at most two repeats, watchdog of at most 120 s per arm, failures recorded as failures.

| ID | Benchmark | Record |
| --- | --- | --- |
| B1 | Forward latency by candidate bucket at 1/8/16/32/64 pinned workers | p50, p95, session load, RSS, scaling |
| B2 | Teacher count mode (N=64 full, 16 fast), 4 tables | Trajectory cost; divergence from GPU |
| B3 | Teacher timed (4 s/K16, 0.8 s/K4), 2 tables | Trajectories, effective K, rounds, deadline hits, provenance mix, root and continuation labels |
| B4 | Ordinary, 24 frozen parents | Accepted-work wall, lane-seconds by phase; first-divergence index per game |
| B5 | Combined at C64: 8 tables plus 64 parents, overlap on/off | Wall, core usage, timed-search stability |

Tiny numerical differences between runtimes can change a sampled action, so real-model runs report the rate and position of divergence and make no claim of equality. Exact-equality claims rest on the scripted-logit runs (R4, R5).

search_v3 gates:
1. All count-mode golden traces identical.
2. In a matched timed smoke, completed rounds are up, deadline-cut rounds down, and no new fallback class appears.
3. Bit-exact tests pass for hot-path cleanup.
4. If waves are used, they pass their own matched comparison.
5. Local-game smokes pass.

### 8.3 Adoption criteria

Parity within tolerance; no RNG or binding failures; the first CPU iteration passes PPO likelihood verification and teacher logit checks; root-label yield at or above today's; deadline hits and fallbacks down; resources back to baseline after each round; rollback rehearsed. Speedups are reported as measured, and regressions are listed rather than hidden.

## 9. Risks

| Risk | Handling |
| --- | --- |
| Memory or L3 limits scaling at 64 workers | B1 first; session-cache cap; shared initializers only if needed; RAM confirmed in C0 |
| Teacher labels change character: about 10× more trajectories gives sharper targets, more continuation labels, a larger teacher gradient | Keep the fixed β; watch gradient share (warn band 0.125–0.5), KL guard and held-out KL. See D2 |
| ONNX vs PyTorch numeric drift | Existing cross-runtime tolerance; GPU `check_logits` on labels; PPO likelihood check on every row (`rtol 3e-3`) |
| Export time grows with pool churn | Cache by `model_id`, parallel historical export; a weight-patched template only if measured necessary |
| 64-process startup cost | Measure in C5; persistent pool or preload |
| Search head enters before CPU path ready | Support `w>0` from C3; do not enable CPU path during Stage B without R3 passing |
| Sealed-core and source-review friction | Additive binaries, reviewed source entries, no sealed-core edits in C0–C6 |

## 10. Decisions for you

- **D1. Teacher budgets on CPU.** Recommended: keep 4 s / 0.8 s for the first canary so the comparison is clean, then move to fixed-count budgets (for example N=256 full, N=48 fast), which cut teacher wall time and make labels independent of machine speed. The alternative is keeping time budgets and taking the quality gain only.
- **D2. Fixed β with stronger labels.** Recommended: keep β fixed for the first canary, consistent with your standing rule, and only reconsider after watching gradient share for several iterations.
- **D3. search_v3 go/no-go** after the gates in 8.2. The local game is patched only on a yes.

## 11. Adjacent findings (outside this plan)

- **PPG value phase rarely retains updates.** 12 of the 14 iterations checked in the exports since g1254 retained 0 steps, nearly always `heldout_loss_increased`. Worth reviewing under parameter tuning.
- **First-iteration overhead.** Iterations 1280 and 1289 spent 59 s and 94 s before ordinary collection, against the usual 12–13 s. Both look like fresh invocations registering historical checkpoints; this is unconfirmed.
- **Rating work.** Evaluation and rating cost 136–296 s in the iterations that run it. It keeps the GPU route here and could later use CPU inference.
- **The retained GPU path keeps the idle-calibration behaviour**, deliberately, so it stays a true comparison arm.

## External references

- [AMD EPYC 9335 product page](https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9335.html)
- [EPYC 9335 specs and benchmarks](https://technical.city/en/cpu/EPYC-9335)
- [L3 layout listing (4 × 32 MB)](https://www.regard.ru/product/735895/servernyi-processor-amd-epyc-9335-oem/description)
- [EPYC 9005 512-bit AVX-512 datapath (Phoronix)](https://www.phoronix.com/review/amd-epyc-9755-avx512)
- [ONNX Runtime session option keys](https://huggingface.co/csukuangfj/ios-onnxruntime/blob/main/1.18.0/onnxruntime.xcframework/Headers/onnxruntime_session_options_config_keys.h)
- [ONNX Runtime memory-mapped model commit](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/9d1492a44972f2d4b6415b49811db0034bae7f75)
