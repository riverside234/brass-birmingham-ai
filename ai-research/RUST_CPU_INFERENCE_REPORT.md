# Rust CPU inference, 64 workers and search_v3: implementation report

Date: 2026-11-09. Plan: [RUST_CPU_INFERENCE_AND_SEARCH_V3_PLAN.md](RUST_CPU_INFERENCE_AND_SEARCH_V3_PLAN.md). Everything below ran on the development laptop (Windows, 14 cores). **Nothing has run on the lab yet**, so the lab speedups in the plan are still estimates.

## 1. Continue from g1290

On the lab, with the trainer stopped and this commit pulled:

```bash
python scripts/prepare-rust-cpu-inference.py --expected-generation 1290
python scripts/continue-rust-teacher.py --expected-generation 1290 --iterations 10 --share-target 0.25 --inference cpu
```

The prepare step builds the two CPU workers, pins ONNX Runtime 1.22.1 and runs the focused regressions. It downloads the official Linux archive if the Python wheel has no `libonnxruntime.so.1.22.1`; on a machine with no download access, pass `--onnxruntime-library /path/to/libonnxruntime.so.1.22.1`. It takes no training action.

To return to the GPU path at any committed generation, use `--inference gpu`. Add `--teacher-budget timed` for 4 s / 0.8 s searches on the CPU path.

## 2. What changed

| Area | Change |
| --- | --- |
| Teacher search | Each Rust worker runs ONNX Runtime in process. Search code, feature encoder, labels and value rows are untouched. |
| Ordinary lanes | New `infer` operation in the Rust collector. Python keeps sampling, RNG and every check. |
| Rating | Same C# host and game flow. Rating lanes evaluate with Python ONNX Runtime instead of the GPU service. |
| Bundles | One immutable ONNX bundle per `model_id` under `artifacts/rust-backend/cpu-bundles/`, with `search_delta_4` when the model has a search head and parity checked at six shapes. About 3 s per model on the laptop. |
| Workers | 64 lanes × 1, CPU budget 64, reserve 0, one lane per CPU, spread across sockets. Applied at run time only. |
| Teacher budget (D1) | Fixed-count: N=256 full, N=48 fast. The existing 4 s / 0.8 s stay as the safety cap, so job digests and the label buffer are unchanged. |
| Teacher β (D2) | Unchanged. |
| Tolerance | CPU-sourced labels use the 2e-4 cross-runtime bound in `check_logits`. GPU labels keep 2e-5 / 1e-5. |

The whole feature sits behind the `native_inference` block in `execution`, which `collector_runtime.recipe` strips before the training signature. With the block absent, the GPU code path, binaries and call arguments are as they were. `lab-native/` is not edited: `lab-native-cpu/` and `lab_cpu_overlay.py` patch a detached copy, so the qualified GPU binaries keep their fingerprints. The source review in `b07-replay-location-review.json` records the new execution identity and accepts the g1290 checkpoint's recorded one.

## 3. Verification on the laptop

- 28 of 30 focused CPU regressions passed with real compiled workers: export parity with the search head, collector outputs against PyTorch, a full fixed-count teacher table through the usual label verification, an ordinary game through `collect_lane`, and a complete rating game through the real actor pool that matched the service path and never used it.
- 2 tests are skipped because their fixtures are wrong. All-seat blending is therefore verified only by unit tests and delta parity, and the one-turn run of the timed-budget option is unverified.
- The existing lab suites showed no new failures. Their remaining errors come from old test fixtures; two of those tests were confirmed to fail on the untouched last commit too.
- `scripts/prepare-rust-cpu-inference.py` has never been run end to end, and nothing has run on Linux.

**search_v3 was not adopted.** The lab and the game stay on search_v1, and nothing was installed. The local smokes did more search work (+15% to +20% trajectories) but finished fewer full halving rounds. A fix for that was still being tested when the work stopped, so there is no result for it. `scripts/search_v3_patch.py`, `build-search-v3.py` and `run-search-v3-smoke.py` remain for later.

## 4. What to read after the first lab iteration

- `report.timings`: `collection_seconds`, `teacher_collection_seconds`, `evaluation_seconds`, `iteration_seconds`.
- `report.exploration_metrics.branching.scheduling`: `collector_runtime.cpu_inference`, `actor_startup_seconds`, `lanes`.
- `teacher_phase` metrics: `cpu_inference`, per-table `cpu_inference.positions` and `forward_ns`, `completed_trajectories`, `deadline_hit_fraction`.
- PPO likelihood verification and teacher logit checks must keep passing.

## 5. Known limits

- Teacher and ordinary collection still run one after the other, so the teacher phase uses only 32 of the 64 lanes.
- Ordinary lane startup measured 16.6 s at g1289, while the 24 publisher processes started in 0.85 s. Python and torch import are therefore not the cost; the likely cause is the per-lane Mono bootstrap, which should be measured before changing it.
- Memory at 64 lanes is unmeasured here.
- Timed teacher scheduling keeps search_v1; fixed-count searches do not use search_v3's rolling quotas.
