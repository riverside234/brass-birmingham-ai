# Self-play input reuse and streaming inference queues

Implemented the three requested performance changes against commit `f73471ab6ce4815ce1e3058d888a6fbfc2abd564` (implementation `e0dc28ad559ddcd2d6b0eff95d1dac61c4ede949f70bf341a16d08e2ca71dca3`). Existing training configurations, model/feature identities and checkpoint files are unchanged. No production training or evaluation was launched.

## Changes

1. **Fixed PPO categorical probe inputs.** Store the collated inputs, legal masks, exploration distributions, indices and old policy probabilities for the selected probe microbatches. The device cache is capped at **256 MiB**, further limited on CUDA to one eighth of currently free VRAM. CUDA overflow can use a separate **256 MiB pinned-host** budget; remaining parts use normal packing. CPU uses a single 256 MiB cache. Current policy outputs are always recomputed, including exact categorical KL after **every optimizer step**. Probe positions, masking, temperatures, epsilon mixing and the early-stop threshold are unchanged.
2. **Repeated scan/admission reuse.** Dataset-local SHA-256 content proofs reuse successful selector admission; shape/layout validation remains active. Repeated probability-measurement scans reuse owned packed CPU/pinned buffers up to **128 MiB / 256 entries** per Dataset. The one-time full-row verification does not occupy that cache, preserving its space for repeated nonforced actor scans. A bounded prefix survives scans that are larger than the cache. Random PPO minibatches retain their existing ordering and candidate bucketing. Modified input arrays, distributions or layouts cannot reuse an old packed entry. Source replay/freshness checks and the strict full-row behavior-probability verification remain active. Temporary state-only slices use weak references so admission proofs do not keep them alive.
3. **Streaming collection inference queues.** Between model forwards, drain already-ready envelopes into separate frozen-model queues, without adding another flush delay. Prefer a fuller compatible queue while requests are young; service near deadlines first and use waiting age after 20 ms to prevent starvation. Inspect every queued request for deadline urgency, including younger requests behind an older model head. Candidate-size grouping prevents a pathological long choice list from padding an entire batch; every legal candidate and each observer/model identity remain intact. A multi-model envelope gets one reply only after all its frames complete. Cancellation, deadlines, shared-slot ownership, failure handling and multi-replica accounting remain enforced. Configured workers, flush window, opponent sampling and game counts are unchanged.

Caches last for the Dataset/probe, are not checkpointed, and never retain network predictions across optimizer steps or model changes. Reports and compact history include `actor.input_cache`; collection telemetry includes per-collection batching deltas and a clearly labelled cumulative queue peak. Scheduler dispatch counts include stopped-frame retirement; actual GPU/CPU forward counts remain in `inference_batches`. Existing `mean_inference_batch`, padding and collection/PPO timing remain directly available. No LR, `target_kl`, entropy, reward, GAE, PPG, rating, teacher or opponent settings were changed.

Files: `recovery/{batch_cache,inference_batches,inference,review_metrics,updates,telemetry,history}.py`, the optional audited path in `learning/batching.py`, exact source/handoff approvals, regression fixtures/tests, the isolated benchmark/smoke scripts and lab guide. Public/untrusted collation keeps its full selector validation. Several old tests used synthetic link endpoints incompatible with the current v11 topology; affected inference/value tests now use the existing native-topology fixture with three legal choices, without weakening production validation.

## Bounded measurements

The final CPU benchmark uses fresh **1,511,464-parameter small UtilityNet** weights, 2 PyTorch threads, **2,048 authored positions** with 2–32 candidates, and three timed repetitions. Baseline code is loaded directly from the Git commit above. The policy is deliberately perturbed: exact probe KL is approximately **0.01355**; cached and baseline computations agree within 1e-7. This is a computation benchmark, not a measured improvement in playing strength.

| Measured path | Baseline median | Revised median | Time reduction |
| --- | ---: | ---: | ---: |
| One 2,048-position KL probe pass | 2.0941 s | 2.0875 s | **0.3%** |
| One repeated probability scan | 2.1425 s | 1.8841 s | **12.1%** |
| Synthetic concurrent inference stream | 3.6077 s | 3.4374 s | **4.7%** |

The inference stream has **16 concurrent lanes × 24 decisions = 384 positions**, four distinct frozen policies, inline transport and the current **1 ms** flush setting. Three trials alternate baseline/revised order. The median runs use **121 → 101 forwards**, mean batch **3.17 → 3.80**, and padding **33.7% → 19.9%**. Wall times are variable: paired baseline/revised trials were 3.920/2.531, 3.608/3.437 and 3.444/3.676 seconds. One pair regressed; these CPU results do not establish a stable production collection speedup.

Probe inputs occupy **64,816,128 bytes**, with all 16 microbatches cached and no repacking on measured hits. Repeated-scan buffers occupy **64,697,344 bytes** under the 128 MiB cap. Earlier short CPU measurements ranged from roughly 0–12% probe time reduction and showed substantial inference timing variation. The final table is the longer, fixed-workload result; it must not be presented as an end-to-end GPU gain. The local runtime is **PyTorch 2.8.0+cpu**; CUDA residency/fallback tests are included but skipped here because CUDA is unavailable.

Artifacts: `artifacts/selfplay-speed-review/cpu-benchmark-stable/report.json` and `benchmark-stable.log`. The benchmark validates route identity, legal-output lengths, normalization and numerical agreement; the regression suite additionally checks mutation rejection, per-row opening temperatures, zero-budget fallback, oversized positions, cancellation/expiry, exactly one reply, fairness and exact PPO weights/Adam/RNG.

The final focused suite ran **100 tests in 109.246 s: 98 passed, two CUDA-only cases skipped**. It covers unchanged PPO weights/Adam/RNG with caches on/off, strict behavior checking, probe and scan limits/fallbacks, current epsilon-0.02/per-row temperature-2 behavior, old source receipt preservation, pending collection refusal, all four observer mappings, mixed-model/shared/inline replies, failure handling and CPU-backed multi-replica routing. See `artifacts/selfplay-speed-review/final-tests-2.log`.

The isolated smoke uses the current **5,928,616-parameter medium UtilityNet**, 64 authored rows, two PPO steps, and a repeat from its saved checkpoint. It passes strict behavior verification and exact categorical guards, produces identical weights/Adam/RNG after restart, and leaves the incoming checkpoint unchanged. The final run took **17.14 s** on CPU. It uses synthetic terminal labels and authored epsilon-0.08/temperature-1 behavior, while the dedicated regressions also cover the current epsilon-0.02/opening-temperature-2 mixture. It does not simulate complete games. See `artifacts/selfplay-speed-review/smoke-final/report.json`.

All **14,624 prior source approvals** retain their original expanded contents and order; 3,953 exact continuation routes are appended. The learning compatibility identity remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. Checkpoint architecture admission is unchanged; archived v11.1 source records are retained as historical evidence rather than relabelled as compatible with v11.2.

## Continuation and lab measurement

Reviewed same-config source transitions retain previous exact migration chains and completed-IL compatibility. Existing runs use their current config, run and IL paths with `--resume-reviewed-source`. The caches/queues need **no configuration regeneration or hotfix script**. Finish/recover a pending old-source collection before changing source; the existing resume guard refuses to mix it into a new collection.

For a run that already adopted entropy95:

```bash
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-entropy95.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

To measure the input/inference paths on the actual lab GPU in a **new** output directory:

```bash
PYTHONPATH=src:tests/Recovery:tests/FeatureEngineering:tests/R2 \
python -B tests/Recovery/benchmark_selfplay_inputs.py \
  --baseline-ref f73471ab6ce4815ce1e3058d888a6fbfc2abd564 \
  --device cuda:0 --output artifacts/selfplay-input-cache-gpu-benchmark
```

This benchmark uses fresh weights and does not open existing runs. Production collection includes native simulation, validation and shared transport; its whole-iteration benefit must be assessed separately from this inference-only workload. FP32 precision, training batches, two epochs and per-step KL checking remain unchanged.
