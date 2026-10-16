# Shared-state policy scorer factorization

Implemented on 2026-10-15. The optimization is automatic in base/utility/personality forwards, ordinary and teacher training, and normal/oversized inference. It does not change the model layout or training recipe. Completed compatible v9 IL is reusable without retraining. Existing ONNX files need re-exporting to include the new graph.

## Computation and preserved tensors

The first scorer layer still owns one `policy_scorer.0.weight` of shape `[128,432]` and one bias. Its existing columns are used as views:

```text
shared = Linear(state_context, weight[:, :304], bias)  # once per position
candidate = Linear(candidate_embedding, weight[:, 304:])
logit = final_linear(GELU(candidate + shared[:, None]))
```

The contributions are added before the original GELU. There are no new Parameters, reordered optimizer slots, persistent activation caches or detached gradients. Oversized candidate inference computes `shared` once outside the chunk loop. Personality scoring projects the context after the personality conditioning, preserving its gradients. Legal masks, stable-seat outputs, probabilistic sampling and behavior-probability verification are unchanged.

Base/utility/personality parameter counts remain **2,005,917 / 2,015,937 / 2,081,770**. For the width-304 models, the candidate work changes from 97,280 to **58,368 MACs**, adding 38,912 MACs once per position. Savings are `38,912 * (padded_candidates - 1)`. New base/utility/personality fixed work is **1,938,080 / 1,948,064 / 2,013,712 MACs**. The counter excludes nonlinearities, masking, encoding, transport and search. A single-candidate position saves no MACs and can incur extra dispatch cost.

Both formulations have the same real-number algebra. Floating-point accumulation order changes, particularly under autocast; bit-identical logits or future training trajectories are not promised. The largest float32 logit difference in the inference benchmark was **2.0862e-7**. Unchanged value/goal heads were also checked independently.

## Checkpoint and source admission

The model-source fingerprint was previously also its checkpoint architecture identity. `configs/learning/architecture-source-equivalence.json` maps only the exact reviewed factored source digest to the original v9 architecture digest. Unknown model/layout source edits still produce a different identity and reject old checkpoints. Architecture/layout/feature validation, strict state-dict loading, weight digests and optimizer/RNG restoration remain active. The full implementation fingerprint includes the new code and equivalence receipt, so collection/resume freshness remains source-bound.

- Implementation: `67174f91c5fe6f5e8c087949946a98a065f6a3f2e8e441f8681e7fd3812c14cd`.
- Preserved learning compatibility: `747216b5ab873126a6e048ade198883593113f70a815b9012ce40b8f66d443cb`.
- Preserved architecture fingerprint: `9c5ad1e2a20f7dfb0483e04ecbf7b77b6d1c6808641283e1a5e72b23a3bb2173`.
- Exact optimized model/layout source digest: `0b88759d9e988dddb27e03b340eb2b7fe6d831205c21466ae7b9bb4d59c324f9`.

Completed IL from the preceding default-enabled source `d3667950...` is explicitly admitted. Earlier reviewed v9 IL handoffs and self-play transitions retain their restrictions. Existing committed self-play uses its normal command with `--resume-reviewed-source` once; old pending collections still block the transition. The existing source-change utility-readiness safeguard remains active. A separate exact personality-source migration preserves its checkpoint tensors and Adam/RNG while recording its new implementation identity. No unfinished IL is re-signed and no production checkpoint was edited.

## Validation

**52 tests passed, with one CUDA BF16 test skipped because this host has no CUDA.** Coverage includes:

- Float32 forward parity for all three models, legal masks, forced choices, candidate permutations and unchanged value/goal outputs.
- Float64/float32 gradients for both scorer weight slices, bias, candidate embeddings and state context; small and medium presets; one and many candidates.
- CPU BF16 finite gradients and bounded numerical differences; this does not substitute for CUDA validation.
- Exactly one state projection across large-candidate chunks; normal and packed inference consistency.
- Native-board fixture likelihood collection/reconstruction and a bounded PPO update, including rejection of deliberately stale probabilities.
- Dynamic ONNX parity for base/utility/personality, including single-candidate tracing followed by larger dynamic batches.
- Checkpoint/config/layout admission, source-review restrictions, pending-work rejection and completed IL handoff.
- Existing teacher auxiliary update/calibration/diagnostic parity and the bounded IL loader smoke.

Checkpoints were also saved **before editing the model** and then loaded after the refactor. Base, utility and personality retained exact state tensors, Adam state and RNG; their original files were unchanged. The completed IL fixture retained its exact origin and file hashes. Those are small authored checkpoints, not the user's production IL.

Evidence is under `artifacts/feature-engineering/factored-scorer-20261015/`: `unit-tests.txt`, `integration-tests.txt`, `training-tests.txt`, `checkpoint-smoke.json` and `verification.json`. The first integration command included a nonexistent unittest class selector; its 30 actual tests passed, and the corrected compaction test passed in `training-tests.txt`. No training/evaluation campaign was launched.

## Bounded CPU benchmark

Windows 11, Intel64 Family 6 Model 170 Stepping 4, PyTorch 2.8.0 CPU, ONNX Runtime 1.22.1, two threads. The comparison used the same untrained width-304 utility weights, pre-encoded inputs, eight warm-up calls per mode, nine alternating rounds and 20 calls per round. Each reported median/p95 has 180 measured calls per mode. Reference scoring uses the original concatenated calculation with the same Parameters. No hardware settings were changed.

| Mode | Positions × padded candidates | Original median ms | Factored median ms | Median reduction |
| --- | ---: | ---: | ---: | ---: |
| PyTorch forward | 1 × 1 | 1.7128 | 1.7434 | -1.8% |
| PyTorch forward | 1 × 8 | 2.5038 | 2.1988 | 12.2% |
| PyTorch forward | 1 × 32 | 1.9009 | 1.8976 | 0.2% |
| PyTorch forward | 1 × 128 | 1.9920 | 1.9354 | 2.8% |
| PyTorch forward | 1 × 512 | 2.5758 | 2.2653 | 12.1% |
| PyTorch forward | 8 × 32 | 5.1110 | 3.9989 | 21.8% |
| PyTorch forward | 8 × 128 | 6.2981 | 3.8965 | 38.1% |
| PyTorch forward | 32 × 128 | 14.4071 | 10.6365 | 26.2% |
| ONNX forward | 1 × 1 | 0.4106 | 0.4246 | -3.4% |
| ONNX forward | 1 × 32 | 0.6073 | 0.5713 | 5.9% |
| ONNX forward | 1 × 128 | 0.9354 | 0.7933 | 15.2% |
| ONNX forward | 1 × 512 | 1.4366 | 1.3935 | 3.0% |
| ONNX forward | 8 × 128 | 2.3540 | 1.8801 | 20.1% |
| PyTorch forward/backward | 8 × 32 | 13.3983 | 14.4550 | -7.9% |
| PyTorch forward/backward | 8 × 128 | 18.6052 | 17.3591 | 6.7% |

The separate corpus sweep visits 14 authored native-board positions with **1–387** legal candidates, covering Canal/Rail action and continuation phases. Median time for the complete pre-encoded sweep was **45.7581 → 45.3578 ms**, just 0.9% lower. It does not establish a meaningful aggregate gain: round medians varied substantially. The JSON includes per-round medians, p95 and paired deltas; e.g. the 1×8 pooled median looks better while its median paired-round delta is slightly slower. Treat the small or inconsistent differences as noise rather than a guaranteed speedup.

The larger inference batches show useful reductions, but smaller training batches can regress. Forward/backward excludes optimizer stepping and uses a bounded synthetic loss. These measurements exclude feature encoding, queueing, device transfer, native gameplay and search. **GPU and full-iteration speedups are unmeasured. No playing-strength improvement is claimed.**

Reproduce in a new isolated output directory:

```bash
PYTHONPATH=src:tests/FeatureEngineering python tests/FeatureEngineering/benchmark_factored_scorer.py \
  --output artifacts/feature-engineering/factored-scorer-local --rounds 9 --calls 20
```

## Starting self-play

No new optimization flag is needed. After syncing all changed/new files, use the [fresh self-play instructions](../docs/FEATURE_ENGINEERING_LAB.md) with your already-completed IL and generated config. Teacher auxiliary value remains enabled in new default configs; clocks remain 9/12/12/12 seconds, five teacher games, eight search workers per game, eight distillation steps and a null cumulative wall fraction. PPO, rewards, exploration, model width/depth and feature encoding remain unchanged.

For the example generated config and an unused self-play directory:

```bash
python scripts/step5.py \
  --config configs/step5/joint7-features4-compact-v9-batch7-teacher-value.json train --fresh \
  --run artifacts/step5/joint7-features4-compact-v9 \
  --imitation-run artifacts/step5/features4-compact-v9-batch7-il \
  --iterations 1
```

Use your actual paths. If the run already has committed iterations, omit `--fresh` and add `--resume-reviewed-source` for this source transition. Existing exports retain their original computation until re-exported.

## Files changed for this optimization

- `learning/model.py`, `utility_model.py`, `batching.py` and `personality/model.py`: factor and reuse the scorer state contribution; correct dense MAC accounting.
- `learning/contracts.py`, `configs/learning/architecture-source-equivalence.json`, `step5/league.py`: exact reviewed source equivalence and full implementation binding.
- Completed-IL/self-play review receipts and personality contract/migration receipt: preserve compatible handoffs and committed continuation with existing guards.
- New scorer/regression/resume tests and benchmark, plus updated MAC assertions in grouped/compaction tests.
- This report, the active grouped-model contract and lab instructions. The separate user-authored architecture-comparison plan was not changed.
