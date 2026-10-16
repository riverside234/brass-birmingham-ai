# V11 ONNX Runtime CPU benchmark

Date: 2026-10-16. Measured initialized v11 utility and personality graphs on the same laptop and saved input fixtures used by the prior v9 deployment microbenchmark. ONNX Runtime was faster than PyTorch in every measured cell. The v11 measurements were substantially slower than the historical v9 timings; the exact architecture cost cannot be isolated from measurements collected at different times.

## Setup

- Windows 11; Intel Family 6 Model 170 (the previous report identifies this laptop as Core Ultra 5 135U).
- ONNX Runtime **1.22.1**, CPUExecutionProvider, FP32, opset 17; **two intra-op threads, one inter-op thread, sequential execution, intra-op spinning disabled**.
- PyTorch **2.8.0+cpu**, two intra-op threads and one inter-op thread.
- Exact prior `rail-action` / `settled-next-round` fixtures: **119 / 387 candidates**, respectively. Native rules and feature extraction ran outside timing.
- Batches **1, 2, 3, 4, 6, 8 and 16**, utility and personality graphs. Three warmups, then three alternating ORT/Torch blocks of seven measurements: **21 samples per backend/shape**.
- Two bounded runs: utility then personality (41.36 seconds), then personality then utility (31.58 seconds). The runs use identical initialized model identities and input hashes.
- No training, search campaign, production checkpoint changes or deployment package installation. The v9 model was not reconstructed; its recorded measurements are historical references.

Ranges below are the **two per-run medians**, not confidence intervals or the minimum/maximum individual call. Timings are milliseconds for the **entire batch**.

## Utility graph: 5,921,320 parameters

| Batch × candidates | Historical v9 ORT | V11 ORT median range | V11 Torch median range |
| --- | ---: | ---: | ---: |
| 1 × 119 | 0.37 | 2.75–5.68 | 11.00–29.51 |
| 1 × 387 | 0.57 | 3.85–8.47 | 11.57–26.84 |
| 2 × 119 | 0.67 | 4.03–7.28 | 13.87–24.56 |
| 2 × 387 | 0.91 | 6.07–8.33 | 15.42–23.59 |
| 4 × 119 | 1.03 | 5.52–9.66 | 15.51–30.60 |
| 4 × 387 | 1.63 | 9.67–16.79 | 19.27–32.95 |
| 8 × 119 | 1.50 | 9.00–11.68 | 17.98–33.09 |
| 8 × 387 | 3.07 | 20.81–22.41 | 27.42–35.84 |

## Personality graph: 6,023,137 parameters

| Batch × candidates | Historical v9 ORT | V11 ORT median range | V11 Torch median range |
| --- | ---: | ---: | ---: |
| 1 × 119 | 0.43 | 2.86–2.91 | 10.74–12.90 |
| 1 × 387 | 0.86 | 4.35–4.86 | 12.08–12.96 |
| 2 × 119 | 0.71 | 3.93–6.84 | 13.66–23.11 |
| 2 × 387 | 1.06 | 5.97–7.57 | 16.11–20.89 |
| 4 × 119 | 0.86 | 6.05–6.96 | 15.01–16.27 |
| 4 × 387 | 2.12 | 9.79–11.51 | 19.29–25.59 |
| 8 × 119 | 1.57 | 9.68–10.54 | 19.15–23.15 |
| 8 × 387 | 3.32 | 17.03–17.50 | 27.04–27.21 |

The utility graph showed pronounced order/time sensitivity: batch-one, 119-candidate ORT medians were 5.68 ms and 2.75 ms. Its corresponding Torch medians also changed from 29.51 ms to 11.00 ms. The personality graph was steadier at that shape (2.91 / 2.86 ms ORT). This makes a single exact slowdown factor against the old v9 run inappropriate. The paired ORT-versus-Torch result supports using ONNX Runtime for these CPU graphs; the measurements do not establish a universal deployment latency.

## Correctness and scope

- All four exports passed production named-output Torch/ORT parity, including legal-mask checks, at dynamic batch/candidate shapes **(1,1), (2,7), (4,31), (1,257)**.
- Every real-fixture timing shape also passed Torch/ORT output comparison before measurement.
- Maximum absolute difference across all exported outputs: **0.00002719**. Same-row ORT outputs alone versus batched were identical (maximum difference 0.0).
- Source fingerprints, saved v9 inputs and historical results remained unchanged. Both runs used identical initialized model identities.
- Initialized weights match the earlier v9 benchmark method. A trained export should be measured again before setting deployment latency expectations; these results do not show playing strength.
- Timing excludes feature extraction, native rules, inference queues/IPC, action sampling and search. Larger batches are measured per batch; divide by batch size only for a throughput calculation.
- No additional runtime optimization or model architecture changes were made in this measurement task.

## Artifacts and reproduction

- [First run, raw samples and parity](../artifacts/feature-engineering/mlp-v11-ort-20261016/report.json).
- [Reversed-order repeat, raw samples and parity](../artifacts/feature-engineering/mlp-v11-ort-repeat-20261016/report.json).
- [Reusable benchmark](../tests/FeatureEngineering/benchmark_ort_v11.py). Both artifact directories retain an executed-script copy.
- Both runs deleted only their own temporary ONNX files after successful measurement; hashes, sizes and deletion receipts remain in their JSON reports. Existing v11 smoke exports/checkpoints were preserved.

```bash
PYTHONPATH=src:tests/FeatureEngineering python -B tests/FeatureEngineering/benchmark_ort_v11.py \
  --output artifacts/feature-engineering/v11-ort-new-measurement --delete-models

PYTHONPATH=src:tests/FeatureEngineering python -B tests/FeatureEngineering/benchmark_ort_v11.py \
  --output artifacts/feature-engineering/v11-ort-new-repeat --reverse-head-order --delete-models
```

Use fresh output directories. The saved prior-v9 fixture/report directory must be present. See [the v11 model report](MLP_V11_REPORT.md) for architecture, training and dataset compatibility.
