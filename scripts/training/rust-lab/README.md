# Rust lab setup and validation

The CPU worker preparation and qualification commands use the shared
helpers listed below. Historical bundle tools are indexed under legacy.
See [training](../README.md) for the current continuation entry points.

Reference corpus producers live in `reference-capture/`, fixture preparation
in `fixtures/`, and milestone checks in `validation/`. Shared functions still
needed by current lab and local-game commands keep the top-level paths below.

## CPU inference and collector setup

- [prepare-rust-cpu-inference.py](../../prepare-rust-cpu-inference.py)
- [qualify-rust-cpu-inference.py](../../qualify-rust-cpu-inference.py)
- [prepare-rust-collector-performance.py](../../prepare-rust-collector-performance.py)
- [prepare-rust-teacher-verification.py](../../prepare-rust-teacher-verification.py)
- [prepare-rust-post-transfer-reward.py](../../prepare-rust-post-transfer-reward.py)

## Shared build, qualification, and export helpers

| Helper | Current use |
| --- | --- |
| [qualify-rust-lab.py](../../qualify-rust-lab.py) | CPU preparation environment, native test inputs, and bounded commands. |
| [qualify-rust-lab-b02.py](../../qualify-rust-lab-b02.py) | CPU build dependency checks and historical bootstrap support. |
| [rust_lab_results.py](../../rust_lab_results.py) | Continuation and qualification evidence export. |
| [rust_lab_b06_metrics.py](../../rust_lab_b06_metrics.py) | CPU qualification and profiling measurements. |
| [rust_lab_b07_retry.py](../../rust_lab_b07_retry.py) | Collector, teacher-verification, and reward runtime preparation. |
| [rust_lab_b07_completion.py](../../rust_lab_b07_completion.py) | Verification of reusable runtime build receipts. |

## Retired lab bundles

The [legacy Rust lab tools](../../legacy/rust-lab/README.md) group the former
bundle publishers, B01–B07 launchers, and historical qualification helpers.
Current CPU preparation still reads `distribution/rust-lab-g1250/inputs.zip`.

## Rust backend inputs and fixtures

- [capture-rust-privacy.py](reference-capture/capture-rust-privacy.py)
- [capture-rust-privacy-edges.py](reference-capture/capture-rust-privacy-edges.py)
- [capture-rust-rule-branches.py](reference-capture/capture-rust-rule-branches.py)
- [capture-rust-rules.py](../../capture-rust-rules.py)
- [capture-rust-state-regressions.py](reference-capture/capture-rust-state-regressions.py)
- [capture-rust-states.py](reference-capture/capture-rust-states.py)
- [complete-rust-state-capture.py](reference-capture/complete-rust-state-capture.py)
- [prepare-rust-feature-layout.py](fixtures/prepare-rust-feature-layout.py)
- [prepare-rust-inference-fixtures.py](fixtures/prepare-rust-inference-fixtures.py)
- [prepare-rust-onnx.py](fixtures/prepare-rust-onnx.py)
- [prepare-rust-state-catalog.py](fixtures/prepare-rust-state-catalog.py)

## Rust backend validation

- [qualify-rust-backend.py](validation/qualify-rust-backend.py)
- [qualify-rust-coordinator.py](validation/qualify-rust-coordinator.py)
- [qualify-rust-features.py](validation/qualify-rust-features.py)
- [qualify-rust-inference.py](validation/qualify-rust-inference.py)
- [qualify-rust-privacy.py](validation/qualify-rust-privacy.py)
- [qualify-rust-rules.py](validation/qualify-rust-rules.py)
- [qualify-rust-scaffold.py](../../qualify-rust-scaffold.py)
- [qualify-rust-search.py](validation/qualify-rust-search.py)
- [qualify-rust-sessions.py](validation/qualify-rust-sessions.py)
- [qualify-rust-simulation.py](validation/qualify-rust-simulation.py)
- [qualify-rust-state.py](validation/qualify-rust-state.py)
- [replay-native-failure.py](../../replay-native-failure.py)
- [verify-rust-coordinator.py](validation/verify-rust-coordinator.py)
- [verify-rust-features.py](validation/verify-rust-features.py)
- [verify-rust-inference.py](validation/verify-rust-inference.py)
- [verify-rust-search.py](validation/verify-rust-search.py)
- [verify-rust-sessions.py](../../verify-rust-sessions.py)
