# Training

See [migrations](migrations/README.md) for historical recipe changes and
[rust-lab](rust-lab/README.md) for current Rust workers and checks.
The migration folder contains 54 historical recipe and restart commands.
The 40 retired bundle tools live in [legacy Rust lab](../legacy/rust-lab/README.md).
Use the [command index](../README.md#common-commands) for current entry points.

## Step 5 and current continuations

- [step5.py](../step5.py)
- [continue-rust-g1190-restart-2.py](../continue-rust-g1190-restart-2.py)
- [continue-rust-g1491.py](../continue-rust-g1491.py)
- [continue-rust-shadow.py](../continue-rust-shadow.py)
- [continue-rust-teacher.py](../continue-rust-teacher.py)
- [recover-step5-iteration.py](../recover-step5-iteration.py)
- [resolve_checkpoint.py](../resolve_checkpoint.py)
- [joint7_preflight.py](../joint7_preflight.py)

## Machine transfer

- [step5-transfer.py](../step5-transfer.py)
- [step5_transfer_frozen.py](../step5_transfer_frozen.py)
- [step5_transfer_restore.py](../step5_transfer_restore.py)
- [step5_transfer_runtime.py](../step5_transfer_runtime.py)
- [check-transferred-legacy.py](../check-transferred-legacy.py)

## Rules hosts and contracts

- [build-training-host.py](../build-training-host.py)
- [build-search-host.py](../build-search-host.py)
- [build-shared-search.py](../build-shared-search.py)
- [build-local-scoring-host.py](../build-local-scoring-host.py)
- [export-catalog.py](../export-catalog.py)
- [define-r1-contracts.py](../define-r1-contracts.py)
- [verify-r0.py](../verify-r0.py)
- [verify-r1.py](../verify-r1.py)
- [test-r1.py](../test-r1.py)
- [check-r1-traces.py](../check-r1-traces.py)
- [build-host.ps1](../build-host.ps1)
- [check-r1-runtime.ps1](../check-r1-runtime.ps1)
- [smoke-host.py](../smoke-host.py)

## Imitation learning and model training

- [r2.py](../r2.py)
- [feature-engineering.py](../feature-engineering.py)
- [il_replay.py](../il_replay.py)
- [smoke-s1-training.py](../smoke-s1-training.py)
- [verify-s1-training.py](../verify-s1-training.py)
- [verify-r2.py](../verify-r2.py)

## Training checks

- [check-step5-distribution.py](../check-step5-distribution.py)
- [check-step5-evaluation.py](../check-step5-evaluation.py)
- [check-step5-scaling.py](../check-step5-scaling.py)
- [smoke-dcpg.py](../smoke-dcpg.py)
- [smoke-local-scoring.py](../smoke-local-scoring.py)
- [smoke-ppg-throughput.py](../smoke-ppg-throughput.py)
- [smoke-ppg-phasic.py](../smoke-ppg-phasic.py)
- [verify-recovery.py](../verify-recovery.py)
- [verify-step5.py](../verify-step5.py)

## Historical opponents and legacy banks

- [prepare-legacy-bank.py](../prepare-legacy-bank.py)
- [freeze-legacy-bank.py](../freeze-legacy-bank.py)
- [collect-legacy-bank.py](../collect-legacy-bank.py)
- [check-frozen-bank.py](../check-frozen-bank.py)
- [check-legacy-admission.py](../check-legacy-admission.py)
- [sample-legacy-pool.py](../sample-legacy-pool.py)
- [qualify-legacy-bank.py](../qualify-legacy-bank.py)
- [amend-v3-qualification.py](../amend-v3-qualification.py)
- [promote-v3-bank.py](../promote-v3-bank.py)
- [verify-v3-bank.py](../verify-v3-bank.py)
- [verify-r3.py](../verify-r3.py)
- [capture-h0.ps1](../capture-h0.ps1)
