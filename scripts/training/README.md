# Training

See [migrations](migrations/README.md) for historical recipe changes and
[rust-lab](rust-lab/README.md) for current Rust workers and checks.
The migration folder contains 56 historical recipe and restart commands.
The 40 retired bundle tools live in [legacy Rust lab](../legacy/rust-lab/README.md).
Use the [command index](../README.md#common-commands) for current entry points.

The [further migration plan](../../ai-research/migration-plans/FURTHER_SCRIPT_MIGRATION_PLAN.md)
records the completed grouping of imitation tools, opponent-bank maintenance,
training checks, reference-host commands, and checkpoint tools. Current
continuation, transfer, and source-bound runtime builders retain their paths.

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
- [export-catalog.py](rules-host/export-catalog.py)
- [define-r1-contracts.py](rules-host/define-r1-contracts.py)
- [verify-r0.py](rules-host/verify-r0.py)
- [verify-r1.py](rules-host/verify-r1.py)
- [test-r1.py](rules-host/test-r1.py)
- [check-r1-traces.py](rules-host/check-r1-traces.py)
- [build-host.ps1](rules-host/build-host.ps1)
- [check-r1-runtime.ps1](rules-host/check-r1-runtime.ps1)
- [smoke-host.py](rules-host/smoke-host.py)

## Imitation learning and model training

- [r2.py](imitation/r2.py)
- [feature-engineering.py](imitation/feature-engineering.py)
- [il_replay.py](imitation/il_replay.py)
- [smoke-s1-training.py](imitation/smoke-s1-training.py)
- [verify-s1-training.py](imitation/verify-s1-training.py)
- [verify-r2.py](imitation/verify-r2.py)

## Training checks

- [check-step5-distribution.py](checks/check-step5-distribution.py)
- [check-step5-evaluation.py](checks/check-step5-evaluation.py)
- [check-step5-scaling.py](checks/check-step5-scaling.py)
- [smoke-dcpg.py](checks/smoke-dcpg.py)
- [smoke-local-scoring.py](checks/smoke-local-scoring.py)
- [smoke-ppg-throughput.py](checks/smoke-ppg-throughput.py)
- [smoke-ppg-phasic.py](checks/smoke-ppg-phasic.py)
- [verify-recovery.py](checks/verify-recovery.py)
- [verify-step5.py](checks/verify-step5.py)

## Checkpoint export and calibration

- [export-test-nn.py](checkpoints/export-test-nn.py)
- [test_nn_calibration.py](checkpoints/test_nn_calibration.py)

The exporter and calibration helper remain siblings. Frozen Rust reference
verification supports both the original and relocated exporter provenance.

## Historical opponents and legacy banks

- [prepare-legacy-bank.py](opponent-banks/prepare-legacy-bank.py)
- [freeze-legacy-bank.py](opponent-banks/freeze-legacy-bank.py)
- [collect-legacy-bank.py](opponent-banks/collect-legacy-bank.py)
- [check-frozen-bank.py](opponent-banks/check-frozen-bank.py)
- [check-legacy-admission.py](opponent-banks/check-legacy-admission.py)
- [sample-legacy-pool.py](opponent-banks/sample-legacy-pool.py)
- [qualify-legacy-bank.py](opponent-banks/qualify-legacy-bank.py)
- [amend-v3-qualification.py](opponent-banks/amend-v3-qualification.py)
- [promote-v3-bank.py](opponent-banks/promote-v3-bank.py)
- [verify-v3-bank.py](opponent-banks/verify-v3-bank.py)
- [verify-r3.py](opponent-banks/verify-r3.py)
- [capture-h0.ps1](opponent-banks/capture-h0.ps1)
