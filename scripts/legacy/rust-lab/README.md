# Legacy Rust lab tools

These 40 scripts belong to the retired lab bundle workflows. Their original
transfer ZIPs have been removed. Use the recorded checkout and matching inputs
when reproducing historical evidence.

The [current Rust lab index](../../training/rust-lab/README.md) lists the live
CPU preparation, qualification, and shared runtime helpers.

## Bundle preparation, transfer, and recovery

- [prepare-rust-lab-b01.py](prepare-rust-lab-b01.py)
- [prepare-rust-lab-b01-transfer.py](prepare-rust-lab-b01-transfer.py)
- [prepare-rust-lab-b02.py](prepare-rust-lab-b02.py)
- [prepare-rust-lab-b025.py](prepare-rust-lab-b025.py)
- [prepare-rust-lab-b03.py](prepare-rust-lab-b03.py)
- [prepare-rust-lab-b04.py](prepare-rust-lab-b04.py)
- [prepare-rust-lab-b05.py](prepare-rust-lab-b05.py)
- [prepare-rust-lab-b06.py](prepare-rust-lab-b06.py)
- [prepare-rust-lab-b07.py](prepare-rust-lab-b07.py)
- [prepare-rust-lab-results.py](prepare-rust-lab-results.py)
- [restore-rust-lab-b03.py](restore-rust-lab-b03.py)
- [rust_lab_toolchain.py](rust_lab_toolchain.py)

## Launchers and qualifications

- [continue-rust-lab-b07.py](continue-rust-lab-b07.py)
- [manage-rust-lab-b07.py](manage-rust-lab-b07.py)
- [qualify-rust-lab-b025.py](qualify-rust-lab-b025.py)
- [qualify-rust-lab-b03.py](qualify-rust-lab-b03.py)
- [qualify-rust-lab-b04.py](qualify-rust-lab-b04.py)
- [qualify-rust-lab-b045.py](qualify-rust-lab-b045.py)
- [qualify-rust-lab-b05.py](qualify-rust-lab-b05.py)
- [qualify-rust-lab-b06.py](qualify-rust-lab-b06.py)
- [qualify-rust-lab-b07.py](qualify-rust-lab-b07.py)
- [run-rust-lab-qualification.py](run-rust-lab-qualification.py)
- [run-rust-lab-b01-b02.sh](run-rust-lab-b01-b02.sh)
- [run-rust-lab-b01.sh](run-rust-lab-b01.sh)
- [run-rust-lab-b02.sh](run-rust-lab-b02.sh)
- [run-rust-lab-b025.sh](run-rust-lab-b025.sh)
- [run-rust-lab-b03.sh](run-rust-lab-b03.sh)
- [run-rust-lab-b04.sh](run-rust-lab-b04.sh)

## B06 and B07 helpers

- [rust_lab_b06_ordinary.py](rust_lab_b06_ordinary.py)
- [rust_lab_b06_teacher.py](rust_lab_b06_teacher.py)
- [rust_lab_b07_publication.py](rust_lab_b07_publication.py)
- [rust_lab_b07_r3_publication.py](rust_lab_b07_r3_publication.py)
- [rust_lab_b07_r4_publication.py](rust_lab_b07_r4_publication.py)
- [rust_lab_b07_sealing.py](rust_lab_b07_sealing.py)
- [rust_lab_b07_smoke.py](rust_lab_b07_smoke.py)
- [rust_lab_b07_transport.py](rust_lab_b07_transport.py)
- [rust_lab_b07_v4.py](rust_lab_b07_v4.py)

## Generation-specific bundle builders

- [prepare-rust-lab-g370.py](prepare-rust-lab-g370.py)
- [prepare-rust-lab-g570.py](prepare-rust-lab-g570.py)
- [prepare-rust-lab-g1250.py](prepare-rust-lab-g1250.py)

The ordinary and teacher helper modules still supply functions to collector
profiling and verification tests. Those callers load this directory explicitly.
The compiler restore helper remains available to the retained B02 bootstrap
builder.
