# Benchmarks

Measure collection, search, inference, or matchup performance. See the
[Rust and lab collection performance plan](../../docs/RUST_AND_LAB_COLLECTION_PERFORMANCE_PLAN.md)
for the Windows profile and local collection probe.

## Training throughput and lab collection

- [benchmark-feature-engineering.py](benchmark-feature-engineering.py)
- [benchmark-step5-performance.py](benchmark-step5-performance.py)
- [profile-rust-collection-local.py](profile-rust-collection-local.py)
- [profile-rust-collector-performance.py](profile-rust-collector-performance.py)

## Rust backend and local inference

- [profile-rust-windows.py](profile-rust-windows.py)
- [profile-onnx.py](profile-onnx.py)

## Search controls and matchups

- [benchmark-search-v2.py](benchmark-search-v2.py)
- [benchmark-search-v205.py](benchmark-search-v205.py)
- [benchmark-v3-vs-hard.py](benchmark-v3-vs-hard.py)
- [seal-search-v205.py](seal-search-v205.py)
- [run-nn-k-tournament.py](run-nn-k-tournament.py)

## Rust qualification studies

- [benchmark-rust-backend.py](benchmark-rust-backend.py) — L14 source archive and host-delivery qualification.
- [benchmark-rust-coordinator.py](benchmark-rust-coordinator.py) — L12 qualification.
- [benchmark-rust-inference.py](benchmark-rust-inference.py) — L09 qualification.
- [benchmark-rust-providers.py](benchmark-rust-providers.py) — L14.5 provider qualification.
- [profile-rust-simulation.py](profile-rust-simulation.py) — L07 qualification.
- [l14-search-controls.py](l14-search-controls.py) — L14 pinned runner and its qualification callers.
- [l14-smoke.py](l14-smoke.py) — L14 source archive.
- [seal-l14-smoke.py](seal-l14-smoke.py) — L14 source archive.

## Retained release source

- [measure-mono-allocation.py](../measure-mono-allocation.py) — Local Rust release source pin.

New study source archives use the current script paths. Sealed historical
receipts and archives retain their original member names and hashes.
