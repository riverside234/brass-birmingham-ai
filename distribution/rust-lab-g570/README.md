# B01–B04 qualification at g570 (r5)

Updated 2026-11-03. **B01 passed on g570 / cuda:0. B02–B04 lab gates remain Pending. Production teacher remains off.**

The latest received attempt (`20261103T232335735601Z-20261103T232040081783Z-g570-all-fc1a5f74-025ddb53`)
passed B01, including real CUDA output parity on 13 roots and nine staged
transitions. B02 built/tested its Rust collector and passed 20 adapter tests,
then stopped because the reference-host C# compiler was missing.

**r5 bundles the pinned public Roslyn 4.12.0 package.** B02 restores it into its
new evidence directory, checks compiler startup, builds the bootstrap and tests
one action/checkpoint restoration before compiling Rust. No compiler download or
Windows developer cache is needed. B04 reuses that verified bootstrap.

The [B01–B04 review](../../ai-research/RUST_BACKEND_B01_B04_G570_REVIEW.md) also
documents the receipt-reuse and timeout-cleanup fixes. Trainer, native core,
checkpoint, recipe and source approvals are unchanged. The earlier native-input
and Python-path fixes remain in place.

## Continue from the passing B01

Pull the matching checkout, including added files and deletions. Keep training
paused at the same clean g570 boundary. From the repository root:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-g570/rust-lab-b01-b04-g570-r5.zip
python scripts/run-rust-lab-qualification.py --stage all --start-at b02 --device cuda:0 \
  --expected-generation 570 \
  --checkpoint-sha256 06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9 \
  --b01 artifacts/rust-backend/20261103T232040081783Z-g570-all-fc1a5f74/b01/qualification.json
```

This runs B02, B02.5, B03 and B04. It verifies the original B01 receipt, native
binary/source, GPU receipt, executable source/runtime inventory and unchanged
run/checkpoint/population. Harness-only repairs may reuse B01; changed executable
inputs or training state require a new B01. A failed/deferred stage still stops
the chain and exports its evidence. The Git-visible review copy is not a
replacement for the original lab receipts and build products.

## Start again or qualify a later generation

If the original evidence is missing, rerun the chain:

```bash
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0 \
  --expected-generation 570 \
  --checkpoint-sha256 06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9
```

If training has advanced, capture the actual latest committed state instead:

```bash
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0
```

The default run is `artifacts/step5/joint7-v11-2-g1290-trained-restart`. The default
config is the [exact recorded g570 config](../../results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z/recorded-config.json),
which matches the g370 recipe. The launcher captures generation/hash/state once
and rejects advancement. Explicit pins assert the latest state; they never rewind
training. If a different reviewed config was adopted, supply that exact config
with `--config`. Qualification does not adopt a recipe, change production state
or run a training iteration. B03 restores the full checkpoint into an isolated fork.

## Individual gates and evidence

```bash
python scripts/run-rust-lab-qualification.py --stage b01 --device cuda:0 --expected-generation 570
python scripts/run-rust-lab-qualification.py --stage b02 --device cuda:0 --expected-generation 570 --b01 ORIGINAL_B01_QUALIFICATION_JSON
python scripts/run-rust-lab-qualification.py --stage b025 --device cuda:0 --expected-generation 570
python scripts/run-rust-lab-qualification.py --stage b03 --device cuda:0 --expected-generation 570 --b02 ORIGINAL_B02_QUALIFICATION_JSON
python scripts/run-rust-lab-qualification.py --stage b04 --device cuda:0 --expected-generation 570 \
  --b02 ORIGINAL_B02_QUALIFICATION_JSON --b03 ORIGINAL_B03_QUALIFICATION_JSON
```

Each attempt exports a new Git-visible folder under `results/rust-lab-qualification/`.
Open `evidence/runner.json` first, push the folder and pull it locally for review.
Retain original `artifacts/rust-backend/...` receipts on the lab for subsequent
gates. The bundle never overwrites original result folders.

Search remains task-only `search_v1`, K16, four seconds of search plus at most
one second total post-search work. B04 uses two independent one-thread sessions,
two shadow tables and bounded typical/complex root probes. DOMiNO/AZ_DB and
search_v2/V205 remain disabled. Current-reward utility readiness still gates B04.

Prerequisites: existing CUDA Python environment, Rust 1.99.1, Linux linker, Mono,
admitted private game runtimes, full checkpoint, retained PPG caches and frozen
opponent/seed references. The [public compiler](../rust-lab-toolchain/README.md)
is included; checkpoint tensors and proprietary game binaries are not.

Extract only the current supplement over its matching checkout. Original
g570/r2/r3/r4 and six stage ZIPs stay immutable; current `bundle.json` and
`transfer-manifest.json` select r5. `local-development.json` records short local
checks, and `package-check.json` records fresh extraction verification. These do
not certify B02–B04 Linux/CUDA execution or B05–B07 production adoption.
