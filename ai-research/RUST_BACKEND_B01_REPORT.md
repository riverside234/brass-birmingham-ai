# B01 — Lab Linux/CUDA qualification tooling

Implemented 2026-10-28; search_v1 bundles refreshed 2026-10-29. **B01 lab qualification: Pending.** The user has not run it on the lab server and will do so later. No local native qualification, CUDA qualification, self-play, optimizer update, source adoption or game installation was performed for B01. Short mocked development tests and input-package integrity checks are separate from the B01 exit gate. The [B02 collector](RUST_BACKEND_B02_REPORT.md) can be implemented meanwhile; its qualification requires a passing B01 receipt.

## What is implemented

- `configs/search/lab-search-v1.json` admits exactly `search_v1`, `brass-gumbel-is-mcts/count-v1`, `brass-gumbel-is-mcts/session-v1`, and V205 disabled. Missing methods, every search_v2 variant, focus/composite fields and subtree-reuse overrides are rejected. The general research worker retains its separately identified experimental modes; the lab profile does not admit them.
- `scripts/qualify-rust-lab.py` verifies the 1.4.42 search_v1 deployment, candidate qualification, all four installed NN policy pins, installed smoke and source archive, extracts that archive into a new evidence directory, and builds Linux release binaries with Rust 1.99.1 and the frozen Cargo lock. Command-line Linux target/output overrides preserve the original source fingerprint. The mutable native working tree is not the build input.
- The runner holds the existing training-run lock. It reads the actual committed signature/config, generation, latest checkpoint, population/anchors and configured host runtime; hashes weights, optimizer, RNG and all embedded training/PPG/auxiliary state; records runtime/device/resource settings and current Python sources; and rejects unfinished collection/checkpoint work. It never calls training, optimizer restore, source transition or recipe migration. Hashes and pending-work status are checked again before exit.
- `lab_inference.py` adapts permitted native feature inputs to the **existing** `InferenceService` scheduler, private pipes and shared slots. Reply validation binds owner/request/epoch, frozen model/reward/profile, permitted input, legal order, revision and observer. Cancellation/expiry before encoding skips encoding; cancellation after dispatch drains the original slot; stale/misrouted/device-failure replies retire the lane. Epoch changes cannot refresh the original deadline or change model/context identities.
- The lab smoke compares native and Python tensors exactly, then compares CPU/PyTorch and real CUDA outputs at `atol=2e-5`, `rtol=1e-5`, with TF32 off. It exercises two distinct committed frozen models, variable full candidate sets, both native diagnostic owners, unknown actor descriptors, and separate explicit critic-descriptor diagnostics. The production inference scheduler and ordinary policy collection code are unchanged.
- Portable native regressions cover rules, observations/privacy, features, Gumbel selection, whole-turn sessions, frozen quotas, completed-round snapshots, supported plans, cancellation and exact NN cache behavior; optional shared-core experiment tests do not enable those methods in the lab profile. Existing inference lifecycle, deadline, output-packing and failure-isolation tests also run on the lab host.

This is **B01 input/evaluator/platform qualification**. It is not B02 ordinary collection, B04 Rust coroutine-to-GPU search integration, teacher admission, a search latency/strength benchmark, or B07 adoption. The service adapter returns the existing policy/value response contract; B04 must supply and qualify the search evaluator's logits/complete-round target contract without deriving logits from rounded probabilities. The local g890 utility exception does not admit a lab teacher.

## Transfer and prerequisites

Update the lab checkout to the current project source first, including the existing native Python clients and Recovery tests. Copy `distribution/rust-lab-b01/b01-search-v1-update.zip` to that checkout and extract it there. For B02 as well, extract `distribution/rust-lab-b02/b02-search-v1-update.zip` second, before running either qualification. All current input and transfer ZIPs are under `distribution/` and explicitly Git-allowed; older `b01-lab-update.zip`/`b02-lab-update.zip` remain historical and must not be extracted over this update. It contains the new Python tooling/tests/profile, this report, and the immutable input ZIP at:

`distribution/rust-lab-b01/b01-search-v1-inputs.zip`

The input ZIP SHA-256 is:

`935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76`

Its source archive is byte-identical to the installed 1.4.42 package: native fingerprint `fa0b3f7b74e9ed638c8b6f96d3f23b0910b4e5df9387ff8df58d5472a8014dd2`, ZIP SHA-256 `6dc6e230b96a4c57bd14d85fb994c8d0bd624b9b504d83647ff8d157842179c7`. Relative to the earlier shared source, only the already-released local temperature adapter files differ; the rules/search core and Cargo lock are unchanged. The new input/qualification contract is version 2; old cache-only inputs/receipts cannot qualify this profile. The input package contains **13 roots and 9 staged transitions**, the local receipts, and vendor attribution. It contains no model checkpoint, saved game, private lab data or proprietary Unity assembly. It does not copy or overwrite a lab training run.

Use the existing `ai-training` environment with Python 3.11 or newer, the project's CUDA-enabled PyTorch and dependencies. A real visible CUDA device and Linux C/C++ linker are required. If the pinned Rust toolchain is missing, install it once:

```bash
rustup toolchain install 1.99.1 --profile minimal
```

The first build may fetch dependencies pinned by `Cargo.lock`. Add `--offline` only when the lab Cargo cache is already populated. No ONNX Runtime GPU provider or model re-export is required.

## Run on the lab

Stop/pause training **after a committed iteration**, using its normal shutdown/recovery procedure. B01 refuses an active run or an unfinished collection; it does not kill training or delete pending work. Replace `YOUR_RUN` with the actual existing run directory. Change the checkout path below if the server uses a different location.

To run both gates in order, use `bash scripts/run-rust-lab-b01-b02.sh artifacts/step5/YOUR_RUN cuda:0` after extracting both updated bundles. It stops if B01 fails and prints both receipt paths. Keep training paused throughout.

The standalone B01 wrapper supplies the pinned input path/hash and a fresh output directory:

```bash
conda activate ai-training
bash scripts/run-rust-lab-b01.sh artifacts/step5/YOUR_RUN cuda:0
```

Or use the complete equivalent command from the checkout:

```bash
cd /data/not_backed_up/yxu209/ai-training
conda activate ai-training

python scripts/qualify-rust-lab.py \
  --run artifacts/step5/YOUR_RUN \
  --device cuda:0 \
  --profile configs/search/lab-search-v1.json \
  --bundle distribution/rust-lab-b01/b01-search-v1-inputs.zip \
  --bundle-sha256 935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76 \
  --out "artifacts/rust-backend/$(date -u +%Y%m%d-%H%M%S)-b01/lab"
```

Repeat `--device cuda:N` for each device to qualify. Device numbers refer to the process's visible CUDA devices. Each device is checked independently; this is not a scaling study. The signed production lane/batch/resource recipe is captured and its batching settings are retained; the smoke itself creates only two diagnostic reply lanes. Pass `--runtime-manifest PATH` for any additional server runtime receipt outside the configured host/runtime inventory.

The run needs a distinct retained population/anchor checkpoint for mixed-model inference. Missing/incompatible frozen models fail explicitly; the script does not synthesize a substitute or change population membership. The actual committed learner is used for mechanical tensor/output checks even if it is not teacher-ready; `teacher_admitted` stays false.

Native compilation/test commands have a 900-second watchdog, configurable with `--build-timeout` (maximum 1800). The existing inference regression batch has a 90-second watchdog; the real GPU smoke has a **120-second watchdog per requested device**. Every diagnostic NN envelope keeps its original **4-second allowance**. No full games or training iterations are run. B04's later matched search benchmark remains **4s/K8**; these NN-envelope checks are not that benchmark.

## Results and failures

The output directory is new on every invocation. Inspect `qualification.json`; only **`passed: true`**, contract **`brass-lab-b01/2.0.0`**, and method **`search_v1`** qualify the refreshed B01. It links `local-entry.json`, `profile.json`, `toolchain.json`, `linux-build.json`, `lab-state.json`, `python-runtime-sources.json`, per-device `gpu-N.json`, and command logs. The full original config/signature is retained in the snapshot, along with checkpoint-component hashes and unchanged-state checks. Keep this evidence private with the training run.

A failure writes `passed: false` and retains logs. A timed-out CUDA diagnostic process group is terminated before the run lock is released. Resolve the reported issue and rerun in a new output directory; there is no automatic fallback to V205, CPU-only success, recipe rewrite, pending-work deletion or weakening of numerical tolerances. Interrupted qualification never creates a successful receipt.

To check the small development tests alone after a source edit:

```bash
PYTHONPATH=src python -m unittest discover -s tests/RustBackendLab -v
```

The original 19-test development receipt remains historical. All **42 refreshed B01/B02 Python tests passed**. The suite adds current-source/evidence-chain tampering tests, exact search_v1 admission and B02 rejection of old receipts; see `artifacts/rust-backend/20261029-search-v1-bundles/`. Bash syntax and mocked combined-runner success/failure checks also passed. The Python tests use mocked native/GPU endpoints and do not satisfy Linux/CUDA qualification. Input archive hashes, extracted native fingerprint and required fixture membership were also verified without compiling or executing the native engine. The B01 exit checkbox remains open until the server receipt passes. B02 is implemented separately with lab qualification Pending; B03–B07 remain pending.
