# B01–B04 qualification at g570 (r9)

Updated 2026-11-03. **B01, B02 and B02.5 passed on the lab. Do not repeat them.**
The latest r8 attempt passed B03's optimizer comparisons, then failed with
`CUDA device topology differs on resume`: checkpoint and visible GPU counts
differ. Training already supports this through logical-GPU RNG restoration;
the unified launcher did not expose B03's existing opt-in flag.

r9 forwards `--allow-cuda-topology-change` to B03 and records it in the receipts.
It retains the exact r8 trainer, native code, source approvals and prerequisite
reuse review. See the [topology repair](../../ai-research/RUST_BACKEND_B03_CUDA_TOPOLOGY_FIX.md).

## Resume on the lab at B03

Sync the matching checkout, including added/deleted files. Keep training paused
at the same g570 checkpoint and retain the original lab artifacts. From the
repository root:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-g570/rust-lab-b01-b04-g570-r9.zip
python scripts/run-rust-lab-qualification.py --stage all --start-at b03 --device cuda:0 \
  --allow-cuda-topology-change \
  --b02 artifacts/rust-backend/20261104T011422740838Z-g570-all-8bee9d91/b02/qualification.json \
  --expected-generation 570 \
  --checkpoint-sha256 06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9
```

This runs B03 and then B04. It reuses the original B02 families, native binaries
and B01/device receipts. No B01/B02/B02.5 qualification or B02 collection is
repeated. Reuse requires the exact original receipt, unchanged run/checkpoint
and all existing file pins; only the reviewed diagnostic-source delta is
allowed. Do not substitute the Git-visible review copy for the original receipt.

The run defaults to `artifacts/step5/joint7-v11-2-g1290-trained-restart`, with its
exact recorded g570 recipe. Qualification never adopts another config, advances
training or rewinds a run. B03 records the prerequisite review in its new receipt.
If training has advanced, these g570 receipts cannot qualify that different state.

Keep the original GPU visibility/allocation and qualified `cuda:0` device. The
flag restores saved RNG streams to matching logical indices, including learner
GPU 0. Additional visible devices keep their initialized streams; unavailable
saved streams remain in the unchanged checkpoint. `restoration.json` and the
lab receipt record saved/visible/restored counts and verify restored values.
The strict equal-count default remains for callers without the flag. This is
an isolated, zero-update restoration check, not a bitwise equivalence claim for
training on a different GPU allocation.

Each attempt exports a new Git-visible `results/rust-lab-qualification/` folder.
Open `evidence/runner.json` first. Push that folder and pull it locally for review.
No original results, source checkpoint, old ZIP or passing receipt is overwritten.

The 60-worker B02 ceiling and 240-second watchdog remain unchanged for future
collection. Existing Mono, CUDA Python, Rust 1.99.1, private runtimes, full
checkpoints, opponent references and retained PPG caches remain required.
Current bundle metadata selects r9; do not extract older supplements afterward.

Task-only `search_v1`, K16, DOMiNO/AZ_DB off and search_v2/V205 disabled remain.
B04 retains two independent one-thread sessions, 4 seconds search and at most
1 second cumulative post-work. Its utility readiness gate still applies.
**B03/B04 lab qualification and B05–B07 production adoption remain Pending.**
