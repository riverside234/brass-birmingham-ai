# B01–B04 qualification at g570 (r7)

Updated 2026-11-03. **r7 lab qualification: Pending; production teacher off.**
The latest B01 pass remains valid evidence for its r6 source only.

r7 fixes a collection transport bottleneck: large C# JSON replies were read one
byte at a time. It buffers text reads, reuses complete step/restore replies and
adds progress snapshots and thread stacks for watchdog diagnosis. All existing
correctness checks, the r6 runtime fix and the 240-second watchdog remain.
See the [diagnosis and measured evidence](../../ai-research/RUST_BACKEND_B02_TIMEOUT_FIX.md).

The B02 smoke now defaults to a **60-worker ceiling**, bounded by the saved
training cap and CPU budget. It allocates only lanes with useful work: three
frozen table families and their eligible continuations, normally four or five
lanes instead of two. It retains the 240-second collection watchdog and all
coverage checks. Worker limits and actual lanes are saved in the results.

## Run on the lab

Sync the matching checkout, including added/deleted files. Pause training at a
clean committed boundary. From the repository root:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-g570/rust-lab-b01-b04-g570-r7.zip
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0 --b02-workers 60 \
  --expected-generation 570 \
  --checkpoint-sha256 06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9
```

**Start from B01 for r7:** executable source changed, so the old B01 source
inventory must not authorize this checkout. B01–B04 all use the new exact
implementation; source-continuation reviews preserve the full g570 checkpoint and
recipe. Qualification never changes the production run or starts training.

If training has advanced, capture its actual latest committed state instead:

```bash
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0 --b02-workers 60
```

The run defaults to `artifacts/step5/joint7-v11-2-g1290-trained-restart` and the
config to the exact recorded g570 recipe. Supply `--config` only for a different
already-adopted reviewed recipe. Generation/hash assertions never rewind a run.

After an r7 B01 pass, an independent B02 retry can use its original receipt:

```bash
python scripts/run-rust-lab-qualification.py --stage all --start-at b02 --device cuda:0 \
  --b02-workers 60 --expected-generation 570 --b01 ORIGINAL_R7_B01_QUALIFICATION_JSON
```

Each attempt exports a new Git-visible `results/rust-lab-qualification/` folder.
Open `evidence/runner.json` first. Push that folder and pull it locally for review.
For a collection timeout, inspect `evidence/b02/collection-progress.json` and
`evidence/b02/collection-smoke.log` for the last stage and thread stacks.
Retain original lab artifacts for dependent stages. No original results or old
ZIPs are overwritten.

The public pinned Roslyn compiler is included. Existing Mono, CUDA Python,
Rust 1.99.1, the matching checkout/private game runtime, full checkpoints,
opponent references and retained PPG caches remain required. Current bundle
metadata selects r7; do not extract older supplements afterward.

Task-only `search_v1`, K16, DOMiNO/AZ_DB off and search_v2/V205 disabled remain
unchanged. B04 retains two independent one-thread sessions, 4 seconds search and
at most 1 second cumulative post-work. Its utility readiness gate still applies.
B05–B07 production adoption remains Pending.
