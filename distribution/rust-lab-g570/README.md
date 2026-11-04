# B01–B04 qualification at g570 (r10)

Updated 2026-11-03. **B01, B02, B02.5 and B03 passed. Do not repeat them.**
B03 restored the full g570 state and admitted the sealed B02 replay successfully.
Its CUDA report shows one saved RNG device, eight visible devices and one restored
stream. B04 compiled and passed fixtures/native tests, then deferred before any
teacher table because the utility-calibration receipt was null.

For this exact g570 checkpoint, the user explicitly accepts null as ready. r10
records `receipt: null, ready: true` with `user-authorized-g570-null-utility`
evidence. It keeps the original checkpoint/model calibration unchanged and does
not claim that a calibration measurement was performed. See the
[review](../../ai-research/RUST_BACKEND_B04_G570_READINESS_FIX.md).

## Run only the remaining B04 qualification

Sync the matching checkout, then run from the lab repository root with training
paused at the unchanged g570 checkpoint. Keep both original evidence directories.

```bash
conda activate ai-training
unzip -o distribution/rust-lab-g570/rust-lab-b01-b04-g570-r10.zip
python scripts/run-rust-lab-qualification.py --stage b04 --device cuda:0 \
  --b02 artifacts/rust-backend/20261104T011422740838Z-g570-all-8bee9d91/b02/qualification.json \
  --b03 artifacts/rust-backend/20261104T015819818200Z-g570-all-4c81aca9/b03/qualification.json \
  --b04-build artifacts/rust-backend/20261104T015819818200Z-g570-all-4c81aca9/b04/build.json \
  --expected-generation 570 \
  --checkpoint-sha256 06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9
```

This runs B04 only and reuses its verified native binary instead of recompiling.
The original B01/B02/B02.5/B03 qualifications and B02 games are not repeated.
Original receipt, checkpoint/state, runtime, source-delta, overlay, lockfile,
binary and dependency pins remain mandatory. The build's previous source
identity stays in the old receipt; the new diagnostic binding records
`reused_from`. No original report is rewritten.

The run remains `artifacts/step5/joint7-v11-2-g1290-trained-restart` with its exact
recorded recipe. Keep the original qualified `cuda:0` allocation. This command
does not restore RNG or run B03, so no topology-change flag is needed.

Null acceptance is scoped to B04 shadow testing of the pinned g570 SHA/model/run
and matching utility reward. Explicit false or reward-mismatched receipts remain
rejected; other checkpoints retain the existing readiness checks. This does not
change production teacher mode or the stored utility calibration.

B04 retains two separate one-thread sessions, 4s/K16 search and at most one second
of cumulative post-work, with bounded GPU probes and two shadow tables. The
existing 240-second collection watchdog remains. Search is task-only `search_v1`;
search_v2/V205, DOMiNO and AZ_DB remain off. No production training update occurs.

Each attempt exports a new Git-visible `results/rust-lab-qualification/` folder.
Push that folder for review. **B04 GPU/search qualification and B05–B07 production
adoption remain Pending.** The null receipt no longer blocks this g570 smoke.
Current bundle metadata selects r10; do not extract older supplements afterward.
