# B04.5 — g1250 search-value qualification

Use the current checkout and `rust-lab-b045-g1250-r3.zip`. Earlier qualification
ZIPs are retired from Git tracking; the necessary sealed baseline inputs are
included here. Training must be paused at the recorded g1250 checkpoint:

`de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`

From the lab repository root:

```bash
conda activate ai-training
git pull --ff-only
unzip -o distribution/rust-lab-g1250/rust-lab-b045-g1250-r3.zip
python scripts/qualify-rust-lab-b045.py --device cuda:0
```

This runs **B04.5 only**. It does not repeat previously successful qualifications
or modify the production checkpoint/config/state. It compiles the new Rust
teacher and runs its short unit checks, then recalibrates with warm probes,
checks both search profiles and zero-blend parity, collects two all-seat shadow
tables, and applies one diagnostic head-only update on copied weights. Full
search is 4s/K16 plus at most 1s post-work; fast search is 0.8s/K4 plus at most
0.2s post-work. Two sessions, each with one thread. The CUDA child has a
240-second watchdog. No 256-game evaluation runs in this command.

R3 fixes absolute deadlines on later turns and second actions, terminal actor
handling, and a worker-cancellation race that hid the initiating error. Before
the GPU smoke, two bounded scripted games exercise all 64 turns, both eras,
terminal settlement and value-buffer sealing. These use zero search counts
and synthetic values solely as fixtures; they produce no training evidence.
On collection failure, `collection-failure.json` retains the original exception,
job, stderr and last native frame; `inference.json` is also saved.
Value sealing avoids redundant tensor copying/decoding and measures each game's
shard against its own final-turn allowance; batch size does not multiply that
charge. Existing value-buffer shards remain readable.

The user-approved null readiness assumption is limited to the exact g1250
checkpoint and shadow qualification. Explicit false or reward mismatch still
fails. Actual GPU timing and performance remain **Pending** until this run.
The existing reference runtime and Mono must be present on the lab; all runtime
dependencies are checked. Rust toolchain `1.99.1` is required as before.

On success or failure, the launcher prints a new
`results/rust-lab-qualification/...` folder. Commit/push that folder in the
**results repository** to bring evidence back here. The exporter checks Git
visibility there, adds local rules preserving log bytes and allowing diagnostic
files, and prints the exact `git -C results` commands. The code repository
intentionally ignores `results/`; pushing the code repository alone does not
publish qualification evidence. If `results/.git` is missing or a parent rule
ignores the whole qualification directory, the exporter reports that explicitly.

Recover an attempt's export **without rerunning any qualification**:

```bash
git pull --ff-only
python scripts/rust_lab_results.py artifacts/rust-backend/PASTE_ATTEMPT_DIRECTORY
```

Run the exact `git -C results add`, `commit` and `push` commands printed by that
export command. Its `evidence/qualification.json` and logs preserve the
original failure. R2's cleanup hashing fix remains in place. Run the B04.5
command above; previous B01–B04 qualification need not be repeated.

If only the smoke failed, reuse a matching **r3** B04.5 build on retry:

```bash
python scripts/qualify-rust-lab-b045.py --device cuda:0 \
  --build artifacts/rust-backend/PASTE_PREVIOUS_G1250_B045_DIRECTORY/build.json
```

The build's source, binary, bootstrap dependencies and fixtures must still match.
The r2 native binary contains the deadline/terminal bugs and must be rebuilt;
do not pass its `build.json`. No g570 build can substitute either.

After stage B (not now), the explicit one-off evaluation tool is:

```bash
python scripts/evaluate-rust-search-value.py \
  --checkpoint PATH_TO_TRAINED_CORRECTION_CHECKPOINT \
  --config PATH_TO_MATCHING_RECORDED_CONFIG \
  --build PATH_TO_CURRENT_B045_BUILD_JSON --device cuda:0 --lanes 48
```

That tool runs 256 games and never changes deployment. See
[implementation details](../../ai-research/RUST_BACKEND_B045_IMPLEMENTATION.md).
