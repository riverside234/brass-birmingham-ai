# B06 g1250 matched workload qualification

Implemented; **Linux/CUDA qualification Pending**. B05 r2 passed at
`20261107T070806231803Z-20261107T070539193547Z-g1250-b05-88497ac1-cf4704fd`.
No successful earlier qualification is repeated. This bundle adds diagnostic
scripts and fixes the native adapter's current branch opening window. Native
binaries, PPO/PPG objectives, installed game and production settings stay intact.

Run from the lab **code repository** in the existing training environment:

```bash
git pull --ff-only
unzip -o distribution/rust-lab-b06-g1250/rust-lab-b06-g1250-r4.zip
python -m pip install -r distribution/rust-lab-b06-g1250/requirements.txt
python scripts/qualify-rust-lab-b06.py --device cuda:0
```

Keep training paused at the preserved original **g1250** state, checkpoint
`de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`, exact
recorded config, full resume dependencies and original B02/B04.5 lab binaries.
The default run is `artifacts/step5/joint7-v11-2-g1290-trained-restart`.
`--run PATH` selects an already preserved compatible run; never rewind a live
run or rewrite its signed paths. Native runtime paths/hashes are in the bundled
`evidence/ordinary-build.json` and `teacher-build.json`. Missing/changed files
fail before collection; no silent rebuild, source override or qualification rerun.

R4 fixes the next lab attempt (`2f3176b0-a1c6d453`): all 53 CPU fixtures passed,
but the fresh ordinary benchmark process reset CUDA allocator statistics before
initializing CUDA, causing `RuntimeError: Invalid device argument`. No collection
or update started. Every CUDA meter now synchronizes the selected logical GPU
(which initializes PyTorch's allocator) before resetting peaks. Each benchmark
child also runs a 256-byte allocation/execution/peak-counter preflight and exports
its own environment receipt. CUDA setup is reported separately from timed work;
CPU fixtures still require no GPU. Earlier qualification receipts remain valid.

R3 fixed the first lab attempt (`06c4bb94-be70601a`), which stopped in CPU
fixtures because `psutil` was absent; no benchmark or training work ran.
The bundle now declares its extra measurement dependency (`psutil==7.2.2`).
Install it with the **same Python interpreter/environment** used for qualification;
the requirements file does not install Torch/CUDA or change the training recipe.
A preflight imports the native extension, verifies its version and exercises the
real process sampler before fixtures, locks or collection. Missing/broken installs
export an `environment` failure with the install command. No implicit install or
silent memory/CPU fallback occurs. Unexpected background sampler errors invalidate
measurements, disappearing child processes are handled, and initial child CPU
time is excluded from benchmark work. Use r4; do not rerun B01–B05.

R2 corrected a compatibility gap found before lab release: current v9 branches use
T1.5 for their first two actions per seat while normal turns use T1.3 for three.
The native adapter and sealed-family validator now use that existing reference
behavior, retaining frozen policy identities and the excluded fork row. The exact
two-file source review preserves B05's unaffected update/restore evidence and
existing restricted resume routes; the source run is not adopted or modified.

The smoke is bounded before execution:

- Ordinary: at most three frozen parent families **per backend**, selected by
  actual table kind and whole-table route, plus naturally occurring branches.
  Current 60-slot CPU cap, inference settings and branch probability stay intact;
  active lanes are limited by this small workload. No forced branch or resampling.
- Teacher: eight fixed-root full-profile turns, typical/complex roots, ABBA order,
  **two independent one-thread sessions**, **4s/K16 + at most 1s post-work**.
  Both arms use search_v1 and identical frozen calibration/model/root inputs.
  These compare the policy-only and all-seat w=0 export paths, not full schedules.
- Frozen B04.5 tables: batch versus streamed validation/reuse, policy/value buffer
  costs, one isolated head update and two teacher-KL gradient kernel samples.
- Ordinary updates: bounded authored PPO/PPG component timings with actual
  g1250 model, Adams and current recipe. No production iteration or teacher enablement.

Phase watchdogs: ordinary 360s, teacher 150s, update kernels 180s. A timeout stops
the attempt and exports its first failure and partial evidence. Bounds never
extend automatically. Build/startup, GPU queues, validation, preparation/cache,
memory, IPC and update costs remain separately visible. Historical all-seat
timings retain their original receipt; no new teacher games are collected.
Whole-iteration speedup and population tails remain unclaimed. Missing natural
branches are reported, not manufactured. B07 still requires a review of useful
measured performance and any coverage gaps before adoption.

Every attempt exports a fresh, non-ignored folder under
`results/rust-lab-qualification/`, including failures. Run the exact
`git -C results add/commit/push` commands printed by the launcher. Pull results
on this computer with `git -C results pull --ff-only`.

For the short local fixture check only:

```bash
python -m pip install -r distribution/rust-lab-b06-g1250/requirements.txt
python scripts/qualify-rust-lab-b06.py --fixtures-only
```

This leaves lab qualification Pending. See
[implementation and scope](../../ai-research/RUST_BACKEND_B06_IMPLEMENTATION.md).
