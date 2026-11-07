# B07 g1250 R4 qualification

R4 fixes the R3 matched-probability failure and optimizes ordinary collection.
**Lab qualification, useful accepted-work gain, canaries and adoption remain Pending.**
The installed game and search_v1 behavior are unchanged.

The latest results were pulled inside `results/` at `e71dc25`. R3 completed
3 families / 1,161 decisions, but comparison failed on a 4.68e-7 probability
difference. Its native accepted-work time was 32.684s versus reference 27.212s.
R4 checks exact prepared features/targets before the probability comparison;
float32 batch tolerance applies only to that comparison. Independent replay
sampling verification remains strict and checks the original RNG draws.

R4 keeps Rust authoritative for V4-containing ordinary tables and retains the
original C# V4 decision worker. It preserves encoded features through a single
sealed actor handoff, publication and reading; simplifies forced verification;
drains record buffers; and reuses collector processes and original policy objects
with per-game/seat resets. See [design, evidence and limits](../../ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md#r4-v4-tables-retained-features-and-reusable-lanes).

From the lab **code checkout**, in the existing training environment:

```bash
git pull --ff-only
unzip -o distribution/rust-lab-b07-g1250/rust-lab-b07-g1250-r4.zip
python scripts/qualify-rust-lab-b07.py --device cuda:0 --ordinary-only
```

If the existing pinned measurement dependency is missing:

```bash
python -m pip install -r distribution/rust-lab-b06-g1250/requirements.txt
```

Use the original paused g1250 clean boundary in
`artifacts/step5/joint7-v11-2-g1290-trained-restart`, checkpoint SHA256
`de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`.
The saved config, opponent identities, model and original C# runtime pins remain
required. Never rewind a newer live run; a later checkpoint requires refreshed gates.
The reviewed source transition preserves the same run/checkpoint/optimizer/recipe.
No qualification command changes production state or starts a training iteration.

This reruns only affected checks: short Python collection/admission/recovery
fixtures, the rebuilt ordinary Rust adapter, binary-feature/forced-RNG/draining
checks, a 24-stage V4 progression comparison, and the bounded matched batch of
self-play, historical NN and historical V4 tables on the exact same g1250 checkpoint.
V4 presence remains an explicit stratum even though all three now use Rust state.
Natural branch draws and the saved worker/CPU cap are unchanged. B01–B06 passing
qualification is reused; teacher canaries and continuation updates are not run
by `--ordinary-only`. Watchdogs: ordinary comparison 360s, V4 smoke 90s, native
wire checks 60s. No long session or additional full game is added.

Reports separate collector/policy startup and reuse, IPC/serialization, binary
feature transfer, collection/export, transient handoff, publication/validation,
feature-cache I/O, first/warm replay reads, preparation and accepted-work total.
Nested timings overlap. A passing ordinary result still leaves teacher/adoption
Pending and cannot authorize production adoption alone.

If this exact R4 build passed before a later failure, reuse its receipt:

```bash
python scripts/qualify-rust-lab-b07.py --device cuda:0 --ordinary-only --ordinary-build artifacts/rust-backend/REPLACE_WITH_R4_ATTEMPT/ordinary-runtime/build.json
```

Old R3 builds cannot be reused: source, producer, core, binary, test/build-log and
C# pins are checked. Rust 1.99.1 and the existing Cargo cache are required; add
`--offline` to prohibit dependency downloads.

Both failures and successes export to **`results/rust-lab-qualification/`**.
Use the exact `git -C results add/commit/push` commands printed by the launcher.
These exports are not ignored in the results repository; commit/push from that
folder, then pull from that folder on the other computer.

Short CPU-only development checks (no checkpoint or GPU):

```bash
python scripts/qualify-rust-lab-b07.py --fixtures-only
python scripts/rust_lab_b07_r4_publication.py --out artifacts/b07-r4-publication.json
```

The optional publication comparison reads the immutable R3 ZIP retained in this
repository. Local evidence: a 24-stage V4 comparison matched plans, intent, RNG and
work budgets; Rust progression took 0.415s versus C# 1.344s, while the unchanged
V4 decision worker took about 11.3s per arm. Two alternating small-fixture pairs
reduced publication plus first read from 3.014s to 2.427s (19.5%). This is component
evidence, not a g1250 GPU accepted-work speed claim. The unsuccessful intermediate
separate-file feature handoff measurement is retained as well.

R4 includes the reviewed sources, exact prior qualification evidence, latest R3
failure/frozen jobs and new local receipts. Archives are immutable: future fixes
must use a new revision. No checkpoint weights or replacement native binaries
are shipped. Production adoption remains the separate, explicitly gated B07
workflow described in the migration/implementation plans.
