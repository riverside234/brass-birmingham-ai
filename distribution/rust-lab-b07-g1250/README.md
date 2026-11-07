# B07 g1250 qualification and backend adoption

Implemented; **lab qualification and production adoption Pending**. The pulled
B06 run passed correctness but native accepted work was 55.265s versus 28.596s
reference. B07 isolates Python collection/publication and removes duplicate full
validation. See the [implementation and limits](../../ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md).

Keep the original g1250 run paused at its clean committed boundary. This bundle
uses checkpoint `de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`
in `artifacts/step5/joint7-v11-2-g1290-trained-restart`, the exact saved config and
the original qualified lab binaries. Never rewind a newer live run; a later
checkpoint requires refreshed, bound gates. No earlier qualification is rerun.

From the lab **code checkout**, using the existing training Python environment:

```bash
git pull --ff-only
unzip -o distribution/rust-lab-b07-g1250/rust-lab-b07-g1250-r2.zip
python -m pip install -r distribution/rust-lab-b06-g1250/requirements.txt
python scripts/qualify-rust-lab-b07.py --device cuda:0
```

This runs short CPU fixtures, at most three ordinary parent families per backend
plus natural branches, two shadow passes of at most two tables each, and a
fixed-data restore/update/reload check. Full teacher turns use **4s/K16**, two
independent single-thread sessions and at most 1s post-work; existing fast
all-seat turns retain 0.8s/K4 + 0.2s post. No extra long session, production
teacher enablement or training iteration starts. Watchdogs are 360s for the
ordinary comparison, 210s per shadow pass, and 180s for the fixed-data check.

Use the printed `git -C results add/commit/push` commands to publish the fresh
`results/rust-lab-qualification/` export. Failures export the original error and
partial evidence too. This repository intentionally ignores `results/`; the
**results repository** does not ignore these exports.

Only after the report has **`passed: true` and `adoption_ready: true`**, explicitly
adopt the backend into the same run with the teacher still off. Set `B07_Q` to
the original lab `artifacts/rust-backend/.../qualification.json` path printed by
the passing attempt, not a portable export or an earlier qualification:

```bash
B07_Q='artifacts/rust-backend/REPLACE_WITH_PASSING_B07_RUN/qualification.json'
python scripts/manage-rust-lab-b07.py adopt --expected-generation 1250 --qualification "$B07_Q"
python scripts/manage-rust-lab-b07.py verify-rollback --expected-generation 1250
```

The adoption command restores a complete rollback copy before changing state;
it does not advance training. Run **one next iteration using your existing
authorized training command and saved recipe**, then stop at g1251 and verify:

```bash
python scripts/manage-rust-lab-b07.py verify-next --expected-generation 1251
```

That check proves the normal commit recorded native families/backend identity and
can restore its full state. It performs no extra collection. If you need to
return to reference, pause at a clean committed boundary and use its actual
generation (1250 before the first iteration, 1251 after it):

```bash
python scripts/manage-rust-lab-b07.py reference --expected-generation 1251
```

Reference rollback preserves current ordinary progress and can run even if the
native executable is unavailable. `verify-rollback` restores the original full
pre-adoption checkpoint in isolation; it never overwrites the current run.
An in-progress iteration must be completed/recovered before switching.

For CPU-only development checks, with no games or GPU:

```bash
python scripts/qualify-rust-lab-b07.py --fixtures-only
python scripts/rust_lab_b07_publication.py --out artifacts/b07-publication.json
```

These do not close Linux/CUDA qualification. The `r2` archive is immutable;
future fixes get a new archive revision. Source review, exact prerequisites,
local checks and all necessary overlay code are included. No checkpoint weights
or replacement native binaries are shipped.

R2 includes the original pinned `inputs.zip` needed by the native overlay CPU
test; it does not require an earlier B04.5 extraction. R1 was caught during local
clean-package verification and was not published. On Windows, use a short
checkout path for nested fixture exports (the lab runs on Linux).
