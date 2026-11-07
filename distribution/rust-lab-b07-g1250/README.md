# B07 g1250 qualification and backend adoption

R3 repairs the first B07 canary failure: default Rust JSON decoding changed
`0.015000000000000001` to `0.015`, falsely rejecting an unchanged GPU binding.
The teacher build enables `serde_json/float_roundtrip`; exact stale/model/epoch,
legal-action and budget checks remain strict. **Canaries and adoption Pending.**
The completed B07 ordinary phase passed correctness (3 families/1,161 decisions),
but accepted work was 40.246s native versus 28.833s reference. This negative
performance result is preserved as historical evidence. R3 also removes repeated
row checks/deep copies and permits at most two publishers under the shared CPU
cap. Ordinary collection uses compact action/revision/context acknowledgements,
one binary feature-array reply per decision, bounded forced-action batches with
unchanged Python RNG, and batched byte-sealed completed-record export. Its affected ordinary performance scope must be remeasured before adoption.
Local wire-only smoke: 24-decision collection 1.213s → 0.401s, feature
transfer 0.715s → 0.057s, 119 → 50 pipe calls. Small serial publication plus
first-read median 1.719s → 1.577s, with equal features/targets. These do not
establish total g1250 accepted-work gain; that remains Pending on the lab.
See the [implementation and limits](../../ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md).

Keep the original g1250 run paused at its clean committed boundary. This bundle
uses checkpoint `de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`
in `artifacts/step5/joint7-v11-2-g1290-trained-restart`, the exact saved config and
the original C# binaries. Both native adapters are rebuilt from reviewed source:
the ordinary collector against the unchanged sealed B01 core, and the teacher
against its unchanged sealed search_v1 overlay. Never rewind a newer live run; a later
checkpoint requires refreshed, bound gates. No earlier qualification is rerun.

From the lab **code checkout**, using the existing training Python environment:

```bash
git pull --ff-only
unzip -o distribution/rust-lab-b07-g1250/rust-lab-b07-g1250-r3.zip
python -m pip install -r distribution/rust-lab-b06-g1250/requirements.txt
python scripts/qualify-rust-lab-b07.py --device cuda:0
```

This runs short CPU fixtures, builds/tests both adapters, checks native wire
parity/cancellation/recovery and a bounded transport microbenchmark, then remeasures only
the changed ordinary scope (at most three frozen parent families per backend and
natural branches), then runs the two unfinished shadow passes of at most two tables each and the fixed-data
restore/update/reload check. Rust 1.99.1 and the existing Cargo cache are required;
add `--offline` to prohibit downloads. Native tests cover the reproduced failure,
budget steps, 4,096 finite-value candidates and changed bindings before games.
Full teacher turns use **4s/K16**, two
independent single-thread sessions and at most 1s post-work; existing fast
all-seat turns retain 0.8s/K4 + 0.2s post. No extra long session, production
teacher enablement or training iteration starts. Watchdogs are 360s for the
ordinary comparison, 210s per shadow pass, and 180s for the fixed-data check.

To measure only this ordinary optimization without teacher canaries or updates:

```bash
python scripts/qualify-rust-lab-b07.py --device cuda:0 --ordinary-only
```

This leaves teacher qualification and adoption Pending. It uses the exact same
frozen jobs/checkpoint in both live arms and preserves the original negative R2
performance result as historical evidence. Reports include process startup,
request encoding/reply decoding, pipe bytes/calls, feature transfer, collection,
record export, part I/O, publication/validation, replay reads, preparation and
total accepted-work time. Nested timings overlap; do not add them together.

If another attempt stops after the repaired builds succeeded, reuse only their
current-source build receipt to avoid recompiling:

```bash
python scripts/qualify-rust-lab-b07.py --device cuda:0 --ordinary-build artifacts/rust-backend/REPLACE_WITH_R3_RUN/ordinary-runtime/build.json --teacher-build artifacts/rust-backend/REPLACE_WITH_R3_RUN/teacher-runtime/build.json
```

The launcher verifies the new producer, unchanged shared-core source, binary,
test/build logs and original C# dependencies. It cannot silently reuse the old
teacher or ordinary adapter. All B01–B06 successful qualifications remain reused.

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
python scripts/rust_lab_b07_r3_publication.py --out artifacts/b07-r3-publication.json
```

These do not close Linux/CUDA qualification. The `r3` archive is immutable;
future fixes get a new archive revision. Source review, exact prerequisites,
local checks and all necessary overlay code are included. No checkpoint weights
or prebuilt replacement native binaries are shipped.

R2 includes the original pinned `inputs.zip` needed by the native overlay CPU
test; it does not require an earlier B04.5 extraction. R1 was caught during local
clean-package verification and was not published. On Windows, use a short
checkout path for nested fixture exports (the lab runs on Linux).
R3 preserves R2's source and original ordinary report inside the sealed bundle;
the old ordinary result is historical, since publication/reads changed. It does not reinterpret a correctness pass
as useful speedup or authorize adoption after the shadow checks alone.
