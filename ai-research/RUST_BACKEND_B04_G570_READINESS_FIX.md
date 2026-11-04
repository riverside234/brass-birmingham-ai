# B04 g570 null-readiness qualification (r10)

Reviewed 2026-11-03, America/Chicago. Received report:
`results/rust-lab-qualification/20261104T020129387978Z-20261104T015819818200Z-g570-all-4c81aca9-ceb8ad68`.

**B03 passed** full isolated restoration and replay admission. The CUDA topology
was one saved device, eight visible devices, one restored stream. B04 compiled
and passed its native and Python tests. It deferred before GPU/search work or
table dispatch because g570's utility receipt was null and generation 568 had
been retired. The user explicitly authorizes using g570 with null treated as ready.

r10 implements that authorization only in B04's diagnostic readiness path. It
requires generation 570, checkpoint SHA
`06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9`,
model `18f42a48933c18095d96f1cc64b836d3f9aac92361a4e4ee7c423830d9a11a91`,
the `joint7-v11-2-g1290-trained-restart` run and matching utility reward. Existing
checkpoint/state/device and source/runtime checks still run. It does not scan
retired history to establish an assumption the user has explicitly supplied.

`readiness.json` preserves `receipt: null` and sets `ready: true`; evidence records
`user-authorized-g570-null-utility`, `calibration_measured: false` and the exact
identity/reward. The isolated shadow phase receives a ready assertion carrying
that evidence so its normal begin/finish lifecycle retains it. The model's null
calibration and all original checkpoint bytes remain unchanged. Explicit false
or mismatched receipts are not overridden. Other checkpoints keep their existing
readiness rules. The production teacher/readiness implementation is unchanged.

The implementation changes only B04 diagnostic readiness/receipt admission and
the exact prerequisite-reuse verifier. Source-continuation reviews cover this
diagnostic revision without a recipe change. The Rust producer, inference, PPO,
PPG, companion heads, optimizers, RNG restoration and game/search rules are unchanged.

The new sealed reuse review retains original B01/B02/B02.5 receipts and adds the
exact passing r9 B03 receipt. Every other executable/runtime inventory entry
still matches the original baseline. B03's preflight, restoration, replay and
fork pins and unchanged-run binding remain mandatory. Older evidence is not
rewritten or represented as a newly executed qualification.

The optional `--b04-build` argument reuses the exact already-tested native build
from this report. Verify its original receipt, B02/B03 pins, native producer,
base/core overlay, lockfile, executable, bootstrap and dependencies. A new
diagnostic build binding records `reused_from`, retaining the original build
identity and source receipt. This avoids recompilation; it does not waive actual
GPU calibration, timed search, label correctness or shadow-table requirements.

Use the [B04-only command](../distribution/rust-lab-g570/README.md). No B01/B02/
B02.5/B03 qualification or B02 collection is repeated. Keep training paused at
g570 and retain both original lab artifact directories. B04 still runs bounded
GPU probes and two concurrent one-thread shadow tables with search_v1, 4s/K16
plus at most 1s post-work and the existing watchdog. No training update occurs.

Local regression coverage includes exact-null acceptance, no historical scan,
identity/reward/false-receipt rejection, shadow lifecycle without model mutation,
original B03 evidence/state validation, binary/dependency tampering rejection,
launcher receipt/build forwarding, existing B04 inference/cancellation/timing
fixtures and Git-visible result export. Hardware-dependent cases remain for the
lab. B04 GPU/search qualification and B05–B07 production adoption remain Pending.
