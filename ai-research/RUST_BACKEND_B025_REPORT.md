# B02.5 — policy-only teacher phase (T1)

2026-10-29. **Implemented; lab qualification Pending.** No production run, checkpoint, installed game or immutable B01/B02 transfer was changed. B01/B02 lab qualification, B04 native exports and B05/B07 training adoption remain open.

## Implemented

- `recovery/teacher_phase_config.py`: explicit `teacher-phase/1.0.0`, default off, legacy-path exclusion, policy-only teacher value default, exact 4s/K16 search_v1 profile, bounded term/buffer options and reviewed adoption. Historical explicit teacher-value configurations still validate. New fresh-config generation adopts off; saved presets and lab run recipes are not silently rewritten.
- `recovery/teacher_labels.py`: float64 improved-policy oracle verified against the reference/golden trace; ordered all-legal targets; generation/model/export/source/reward/job/epoch/observer/revision binding; encoder and logits parity. Partial rounds, short quotas, overrun and fallback provenance are accepted. Forced/zero-evidence/canceled labels are dropped; integrity failures fail admission.
- `recovery/teacher_buffer.py`: separately owned train/held-out gzip shards and receipts, hash/lineage checks, four ages, per-label 4:3:2:1 sampling with independent RNG, profile invalidation, immutable publication and post-commit cleanup limited to known receipts. Missing/changed committed shards fail; unknown files survive cleanup. Pending differing recollections require explicit recovery rather than overwriting evidence.
- `recovery/teacher_term.py`: cached admitted encodings, staleness by age, raw T1 policy-only KL, shared-encoder gradient share/cosine, shadow measurements, bounded beta calibration and held-out before/after measurements. No critic, utility, descriptor or local-scoring graph is constructed. The term uses the existing PPO optimizer before clipping and its existing KL probes; it adds no Adam steps. An active term requires a nonforced ordinary KL probe.
- `recovery/teacher_phase.py`: frozen 28-train/4-held-out job contract, a separate setup/sampling stream, versioned four-phase reservations, teacher-first collection order and a separate admission barrier. Teacher results never become ordinary games or actor/critic targets. It persists teacher cursor, readiness, beta and buffer receipts in checkpoint/state, with current-reward latest-non-null readiness. Explicit not-ready receipts still defer collection; admissible older labels can still train.
- `updates.py`, `league.py`, `history.py`: PPO term hook, off-mode lifecycle, checkpoint/report/console integration. The original LR/IL-KL/PPG/local-scoring flows remain in place. Live non-off collection is explicitly unavailable until B04; Python enabled-mode tests use authored fixture labels through the new interface.
- `resume.py`, reviewed-source receipts and `configure-teacher-phase.py`: preserve full same-run state; exact off adoption, narrow mode/share toggles, and readiness preservation only for those toggles (also across a restart before the next checkpoint). Other source/reward/recipe changes retain the existing reset. Reviews carry the prior g1324 routes forward and pin this implementation; there is no generic source-admission bypass.

`TEACHER_SEARCH_PHASE_REVISION_PLAN.md` remains the detailed schema/equation/algorithm specification. Native B04 should call `TeacherPhase.begin()` with its frozen export/epoch, prepend its returned jobs using `collection_order()`, return only teacher results to `admit_collection()`, and keep ordinary records on their existing ingress path. `prepare()` provides the optional PPO term; `finish()`, `payload()` and `committed()` integrate diagnostics and the durable transaction. Replace the explicit live guard only after B04's real exports and cancellation/identity tests pass.

## Short evidence

The B02.5 suite has 24 authored fixture tests covering config/defaults, golden-trace math, label tampering and dropped-evidence cases, weighted sampling/staleness, ownership/expiry/resume, separate seeds/held-outs, CPU off/shadow equivalence, head-gradient isolation, KL participation, exact model/Adam/RNG restart, DCPG shared-encoder shadow equivalence, source/toggle receipt handling and the locked helper. It uses no full games. The additional focused regression batch checks the prior g1324 recipe/helper/PPG-state preservation and seven existing categorical-KL guard cases (10 tests total).

Local logs/receipt: `artifacts/rust-backend/b025-development/`. The sealed transfer manifest records the actual tested implementation and outcomes. CPU fixture evidence does not certify Linux/CUDA, native teacher exports, GPU search throughput or end-to-end training continuation.

## Remaining gates

- **Lab B02.5:** run the [bundle commands](../distribution/rust-lab-b025/README.md); optional CUDA fixture and read-only actual joint7 adoption inspection are still Pending here.
- **B04/T2:** real Rust root/continuation/replanned label exports, GPU trajectory calibration and 4s whole-turn scheduling; cancellations/late replies; two-table shadow smoke. Collection phase/search/lane/inference metrics must come from that collector; fixture-derived label metrics are not measured search throughput.
- **B05/T3:** full fixed-data PPO/PPG/auxiliary continuation, partial-buffer restart/toggles and rollback on the actual lab recipe.
- **B06/B07:** measured cost and two isolated shadow iterations before any production enablement. No long late-training campaign is started by this patch.

Transfer: `distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip`. `bundle.json` and `transfer-manifest.json` carry hashes. Lab results retain `method=search_v1`, `v205_enabled=false` and an explicit Python-only scope.
