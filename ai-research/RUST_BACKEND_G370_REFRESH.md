# Latest-restart task-only lab qualification refresh

Date: 2026-11-02 (America/Chicago). B01–B04 tooling updated; **Linux/CUDA qualification Pending**. No training-run mutation or installed-game deployment.

The training run is `joint7-v11-2-g1290-trained-restart`. Qualification uses its **latest committed iteration**, expected to be later than the supplied report (for example g390). The launcher captures the actual generation/checkpoint/state once, passes exact pins to dependent gates and rejects advancement during qualification. Optional generation/hash flags assert that latest state; they never rewind. Continue the run with all optimizers, RNG, counters, population, local-scoring companion, PPG ages and gates intact. The historical g1290 weights-only seed was its original startup, not permission to repeat a restart.

The supplied g370 / 118,400-game report remains historical recipe evidence: `generation-0370.pt`, SHA-256 `198af418247e6559eb4e2413acb02afd54e7baa5e446a30fb6ce1442c37dfcdc`. It is not a hardcoded qualification target.

## Current recipe and restoration

The exported report predates the requested `joint7-v11-2-g1290-trained-restart-g370.json`. The bundled config is generated using the existing `training_revision_g370.configure` transformation: GAE 0.95 with floor 0.95, self-play branches 0.5% with T2-only alternate choice, and PPG v20 beta floor 1. Other settings retain the actual restart recipe: neutral contexts/fingerprints, DOMiNO and context panels off, fixed entropy 0.005, T1.3 three-action openings, epsilon 0.05, insolvency 0.02/event capped at 0.075, economic-only LR control, current PFSP/manual-startup provenance, local-scoring auxiliaries, PPG two-age cadence and 16-step loss / 32-step KL checks.

Qualification expects the later iteration to have adopted the g370 recipe and requires its lab signature to match the requested config. B03 then restores its exact full phase state without repeating adoption. At a signature/checkpoint boundary only, B03 understands the exact already-authorized g370 GAE/beta restoration and verifies the projected phase payload in an isolated fork. Unrelated pending recipe migrations remain rejected. All actual checkpoint bytes and source-state hashes stay unchanged.

## Removed diversity path

Removed the AZ_DB teacher context/evaluator module, intrinsic fit receipt module, Rust intrinsic reward/continuation overlay, GPU companion, normalization gate, conditional label/buffer branches, CLI enablement, producer and qualification paths. Current teacher jobs are neutral and use only the task utility. Active lab capture, planner and teacher validation reject DOMiNO-enabled configs.

Historical ZIPs, reports and source approvals remain immutable. Dormant ordinary DOMiNO schema/legacy support remains where older signed training records require it; current qualification never creates its heads, optimizers, contexts or jobs. No AZ_DB production path remains. Old conditional teacher states are not silently converted into task-only labels.

## Timing and finalization

`teacher-phase/1.1.0`, `brass-lab-teacher/1.1.0` and `search-plus-post/1.0.0` bind the new timing. Search starts with an absolute 4-second work cutoff and a 5-second delivery cutoff. It receives no 0.4-second deduction. All second actions, payment/continuation work and replans narrow the same budget. Finite quota/admission rules may finish early; they cannot use the post-search second for more trajectories.

Native finalization measures selection, label construction, main-game commit, IPC/verification and total post-search work separately. The whole turn shares a cumulative one-second post-work cap, including work between searched actions. Late or canceled verification fails the unpublished table, rather than extending the clock or publishing partial results. The 30-second stuck-process watchdog remains a failure-recovery mechanism, never a search allowance.

Labels from a completed turn cross IPC together. Python checks the frozen job/actor/revision/legal order, recomputes the full improved policy, independently encodes rows and batches the frozen-model logit parity check. A private, one-use verified-table cache avoids repeating that work at the phase boundary. Changed model/job/table bytes, duplicate consumption, cancellation and deadline expiry fail closed. Verification serializes the reference model/RNG context across two lanes. Shards use level-1 gzip; deferred sealing is measured and charged against the final searched turn's remaining post-work allowance. There is no separate fresh sealing allowance.

Search_v1 recommendation semantics, completed-round selection, task-utility arithmetic, full legal denominator and unknown search descriptors are unchanged. Label sources still use the search-v1 improved-policy equations; timing/profile/native producer hashes prevent mixing new and old qualification. Search_v2/V205 are disabled.

## Qualification and limits

Use the [current bundle commands](../distribution/rust-lab-restart/README.md). Results export to a new Git-visible folder on success, failure or deferral. B01 is bounded adapter/feature/device checks; B02 uses at most three ordinary families and a branch; B02.5 uses authored policy/resume checks; B03 reuses B02 families and restores the full incoming checkpoint without training; B04 calibrates on two roots and collects exactly two shadow tables using independent one-thread sessions.

Local development evidence is recorded in `distribution/rust-lab-restart/local-development.json`. Authored fixtures and count probes do not establish g370 CUDA timing, full-checkpoint restoration or training strength. B04 must report search and post-search times separately and pass the one-second post-work cap on the lab. B05–B07 retain the short-smoke requirement, ordinary-only PPO/PPG preservation, exact continuation/rollback and explicit production enablement gates.

The bounded Python suite passed 125 tests with two expected skips. After the final native work-accounting adjustment, 17 focused checks passed with one Windows symlink skip; these included actual native count and timed-turn probes. A further 16 checks passed with one skip after adding latest-state capture, including g390 receipt/argument wiring and rejection if the run advances to g391. The current Rust release build passed 32 tests. With the authored small CPU model, the typical turn used 2.966 seconds of search and 67 ms of post work; the complex root (413 legal actions) used 3.991 seconds and 521 ms. Early search completion reflects the frozen quota. These are single-turn diagnostics, not a g370/CUDA benchmark or a demonstrated overall speedup. Historical archive hashes and all 78,357 existing source-approval entries were preserved.
