# DOMiNO debug and performance review

Reviewed 2026-10-30 after the initial D0–D4 implementation. This follow-up keeps
the selected run's configuration, model architecture and training objective.
It does not repeat DOMiNO adoption, reset PFSP evidence or clear compatible PPG
replay. No production checkpoint was edited and no long training/evaluation ran.

## Correctness fixes

- **Context assignment must be complete.** Every neural physical seat must have
  its own valid assignment, including frozen history/IL seats. Validate all four
  seats before installing any context. A missing receipt can no longer silently
  use the old mixed-context fallback or neutral historical context.
- **Collection, critic and replay receipts must agree.** Cross-check the exact
  strategy receipt in critic descriptors against the collection assignment.
  Reject a changed receipt within one cached seat-game instead of letting a
  later row overwrite the earlier context. Learner/frozen roles remain distinct
  even when their base model weights have the same identity.
- **Intrinsic targets stay attached to their original rows.** Bind current
  targets to the collection receipt and reject duplicate decision identities or
  a conflicting role/context. Current ordinary learner training rows alone fit
  the intrinsic critic; prior ages, held-outs, snapshots and branches remain
  excluded. Current held-outs still have targets for diagnostics.
- **Fingerprint caches cannot certify the wrong policy.** Validate the nested
  fingerprint context/probe contract as well as the outer cache key/seal. Before
  an uncached fingerprint pass, reject a model whose parameters or buffers have
  changed since the provider froze its identity. Previously a reused provider
  could write new-weight probabilities under an old model ID.

These were admission and cache-identity gaps found by inspection and deliberately
malformed fixtures. They are not evidence that the live trainer produced
incorrectly labelled games. The existing planned assignments, sampling streams
and correctly sealed replay are unchanged.

## PPO/PPG integration and performance

PPO still combines unnormalized task and intrinsic advantages once, then applies
the existing shared normalization and opening objective. Vectorization replaces
per-row tensor reads/writes; scalar-versus-vectorized checks are bit-exact in
float32 and float64. Coefficients, branch masking and RNG are unchanged.

PPG now reads eligibility from small sealed replay columns instead of hydrating
complete row/view/action metadata repeatedly. An iteration-local weak-reference
cache stores masks, targets and context IDs; it is cleared on each new
collection. It holds no autograd graphs or duplicate encoded policy inputs.

When local scoring heads are absent, inactive or have no labels, the intrinsic
head now reuses the task critic's encoder forward. Task descriptor dropout is
preserved while the intrinsic head still uses true seat descriptors. The usual
active, labelled local-head path already shared this forward before the audit.
Gradient tests compare the optimized path with separate original forwards and
check zero intrinsic gradients on ineligible rows. Rejected PPG intervals restore
base weights, value Adam, the intrinsic head and both intrinsic Adam states.

CPU component benchmark: same authored inputs, two Torch threads, no GPU or
optimizer steps. Medians from the recorded repetitions:

| Component and measurement scope | Before | After |
| --- | ---: | ---: |
| PPO advantage mixture, 32,768 rows | 1,517.7 ms | 11.6 ms |
| PPG eligibility, 4,096 rows, warm iteration-local cache | 44.69 ms | 0.196 ms |
| Task + intrinsic forward, 128 rows, no active local heads | 92.05 ms | 43.53 ms |

The forward benchmark uses the 512-wide, 5,928,616-parameter utility model and
reduces encoder calls from two to one. Sampled process RSS was 424 MB; this is
not a measured OS peak. Artifact:
`artifacts/domino-audit-benchmark-01/report.json`.

These component gains do not establish whole-iteration or GPU speedups. In the
current run with active labelled local heads, the repeated metadata lookup and
advantage mixing are the relevant improvements. Measure the next ordinary
iterations before attributing a change in total PPO/PPG time to this patch.

## Validation

- **111 focused tests passed in 259.75 s.** Coverage includes context assignment,
  native/branch admission, strict behavior probabilities, current-only auxiliary
  fitting, gradients, rollback, shadow-mode parity, source/config transitions
  and retained cache compatibility. Log: `artifacts/domino-audit-regressions.log`.
- **Isolated training smoke passed in 116.50 s on CPU.** Three authored iterations,
  each with 126 synthetic complete tables, one PPO step and up to six PPG steps;
  the third was repeated after save/reload. Model/auxiliary weights, all Adam
  states, committed statistics and RNG matched exactly. This smoke uses a smaller
  model to bound optimizer work; the separate inference benchmark uses the
  representative 512-wide model. Artifact:
  `artifacts/domino-audit-smoke-01/report.json`.
- **28 existing native/reference game families and seven branches revalidated.**
  All 10,021 records passed seals, model/context attribution, original sampling
  RNG/probability and branch inheritance checks. This was a read-only check of
  prior smoke fixtures, not newly collected games. Artifact:
  `artifacts/domino-audit-family-revalidation.json`.
- `git diff --check` passed. The existing source-review prefix and learning
  compatibility are unchanged.

## Continue an already adopted run

After updating the code, use the existing DOMiNO config and normal reviewed
source continuation. Do not rerun the generation-1326 recovery/adoption helper if
the run has advanced. For the config name generated by the original command:

```bash
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-domino-g1326.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This resumes the actual committed generation; it does not rewind to g1326.
Existing pending collections remain subject to the normal source/freshness
checks. The patch preserves weights, all optimizer/RNG state, progress,
population, DOMiNO statistics/duals and compatible two-age PPG replay.

Reviewed source: `d980354d39b02f27aeee147e91ba32c24f55a879647a3d1f27615410ca9e012a`.
Learning compatibility remains
`ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
The prior 75,244 source-review entries are retained exactly, with 121 narrow
entries appended for this source update and the already-approved adoption paths.

Implementation files changed in this review: `domino_sampling.py`,
`domino_auxiliary.py`, `domino_fingerprints.py`, `critic_data.py`, `ppg_replay.py`,
`updates.py`, and `local_scoring_auxiliary.py` under `src/brass_ai/recovery`, plus
the source-review registry, focused tests, benchmark and documentation.

This review verifies implementation behavior. It does not demonstrate stronger
play or seven reliably distinct strategies; those remain empirical questions.
