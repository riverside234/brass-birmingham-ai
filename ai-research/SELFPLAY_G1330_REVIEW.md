# g1327–g1330 DOMiNO review

Reviewed 2026-10-30 (America/Chicago). Inputs are the supplied exports ending in
`g1327-20261031T020511541347Z` and `g1330-20261031T024940047516Z` (UTC export names).
The latter state is committed at **generation 1330 / 425,600 ordinary games**.
Report directories are zero-based: `iteration-1329/report.json` describes the
update that committed g1330. G1328 is available in compact retention history.
The g1330 export reports no pending iteration, no run lock and no export warnings.

## Findings

| Committed generation | DOMiNO evidence entering PPO | PPO steps / planned | Exact PPO raw-T1 KL | Retained / attempted PPG steps | PPG stop |
| --- | --- | ---: | ---: | ---: | --- |
| 1327 | Bootstrap, task-only | 50 / 50 | 0.007385 | 16 / 32 | Held-out loss increased |
| 1328 | Ready | 50 / 50 | 0.008084 | 0 / 16 | Held-out loss increased |
| 1329 | Ready | 52 / 52 | 0.008354 | 0 / 16 | Held-out loss increased |
| 1330 | Ready | 29 / 50 | 0.010177 | 0 / 16 | Held-out loss increased |

All four configured fixed entropy coefficients are **0.10**; adaptive entropy
remains off. Epsilon is 0.08, the first three action decisions use T2, branches
use 1% with 50% T2 / 50% q fork moves, and insolvency remains 0.005 capped at 0.05.
Every collection retains 160 self-play / 154 history / 6 v4 tables and a
24-member history pool. These settings were not changed by this review.

**Context/evidence checks passed in the exported reports.** Each collection has
1,268 assigned and completed neural seat-games, with matching counts by strategy,
table, role and physical seat. There are 960 ordinary learner seat-games, of which
908 across 302 training games enter DOMiNO statistics. The other 18 whole games
are held out. No terminal phase is missing. Each strategy already exceeds the
32-seat-game / 16-distinct-game readiness thresholds after g1327. The recorded
coefficients, EMA and dual direction behave as specified; no evidence-regime reset
occurred. By g1330, coefficients for nonneutral strategies are approximately
0.493–0.508 for the next iteration; neutral remains exactly 1.

**PPG is not retaining fresh value learning after bootstrap.** The first pooled
held-out loss changes in g1328–g1330 are respectively +0.003126, +0.001401 and
+0.007757. These violate the current non-increasing-loss rule. The intervals are
correctly rolled back before the first scheduled 32-step KL check. The small
increase at g1329 still violates that rule; this recipe has no one-standard-error
tolerance. This was already intermittent before DOMiNO (zero retained steps at
g1322, g1323 and g1325), so the four new iterations do not establish a DOMiNO cause.
No acceptance threshold, LR, beta or patience was loosened.

The intrinsic head was inactive during g1327's retained PPG interval. Subsequent
PPG intervals all roll back, and intrinsic PPO preservation drift remains zero.
Thus there is no retained intrinsic target fitting in these four iterations.
PPO preservation-step counts alone are not evidence that the new head learned.
The new reports make this distinction explicit.

**The intrinsic signal is currently very small.** At g1330, raw task-advantage RMS
is 0.21731, intrinsic-advantage RMS 0.00006546, and the weighted intrinsic
contribution RMS 0.00003287. The implementation follows the frozen descriptor and
phase-normalization contract. It does not provide evidence that useful distinct
strategies are emerging. Any change to descriptor/reward scaling should be a
separate versioned experiment, with task quality and matched-opponent diversity
measurements; this review does not silently rescale stored centroids or rewards.

**The PPO guard worked.** At g1330, exact raw-T1 KL crossed 0.01 and stopped PPO
after 29/50 steps. Actual-behavior KL was 0.008050. One stop after more than half
the planned steps does not require an immediate LR cut under the existing rule;
the three-iteration median is 0.008354. LR remains 7.8125e-6. Economic health
passes: the five-iteration self-play window has 124.70 VP/seat, 0.19% zero-VP seats
and 1.84% insolvent seats. Anchor KL is 0.10857 versus its 0.15 overall allowance;
the group allowance remains 0.30. An anchor-only permission flag does not override
the PPO controller's early-stop/median checks.

**Playing strength is unresolved.** G1330 is 781.6 PL points (95% interval
733.5–833.2). Within that same fit/bootstrap, g1330 minus g1320 is −8.9 points,
95% interval −75.5 to +39.5. This establishes neither improvement nor deterioration.
It is not a seven-context strategy qualification. No new evaluation campaign ran.

## Fixes

1. **Remove rowwise GPU weight writes.** Critic preparation wrote each snapshot's
   weight separately, including roughly 29,000–30,000 redundant writes per
   iteration at the current fixed weight of 1. The all-one case now uses the
   tensor already created by Dataset; other weights use one bulk transfer.
   Values, role masks, holdouts and RNG are unchanged. Preparation reports now
   separate dataset construction, weight assignment, splitting, local inventory
   and DOMiNO preparation, so remaining bottlenecks can be identified.
2. **Keep loss attribution separate.** The combined PPG companion previously
   passed local-head plus intrinsic loss to the local-head telemetry accumulator.
   It now reports only the local loss there. The summed backward objective is
   unchanged.
3. **Do not report an unchecked attempted KL as zero.** A phase rejected at step
   16, before its first KL check, now reports
   `maximum_attempted_probe_kl: null` and `attempted_probe_kl_checks: 0`.
   The retained policy's zero KL remains correct after full rollback. Periodic,
   epoch-tail and retention checks still enforce the same 0.002 safeguard.
4. **Report actual intrinsic fitting.** `diversity.intrinsic_fit` distinguishes
   attempted/retained steps, presentations and distinct rows, with retained
   counts restored together with the selected weights/Adam. Held-out diagnostics
   now include prediction RMS and target RMS. Compact history retains these fields.

## Performance and validation

Collection took 220.0 / 301.4 / 298.6 / 292.8 seconds. PPO took 83.3 / 183.0 /
181.4 / 160.0 seconds. The full reports attribute 89.2 and 81.6 seconds to critic
preparation at g1329/g1330, but do not split that time further. Redundant GPU
writes are a confirmed avoidable cost, not a proven explanation for that entire
delay. The new timing breakdown will resolve the remaining contribution.
The g1330 total also includes 309.6 seconds of scheduled evaluation.

Context-fingerprint cache reuse is working: misses fall from 144 to 30 to 10 to
6; measured fingerprint work falls from 19.07 seconds to 0.84 seconds. Only six
nonneutral current-model fingerprints require fresh passes in g1330.

The isolated authored smoke passed in **187.08 seconds on CPU**: three iterations
with 126 synthetic complete tables each, one PPO step and at most six PPG steps,
plus an exact repeat of iteration three after save/reload. Weights, all Adam
states, DOMiNO statistics/duals and RNG matched. The active iterations retained
six intrinsic fitting steps and 1,280 distinct eligible rows each; current-only
masking and joint rollback work. This uses a smaller 1,511,464-parameter model
(10,369 intrinsic parameters), not a full production training/evaluation run.
Artifact: `artifacts/domino-g1330-smoke-01/report.json`.

The read-only export audit verified all four generations' assignment/completion
counts, per-ID support accumulation, coefficient continuity, EMA arithmetic,
unchanged composition/entropy, cache ages and retained KL limits.
Artifact: `artifacts/domino-g1330-evidence-audit.json`.

**62 focused regressions passed in 326.39 seconds**, including five new tests for
bulk snapshot weights and masks, local/intrinsic loss attribution, unchecked KL,
rollback-aware fitting counts and the real g1330 source-only continuation. The
suite also covers context assignment/rotation, strict behavior verification,
current-only intrinsic fitting, base/companion gradients, PPO/PPG rollback,
shadow parity, historical/forced/branch eligibility, checkpoint and source
compatibility. Log: `artifacts/domino-g1330-regressions.log`.

The CPU-only weight-construction microbenchmark uses g1329's 147,359 rows,
including 29,665 snapshots, two Torch threads and seven repetitions after warmup:

| Snapshot weight | Before, median | After, median | Indexed writes after |
| --- | ---: | ---: | ---: |
| 1.0 (current) | 564.73 ms | 0.20 ms | 0 |
| 0.5 | 533.81 ms | 22.22 ms | 0 |
| 0.0 | 202.55 ms | 21.54 ms | 0 |

All output tensors match exactly. These are isolated construction timings, not
GPU or whole-PPO latency estimates. Artifact:
`artifacts/domino-g1330-weights-benchmark.json`.
No production model, optimizer, population, probe, cache or run state was edited.

`git diff --check` passed. Reviewed source:
`fa4514f0c562e7eb247e3d4a75b4b38e2304ee11cfaadeaee3702ae76310447d`.
Learning compatibility remains
`ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
The exact ordered prefix of 75,608 source-review entries is preserved, with 244
narrow entries appended for the source update and existing approved adoption paths.

## Continue from g1330

After updating the code, use the already generated configuration. Do not rerun
the g1326 recovery/adoption helper or clear the compatible two-age PPG cache.

```bash
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-domino-entropy010-g1326.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1330 --iterations 20
```

This requests g1331–g1350. The source-only transition preserves the current
configuration, DOMiNO adoption origin/evidence/duals/head, model/optimizer/RNG,
training progress, pool, GAE and compatible PPG replay. The trainer still rejects
unreviewed configuration edits and incomplete pending collections. Full production
checkpoint files are not in these exports; normal continuation verifies them on
the lab machine before training.

Implementation files: `recovery/actor_value.py`, `domino_auxiliary.py`,
`ppg_g210.py`, `value_phase.py`; focused tests in
`tests/Recovery/test_domino_g1330_review.py`; source review registry and this report.
