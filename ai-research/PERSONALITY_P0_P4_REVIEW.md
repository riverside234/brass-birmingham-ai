# P0–P4 correctness review — 11 October 2026

Reviewed the handoff, auxiliary goal training, search/teacher integration, learned repertoire and unified deployment workflow. Confirmed defects were fixed in the personality packages and covered by focused regressions. This was a code/data-integrity review, not a strength or 16-style qualification campaign.

## Findings and fixes

| Stage | Issue and consequence | Correction |
| --- | --- | --- |
| P0 | No confirmed new defect in stopped-run admission, accepted-parent selection, pinned evidence or compatible weight import. | Existing rejection/restore tests passed. The P1 source fix changes the shared personality fingerprint, so create a fresh handoff. |
| P1 | Matching model IDs alone could let externally selected actions or a different sampling temperature be treated as on-policy PPO data. | Require recorded model sampling at temperature 1 with no teacher before assigning personality collection rows. Legacy auxiliary-only input remains separate. |
| P2 | `prepare --distilled` could attribute an older teacher branch to a different pilot or newer generation sharing the same Step 5 parent. This could pair incompatible weights and PPO state. | Verify the pinned originating readiness bundle, exact pilot path and pilot model hash before importing the distilled weights. |
| P2 | A forced first action could defer goal generation to a payment subchoice or the unsupervised second action. | Only the first action root proposes a goal. A forced root retains any surviving intent and defers fresh proposals until the next turn. |
| P3 | Public effects were measured until the next own turn, misattributing opponent consumption/flips and crossing the Canal clear. | Aggregate adjacent observations after own choices; censor unavailable terminal/era-clear effects with explicit coverage/boundary fields. |
| P3 | Merchant beer was compared with bare merchant IDs, although native choices use `merchant:`-prefixed targets. | Count the actual native target format. |
| P3 | Archive refits could silently replace retained source receipts before re-encoding. | Verify the previous archive's pinned manifest and receipt hashes before reusing its evidence. |
| P3 | Large proposals were allowed, but an entire proposal had to fit within the 4,096-game collection limit. A 2,048-candidate proposal could not be collected even with one setup. | Add bounded candidate slices preserving the full proposal, controller IDs and matched setup/seat/opponent/sampling streams. Only the slice containing candidate zero collects the neutral control. |
| P4 | Neighborhood selection compared the nearest distance with the largest unrelated pair's noise, and could stop before trying another valid candidate. | Check every candidate against each selected member's own pairwise noise threshold, then choose among remaining compatible neighbors. |
| P4 | Blind examples all came from the first setup and seat; partial chunks could change the chosen examples. | Freeze outcome-independent cases from the planned jobs, cover every member and balance setups/seats. Missing planned cases stay missing. |

No reward coefficients, PPO settings, search budgets, worker counts, release thresholds or installed game code changed. P3 also avoids extracting unused opponent trajectories when only active candidate records are requested.

## Compatibility and lab use

The Step 5 implementation identity remains `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`. Existing Step 5 checkpoints, main self-play, IL handoff and base generated-data formats are unchanged. The personality additions remain separate post-Step-5 commands.

Personality contracts intentionally change when their source changes. After updating, finish Step 5 and create a **new P0 handoff → P1 pilot → P2 readiness/teachers → P3 discovery/descriptor/archive → P4 roster/panel** in new directories. Do not rewrite old hashes or reuse qualification receipts for changed code. The new P0/P1 source identity is `3bb0eab8a753a5fb29e33b595e19e6ac0ab6d2e2d09875c059e93c9387bee068`.

P3's derived schema is **`public-behavior-trace/1.1.0`, 32 turns × 95 fields**, previously 93 fields. The additional `effects_observed` and `era_boundary` fields distinguish missing/censored consequences from observed zero effects. Old descriptor weights, latent coordinates and downstream rosters require regeneration; no converter pretends they are equivalent. This does not change the underlying game's replay format.

Instructions: [P0/P1](../docs/PERSONALITY_P0_P1_LAB.md), [P2/P3, including candidate slicing](../docs/PERSONALITY_P2_P3_LAB.md), [P4](../docs/PERSONALITY_P4_LAB.md). For 2,048 candidates and two discovery setups, use four 512-candidate slices with the same seed and distinct output directories. For eight validation setups, use 128-candidate slices. Supply all matching slices to fitting/archive construction; duplicates are rejected.

## Verification

- **59 focused tests passed**: 17 P0/P1, 14 P2, 12 P3 and 16 P4. This includes 12 additional regression methods plus expanded existing coverage. The combined run took 84.5 seconds.
- A **single native four-player direct-preview game** completed both eras through the actual rules host, collection, updated 95-field extraction and P4 report generation. It correctly remained incomplete and unqualified. The fixture report is `artifacts/personality/p0-p4-review-20261011/smoke-report.json`.
- That smoke took **72.5 seconds while the test suite ran concurrently**. This is not a search/GPU throughput comparison or evidence of a performance regression.
- Independent cross-review checked P4's pairwise filtering and frozen blind selection, and the P1/P2/P3 interactions. The new P4 regressions also passed independently.

Reproduce the focused tests with the project's dependencies available:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
export PYTHONPATH="$PWD/src:$PWD/tests/R2:$PWD/tests/Recovery:$PWD/tests/Personality${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest test_personality test_review_p01 test_search test_search_training test_repertoire test_deployment -v
```

## Remaining limits

The actual trained parent is still on the lab. This review cannot establish goal calibration, preserved strength, human-recognizable styles, stable 16-by-8 neighborhoods or CUDA utilization. Small authored fixtures must not supply those qualifications. P4 remains an intermediate evidence gate; P5 difficulty calibration and P6 laptop deployment are later work. Era-clear and final-action consequences without an observable successor remain explicitly censored rather than reconstructed from hidden state.
