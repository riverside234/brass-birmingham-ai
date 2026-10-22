# VP reward capped at 180

22 October 2026. The prepared continuation now uses:

```text
terminal utility = win_share + 0.10 × pairwise_placement_credit
                 + 0.001 × min(actual_VP, 180)
return           = terminal utility − future capped insolvency costs
```

The VP term contributes at most **0.18**. Insolvency remains **event_cost=0.020, max_cost=0.080**. Official VP, win/tiebreak decisions, rankings and the separate VP prediction target remain uncapped. Scores of 250 and 240 still produce different official ranks even though both receive the same additive VP reward.

## Versioning and coverage

- `brass-combined-return/2.0.0` requires `vp_cap=180`. Archived v1 specs and coefficient-only configs keep their original uncapped behavior. Old files are not silently reinterpreted.
- Ordinary replay returns and search terminal returns share the capped computation. Reward-component reports and weighted VP reward use the capped value; official gameplay/health statistics continue using true VP.
- The utility prior caps its VP contribution for v2 only. The existing learned correction, official heads and all parameter shapes remain unchanged. New reward metadata changes model identity/provenance. The exact architecture-source receipt preserves tensor/IL compatibility while admitting only this reviewed conditional source change; unrelated source edits still change compatibility.
- Fresh self-play config generation explicitly enables the cap. Also corrected the fresh generator's use of an already-v7 template: its intermediate PPG-disabled scaffold no longer leaves fresh PPG disabled. The signed continuation converter changes only `reward` and matching `search.reward`.

## Continuation and caches

The explicit migration restores weights, both Adams, RNG and ordinary progress. It invalidates utility calibration so the existing bounded head-only recalibration uses new targets. It retains the completed IL identity, opponent pool, LR/warmup, GAE state and economic/entropy baselines.

Old PPG Monte Carlo targets are excluded, never relabeled. At the reward transition's committed generation N, reset the PPG replay/schedule origin and target-dependent epoch votes/KL-trip history; preserve the attained epoch cap and beta. After durable source/config adoption, verify the committed reward identity, audit both owned caches, then delete recognized cache files. Eight fresh collections precede the next PPG phase. Both rolling v6/v7 cache formats are covered. Later restart retains the new origin and caches and does not repeat the reset. Normal expiry still follows checkpoint commit.

For the pending g540 recipe, its existing explicit cap hotfix remains two, floor two and maximum six; phases are due at **g548 and g556**. No production checkpoint, progress or exported result was edited locally.

## Commands

After syncing this revision, from the lab repository, when committed progress is still g540:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 540 --apply && \
python scripts/joint7-g540-ppg.py --expected-generation 540 && \
python scripts/joint7-vp-cap.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-one-se-g540.json \
  --expected-generation 540 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-vp180-g540.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This continues g541–560. The new tracked VP-180 config is the prepared one-SE g540 config with only the VP-cap and matching search-reward fields added. Its helpers refuse conflicting outputs and do not modify run/checkpoint files. Repeat only the train command for later clean continuations.

If the g540 recipe has already advanced, recover an incomplete collection using its actual committed generation. Then run `scripts/joint7-vp-cap.py` without `--config`, using `--expected-generation N` and a new `--output configs/step5/joint7-v11-2-grouped-lr-critic-vp180-gN.json`. That derives only the cap from the current committed config. Use that output in the same reviewed train command. Keep the existing completed IL directory; fresh IL is not needed.

## Validation

**41 distinct focused regression cases passed across runs and corrected reruns**, including eleven new VP-cap tests:

- 179/180/181/high-VP boundaries; the capped term saturates while real official ordering/ties and raw VP targets remain intact;
- unchanged archived linear rewards, causal .020/.080 insolvency accounting and reward reporting;
- strict schema, reward/source identity, search/learner agreement, preservation of other settings and fresh defaults;
- utility prior consistency and ONNX Runtime parity at batches one/two, including VP over 180 and different seat mappings;
- all weights, both optimizer states, RNG and checkpoint bytes preserved across explicit/interrupted adoption;
- owned-cache cleanup only after reward adoption, old-target rejection, fresh schedule origin and restart continuity for rolling v6/v7;
- existing PPG one-SE selection/rollback, GAE, search-return and rating-resume regressions.

Two pre-existing test assertions still expected an obsolete fresh recipe and a sub-2M model. Their checks now assert the current fresh recipe and actual declared parameter budget while retaining checkpoint/Adam/RNG and old-reward coverage. Initial new-fixture issues were corrected; no unresolved test failures remain.

The isolated smoke used authored mixed-table/branch fixtures, two CPU threads and a **1,511,464-parameter small model**. Real PPO updates ran for g541–548; PPG retained **56/56 steps**, **869 distinct critic rows**, and used **48 ordinary held-out games across all eight fixture ages**. Production retains its normal 144-game holdout and full tensor budgets. Retained overall/age-0 exact KL were **0.0000014781 / 0.0000014775**, below both 0.002 limits. Repeating g548 from g547 reproduced weights, both Adams, RNG, phase clock and epoch control exactly; incoming fixture checkpoint bytes were unchanged. Total smoke wall time was **98.1 seconds**, alongside tests, not a speed benchmark.

Evidence: `artifacts/vp180/smoke/report.json`, `source.json`, `before.json` and `validation.json`. Reproduce with:

```bash
PYTHONPATH=src:tests/Recovery python -m unittest -v test_vp_reward_cap
PYTHONPATH=src:tests/Recovery python tests/Recovery/smoke_ppg_one_se.py \
  --vp-cap --output artifacts/vp180-smoke-new
```

The scalar/tensor clamp adds negligible expected computation; the material one-time consequence is discarding old-target PPG caches and waiting for eight fresh ordinary collections. No long training, full GPU benchmark or playing-strength claim was made.

Implementation: `b95f45e2b0be6672a4efb7a3cabf9e2fab3e152a667e7f229b6c09ab5cb01c71`. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
