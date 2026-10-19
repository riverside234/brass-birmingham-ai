# Snapshot critic weight recovery

Date: 2026-10-19

The generation-85 review fixed reductions before actual snapshot training, but the controller could only decrease weight. It now restores weight in two stages after sustained recovery. This changes critic weighting only; frozen opponent rows still never enter the PPO actor loss.

## Rule

At weight 0.5, require five consecutive iterations with:

- Actual retained snapshot training in PPO or accepted PPG steps.
- Finite learner held-out EV at least the fixed Phase-1 learner EV recorded when snapshot rows were enabled.
- Existing economy/pass health, PPO warmup completed, a nonempty PPO update, no PPO early stop, exact PPO KL within 1.5 times target, and no PPG KL trip.

Then use weight 0.75 from the next iteration. Require another five qualifying iterations at that weight before using 1.0. Each stage starts a new streak. Missing evidence or a failed condition resets it. Merely attempted or available snapshot rows cannot qualify. Retained PPO snapshot rows can qualify when a PPG held-out-loss rejection restored its attempted changes. The LR controller's IL-distance limit/latch does not veto this gate; no IL sale-probability check is added.

The existing reduction remains: actual snapshot training plus EV below `baseline - 0.02` immediately selects 0.5 for the next iteration, clearing recovery progress. Recovery requires reaching the baseline itself. The resulting 0.02 gap avoids changing direction at a single threshold. Missing Phase-1 baseline, including forced snapshot mode without an adoption measurement, cannot authorize increases. No new automatic playing-strength rollback is introduced.

## State and reporting

`actor-value-phases/5.0.0` adds the versioned `critic_data.weight_recovery` state, containing origin, healthy streak and last weight-change generation. Legacy v3/v4 checkpoints migrate explicitly after the existing reviewed source/config check. They preserve weight, Phase-1 baseline, activation, fingerprints, model/Adam/RNG state and all unrelated progress. Only the new streak starts empty at committed progress. Restarting before the first new commit repeats the same migration; new checkpoints restore the exact streak and reject missing/corrupt recovery state.

The config and model architecture remain unchanged. The selected weight applies to snapshot critic rows in PPO and both generations of the PPG buffer. Learner weights remain 1.0. This does not change row sampling, holdouts, GAE, losses, policy KL caps, optimizer settings, population, reward, teachers or opponents. It adds no model forward passes or tensor caches.

Full and compact reports retain weights used/next, decision, reason, recovery eligibility, health, baseline/reduction thresholds, observed streak, next streak, required count and next target weight. Console output adds `snapshot_recovery=<next streak>/5:<decision>`. Streak 0 immediately after an increase means the next stage has started, not that the increase failed.

Implementation lives in [critic_data.py](../src/brass_ai/recovery/critic_data.py), with checkpoint migration in [actor_value.py](../src/brass_ai/recovery/actor_value.py) and console integration in [league.py](../src/brass_ai/recovery/league.py). [Focused regressions](../tests/Recovery/test_snapshot_weight_recovery.py), the existing generation-55 test fixture, [exact source approvals](../configs/learning/selfplay-resume-sources.json), [IL handoff approvals](../configs/learning/imitation-handoff-sources.json) and [lab instructions](../docs/MLP_V11_LAB.md) are updated together. Existing generation-85 fixes and unrelated game changes are preserved.

## Validation

**155 focused and adjacent regressions passed in 377.32 seconds** ([test log](../artifacts/snapshot-weight-recovery-20261019/regressions.log)). The 12 new tests cover both stage boundaries, actual exposure, health/EV gaps, reductions from 0.75 and 1.0, restart at both increase boundaries, exact checkpoint migration, missing/corrupt state, current/cached row weighting, compact reports and console output. The suite also checks the generation-85 fixes, bounded mixed EV, GAE/PPG, two-iteration cache, IL health/limits, LR control, history sampling, source continuation and five-profile search-free v4 behavior.

The [isolated two-iteration compute smoke](../artifacts/snapshot-weight-recovery-20261019/compute-smoke/report.json) passed in **37.43 seconds**, using authored mixed-table data, the current 5,928,616-parameter v11.2 model and two CPU threads. Each iteration completed one PPO step; PPG accepted two steps in the first and restored a held-out-loss rejection in the second. Restored model weights, both optimizers and gate states matched the uninterrupted next update exactly. The prior checkpoint remained unchanged and expired PPG cache deletion passed. This smoke exercises actual tensor updates and restart integration; the focused controller tests exercise the two recovery boundaries. It is not a native-game run or production-checkpoint reload. It ran alongside regressions, so its elapsed time is not a controlled throughput comparison.

## Continuation

Use the existing run and config; no new IL, config generation or training reset is required:

```bash
python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 20
```

The change is an experimental recovery controller. Passing tests establishes implementation behavior, not improved stability or playing strength.
