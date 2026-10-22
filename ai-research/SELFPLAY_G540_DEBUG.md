# Self-play debug: g540 continuation

22 October 2026. Inspection used the latest local export, `results/joint7-v11-2-grouped-lr-critic-g0540-20261022T144917069131Z`, and the implementation containing the completed g540 one-SE PPG revision. No later production log or live lab checkpoint was available locally.

## Bugs fixed

1. **A pending PL rating slot blocked continuation after a reviewed code update.** `ratings.schedule` always reused `ratings/slots/generation-N`, but its immutable `spec.json` includes the implementation hash. A code update after interruption therefore raised `Immutable rating file changed` before collection could resume. A focused test reproduced the original exception before the fix.

   Rating slots now share the existing monitor/champion source-transition guard. An unfinished slot from a verified earlier source gets a deterministic `generation-N-source-<implementation>` directory. The entire recorded transition chain must reproduce pinned reviews. Old files remain unchanged; unknown sources fail; a changed recipe under the same source still fails the ordinary immutable-spec check. Completed rating slots remain completed and are not recollected.

2. **Several previously approved continuation routes had been omitted from the latest source review.** Input-cache continuations at g320/g412 and the g470 throughput helper failed with `This self-play source transition has not been reviewed`. Restored their exact previously approved routes and carried the existing g540 routes to this implementation. No general transitive approval or arbitrary configuration allowance was introduced.

The rating fix also updates compaction/retention to record and validate the actual committed slot directory. Older archives without this optional path still use their canonical directory. Cleanup cannot substitute another generation, nested folder or arbitrary path, and continues respecting checkpoint/evidence pins. This prevents the new namespace from causing cleanup of the preserved old pending directory or leaking the completed replacement slot forever.

## Scope preserved

- No change to PPO objectives, behavior checks, opening temperatures, exploration, opponent selection, reward targets or evaluation game budgets.
- PPG keeps its eight-iteration buffer/cadence, all-eight-age holdout, 16-step acceptance interval, patience one, one-SE coverage retention and independent overall/age-0 KL caps of 0.002.
- The prepared g540 config keeps insolvency `event_cost=0.020, max_cost=0.080` and the epoch cap/floor of two, with maximum six. The original g540 export predates that explicit migration and is preserved as recorded.
- Training counters, model/IL layout, weights, optimizer state, RNG, cache receipts, economic/entropy baselines, population and lineage are not edited by this debug fix. Existing reviewed-source utility-readiness checks remain active.

Principal files: `recovery/ratings.py`, `recovery/resume.py`, `recovery/retention.py`, both source-review registries and `tests/Recovery/test_selfplay_debug.py`.

## Validation

**97 distinct regression cases passed across the focused runs and repaired reruns**, including six new cases for source-isolated rating recovery, unchanged old bytes, same-source recipe rejection, unknown-source rejection, confined archived directories, pinned evidence, compaction/retention and actual g540 continuation admission. Existing PPO opening/behavior, PPG selection/retry/restart, rating, inference-queue and four-worker CPU collation tests also passed.

The initial audit exposed the missing archived reviews; their failing tests passed after the exact routes were restored. One new fixture initially counted main games rather than all main-plus-panel records, and was corrected. No unresolved test failures remain. The production rating campaign was not launched: rating tests use authored outcomes, two setups and four bootstrap refits.

The isolated CPU smoke ran real small-model PPO/PPG updates over authored mixed-table/branch fixtures, using a separate directory:

| Check | Result |
| --- | --- |
| Small-model parameters | 1,511,464 |
| Total smoke wall time | 100.2 seconds, alongside regression tests; not a speed benchmark |
| Fresh data generations / first PPG | g541–548 / g548 |
| PPG retained / attempted steps | 56 / 56 |
| Distinct critic rows retained | 869 |
| Ordinary paired holdout games | 48 across all eight fixture ages; production remains normally 144 |
| Overall / age-0 exact raw-policy KL | 0.0000014781 / 0.0000014775 |
| Restart comparison | Identical model weights, PPO Adam, value Adam, RNG, phase schedule and epoch-controller state |
| Incoming fixture checkpoint bytes | Unchanged |

Artifacts: `artifacts/selfplay-debug-g540/smoke/report.json`, `source.json` and `validation.json`. No production checkpoints or exported results were modified, and no full-size GPU training/collection benchmark was performed. These checks establish implementation behavior, not playing-strength improvement.

Reproduce focused recovery regressions with:

```bash
PYTHONPATH=src:tests/Recovery python -m unittest -v \
  test_selfplay_debug test_selfplay_input_cache_resume test_rating_retention \
  test_ppg_one_se test_ppg_eight test_opening_ppo test_throughput_g470 \
  test_resume_evaluations
```

The existing [g540 recovery and continuation command](PPG_ONE_SE_G540.md#continuation-command) remains valid after syncing this fix. Use the same `joint7-v11-2-grouped-lr-critic-one-se-g540.json` config and retain `--resume-reviewed-source`. No new config generation or additional hotfix is required for this debug revision.

Implementation: `5828457e258f104e9abf2d37350275c11b1cb1f0209b97e24204a0bb99764788`. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
