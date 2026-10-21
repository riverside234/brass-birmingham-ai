# g320 review and fixed entropy floors

Reviewed `results/joint7-v11-2-grouped-lr-critic-g0320-20261021T052141479749Z`, including compact g271–320 history. The export ends at **320 committed iterations / 102,400 ordinary games**. The live Linux checkpoint is not present locally; the helper below applies the hotfix there after verifying the actual checkpoint. No live training files or exported evidence were rewritten by this revision.

## Quick findings

- No new training-state or probability-accounting failure was identified in the exported reports. Economic health passes: the latest five-iteration window has **116.83 VP/seat**, **0.44% zero-VP seats** and **2.63% insolvent seats**. PPO early-stopped in **5/50 iterations**; its guard is active. PPG remains disabled, while PPO trained the critic on 74,257 distinct rows at g320.
- The existing entropy floor decayed to **0.79579 × baseline**, leaving all coefficients at 0.005. The requested change restores it to 0.95 permanently. Three of four latest measured entropies are below the restored floors; the controller will respond under its existing median, health, cap and hysteresis rules. A fixed target is not a guarantee that measured entropy stays above it.
- The g320 KL-reference promotion is valid in the saved evidence: **+62.37 PL points over g260, paired 95% interval [24.35, 104.78]**, from one current fit and its 200 setup-bootstrap samples. The post-rating reference transition is pending its next PPO checkpoint; zero KL to the identical g320 reference and reset 0.15/0.30 allowances are expected. This hotfix preserves that transition and the independent economic baseline.
- Remaining watch item: pooled v4 learner EV is **0.141 versus 0.337 self-play EV**, over 32 v4 games in 16 qualifying iterations of the g301–320 window. Its latest two-game EV is −0.215. Snapshot weight correctly holds at **0.75**: rolling learner R² **0.35281** is below the recovery threshold **0.35929**, but above the reduction threshold **0.34929**. These are diagnostics, not reasons to silently change GAE or retention rules.
- Branch overlap works: g320 has **67 branches**, **904.98 branch worker-seconds / 48 workers = 18.85 seconds**, a **157.30-second collection scheduling wall time**, and **zero branch tail seconds**. Branch rows actually used by PPO: 6,023 actor and 7,705 critic rows. Those worker-seconds are overlapped work, not a measured no-branch timing difference.

## Applied code/config changes

The new versioned recipe sets `floor_start_ratio=0.95`, `floor_min_ratio=0.95`, `floor_decay_per_iteration=1.0`; ordinary neural seats use **temperature 2.0 for their first two action decisions**. Each self-play branch resets all four counters after the excluded fork move, giving each seat two fresh temperature-2 action decisions. Subdecisions remain at 1.0 and do not consume that allowance. Fork selection is **0.015 per eligible decision**, still capped at one branch per ordinary self-play game. The alternative fork move's existing 70/25/5 mixture is unchanged.

| Fixed probe group | Existing baseline | Old g320 floor | New frozen floor |
| --- | ---: | ---: | ---: |
| Canal action | 0.978154 | 0.778403 | **0.929246** |
| Canal other | 0.378625 | 0.301306 | **0.359694** |
| Rail action | 1.364130 | 1.085558 | **1.295923** |
| Rail other | 0.490439 | 0.390286 | **0.465917** |

The baseline, alpha/window state, origin g185 and arming generation g197 stay unchanged. Alpha updates continue on committed iterations; the helper does not fabricate an observation. Epsilon remains **0.02**, the table allocation remains **160/154/6**, and PPO/critic, reward, teacher, evaluation, PFSP and anchor settings remain unchanged. The existing LR controller clears its short data-dependent decision window on the new collection regime, preserving multiplier, cooldown and warmup.

Implementation adds strict temperature-2 probability admission, per-seat branch counter validation and branch-contract/provenance matching. Old signed recipes keep their original floors, temperatures, fork seeds and inherited counters. The reviewed migration also handles pre-exploration source routes without preventing their existing upgrades. Changed policy/collection identities reject stale cached behavior. No model architecture or IL dataset change is required.

Changed files: `recovery/{exploration,branching,execution,replay,resume,league}.py`, `scripts/joint7-entropy95.py`, `scripts/export-training-reports.py`, the main `joint7-latest.json` template and new `joint7-v11-2-grouped-lr-critic-entropy95.json`, exact source/handoff approvals, focused exploration/export tests and native smoke, and `docs/MLP_V11_LAB.md`. Existing concurrent rating-plot edits are preserved. Full/compact entropy reports include the fixed ratios; console output adds `Hfloor=0.95*baseline(fixed)`. The exporter now includes `exploration/hotfix-report.json` automatically.

## Validation

**147 distinct focused tests passed**: the 146-test suite in 109.10 seconds, followed by all 30 exporter tests in 15.03 seconds including one added hotfix-report case. Coverage includes floor freezing, strict config/behavior checks, per-seat fork counters, hotfix preview/idempotence/state preservation, legacy source routes, branch scheduling, anchor/retirement/PFSP, snapshot R², LR and report export. Results are recorded in `artifacts/g320-review/{final-tests,export-tests}.log` and `artifacts/g320-review/native-smoke-2/report.json`.

The isolated native smoke passed in **183.66 seconds** on CPU: two ordinary games, one forced test fork, and one PPO step repeated from the same incoming checkpoint. It verified two post-fork openings for each of four seats, full replay admission, exact collected-probability reconstruction, excluded fork rows, parent-family holdout isolation, and identical restored weights/Adam/RNG. The retained step used **134 branch actor rows and 24 branch critic rows**. The 416-position smoke probe measured post-step KL **0.00000617**. This uses random initial weights and a forced fork for coverage, not the trained g320 model or a production-rate benchmark. An initial added test assertion incorrectly read compact summaries as full replays; it was corrected to inspect the authoritative shard before this passing run.

Higher fork probability will create more and generally earlier continuations, increasing collection, replay and PPO work. The rate is not a percentage of tables; with roughly 100 eligible decisions, the chance of a fork rises from about 39% to 78%. Actual lengths and scheduling change this estimate. No 6–8% runtime-overhead target or strength/stability gain is established by this smoke.

## Current-run hotfix and continuation

With training stopped, from the lab repository:

```bash
python scripts/joint7-entropy95.py --apply --expected-generation 320 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-entropy95.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

Omit `--apply` for a preview. The generation guard is optional; change it to the actual committed generation if later. The helper verifies the checkpoint and restored optimizer, keeps a state backup, records the reviewed transition and its old/new floors, and is idempotent. Subsequent continuations use only the train command. Conflicting configs and uncommitted old collections are rejected; use the documented `recover-step5-iteration.py` flow for an interrupted collection before adopting new behavior. Do not use `--fresh` or rerun earlier g213/qualified-anchor hotfixes on this run.

Fresh self-play config generation in the lab guide now gets the same fixed floor, T=2 openings and 0.015 fork rate, retaining the existing 20-iteration branch activation delay and prospective baseline collection.
