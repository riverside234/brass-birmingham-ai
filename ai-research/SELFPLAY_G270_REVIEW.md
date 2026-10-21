# Generation 270 review and panel/PFSP revision

Reviewed 2026-10-20. Source: `results/joint7-v11-2-grouped-lr-critic-g0270-20261020T233304700657Z`, committed generation **270**, **86,400 ordinary games**. Trend review covers compact reports g214–270 and the latest full reports/ratings. The live lab run was not edited or started. Existing concurrent progress-plot changes were preserved.

## Findings from the saved run

The recorded training does not show an economic collapse. All **57/57** reviewed economic-health checks passed; three iterations recorded a PPO KL stop. The economic quantities below are ordinary learner-seat metrics, not held-out strength measures.

| Iterations | Median categorical PPO KL | PPO stop flags | Ordinary VP/learner seat | Zero-VP seats | Insolvent seats | Median collection / PPO / whole iteration |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 214–230 | 0.00743 | 1/17 | 111.40 | 0.423% | 3.113% | 156.9 / 75.3 / 260.0 s |
| 231–250 | 0.00755 | 1/20 | 112.73 | 0.333% | 3.339% | 156.5 / 74.4 / 261.7 s |
| 251–270 | 0.00718 | 1/20 | 113.65 | 0.464% | 3.182% | 159.7 / 75.0 / 265.6 s |

Both latest diagnostic panels have **32 games, eight setups and 100% learner win share**. The v4 panel has one learner versus three v4 bots; the IL panel measures the combined win share of two learners versus two IL opponents. Their respective VP per learner seat is 141.625 and 134.391. These results already satisfy the requested 90% gates; adoption needs no additional panel games.

Use the **same latest cumulative PL fit** to compare generations. It estimates g250 at **615.39** points, g260 at **622.34**, and g270 at **603.69** (g270 interval **555.22–668.94**). The paired g270−g250 difference is **−11.70**, with a 95% setup-bootstrap interval **−74.54 to +36.76**. This does not establish either a strength gain or a regression. Comparing point estimates published by different historical refits would mix changing fit evidence. The current regression and v4-watch alerts are false.

Snapshot critic weight recovered to **0.75 at g239** and **1.0 at g253**, and remains 1.0. Current rolling learner R² is **0.375447783**, against the frozen reference **0.369290472**:

| Evidence | Iterations | Whole games | Learner rows | MSE including bias | Mean prediction error | Target variance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Current weight stage | g254–270, 17 qualifying | 306: 136 self-play, 136 history, 34 v4 | 84,996 | 0.155600639 | −0.006792222 | 0.249139518 |
| Frozen pre-increase reference | g240–253, 14 qualifying | 252 | 69,983 | 0.152923508 | +0.009544860 | 0.242462656 |

The g270 snapshot health flag is false because the **sampled full-dataset PPO KL safeguard** fired at the final check. All 46 planned steps had completed; categorical probe KL was 0.009374. This is not evidence that PPG was required or that IL KL blocked snapshot recovery. PPG remains disabled, and GAE/snapshot health remains separate from the IL-anchor LR gate.

**A material constraint remains:** overall IL KL is **1.08072** versus the calibrated maximum **1.0**, and Canal-action IL KL is **2.42134** versus group limit **2.0**. The IL recovery latch therefore blocks further LR increases as designed. Current LR multiplier is **1.5625**, effective PPO LR **9.375e-6**. The g270 decision is hold (`early_stop_in_window`). No anchor budgets, LR rules, rehearsal, losses or economic checks were loosened by this revision.

## Implemented behavior

1. **Independent, persistent 90% panel gates.** The new signed `panel_gates` block retires each diagnostic panel at learner win share **>= 0.90** on a complete committed panel. Initial adoption reads the latest completed result, never an older peak. Every trigger records the candidate, fixed-reference identity, setup/game support and checksummed spec/panel receipts. Retirement survives restart and rating-history compaction. Retired reports use `null` metrics and an explicit retired status, rather than fabricated zero wins.
2. **V4 share threshold is now inclusive 90%.** The v3 share contract keeps the existing allocation: **160 self-play / 154 history / 6 v4** per 320 games, with the released ten v4 tables going to history. This run's already-latched g270 activation and allocation origin are preserved. Older signed v1/v2 configs retain their original meaning until explicitly upgraded. An unlatched threshold migration rereads the latest panel; it cannot activate from a stale older result.
3. **IL joins PFSP after its panel gate.** All members, including IL, then receive `p_i = 0.8 * (1 - S_i/C_i)^2 / sum(weights) + 0.2/N`, with N equal to the full pool size. IL no longer has a reserved seat share. Its existing ordinary-game S/C counts, model identity and checkpoint are retained. Panel wins never enter this estimator. The five-iteration decay, capacity schedule, admission, eviction rules and checkpoint protection remain unchanged. The all-zero-weight and IL-only cases are explicit.
4. **Temperature comparison disabled.** Fresh config generation and this continuation set `entropy_control.deployment_temperature_check.enabled=false`. Neither future nor backlogged 1.0-vs-0.7 comparisons run. Existing archived comparisons remain readable. Ordinary opening temperatures, probabilistic sampling and entropy control are unchanged.
5. **Snapshot recovery tolerance.** Recovery uses **R² >= frozen reference − 0.01**, still requiring the existing five newly committed healthy qualifying iterations, the 20-iteration rolling window, ten qualifying iterations and whole-game support. The reduction threshold remains **reference − 0.02**. A tolerated small decline cannot lower the reference at an increase: the higher frozen reference is retained. Each weight stage still gathers its own evidence. Current weights, reference and counters are not reset by the source transition.

Two associated failure paths were handled explicitly. Skipping panels must not cause missing-result errors or create zero-win regression alerts. Also, an interrupted rating backlog can retire the v4 panel and then finish a later main-only slot before ordinary collection resumes; the v4 share gate now reads the verified retirement trigger rather than trying to treat that later absent panel as fresh evidence. Both paths have regressions.

The **128 main PL games every ten iterations remain unchanged**, including IL/v4 reference opponents, fixed setups, bootstrap fitting and GAE rating evidence. Deployment qualification still requires review. Other training settings, both Adam states, RNG, weights, progress, population, checkpoint lineage and source/pending-collection checks are preserved.

## Expected next collection and cost

The deterministic g270 planner test produces **160/154/6** tables, **960 learner seat trajectories**, **308 history opponent seats** and **616 actual learner/history comparisons**. The held-out critic quota remains **8 self-play / 8 history / 2 v4 games**. Existing regime-change rules start fresh critic/LR evidence where the actual composition changes; they preserve the attained snapshot weight, frozen R² reference and LR multiplier.

With the saved 17-member pool, IL's sampling probability changes from reserved **5.5556%** to PFSP **1.2272%** of history opponent seats. The latter includes the same uniform floor as other models (0.2/17 = 1.1765%). Its expected exposure is about **3.78 of 308** opponent seats, not a fixed quota. Actual draws and subsequent S/C updates are reported. More difficult historical models receive the freed probability.

After both panels retire, each scheduled rating slot drops from **192 to 128 games**, a **33.3% reduction in evaluation game count**. Disabling the separate 192-game temperature comparison every 50 iterations removes additional evaluation work. Over a complete 50-iteration period with both gates passed, scheduled evaluation games fall from 1,152 to 640, **44.4% fewer**. Main PL refits and their growing evidence cost are unchanged, so these percentages are not wall-time speedup claims.

The saved g270 rating iteration took 577.0 seconds, including 300.7 seconds of evaluation. Ordinary iterations had median total times around 260–266 seconds. There is no extra ordinary collection or model-forward cost from these gates. Cleanup remains roughly 25 seconds per iteration; profiling that I/O is a possible later optimization, outside this change.

## Changed files and validation

- Gates/defaults: `src/brass_ai/recovery/panel_gates.py` (new), `v4_share.py`, `temperature_rating.py`, `exploration.py`, `config.py`; `configs/step5/joint7-latest.json` and new `joint7-v11-2-grouped-lr-critic-g270.json`.
- Planning/reporting: `pfsp_pool.py`, `quality_pool.py`, `ordinary.py`, `ratings.py`, `league.py`, `history.py`, and `scripts/export-training-reports.py`. Export retains live trigger slots until their evidence moves into the existing checksummed archive.
- Controller/continuation: `snapshot_r2.py`, `resume.py`, both source-compatibility registries, and new `scripts/joint7-panel-gates.py`. The recognized config transition changes only `panel_gates`, `v4_share_gate`, and `entropy_control`'s comparison-enable field.
- Focused regressions: new `tests/Recovery/test_panel_gates.py`; updated archived/default fixtures in `test_v4_share.py`, `test_pfsp_pool.py`, `test_quality_pool.py`, `test_exploration_revision.py`, and trigger coverage in `tests/Reports/test_export_training_reports.py`. `docs/MLP_V11_LAB.md` has current defaults and commands. Concurrent shared plot-renderer changes were preserved separately.

**Validation passed:** 162 focused tests in 170.7 seconds across panel gates, v4 sharing, snapshot R², PFSP, archived quality sampling, strength ratings/retention, PPO-only behavior, exploration, source reviews, interrupted-iteration recovery and report export. The new coverage includes the inclusive 90% boundary, independent panel retirement, full rotated-game support, stale/tampered evidence, retained and archived triggers, interrupted commits, rating backlog, normalized IL sampling and its actual pairwise updates, checkpoint planning/restart, and the R² tolerance without a downward reference ratchet. Historical test fixtures explicitly retain their old configuration contracts. Log: `artifacts/g270-review/final-validation-v2.log`.

The final isolated CPU smoke completed in **7.86 seconds** (128 authored rows, eight authored games, two CPU threads; PPO 1.72 seconds). It performed real PPO and checkpoint restoration, with **zero PPG steps**, unchanged checkpoint bytes, and a **bitwise-identical next update after restore**, including model, optimizer, phase and RNG state. The config-helper smoke matched the shipped g270 template, succeeded on rerun, rejected a conflicting output and left run-state bytes unchanged. Artifacts: `artifacts/g270-review/ppo-smoke-final/report.json` and `helper-smoke.json`. These authored tensors do not constitute native gameplay or playing-strength evidence. No long training or real evaluation campaign was launched. `git diff --check` passes with the repository's CRLF-aware whitespace setting.

Final implementation identity: `10d5817a743acc20eb0e70985f41cc2e07b0b1fa16779c54769d3828d3988e33`. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. Reviewed continuation from the export's `efc26563d6d0daa236d81a4d428a38b36624d7e72391c67bb84236a2515676b2` accepts only the explicitly recognized config changes; unrelated config changes still fail.

## Continue the existing run

After syncing these files, from `/data/not_backed_up/yxu209/ai-training`, with the run stopped at committed **generation 270**:

```bash
python scripts/recover-step5-iteration.py \
  --root "$PWD" --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 270 --apply && \
python scripts/joint7-panel-gates.py && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-g270.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This trains **20 additional iterations**, ending at g290 if all commit. The first command verifies g270 and its checkpoint under the lock and archives any unfinished next collection; it is a no-op if none exists. It refuses an advanced run or an uncommitted checkpoint/report needing commit recovery. The config helper writes only the reviewed config and checks the panel receipts; it is idempotent and refuses conflicting output. Normal startup then persists the gates before planning ordinary collection. The completed IL path and saved training checkpoint are unchanged. Do not repeat the old g213 R² hotfix or use `--fresh`.

For later continuations, reuse only the final `step5.py ... train` command with the desired additional iteration count. Export with:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic
```

This revision verifies scheduling, sampling, accounting and restart behavior. It does not demonstrate improved stability or playing strength. A sensible next review is the subsequent ordinary training/evaluation logs under the unchanged main PL protocol; neither a larger LR nor higher anchor limits are justified solely by saturated easy-reference panels.
