# Self-play g410 review and group entropy hotfix

Reviewed `results/joint7-v11-2-grouped-lr-critic-g0410-20261021T151332842603Z` on 21 October 2026: committed **g410 / 131,200 ordinary games**, including 90 iteration summaries from g321–410 and the latest full reports/PL fit. The full live checkpoints and any Ctrl+C partial collection remain on the lab machine; this checkout contains exported reports. No live run or exported evidence was modified.

## Findings and fix

- No failed invariant in the bounded log audit: finite nonempty PPO updates, planned-step limits, constant entropy baselines/floors, branch caps, disabled PPG and the bias-aware pooled R² formula agree. There were **no PPO KL early stops in g321–410**. The latest 20 iterations average KL **0.00554**, **124.2 branches**, **190.5 s collection** and **333.3 s total**, including periodic ratings.
- Latest five-iteration self-play health is supported and healthy: **118.94 VP/seat**, **0.344% zero-VP seats**, **2.219% insolvent seats**, **0.244 passes/seat**. Actual PPO actor usage is **51,570/51,570 eligible distinct rows**, including 12,677 branch actor rows; there is no unused actor backlog. PPO has 52 accepted steps; PPG/teacher updates are disabled as configured.
- The v4 critic warning remains: pooled learner EV **0.306** from **40 v4 games**, versus self-play **0.369** from 160 games over g391–410. History EV is **0.377** from 160 games. Single-iteration v4 EV is much noisier; existing warnings and GAE safeguards remain active.
- Snapshot weight correctly stays **0.5**. Rolling learner R² is **0.380885** from **20 iterations / 360 ordinary held-out games / 100,406 rows**: MSE **0.150587**, prediction bias **+0.000485**, target variance **0.243229**. The frozen reference is **0.460971**, so recovery requires at least **0.450971** plus existing health/support safeguards. The weight rose to 1.0 at g372 and reduced at g382. No baseline relaxation is introduced.
- **Reporting bug fixed:** nested `weight_recovery.next_weight` previously displayed the next possible recovery stage (0.75) even when effective next weight was 0.5. It now reports the actual next weight; `recovery_target_weight` separately identifies the possible future stage. Controller decisions/state are unchanged; old exports remain intact.
- Active KL reference is **g340**. Anchor KL **0.14210**, overall allowance **0.17763**, group allowance **0.35526**; the recovery latch still blocks LR increases because recovery is stricter than merely being within the allowance. Current LR is **6.25e-6**, multiplier **1.04167**. The latch is retained. The g410–g340 same-fit paired PL difference is **+20.0 points, 95% CI [−17.4, +55.3]**, so no reference promotion is justified. g410–g320 is **+52.9 [16.7, 94.7]** in this fit; these observations do not isolate the effect of exploration changes.

## Requested settings

The new versioned entropy recipe and fresh defaults use these fixed floors on the **original saved baselines**, without decay:

| Group | Multiplier | Effective floor |
| --- | ---: | ---: |
| `canal_action` | 0.95 | 0.9292460742 |
| `canal_other` | 0.90 | 0.3407628226 |
| `rail_action` | 0.95 | 1.2959231842 |
| `rail_other` | 0.90 | 0.4413949833 |

Branching becomes **0.005 per eligible decision**, retaining at most one branch per ordinary self-play game. Temperature remains **2.0 for two action decisions per seat**, both at game opening and after each excluded fork move. The lower probability should reduce continuation work, but 0.5% is not the fraction of tables containing branches. g410 branch work was **2,027 worker-seconds / 48 workers = 42.2 s**, with **zero branch tail** and 95.4% work utilization. This is a sharing reference, not measured counterfactual overhead; no production speedup or strength gain is claimed.

The explicit hotfix changes only `entropy_control` and `branching`. Existing signed recipes keep their original behavior. The helper preserves checkpoint bytes, weights, Adam/RNG, progress, population, entropy baseline/alphas/windows/origin, KL reference/calibration and economic/GAE evidence. It does not immediately rewrite entropy coefficients. On the next update, existing regime handling resets the LR decision window and stage-local snapshot-R² evidence for the new branch rate while preserving the LR multiplier/cooldown/warmup, attained snapshot weight and frozen R² reference. The old entropy95 helper refuses to overwrite the adopted new recipe. Existing input-cache/inference work is preserved.

Files: `recovery/{exploration,resume,league,snapshot_r2}.py`, `scripts/joint7-entropy-groups.py`, the superseded helper guard, `configs/step5/joint7-latest.json`, the new `joint7-v11-2-grouped-lr-critic-entropy-groups.json`, exact source/handoff reviews, focused regressions/smoke harness and lab instructions. The exporter already discovers `exploration/group-floors-hotfix-report.json`; a regression checks byte-preserving export.

## Validation

**139 focused tests passed in 130.011 seconds** (`artifacts/g410-review/focused-tests.log`). They cover fixed group floors, validation, old signed recipes, exact reviewed continuation, state/RNG preservation, hotfix idempotence/failure, cache identity, interrupted-collection recovery, snapshot reports, branch admission and report export. Live g410 weights are unavailable locally, so the hotfix fixture substitutes their loader while exercising real signature checks, Adam restoration and atomic state adoption.

The isolated native smoke **passed in 172.734 seconds** with a fresh **5,928,616-parameter** model, two ordinary games, one forced early test fork and one PPO step repeated from the same checkpoint. All four branch seats used exactly two temperature-2 action decisions. Full replay admission, exact collected-probability reconstruction, identical restored PPO weights/Adam/RNG and unchanged incoming checkpoint bytes passed. The 416-position post-update categorical KL was **0.00000617**. The test forces a fork to cover the path; production probability is explicitly recorded as **0.005**. Results are in `artifacts/g410-review/native-smoke/report.json`; this small random-model smoke establishes neither strength nor production runtime gains.

All **18,577 preexisting source-review records** and completed-IL handoff records were preserved. Exact unchanged-config and opt-in migration routes were appended to implementation `fb4a7c40c8784dff3f1b991cb6eade5e95fbdb2b69172eb2ecb18ed590c411a0`; learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. Final config validation and diff whitespace checks passed. No long training or evaluation campaign ran.

## Apply after Ctrl+C and continue 20 iterations

From the repository root on the lab machine, with training stopped:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 410 --apply && \
python scripts/joint7-entropy-groups.py --apply --expected-generation 410 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-entropy-groups.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The recovery command archives incomplete next-iteration collection files without changing committed progress and does nothing if none exist. It refuses an uncommitted checkpoint/report requiring commit recovery. The hotfix verifies the real checkpoint and opponents before adoption, refuses a conflicting config, and writes an exported receipt. Omit the hotfix's `--apply` to preview. If the live committed generation is different, use its actual generation for both guards. Later continuations need only the final training command with this new config; the completed IL path stays the same.
