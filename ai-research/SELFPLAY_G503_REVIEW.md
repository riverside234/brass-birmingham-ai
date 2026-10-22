# g503 self-play review and continuation — 22 October 2026

Reviewed `results/joint7-v11-2-grouped-lr-critic-g0503-20261022T051624803527Z`: committed **503 / 160,960 ordinary games**, source `3b0546c92357041cde15cc0a4040bddbd0685e40cf3913bb22a52acc73bf6f65`. The two full reports are iteration directories 0501/0502, which produced generations 502/503. The compact history covers the earlier iterations; incremental references still point to earlier export bundles. Reports are not checkpoint backups.

## Findings

Across g471–503, every PPO update completed its planned steps (44–46); no PPO KL early stop or shared economic-health failure was reported. Median post-PPO categorical KL was **0.00537**, range **0.00430–0.00689**, with effective LR **6.25e-6**. At g503, active-reference KL is **0.10043 / 0.15**, and all four group distances are below **0.30**. The active reference is still g440. The four original entropy baselines and promotion counter are unchanged; there has been no new qualified promotion since adoption at g470.

| Metric | g461–470 median | g471–503 median |
| --- | ---: | ---: |
| Collection wall time | 157.14 s | 159.77 s |
| PPO wall time | 75.09 s | 68.56 s |
| Inference batch size | 1.909 | 1.925 |
| Retention cleanup | 27.29 s | 26.72 s |
| Branch tail | 0 s | 0 s |

PPO is about **8.7% faster in these observational windows**, with changed critic weight/row counts and different games; this is not a controlled throughput benchmark. Four CPU preparation workers were used. Their median staging/seal work is 30.58 s and cumulative output is 43.64 GB per iteration (streamed, not simultaneously resident); this is a future profiling opportunity. Retention remains another measurable cost. Collection has not materially improved. No additional queue/cache/retention tuning is enabled from these logs alone.

At g503, 72 branch continuations produced 6,530 additional actor and 15,722 critic rows. 67 started before their parent finished; the branch tail was zero. Recorded branch worker-seconds / 48 lanes was 22.71 s; that is a sharing reference, not a measured counterfactual overhead.

The main training concern is critic fit. The controller cut snapshot weight from 0.5 to zero after g471; it then stayed at zero. Its g503 rolling ordinary-learner **R² = 0.35002**, from 20 iterations / 360 held-out games (160 self-play, 160 history, 40 v4), with **MSE 0.15698**, prediction bias **−0.00241**, target variance **0.24151**. This is below its frozen 0.46097 reference and 0.45097 recovery threshold. It is not a stuck counter: the saved recovery condition is unmet. Reduced snapshot coverage and critic regression occur together; the logs do not establish causation.

The five-iteration self-play economic window at g503 has **121.75 VP/seat**, **0.4375% zero-VP seats**, **2.4375% insolvent seats**. Across all ordinary learner seats, the latest collection has 65 insolvency events / 960 seat-games, affecting 24 seats; branch events are reported separately. There is no economic collapse in this window, though the requested penalty increase is an experiment.

The g500 PL fit converged. In that same fit, g500 is **674.70 points** (95% interval 619.53–740.62), versus g440 at 681.79. The paired **g500 − g440** interval is **[−48.29, +38.01]**, so holding the anchor is correct. The higher g470 point estimate does not establish a statistically demonstrated regression. Competitive coverage retains the previous rated checkpoint and anchor, with 2.75 rated opponents/game and the 128-game budget. No playing-strength benefit is established by the changes below.

## Changes and compatibility fixes

- Insolvency **event_cost 0.0075 → 0.010**, cap **0.075**. Seven full charges cost 0.070; the eighth adds only 0.005. Learner and stored search reward specifications agree. A new explicit migration preserves the older signed .005 → .0075 transition. The utility head's reward identity changes and requires fresh calibration; model tensors and Adam moments remain intact. Ordinary PPO derives new targets from fresh games; frozen opponents and official game results are unchanged.
- Branch probability **0.005 → 0.015 per eligible self-play decision**, with at most one fork per parent. This is not 1.5% of games. The same two temperature-2 action decisions after each fork, exclusions, scheduling, ordinary counters and mixture are retained. Validation now admits this rate with the current anchor-driven entropy contract, without restoring old entropy settings.
- New `critic-rows/3.0.0`, `rollout.mode="fixed"`: **snapshot weight 1.00 immediately on restoration**, including a saved zero-weight state. Both recovery and decay are bypassed. The prior controller, reference and evidence are archived unchanged inside the phase checkpoint. Fixed mode persists across commits/restarts; old contracts remain available for reproducing old recipes. Actual learner/snapshot usage and ordinary-learner R²/MSE/bias/variance remain diagnostics. Snapshot rows remain critic-only; IL/bot exclusions and whole-game family holdouts remain active.
- Fixed-mode restoration no longer enters the automatic R² initializer or rejects a future checkpoint because of the earlier g213 R²-hotfix receipt. The old manual R² hotfix becomes inactive under fixed mode. Other migration paths cannot add recovery streak fields back into fixed state.
- A locked, idempotent config helper admits only the reviewed changes. It does not edit the run or checkpoints. PPO/PPG/GAE, learning-rate state, RNG, population, lineage, original entropy baseline, independent economic baseline and fixed probe remain unchanged. PPG and teaching stay disabled. Fresh config generation also uses these settings; the completed IL directory is unchanged.

Restoring snapshot rows increases critic data available for sampling: at g503, about **27,834** additional non-held-out snapshot rows would join **102,806** learner rows. PPO still uses the configured 2,048-row critic batch and actor-driven step budget, so more available rows do not imply every row trains each iteration. More branches increase collection and PPO work; the one-fork cap prevents a simple 3× runtime extrapolation. Watch actual row usage, held-out fit, branch worker-seconds and iteration time in subsequent logs.

## Validation

**142 focused regression cases passed across the suite and targeted reruns.** Coverage includes the actual g503 signature/config and saved entropy baseline, historical reward migrations, 0.010 charges/cap, fixed-weight validation and immediate sample usage, no recovery/decay calls over 44 unhealthy observations, prior R² evidence preservation, GAE/PPO health and holdouts, branch eligibility/probability/provenance, archived recipes and checkpoint restart. The initial 140-case run exposed two assertions for superseded fresh defaults and an old hand-built fixture missing a migration attribute; those fixtures were corrected and all affected cases passed. The final 31-case run (new g503 checks plus existing R² regressions) passed in 24.08 s.

The actual recovery/load/commit path was exercised in an isolated run: partial collection bytes were archived intact; a pending collection was rejected before recovery; a restart after source/config publication but before the first new model commit restored the same new phase/Adam state without altering its checkpoint.

The **6.00 s CPU smoke** used the current **5,928,616-parameter v11.2 utility model**, authored mixed-table tensors, real PPO, checkpoint serialization and Adam restoration. Snapshot rows trained immediately at 1.00; the next update after reload matched weights, both optimizer/phase states and RNG bit-for-bit. Both saved checkpoint hashes stayed unchanged. This is a compute/resume check, not native gameplay, a GPU benchmark or playing-strength evidence. No long training or evaluations were launched. Actual lab checkpoints are unavailable in this checkout and are validated by the recovery/trainer on the lab host.

Changed implementation files: `recovery/{config,utility,exploration,critic_data,actor_value,snapshot_r2,snapshot_r2_hotfix,resume,league}.py`; new `scripts/joint7-g503.py` and continuation JSON; explicit source/handoff receipts; focused regressions/smoke and lab instructions. Unrelated working-tree rating chart edits were preserved. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`; reviewed source is `0175cf43e34fa0cb1a3425158e13399a868c3fe2bdcd27bef58ee10df2bda02f`.

## Continue from committed g503

After syncing the changed source/config/review files:

```bash
cd /data/not_backed_up/yxu209/ai-training
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 503 --apply && \
python scripts/joint7-g503.py --expected-generation 503 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-fixed-snapshots-g503.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This targets **g504–523**. Recovery preserves an incomplete next collection in `.interrupted-iterations/` and leaves committed state/checkpoints unchanged; it is a no-op if none exists. It refuses a different committed generation or an ambiguous uncommitted checkpoint/report. The trainer then records the reviewed source/config transition and applies fixed weight before collecting/updating. Repeat only the train command on later clean continuations, using the actual committed generation if explicit recovery is needed again. Do not rerun the iter-160 restart or use `--fresh`.
