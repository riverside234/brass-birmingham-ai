# Masked ISMCTS strength repair — 1.4.11

1 October 2026. Installed and checked. Implements F0–F6 of the [strength fix plan](../ai-research/MASKED_ISMCTS_STRENGTH_FIX_PLAN.md).

The six player-facing v3 strategies remain available. Easy/Mid/Hard keep **2/5/12 seconds**, and existing saves keep their original assignments and 10-second compatibility allowance. The rules, city highlighting and statistics readability binaries are byte-identical to installed 1.4.10. The [execution lifecycle](AI_EXECUTION_RESOURCES_1.4.7.md) still caps computation/retirement rendering, restores presentation settings, and waits for actual backend completion.

## Search changes

Visited and unseen moves now compete through PUCT and first-play urgency. Opening more candidates requires completed support; protected setup actions, every industry, selling and beer sources remain eligible. Exact card, route, resource and bonus choices remain legal. The full policy denominator is independent of expansion pruning.

Each complete turn has its own signature, completed return vector and scenario support. Two actions cannot borrow confidence from an unrelated continuation with the same first action. Heuristic challengers must have at least two completed, matched scenarios against the incumbent. Parallel replicas merge raw counts and four-seat return sums; their warmups differ, and pending/cancelled work contributes no visits.

Heuristic fallback planning uses one incremental detached world. Card variants share cached proposal work, forced single-choice transitions avoid redundant observations, repeated complete plans reuse their public ledger, and one conversion forecast serves all four seat ledgers. The forecast reserves scarce beer once rather than promising the same barrels to several unflipped tiles. Payment recovery and critical merchant/own-brewery comparisons remain mandatory.

| Setting | Lab v3 | Game Easy | Game Mid | Game Hard / old save |
| --- | --- | --- | --- | --- |
| Whole-turn clock | 1 second | 2 seconds | 5 seconds | 12 / 10 seconds |
| Root / interior K | 4 / 2 | 4 / 2 | 6 / 3 | 8 / 4 |
| Voluntary opponent-turn prefix | 0 | 1 | 2 | 2 |
| Exploration | 0.6 | 0.6 | 0.6 | 0.6 |
| Q transform | Regularized sibling | Same | Same | Same |

Leaves finish both current actions and mandatory income, liquidation and scoring transitions. An earlier actual next root turn ends the reply prefix. The choice cap is an interruption guard, not an action-depth measure. Q normalization is selection-only, uses a minimum span of 0.05, and waits for two siblings with two completed visits each. Raw backups remain bounded four-seat relative heuristic returns; terminal results use official win shares.

The **MLP-ResNet branch remains 1,840,109 parameters** and has separate settings: root/interior K 16/8, raw neural Q values, CPU/GPU batching, and a new-root-information frontier. A first visit evaluates a valid acting-root observation; later completed visits may expand it deeper. It inherits neither heuristic Q normalization nor a fixed reply/depth cap. No previous neural checkpoint or population is imported.

## Evidence and limits

Twenty four-player games used five independent setups and four seat rotations. Each table contained the repaired Balanced candidate, frozen old Balanced v3 and two rotating frozen styles. All used one second and one private CPU worker per turn.

| Metric, average per game | Repaired Balanced | Frozen control |
| --- | ---: | ---: |
| VP | 47.60 | 36.15 |
| Rank, lower is better | 2.15 | 2.90 |
| Official win share | 0.30 | 0.15 |
| Cash-distress decisions | 0.00 | 0.20 |
| Unused Canal level-one tiles | 0.60 | 0.70 |
| Unflipped endgame tiles | 0.45 | 0.65 |
| Merchant beer consumed | 1.60 | 0.95 |
| Completed simulations | 54.55 | 2.45 |

The candidate's VP advantage averaged **11.45**, with a descriptive setup-block bootstrap interval **0.9–22.5**. Four of five blocks favored it; one favored the control. This is a small repair screen, not a human-strength rating or separate strength qualification of every style. Pottery remained available and was built in 16 of the 17 candidate games with a recorded opportunity.

The final 16-position panel averaged **4.625 completed simulations versus 0.4375** at one second, with no zero-completion candidate decisions and 26 revisited root edges versus zero. Four-position ablations separately record unseen-first selection, wider K, one reply, raw Q, and incumbent-only decisions. They do **not** isolate a statistically proven benefit for Q normalization; many one-second decisions still use the improved incumbent. Selected early/late-era positions also completed repeated plans at the laptop's 2/5/12 clocks. Bounded completion retired 2–38 ms after those deadlines; incomplete paths were excluded.

See the [exit record](MASKED_ISMCTS_STRENGTH_EXIT_CHECKS.md) and the [immutable bank receipt](../training/opponents/evaluations/masked-strength-1.1.0/report.json). Individual game/decision reports and the install archive are private under `artifacts/masked-strength/release-1.4.11/` and `artifacts/patch-releases/1.4.11/`.

## Training and transfer

The active bank is **bank-981744ddae4aaf07f1cf**, with new `*-v3-masked-strength` identities. Five styles are training opponents; Pottery Flex's identity remains evaluation-only. Learner Pottery actions remain unrestricted. Older banks and reports are preserved.

The authored workers, bank sources and configuration are Git-ready. The refreshed private runtime bundle has 339 members and SHA256 `568b53012fd83ebe00014a08804304cee1c3e9296d60c6cdf51a3f5497d7f561`. Sync the new source/`training/` files and transfer that bundle separately. Follow the [conda lab guide](TRAINING_RECOVERY_LAB.md) with a **new run directory**; source-bound older runs cannot resume this revision. Existing audited heuristic demonstrations may be reused; old neural replay and old search teachers may not.

To reproduce the compact checks locally:

```powershell
python scripts/verify-masked-ismcts.py --integration
python scripts/masked-strength-check.py diagnose --positions 16
python scripts/masked-strength-check.py clocks
python scripts/masked-strength-check.py screen --workers 4
```

Keep completed strength evidence in a new output directory if repeating a screen. Windows CPU/native checks were performed here; CUDA performance and Linux execution must pass the lab preflight before training.
