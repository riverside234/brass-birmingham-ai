# Self-play generation 55: review and fixes

Date: 2026-10-19. Evidence: [exported run](../results/joint7-v11-2-grouped-lr-critic-g0055-20261019T053031361382Z/state.json), [latest full iteration](../results/joint7-v11-2-grouped-lr-critic-g0055-20261019T053031361382Z/iteration-0054/report.json), and [analysis](../artifacts/selfplay55-review-20261019/analysis.json).

**Subsequent user-requested revision:** [health and mixed-table EV window](SELFPLAY_HEALTH_EV_REVISION.md) removes the IL sale-probability check and implements pooled EV support. The findings below describe the exported run and the preceding fixes; its sale-check retention recommendation and single-iteration mixed-table gate have been superseded. Archived logs are unchanged.

The run shows improving measured play, with several gates preventing the planned training progression. This review fixes clone-beta escalation, an unintended coupling of the IL-distance budget to GAE/critic adoption, and frozen-rating runtime handling. At the user's request, the experimental v5 bot and all three of its bank snapshots are removed; game and self-play use official **v4, `bank-3dc65253a610191d66a5`**.

## Evidence and scope

The export contains 55 committed iterations / **17,600 ordinary games**, compact reports for generations 1–53, full reports for 54–55, and rating slots 10–50. All 35 files listed by the export manifest match after Git CRLF normalization; there are no other hash differences. Generation 1 used source `de2aac3e…`; generations 2–55 used `e2b6bf87…`. The later PPG safe-prefix and five-profile cycling fixes in the current checkout were therefore **not exercised by these logs**.

At 320 games per iteration, the unchanged 50/45/5 recipe gives 160 four-learner self-play tables, 144 two-learner history tables and 16 two-learner v4 tables: 960 learner seat-games. The current v11.2 model, exploration 0.08, PPO/reward settings, 24-member history capacity, rehearsal schedule, two-iteration PPG buffer and disabled teacher/v4 search remain unchanged. There is no training restart or checkpoint edit.

## Observed progress

The following ordinary-game rates compare ten-iteration windows; they describe training behavior, not a controlled evaluation against identical opponents.

| Self-play learner metric | Generations 1–10 | Generations 46–55 |
| --- | ---: | ---: |
| VP per seat | 85.50 | 99.72 |
| Seats affected by insolvency | 4.14% | 2.67% |
| Insolvency events per seat | 0.1114 | 0.0597 |
| Seats finishing with zero VP | 0.89% | 0.36% |
| Passes with an alternative per seat | 0.655 | 0.373 |
| Selected sell decisions per seat | 2.965 | 3.009 |

There is no aggregate insolvency surge in these windows. Historical-table learner VP rises from 85.82 to 100.70 and per-learner win share from 0.256 to 0.408. The old v4-training profile bug limits interpretation of that table type: only cotton and goods-beer were sampled in this export. The preserved fix now cycles all five training profiles, with 6–7 of the 32 opponent seats per profile; pottery-flex remains held out.

All 55 PPO updates completed their planned steps without a KL early stop. Exact old-to-new behavior KL ranges from **0.002813 to 0.004419**, with median **0.003586**. The LR multiplier stays at 1.0: 50 iterations report failed health checks, three are in warmup and two are filling the initial decision window. Low PPO KL alone does not authorize an increase.

### Strength evidence

Use the **same generation-50 PL fit** to compare checkpoints, rather than comparing changing estimates from different slots:

| Agent | PL points, IL fixed at zero |
| --- | ---: |
| Generation 10 | 74.74 |
| Generation 20 | 131.55 |
| Generation 30 | 188.54 |
| Generation 40 | 215.96 |
| Generation 50 | 282.11 |
| Fixed search-free v4 reference | 203.29 |

The paired difference is **+66.15 points for g50 over g40**, with 95% setup-cluster interval **[19.29, 118.43]**. G50 is **+78.83 over v4**, interval **[34.93, 144.02]**. These support improvement within the fitted evaluation pool. The cumulative fit uses 640 main games on 32 setup clusters; generation 55 has not yet been rated.

The g50 diagnostic panels report 0.375 win share against three v4 opponents (interval [0.219, 0.531]) and 0.781 combined learner win share for two learners against two IL opponents ([0.656, 0.906]). Each panel has 32 games / eight setups. The PL probability of outranking one v4 player is not a four-player table win rate. These results do not establish general strength against humans, searched opponents, or deployment qualification; panel review remains required. [Recorded fit and panels](../results/joint7-v11-2-grouped-lr-critic-g0055-20261019T053031361382Z/ratings/slots/generation-000050/report.json).

## Issues and implemented fixes

**Clone beta reused old trips on clean iterations.** G8–10 hit the value KL cap. G10 raised beta 1→2; g11 and g12 then raised it 2→4→8 despite no new cap trip. `TrainingPhases.finish` now requires a current trip as well as three trips in the last five iterations. Clean iterations hold. The saved beta of 8 and its trip history are preserved on continuation; there is no retrospective rewrite or silent beta reset.

**The calibrated IL-distance budget accidentally gated GAE and snapshot-critic adoption.** The auto-calibration plan scopes this distance budget and recovery latch to LR increases. Adoption now uses the nested economic/behavior health result plus its own PPO/value KL and EV requirements. Every sale/pass check remains required. This fix does not activate GAE or snapshot data immediately in the reported run, whose behavior checks still fail.

**Frozen ratings depended on current source code.** A rating reference is supposed to remain fixed. The checkout's experimental shared bot made the next rating slot fail source freshness against the old reference. Restoring v4 resolves this particular mismatch. Rating execution also now verifies and runs the exact recorded binary, manifest, dependencies and profile independently of newer working sources. This path is restricted to held-out rating jobs; ordinary collection freshness stays strict. Startup verifies the reference before collecting more training games. Missing or altered files fail explicitly, with no substitution under the old policy identity.

**Gate failures were difficult to diagnose from the console/compact history.** New `gate_diagnostics` fields expose sale/pass failures, health and LR blockers, IL-only history fallback, accepted/attempted value steps, beta reason and per-table held-out rows plus **whole-game counts**. Missing EV remains unavailable. Console output includes `health=...` and `sale_fail=failed/eligible`. This adds reporting, not new thresholds.

**Experimental bot removal.** Restored `InvestmentOpportunity.cs` and `TournamentOpenings.cs` match the official v4 bank's frozen files. Removed the experimental source, six profiles, build/test entry points, candidate selectors, three banks, their portable/private runtimes and local experiment folders. The v4 portable worker is restored to `search-76a2f9ad7529c0e5e2117a36`, matching this run's rated reference. The installed game's three DLLs still match the qualified 1.4.31 release; its recorded validation identifies the same official v4 bank. Generic isolated-bank freezing and its tests are retained. Unrelated feature-layout versions numbered v5 are unrelated to this bot and remain intact.

## Remaining bottlenecks and proposed improvements

1. **Review the failing sale positions before relaxing any gate.** At g55 all five economic checks pass, but **10/79 eligible sale positions fail**; pass failures are 0/511. The original rule requires every eligible position to pass, so this is intended enforcement, not a sampler defect. Since baseline establishment at g13 there has been no qualifying five-iteration health window. Consequently the pool is **1/24, IL only**, with no scored snapshots: the 45% history allocation currently faces two IL opponents. Capture the failed positions and compare legal alternatives, IL/current probabilities and later outcomes. A later aggregate/tolerance rule would be a separately reviewed experiment, not part of this fix.

   Automatic tightening has already reduced the mean/group limits from 0.15/0.30 to **0.05/0.10**. G55 raw IL KL is **0.2005**, with the largest group mean **0.3816**. These distance checks also block LR increases and raise the existing alarm. They are fixed-probe raw-policy distances, distinct from the much smaller per-update behavior KL. Neither the limits nor the recovery latch are reset here.

2. **Keep the value KL cap at 0.002 for now.** PPG accepted steps in 20/55 iterations: **1,176 accepted / 3,136 attempted**. The stop reasons are 46 held-out-loss increases, six KL-cap stops and three completed phases. The last 20 iterations have **no KL-cap stops**, but 19 held-out-loss stops. The latest phase accepted 41/82 steps, retained KL 0.000667 and slightly improved held-out loss (1.107294→1.107069). Raising the KL cap does not address the recent rejection reason. Apply the existing safe-prefix fix and collect its next reports first. If loss rejection persists, compare smaller value-update chunks or head-only fitting on an isolated saved batch; a controlled lower clone-beta setting is another hypothesis, not an automatic reset.

3. **Improve mixed-table EV evidence before expecting its GAE gate to open.** V4 held-out learner EV is missing in **29/55 iterations**. With only 16 v4 tables and a 5% whole-game hash holdout, many iterations naturally have no held-out v4 game. The longest observed run with available v4 EV is six iterations, below the ten consecutive healthy iterations required by the gate. A future change could use a deterministic stratified whole-game holdout or a bounded multi-iteration evidence window with minimum game support. Keep those games excluded from all critic fitting and preserve resume/cache identities. No missing evidence is treated as passing in this revision.

4. **Do not interpret dormant GAE as a calculation failure.** Lambda remains 1.0. Only 20 iterations have accepted value fitting, short of the 30-iteration fallback; EV streaks and behavior health also block progression. Mean held-out EV rises from 0.298 in the first ten to 0.396 in the last ten, but is noisy. The late mean descriptor gain is about 0.0057 over unknown descriptors; this is a descriptive probe comparison, not proof that descriptors improved strength.

The PPG paper motivates separating policy/value phases and controlled reuse; it does not establish an appropriate KL cap or acceptance tolerance for this run. Likewise, GAE trades variance against bias rather than guaranteeing stronger play. [PPG](https://arxiv.org/abs/2009.04416), [GAE](https://arxiv.org/abs/1506.02438).

## Validation and runtime

**141 focused tests passed**: 139 in the final self-play suite plus two generic isolated-bank tests. Coverage includes clone escalation/restart, strict behavior failures, GAE/critic gates, exact generation-55 source admission, PPO probability/KL/LR handling, PPG rollback and disk buffers, five-profile cycling, all-v4-search disablement, official-v4 source/bundle identity, quality sampling, ratings, retention and frozen-runtime tamper rejection. Earlier broad checks exposed two stale IL/model fixtures and one transient Windows retention-file replacement failure; the fixtures were corrected, and the final complete suite passes. [Final regression log](../artifacts/selfplay55-review-20261019/final-regressions.log).

Two isolated smoke tests passed without production weights or checkpoint writes:

| Smoke | Result |
| --- | --- |
| Medium v11.2, 5,928,616 parameters; authored eight-game / 128-row CPU batch | Normal value cap: 7/7 steps accepted, KL 0.00007241, held-out loss 1.64324→1.62349 |
| Same weights/Adam/RNG, stricter diagnostic cap 0.000020665 | 1/2 steps accepted; violating step restored; retained KL 0.00001095; PPO Adam unchanged |
| Actual pinned v4 runtime, two native opening decisions | Legal actions, unchanged rated policy identity/files, zero search completions and failures |

The value smoke took 7.38 seconds with two CPU threads; the native smoke took 3.45 seconds. These measure bounded implementation behavior, not production throughput or strength. The actual generation-55 weights and dense probe tensors are not in the export, so no production checkpoint reload/update is claimed. Source admission is tested against the exported state, and optimizer/RNG continuity is covered by authored fixtures. [Value smoke](../artifacts/selfplay55-review-20261019/ppg-smoke/report.json), [native smoke](../artifacts/selfplay55-review-20261019/frozen-v4-smoke/report.json).

The last ten ordinary iterations have median total time **345.6 s**, collection 189.9 s, PPO 62.1 s, rehearsal preparation 39.1 s and value fitting 27.0 s. These component medians need not sum to the total median. Rehearsal already uses 16 workers; the latest 16,384-row sample contains 14,920 distinct rows, of which 6,261 were actually used (32,982 presentations). No additional games or model forwards are added by the new gate diagnostics. Startup adds bounded reference-file verification. The previously implemented safe-prefix transaction uses a bounded model/value-Adam snapshot per step (approximately 68 MiB for initialized FP32 state); useful accepted work may take longer than a rejected phase. Lab GPU/disk impact remains to be measured.

## Continue the existing run

After updating the lab checkout, prepare its restored portable v4 worker and continue with the **existing config and IL directory**:

```bash
python scripts/build-shared-search.py --root .

python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 20
```

This adds 20 iterations from the latest committed generation. Optionally add `--expected-generation 55` if it is still the intended entry; omit it once training has advanced. Do not use `--fresh`, regenerate the config or rerun IL. Preserve the rated v4 runtime under `artifacts/legacy-ai/search/` and the run's existing probe and PPG cache. Fresh collection source checks remain active.

Changed code: `legacy.py`; recovery `actor_value.py`, `execution.py`, `history.py`, `league.py`, `legacy_search.py`, `ratings.py`, `review_metrics.py`, `value_phase.py`; restored v4 strategy/shared-runtime files; exact resume/IL handoff receipts; focused tests and lab documentation. The receipt retains all 2,163 prior approvals and appends four exact unchanged-config source transitions. Tested implementation: `c13907c7564371269b2ae86fa690dcb6a0cd7d18c74d7322e9fcc83ac6a61d80`. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
