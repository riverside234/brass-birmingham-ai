# S1 / Masked v3 comparison and training selection

1 October 2026. **Keep v3 + Masked ISMCTS for imitation generation and rule-based training/evaluation opponents**, following the user's condition: keep Masked if it performs better in the requested quick comparison. Both eight-game screens completed without planning or legality failures. Masked beat the two-worker S1 control on win rate, mean rank and VP in the second screen. Eight games provide a diagnostic choice, not proof of superior play across setups, eras or human opponents.

| Component | Active implementation |
| --- | --- |
| Imitation-data generation | v3 + Masked ISMCTS, one second, 2 private roots |
| Rule-based league opponents | Same frozen bank; v3 table curriculum 40% → 20% → 10% → 5% |
| Rule-based evaluation opponents | Same bank; Pottery Flex identity held out |
| Learned policy and self-play | Fresh 1,840,109-parameter MLP-ResNet; existing batching and PPO |
| Neural search/distillation | Masked ISMCTS + NN; raw Q, K 16/8, observation frontier, batched CPU/GPU inference |
| S1 control | Original frozen v3 S1, retained for reproducible comparisons |
| Installed game | Existing heuristic Masked ISMCTS; no game patch installed by this revision |

## Decision budgets and results

Each screen uses two independent setups × four seat rotations, with one Balanced candidate of each variant and two rotating serial S1 styles. Local runs use two concurrent tables to limit laptop contention. **16×2 means 16 concurrent heavy tables and two private search workers per acting AI**, not a search depth or 16 search replicas. Lab throughput has not been measured here.

The original S1 one-worker control has 240 total node work per turn. The previous lab S1 setting has 720 total work, split across two deterministic root partitions. Both share one second across the complete turn and replanning. The Masked control uses two private trees under the same one-second deadline; the S1 work multiplier does not multiply its simulation allowance.

| First screen, eight games | S1 1s / one worker | S1 1s / two workers |
| --- | ---: | ---: |
| Mean VP | 39.125 | 46.000 |
| Official win rate | 25% | 0% |
| First-place finishes | 2/8 | 0/8 |
| Mean rank, lower is better | 2.625 | 2.500 |
| Mean nodes/game | 287.875 | 541.250 |
| Mean measured planning seconds | 1.111 | 1.132 |
| Maximum measured planning seconds | 1.907 | 1.875 |

More work and VP did not establish an S1 strength improvement. See the [S1 budget receipt](../training/opponents/evaluations/s1-budget-smoke-1.0.0/report.json). The preparation helper snapshot changed after this screen to support clean lab clones; the receipt verifies identical compiled decision bytes and profiles.

Before the second screen, selection was declared as official win rate, then lower mean rank, then higher mean VP; an exact tie keeps S1.

| Second screen, eight games | S1 1s / two workers | Masked v3 1s / two workers |
| --- | ---: | ---: |
| Mean VP | 42.5 | 53.5 |
| Official win rate | 25% | 37.5% |
| First-place finishes | 2/8 | 3/8 |
| Mean rank, lower is better | 3.000 | 1.875 |
| Mean nodes/game | 588.000 | 771.375 |
| Mean measured planning seconds | 1.109 | 1.096 |
| Maximum measured planning seconds | 1.890 | 1.859 |

The [S1-versus-Masked receipt](../training/opponents/evaluations/s1-vs-masked-smoke-1.0.0/report.json) selects `masked_ismcts`. Nodes are implementation work counters, not equally expensive units. Actual retirement can exceed one second because complete incumbents, serialization and selection must finish; both controls drain all workers before reuse. Individual match receipts remain private under `artifacts/s1-training-smoke/switch-2026-10-01/` and `artifacts/s1-vs-masked-smoke/switch-2026-10-01/`.

## Frozen assets and active settings

The selected bank is **bank-981744ddae4aaf07f1cf**, manifest SHA256 `162ac11c52de11fd02214a3122cc9b3418472d13fded20d066724bd9a8aa4a18`. Its original strength qualification remains byte-for-byte unchanged. The new selection receipt is separate and does not reinterpret that evidence. Five styles enter training; Pottery Flex's identity is evaluation-only, while learner Pottery remains unrestricted. Old immutable banks and reports remain.

The S1 control is **bank-d82438ccec287dbc8a64**, manifest SHA256 `7f440ee6af7aa7f4231feda35199c3ef45101f6bba2db597335efb377d538145`. Its compiled runtime is identical to pre-ISMCTS S1 bank `bank-b880ac1fd5f1dfe8bcc2`; the sixth style comes from original v3 parent `bank-af17b1aadbe7f9b21fa7`. Its diagnostic admission preserves original provenance without adding it to the active training pool.

`lab.json`, `lab-16x2.json` and development evaluation allow 16 heavy tables and two private roots per acting v3. The main collection allowance is 28 ordinary actors + 32 heavy search workers + 4 tensor threads = 64 planned CPU threads, not an OS affinity cap. Heuristic v3 retains K 4/2, guarded sibling Q and a settled current-turn leaf. NN distillation retains three seconds, K 16/8, raw Q, warm simulator lanes and batched inference with two private root workers. Standalone neural probe worker settings remain unchanged. PPO/minibatch sizes, curriculum, evaluation cadence and retention are unchanged.

Heuristic actions are external behavior labels: they never become PPO neural likelihoods or tree-visit distillation targets. Only supported, completed Masked ISMCTS + NN visits generate distillation targets. Archived audited heuristic datasets retain verified provenance and may be reused; existing neural weights, replay and populations are not imported.

The private rules transfer bundle contains **452 members**, SHA256 `f69f6d8650af7eb0aac48f52aaf7f5ae2801bfa2c9afc38322fef41f282ad989`. It supplies the selected bank and separate S1 comparison runtime. The five project-authored workers and bank snapshots are Git-ready; game DLLs stay private. Windows CPU preflight and two full-game transfer parity checks passed. Use CUDA/Linux preflight after import on the lab.

The [conda guide](TRAINING_RECOVERY_LAB.md) has generation, fresh training, evaluation, both comparison commands, neural probes and cleanup. Use new dataset/run names `masked-16x2-v3` / `masked-16x2-fresh`; an older source-bound run cannot resume under this revision. The compact fresh NN integration passed BC/PPO/resume, evaluation, real batched searches, separate distillation, ONNX parity and retention. A selected-Masked actor game produced 351 audited rows with legal continuation replay and held-out identity rejection. The [exit receipt](../training/opponents/evaluations/s1-training-exit-1.0.0/report.json) records these checks and their source receipts. Server performance and human-level playing strength remain unmeasured.

## Rail-era weakness: targeted follow-up

The user's observation that game v3 spends too much on rail is consistent with valuation weaknesses in the current heuristic code. This is a diagnosis from source, not a replay-confirmed cause for the reported game. The sampled overall score advantage above does not establish rail-era quality.

- `Evaluator.Proposal` credits network link points and merchant access without an explicit monetary/resource cost term. Build proposals subtract tile cost. `MaskedSearch.Priors` and rollout choice use these proposals, so the difference affects candidate order and exploration. Real simulated rails still pay their legal costs; this is not evidence that the engine forgets payment.
- `Evaluator.Seat` uses a saturating projected-money value: available cash can become cheap to spend even when it is needed for the next useful investment. `CashManagement.Project` guards bills and borrowing-floor recovery, not the cash needed for the next feasible build-and-complete plan.
- Links receive forecast credit for unflipped endpoint industries, including opponents' tiles. These forecasts can be optimistic. `Conversion` also uses a fixed seven-per-missing-rail financing approximation rather than the actual market/resource costs of a complete connection. Rail Tempo adds an explicit link preference once cash is at least five, without testing that full cost.

Keep Masked ISMCTS and first repair its heuristic inputs. The [original ISMCTS paper](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) concerns search over hidden information; it does not supply Brass economic valuation. Our short heuristic horizon makes that valuation particularly important. Q normalization rescales scores and top-K changes which plans get work; neither corrects systematically optimistic rail values.

1. Instrument early/mid/late rail positions: actual paid cash, coal/beer use, secured versus forecast link VP, retained funding, best feasible industry alternative, chosen complete plan and its leaf ledger. Keep two-action build-and-sell, loan-and-build and develop-and-build alternatives in coverage.
2. Price proposals using actual feasible rail/resource paths and beer opportunity cost. Evaluate cash/action opportunity in the resulting state; avoid subtracting the same investment cost twice. Replace the fixed connection estimate where it changes completion feasibility.
3. Value retained funding against the best credible next industry/completion plan, using the AI's own cards and public resources. Include any necessary loan action and income loss. Discount uncertain opponent-dependent endpoint flips by time and competition; retain the full value of already secured VP.
4. Gate Rail Tempo preferences on economically credible plans. Preserve strong late-game rail opportunities; do not impose a blanket rail ban or a fixed rail-to-industry ratio. If immediate leaf evaluation still misses a clear investment sequence, add a bounded own-next-turn economic projection before considering broader rollout changes.
5. Use compact matched four-player checks and era breakdowns: rail spend, completed industry VP, unfinished tiles, forced financing, merchant beer and final rank. Require relevant industry plans to survive pruning and an improvement without canal regression. Then update both the game heuristic and a newly frozen research bank, preserving existing evidence and 2/5/12s game choices. Learned NN values/settings remain separate.

This rail follow-up is proposed work. No rail scoring or game binary is changed by the teacher-comparison revision.


## Superseding era/parallel revision

Use [the current conda guide](TRAINING_RECOVERY_LAB.md) and [Masked v3 era repair](MASKED_V3_ERA_REPAIR.md). The active recipe queues **24 tables with four workers per search decision**, one second for searched v3, under a shared 60-slot admission gate. Early RL uses a separate search-free v3 incumbent; searched v3 resumes at 10,000 games/5% tables. Imitation and v3 evaluation remain searched. NN distillation and the additional Masked probe use four roots, central CPU/GPU batching and selection-only virtual loss. Historical S1/16�2 commands and results above remain evidence, not current recommendations. Read `training/opponents/current.json` for the current bank and `training/runtime/distribution.json` for the new private bundle. Start a new source-bound run.
