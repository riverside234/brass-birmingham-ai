# Training recovery from the results 1.0 run

30 September 2026. The core recovery recipe is now implemented for Step 5 training, evaluation and the additional R4/R5 experiments. Use [the recovery lab guide](../docs/TRAINING_RECOVERY_LAB.md) for commands and migration. This document retains the motivation and acceptance criteria; no trained candidate is claimed qualified. Its companion is the [CPU and GPU throughput plan](TRAINING_THROUGHPUT_PLAN.md).

**Recommendation:** use a short, audited v3 imitation warm start, then resume learning through self-play with a declining v3 opponent curriculum. First change the update schedule: the 320-game report contains only three optimizer steps. Increasing the number of games without increasing useful learning updates is not sufficient. Keep the approximately 1.90M-parameter backbone for this recovery experiment.

All games have four players. Use the existing conda environment `ai-training` and laboratory root `/data/not_backed_up/yxu209/ai-training/`. Preserve the current checkpoint and run as controls. Versioned schemas, audited BC labels, effective-minibatch updates, persistent curriculum counters and shared execution are implemented; playing-strength and lab performance acceptance remain to be measured. Economic reporting currently includes permitted-observation proxies rather than exact per-tile era-end VP/waste attribution. DAgger-style learner-state queries remain a conditional follow-up if distribution shift is observed.

## Evidence and limits

Sources are [report.json](../results/1.0/report.json), [earlier_report.json](../results/1.0/earlier_report.json), and the current [league](../src/brass_ai/step5/league.py), [updates](../src/brass_ai/step5/updates.py), [losses](../src/brass_ai/learning/training.py), and [lab configuration](../configs/step5/lab.json).

The earlier report is invalid JSON. It contains two pasted `iteration-0003` strings, including one inside its game count. Removing exactly those strings **in memory for analysis** recovers a consistent 20-game report. Its values below are provisional; the original file is preserved. It is one iteration, not the complete earlier training history.

The valid report contains **320 total games: 224 self-play, 80 historical, and 16 legacy**. Those 80 historical games are already included in 320. The user's estimate of roughly 400 cumulative games may include earlier iterations, but these two reports alone cannot reconstruct the full learning curve. Recover that curve from the lab run state and all iteration reports, recording parent-run ancestry.

| Measurement | Earlier iteration 3, recovered | Scaled iteration 0 |
| --- | ---: | ---: |
| Games in this report | 20 | 320 |
| Learner seat trajectories | 62 | 992 |
| Learner decisions | 4,902 | 77,095 |
| Mean learner VP | 5.18 | 5.70 |
| Aggregate learner win share | 24.19% | 24.85% |
| Builds per learner seat trajectory | 2.90 | 2.88 |
| Search distillation | Not due | Not due |

These are collection outcomes under the incoming policy, **before that iteration's optimizer updates**. The saved generation is produced afterward. Evaluate the saved checkpoint separately before attributing these scores to its updated weights. Different setups, populations, and sample sizes also prevent treating 5.18 to 5.70 as a reliable improvement.

### Why the apparent 25 percent win rate is misleading

All-current self-play contributes one total learner win share per game regardless of playing quality. In the scaled report, 224 of the learner's 246.5 win shares come from those 224 tables automatically. The historical neural opponents themselves average only 3.07–4.22 VP.

The legacy seats win all 16 legacy tables: the learner wins **0/16**. Their mean scores are 110.19 VP for Rail Tempo, 107.13 for Balanced, and 99.63 for Cotton. Only these three of the five admitted training profiles appear in this collection. The code samples the opponents once for a collection group, so large groups can reduce opponent variety.

This supports a bootstrap from qualified v3 behavior. It does not establish that v3 is strong enough to remain the long-term teacher. Separate score, placement, and win-share reports by table composition and opponent identity; never use the aggregate all-current win rate as a promotion measure.

### What the losses show

| Report and epoch | Actor loss | Combined value loss | Entropy | Approximate KL | Gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier 0 | 2.379995 | 1.402011 | 1.423785 | approximately 0 | 1.2753 |
| Earlier 1 | 2.369034 | 1.401192 | 1.423755 | 0.00004875 | 1.2953 |
| Earlier 2 | 2.353490 | 1.399517 | 1.423621 | 0.00018705 | 1.3107 |
| Scaled 0 | 2.430216 | 1.403157 | 1.432412 | approximately 0 | 0.5831 |
| Scaled 1 | 2.426385 | 1.402398 | 1.432130 | 0.00003606 | 0.5757 |
| Scaled 2 | 2.422065 | 1.401294 | 1.431818 | 0.00013443 | 0.5670 |

Within the scaled update, actor loss falls 0.34% and combined value loss 0.13%. The recorded policy change is small; gradients are finite and well below the current clipping threshold of 5. This does not look like an exploding-gradient failure. The logs do not establish overfitting either: they contain no held-out loss curve.

The implementation accumulates gradients across all 226 memory microbatches before stepping once per epoch. Thus `minibatch_size=512` is a memory limit, not 512 examples per optimizer step. Three epochs produce three steps for all 77,095 decisions.

Two qualifications matter when tuning:

- Actor loss is summed over decisions and divided by seat trajectories; value and entropy are averaged over decisions. There are about 77.7 decisions per trajectory here. This is an intentional episode-style weighting, but changing to conventional minibatch means changes the relative loss scales. Version and measure that change explicitly.
- Combined value loss is win-share cross-entropy plus 0.25 normalized placement Huber loss and 0.1 normalized VP Huber loss. Although `ln(4) = 1.3863` is the uniform four-seat win cross-entropy, a combined value loss near 1.4 cannot establish exactly uniform predictions. Log the components and calibration separately.

Epoch metrics precede the corresponding optimizer step; the report lacks a measurement after the final step. A near-zero first-epoch KL is expected when collection and learner weights agree. Do not increase learning rate solely to force a particular reported loss or KL.

## A0 Establish a trustworthy starting comparison

Freeze the current trained checkpoint, its parent history, optimizer state, config, source fingerprints, and runtime identity. Open a new recovery run; do not overwrite or silently resume a changed algorithm into the old run. Changing objectives or optimizer-step semantics should start fresh optimizer moments while retaining compatible model weights and recorded ancestry.

Add a compact per-composition report and fresh post-update evaluation. Alongside wins, placement, and VP, record:

- Action types and decision phases, separating forced choices from meaningful choices.
- Industries built, sold/flipped, scored, and left unflipped at each era end; useful resource depletion for coal, iron, and breweries.
- Cash, income, loans, liquidation, and repeated pass/economic-stall events.
- Feasible sales, merchant beer use and bonuses, and failed versus abandoned sale continuations.
- Pottery opportunities, completed sales, VP contribution, and performance with Pottery focus.

Use these to identify whether low scores come from failure to build, complete sales, finance an engine, or coordinate a turn. The current build counts establish little productive construction; they do not prove any specific cash or sale defect. Also audit chosen-choice/seat mapping and terminal scoring on a small saved replay sample before diagnosing every problem as learning.

**Exit:** the starting frozen model has fresh neutral-policy results and decision-level economic diagnostics against the same fixed panel used for later candidates.

## A1 Build an eligible demonstration dataset

### Reuse existing games selectively

A limited inventory under `artifacts/legacy-ai`, `artifacts/step5`, and `artifacts/r2` found 28 compressed replay shards, about 67 MiB: 7 marked diagnostic, 11 evaluation, 7 training, and 3 without a purpose field. These are shard counts, not eligible game counts, and are not a complete inventory of every project artifact.

For example, `artifacts/legacy-ai/v3-integration.jsonl.gz` contains one complete game with 366 decisions and no search-teacher rows, but its bank is `bank-c891530a53c0ae07765b`, not the currently admitted `bank-af17b1aadbe7f9b21fa7`. A filename containing “v3” is insufficient evidence of compatibility. The stored [20-game v3 versus Hard evaluation](../training/opponents/evaluations/v3-vs-hard-20/report.json) is an aggregate report, not an observation/action training dataset.

Create a manifest with replay hashes, complete-game counts, rules/feature/choice contracts, bank identity, original purpose, setup group, seats, and exclusion reasons. Admit data only after:

1. Verifying complete permitted observations, the complete legal-choice list, selected semantic choice, outcome, and provenance.
2. Checking the rules version and replay transitions against the current research runtime. Older banks need an explicit behavioral/bug-fix review, or fresh recollection.
3. Preserving existing evaluation, qualification, final-test, and opponent holdouts. Diagnostic data is not automatically training data. If a development benchmark is deliberately retired into training, document that retirement and create a replacement benchmark first.
4. Deduplicating complete games and related setup/seat-rotation groups before assigning partitions. Respect existing partition contracts; do not relabel held-out shards to make the loader accept them.

Scores, screenshots, and summary JSON alone are insufficient. Reconstruct only when a complete, compatible seed/action trace can reproduce each permitted decision; otherwise collect new games.

### Generate additional games on the lab

Start with approximately **1,000 new four-player demonstration games**, balanced across the five admitted v3 training profiles and varied mixtures, seats, and valid setups. Keep losing games, recovery positions, both eras, and all legal action kinds. Exclude original Hard and the held-out Pottery Flex identity from teacher selection.

Reserve whole setup families for development and final testing through the existing partition mechanism; target roughly 80/10/10 for newly generated groups if compatible with that mechanism. Never split decisions from a single game, replay duplicate, or rotated setup across partitions. These demonstration partitions are additional to the fixed gameplay evaluation panel.

Only grow to 2,000–5,000 games if held-out learning curves or state coverage warrant it. Prefer newly visited situations to repeated copies of familiar openings. If a BC policy fails in its own unusual states, query an admitted masked v3 teacher on selected **training-only learner states**, recording its real label and any teacher failure. This addresses distribution shift without using hidden information or treating the heuristic as optimal. The motivation is the sequential-distribution problem studied by [DAgger](https://arxiv.org/abs/1011.0686); this limited repair loop is not a claim to implement its full guarantees.

### Add a separate imitation label contract

Legacy rows currently have `teacher=None` and external behavior probability 1.0. R2's teacher mode therefore does not automatically imitate them, and Step 5 correctly excludes their actions from on-policy PPO.

Introduce a versioned demonstration label, separate from fresh-confirmation search labels:

```text
L_BC = -log_softmax(student_logits over ALL legal choices)[expert_choice]
L_supervised = L_BC + 0.5 * L_value
```

The current coverage-aware search loss normalizes over covered choices. Marking only one demonstrated choice as covered yields probability one and **zero policy gradient**. Do not use that encoding, invent search confidence, or reinterpret external probability 1.0 as the student's behavior probability.

Use only causal inputs. The legacy wrapper currently exposes context after selecting a plan; copying that post-selection intent into the student input can leak the target. Start BC with neutral student context and permitted pre-decision information. Any later translation of legacy style into student style/intent needs an explicit mapping; similarly shaped vectors do not imply the same semantics.

**Exit:** an eligible manifest, disjoint setup groups, reproducible labels, and a focused gradient check showing that BC increases a demonstrated action's probability when multiple legal alternatives exist.

## A2 Run a short imitation warm start

Reuse the current 1.90M-parameter model as the primary initialization; compare a fresh R2 initialization on the same dataset as a small control. Do not assume the current weak policy must be discarded, or that more model capacity fixes data and update problems.

| Setting | Proposed starting value |
| --- | --- |
| Effective optimizer batch | 1,024 demonstration decisions |
| Memory microbatch | Up to 256–512 positions, bounded by candidate cells |
| Learning rate | `1e-4`; compare `3e-4` only if development loss improves slowly |
| Optimizer | AdamW; explicit weight decay `1e-4` |
| Epochs | Start with 3; maximum 10, development-loss patience 2 |
| Gradient clipping | Keep 5 initially; report actual clipping frequency |
| Value targets | Existing four-seat outcomes and placement/VP auxiliaries |
| Selection | Development NLL plus fresh gameplay and economic completion |

Sample across games, profiles, eras, and decision kinds so long subchoice sequences do not dominate. Bound any oversampling and report both natural-frequency and balanced development losses. Exclude forced one-choice decisions from policy accuracy claims; they can still supply useful value targets.

A falling training loss with flat or rising development loss means stop and improve data diversity, not add epochs. Watch development performance on uncommon industries and recovery states. Matching the expert's exact choice is only a diagnostic when several good actions exist; gameplay must corroborate it.

The supervised-then-self-play sequence has a precedent in [AlphaGo's combined training approach](https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/). That supports considering the sequence, not predicting its strength in four-player Brass with imperfect heuristic teachers.

**Exit:** the selected BC checkpoint beats its initialization on the fixed development matchup panel and shows improved productive construction/completion without an information-boundary failure. Exact agreement with v3 is not the release goal.

## A3 Change PPO update semantics and tune conservatively

Implement shuffled effective minibatches with an optimizer step after each effective batch. Split an effective batch into memory microbatches as needed, weighting each contribution by its actual example count. Freeze collection weights, old behavior probabilities, baseline values, and advantages for the entire update.

Use per-decision means and normalize learner advantages once over the declared collection population, recording the rule and its standard deviation. Preserve the actual exploration-mixture likelihood in the PPO ratio. Maintain each seat's own terminal reward; do not combine alternating players into one trajectory. Retain terminal Monte Carlo credit initially so this experiment can isolate BC and update changes.

| Parameter | Proposed recovery setting | Tuning rule |
| --- | --- | --- |
| Games per collection | 320 after a smaller correctness pilot | Keep comparisons at equal collected games and report optimizer steps |
| Effective PPO batch | 2,048 learner decisions | Compare 1,024 only after a stable baseline |
| Memory microbatch | At most 512 and 131,072 padded candidate cells initially | Performance control, independent of optimizer batch |
| PPO epochs | 2 | Raise to 3 only with useful held-out progress and stable KL |
| Learning rate | `3e-5` initially; compare `1e-4` | More steps change the effective update budget |
| PPO ratio clip | 0.2 | Try 0.1 if updates overshoot |
| KL guard | Stop remaining updates near mean KL 0.01 versus collection policy | Measure after updates; this is a guard, not a target to maximize |
| Value multiplier | 0.5 initially | Log win/rank/VP losses and component gradient scales |
| Entropy weight | 0.005 initially | Compare 0.01 if policy prematurely collapses |
| Explicit exploration | 0.05 after BC, compared with current 0.10 | Preserve legal and industry-balanced exploration |
| Gradient clipping | 5 initially | Lower only if observed unstable gradients justify it |

For 77,095 rows, batches of 2,048 give about 38 optimizer steps per epoch, or **76 across two epochs**, before KL early stopping or any explicit row filtering. This contrasts with the current three steps. The proposed normalization and coefficients form a new training recipe; old and new raw actor losses will not be directly comparable.

Stochastic minibatch updates are part of the [PPO formulation](https://arxiv.org/abs/1707.06347). The numerical choices here are project hypotheses, not values guaranteed by that paper.

Add clip fraction, post-update KL, advantage mean/std, policy entropy by phase, separate value losses/calibration, optimizer-step counts, and actual examples per step. Change one tuning dimension at a time. Do not respond to the current tiny KL by multiplying LR, epochs, batch size, and exploration simultaneously.

Forgetting prevention may use a small **separate BC loss**, initially coefficient 0.1 and decaying to zero by the mature self-play phase. Demonstrations must never enter the on-policy ratio as learner actions. Compare this against no rehearsal; retain it only if it prevents measured economic regression.

## A4 Reduce the v3 curriculum to five percent

Use explicit table quotas instead of overloading `legacy_every`. The current validator allows 0 or values at least 20, limiting v3 to at most 5% of ordinary tables. The following schedule needs a new versioned config and scheduler.

| Phase and provisional new training game count | All-current tables | Historical neural tables | v3 tables | Counts in 320 games |
| --- | ---: | ---: | ---: | --- |
| Bootstrap, 0–2,000 | 40% | 20% | 40% | 128 / 64 / 128 |
| Transition, 2,000–5,000 | 55% | 25% | 20% | 176 / 80 / 64 |
| Expansion, 5,000–10,000 | 65% | 25% | 10% | 208 / 80 / 32 |
| Mature, 10,000 onward | 70% | 25% | 5% | 224 / 80 / 16 |

“Game count” here means all completed ordinary training games in the new curriculum, excluding demonstrations and evaluation. Phase transitions use persistent cumulative counters, not local iteration numbers that reset on a fork. Counts are an initial budget; review held-out progress at each boundary. If economic competence collapses, stop for diagnosis or roll back instead of spending unlimited games against v3.

Keep the existing table definition initially: four learners for current self-play, one learner plus three frozen opponents for historical/v3 tables. Thus 40% v3 tables means 30% of all seat assignments are v3; the final 5% means 3.75% of all seats. This deliberately preserves the project's existing **table-based** meaning of 5%.

Rotate learner seats, balance all five admitted v3 profiles across the whole collection, and vary combinations within collection groups. Use the selected BC checkpoint and diverse frozen neural generations as historical opponents. Avoid filling early history only with the low-scoring pre-recovery population; keep a small fixed subset of those for regression diagnostics.

If three v3 opponents produce no informative learner wins even after BC, test a separately identified bridge phase with two learners and two v3 seats. Report its different composition and keep standard one-versus-three evaluation unchanged.

Pottery Flex remains held out as an opponent identity. Keep all Pottery legal actions, training-only Pottery opportunities from admitted policies, and the existing equal-industry learner contexts. Add targeted completion coverage if needed; do not force Pottery builds merely to satisfy a count. A zero-only Pottery warning is insufficient: the current 165 builds and 23 focus-matching builds do not establish useful Pottery play.

## A5 Coordinate search learning and the R4/R5 experiment

Both supplied reports say search is not due. The existing `search_start=10` and 320 games per iteration postpone the first search phase until after roughly 3,200 preceding games; a fork can reset that local schedule. Search cannot explain the current slow iteration or be credited for its learning.

After the BC checkpoint demonstrates basic competence:

1. Run a small S2 teacher audit on independent training games. Start with 2–4 complete games and a bounded number of searched turns; record completed confirmations, label coverage, latency, and teacher-versus-base decisions.
2. Trigger later teacher collection using persisted game/update counts and readiness, not a reset local iteration. Initially reserve approximately 10% of measured training wall time for teachers, then adjust by demonstrated benefit.
3. Require fresh finalist confirmation and valid continuation labels. If budgets produce no usable labels, reduce proposal breadth or increase the diagnostic budget; do not manufacture confidence. Preserve the complete-turn and hidden-information contracts.
4. Keep R4 OpenSpiel comparison and R5 S3 as separate branches using the same frozen BC or recovered checkpoint, opponent panel, setups, and whole-turn budgets as S2. Correctness gates still apply.
5. Compare S0/S2/R4/S3 first on a small development screen. Only collect substantial S3 distillation data after it produces usable labels and an equal-budget advantage. R4 search visits are not automatically confirmed policy-teacher probabilities.
6. Evaluate an S3-distilled checkpoint afresh against the undistilled parent, with and without search. Teacher evaluation games remain excluded from training.

Do not use deeper search as a substitute for repairing a near-untrained policy/value model. Conversely, do not infer S3 is ineffective from these logs: S3 was not measured here.

## A6 Evaluation and experiment decisions

Use three comparisons with the same initialization provenance, setup partitions, and training-game budget:

| Arm | Purpose |
| --- | --- |
| Existing frozen checkpoint | Preserve the actual starting behavior without repeating a full slow run |
| Revised PPO without BC | Test whether update frequency alone repairs the main weakness |
| BC plus revised PPO and v3 curriculum | Primary recovery proposal |

Use two training seeds for a shortlist if resources permit; expand only promising results. Keep meta-RL adaptation off during this recovery comparison, then rerun its separate support/query evaluation on qualified weights. Personality fine-tuning remains after Step 5.

For frequent diagnostics use 8–16 independent development setups with all four candidate-seat rotations against fixed neural references and a rotating admitted-v3 panel. Evaluate the **post-update frozen checkpoint** with exploration disabled. Report score margins and economic progress even before wins become common. This is a screen, not final qualification.

After selecting a recipe, run the established 64-setup Step 5 evaluation and untouched final-test partition. With two neural references and two legacy lineups, the current full configuration represents 1,024 games. Avoid paying that cost after every hyperparameter change. Include held-out Pottery Flex in the final mixed-opponent evaluation; keep its results out of gradient updates and repeated recipe tuning. The current full evaluator includes that held-out lineup, so add an explicit development-panel selector rather than repeatedly running the full panel during tuning.

Record BC checkpoints with their true training mode. The current Step 5 promotion gate requires at least two distinct references marked as population-PPO or Reptile trained. A BC checkpoint can be a diagnostic opponent, but must not be mislabeled to satisfy that gate. Retain the gate for formal promotion or version an explicitly justified replacement; the early recovery screen is not promotion.

For R4/R5, parallel comparisons remain screening. Preserve the current serial `workers=1` equal-budget promotion requirement, source/device correctness gate, at least 30 independent setups, and paired uncertainty analysis. Report completed simulations and confirmations alongside wins. Cluster confidence intervals by setup, not by correlated seat rotations.

**Acceptance:** the recovery candidate improves on the frozen starting model across held-out matchups, gains meaningful results against v3, maintains legal/private observations and economic completion, and does not collapse on Pottery or one opponent family. Beating weak v3 alone still does not demonstrate human-level or enjoyable play.

## Implementation sequence and ownership

| Work | Main files or new components | Required evidence |
| --- | --- | --- |
| A0 reporting | `step5/league.py`, `step5/evaluation.py`, replay analysis | Correct pre/post-update labeling and composition-specific metrics |
| A1/A2 demonstrations | New dataset manifest and BC trainer; versioned replay/label contract alongside `learning/replay.py` | No holdout leakage; causal inputs; nonzero BC policy gradient |
| A3 optimizer changes | `step5/updates.py`, checkpoint/config signatures | Correct likelihoods, effective-batch gradients, KL stopping, resume identity |
| A4 curriculum | `step5/league.py` and new config version | Exact quotas, persistent counters, five-profile/seat coverage |
| A5 search | `step5/search.py` and `searchlab` orchestration | Completed valid teachers and separate experiment lineage |
| A6 selection | Existing Step 5 and searchlab evaluators | Frozen, held-out, equal-budget comparisons |

Paths in this table are under `/data/not_backed_up/yxu209/ai-training/src/brass_ai/` unless explicitly described as new manifests/configs. Preserve frozen bank identities and runtime admission checks; do not edit their qualification reports to accommodate a new pipeline.

Start with A0 and throughput instrumentation, then eligible demonstrations/BC and the corrected PPO pilot. Implement the v3 scheduling isolation in the [throughput plan](TRAINING_THROUGHPUT_PLAN.md) before scaling early high-v3 collections. Only then spend substantial lab time on the curriculum and S2/R4/R5 comparisons.
