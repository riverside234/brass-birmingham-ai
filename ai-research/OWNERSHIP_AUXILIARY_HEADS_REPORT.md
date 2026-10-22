# Ownership and local VP companion heads — g560 adoption

Implemented 2026-10-22 from [the plan](OWNERSHIP_AUXILIARY_HEADS_PLAN.md). Branching is **1% per eligible decision**, with the existing eligibility rules and at most one fork per ordinary self-play game.

## Scope and current run

The g560 export records 179,200 ordinary games, checkpoint `generation-0560.pt`, checkpoint SHA `ff10e995a94b1a27ee605505bea2427bc05348f45a7bd9fc4b88cab66cb3c83d`, and source `b95f45e2b0be6672a4efb7a3cabf9e2fab3e152a667e7f229b6c09ab5cb01c71`. Its live checkpoint is not present here. Full-state validation used isolated, real PyTorch checkpoint fixtures with populated Adam state; the lab helper verifies the actual committed checkpoint before adoption.

The helper copies this run's recorded configuration. Following the debug-review request, only `local_scoring_auxiliary`, branching and `ppo_lr_controller.low_ratio` change; the latter is **0.45 → 0.50**. In particular, the **currently recorded insolvency 0.020 / cap 0.080**, VP cap 180, epsilon zero, T2 openings, fixed snapshot weight 1, PFSP pool/curriculum, teacher/evaluation settings, PPO objective, anchor g440, entropy/economic baselines and GAE evidence are retained. Earlier conversation settings do not replace the newer committed recipe.

No production training, evaluation campaign, checkpoint mutation or live cache cleanup was run here. Concurrent rating-chart display changes were left intact; the explicit source review includes their current implementation without changing rating fitting or the training recipe.

## Capture and targets

`local-scoring-targets/1.0.0` stores one sealed game-level envelope containing current-era targets in canonical stable-seat order. There are 49 slots and 69 links, with era-invalid links masked. Empty/unflipped slots are NONE; a flipped zero-VP industry keeps its owner. Targets retain exact, bounded integers, including values above the reward VP cap. Observer-relative classes and four-seat VP vectors are prepared from those shared maps.

The additive native recorder hooks the official `ScoreLinksPhaseStarted` transaction after the host's required liquidation handling and before scoring cleanup. It reconciles local totals with the official Canal/Rail scoring event dictionaries, records bonuses/outside-scoring adjustments, and checks actual VP changes. Network scoring includes all three endpoints where present and border-region contributions. Final Rail introduces no extra income settlement.

The recorder is a separate capability/RPC and separately pinned binary. It does not add targets to observations, action inputs, gameplay hashes or the core inference graph. Raw replay seals bind the envelope, producer hashes and branch identity. Newly enabled collections require their complete receipts. Old replays/IL without this optional field remain readable; source-bound derived caches can be revalidated/rebuilt.

Branch restoration preserves its capture origin, and each new fork begins its own recorder. Rail-only continuations have Rail-only labels. Existing prefix/fork exclusion and whole-family critic holdouts remain in force.

## Model and optimization

| Component | Parameters |
| --- | ---: |
| Existing width-512 v11.2 utility core | 5,928,616 |
| Slot companion head | 45,705 |
| Link companion head | 45,705 |
| Total auxiliary | **91,410** |
| Training total | **6,020,026** |

Each shared entity MLP combines the existing 512-dimensional context, four 32-dimensional seat descriptors, and 64-dimensional entity embedding. Width 64 feeds five ownership logits and four direct VP predictions. Final projections start at zero, with initialization under a private deterministic seed that restores global RNG streams.

The existing state encoding and descriptor/dropout result are reused for official and auxiliary losses. The original core module, state dictionary, model identity, IL loading and ONNX interface are unchanged. Heads are trainer-owned and omitted from slim opponent checkpoints.

Loss coefficients are slot/link ownership CE **0.05 / 0.05** and VP Smooth L1 **0.01 / 0.01**, with VP divided by 20. They sit outside the existing official `value_weight`. Valid entities are averaged per row; existing row weights and full effective-batch denominators apply once. Missing targets contribute zero without inflating the labeled fraction.

PPO and PPG use separate companion Adam histories, base LR **1e-4**, zero weight decay, clip 1.0. Core optimizer membership is unchanged and core parameters step once. PPG applies its existing warmup/decay to the companion LR and resets only its PPG moments at phase start. PPO companion moments survive PPG. No extra phase/epochs are introduced.

Activation from g560:

| Collections | Auxiliary gradient into core |
| --- | ---: |
| g561–562 | 0; heads only |
| g563–567 | 0.2, 0.4, 0.6, 0.8, 1.0 |
| g568 onward | 1.0 |

Age advances only after commit. `shadow` remains detached; `off` retains companion state without auxiliary optimization. These metrics never become LR, GAE, economic-health or acceptance gates.

PPG snapshots/selection move core weights, companion weights, value Adam and companion PPG Adam together. Rejected intervals and beta retries restore the matching retained sample counters. Both exact policy KL caps remain **0.002**; official held-out value loss, paired one-SE selection and coverage preference remain authoritative.

## Durable adoption and caches

The reviewed route is exact and non-transitive. Its helper locks the run, checks generation/hash/config, refuses conflicting output files, and writes only the requested config. Training startup validates the old full checkpoint before committing the adoption receipt.

Old PPG metadata is archived under `local-scoring/adoption-g000560.json`. After durable state publication, only audited files through g560 in the two owned PPG cache directories are retired. Unknown files, links, changed hashes and ownership mismatches fail before deletion. Owner markers stay. Restart during cleanup skips already removed files and preserves g561+ files. There is no hidden second copy of the old payload: restoring that old buffer later requires an external backup.

The new PPG origin is g560, with empty replay and epoch-vote history. The attained cap 2, bounds 2–6, saved beta 8 and KL-trip history remain. The first new phase is **g568** on g561–568, followed by **g576**. Both existing core Adam states, warmup, progress, population, RNG and other controller evidence survive adoption. Their ordinary configured phase-start behavior still applies later.

New label sidecars store compact shared per-game maps, not a target tensor per decision. They bind source/holdout identities and checksums, commit with the normal cache receipts, and expire only after checkpoint commit. A label envelope is hashed once per shared object during cache creation instead of once per row. Invalid sidecars fail admission.

The smoke exposed an LR endpoint issue: a valid scheduled `3e-6` could be stored as `3.000000000000001e-6` and fail a strict resume bound. Resume now permits only floating-point endpoint tolerance and preserves the saved value exactly. Companion phase-rate multiplication clamps its numerical upper endpoint to the configured base; material LR changes still fail validation.

## Validation and measured cost

**24 focused Python tests passed**, plus the native C# scoring fixture and bounded smoke. They cover dimensions/counts, B=1/2 output parity, all observer rotations, labels above the reward cap, missing/partial labels, shared gradients, head-only core/RNG preservation, off-mode preservation, full checkpoint/Adam restoration, g560 narrow review/helper behavior, crash cleanup, sidecar tampering/expiry and held-out diagnostics. Production PPG tests force each KL guard, beta retry, one-SE rejection and final coverage-selected restoration. Existing core ONNX tests passed at B=1/2 and 1–772 candidates, retaining legal masks and fused operations. Export-library deprecation/binding warnings were emitted; parity assertions passed.

Native tests exercise completed collector replay/derived caches, old replay admission, interrupted/complete receipts, branch restoration and Rail-only futures. Additional native fixtures exercise actual liquidation, overbuilding, flipped zero-VP tiles, unflipped tiles, Canal cleanup, surviving industries, era masks, all link endpoints and unchanged cash during final Rail scoring.

The final bounded smoke is at `artifacts/local-scoring-smoke-final/report.json`; a separate post-PPG restore receipt is `post-ppg-resume.json`. It used eight reduced fixture iterations, 163 cached critic rows per iteration, a restart after iteration 4, and verification of the full checkpoint after iteration 8. PPO retained eight steps each iteration; PPG ran only at iteration 8 and retained **14/14 steps**. Its ordinary holdout had **48 games**; this deliberately smaller fixture does not claim the production 144-game support. Overall / age-0 KL were **1.479e-6 / 1.635e-6**. The final iteration exposed **869 distinct eligible rows from 101 games / 197 era maps** and retained all 869; 16,467 distinct valid entity targets are correlated labels, not independent games.

| CPU measurement | Result / scope |
| --- | --- |
| Native seed 153, recorder off/on | 15.865 / 18.069 s; 318 decisions; identical full trace digest |
| Native seed 154, recorder off/on | 15.663 / 17.264 s; 315 decisions; identical full trace digest |
| Final label RPC | 15.4 / 16.0 ms for the two games |
| Medium model forward/backward median, core only / heads | 124.5 / 174.9 ms, 32 fixed rows, 2 CPU threads, 5 warmed samples |
| Core Adam median, core-only / companion experiment | 99.1 / 127.7 ms; same core parameter count |
| Companion Adam median | 8.6 ms |
| Small fixture PPG exact-policy guard time | 6.261 s across the phase |
| Small fixture held-out loss evaluation time | 5.361 s across the phase |
| Small fixture PPG core optimizer time | 0.631 s across the phase |
| Process lifetime peak RSS during PPG | 481,075,200 bytes (~459 MiB) |
| Eight-age critic / selected-policy arrays | 18,915,496 / 1,348,264 bytes |
| Eight-age auxiliary sidecars | 4,096,206 bytes |
| Observed cache directory peak | 24,600,333 bytes |

The native runs are serial full-trace fixtures with extra inspection/checkpoint RPCs, one pair per seed; they do not model the lab's 48-lane collection. CPU microbenchmarks are small and noisy, and do not predict GPU PPO time. CUDA was unavailable in this validation runtime, so GPU peak allocation and a production-size eight-age peak were not measured. Actual cache-release bytes are computed on the lab at adoption, not inferred from the report export. The implementation adds training work; the reduced branching rate may offset some collection work, but no net wall-time improvement is established.

Full/compact reports retain activation, contracts, available/attempted/retained support, missing/masked targets, CE/Brier/confusion by era/table, positive/NONE support, current occupancy support, VP MAE/RMSE/sum error, norms, loss totals, separate optimizer steps/LRs, rollbacks, cache sizes and timing. `export-training-reports.py` includes compact adoption/runtime metadata and normal metrics, excluding label arrays and private simulator snapshots. Auxiliary diagnostics use ordinary learner critic-held-out families; those families retain their existing actor-update eligibility.

## Continuation

Use the [g560 commands in the lab guide](../docs/MLP_V11_LAB.md#continue-from-g560-auxiliary-heads-1-branches-and-50-lr-threshold), which recover only verified incomplete work, prepare `joint7-v11-2-grouped-lr-critic-local-scoring-lr50-g560.json`, and continue **g561–580** from the unchanged IL folder. The user confirmed the previous adoption command was not run. The helper now includes the 50% LR threshold in the initial adoption. It uses the actual state; a stale generation is refused rather than rewound. Subsequent clean continuations reuse that config and update the generation guard without repeating the fresh-buffer flag.

## Follow-up debug review, 2026-10-22

Two integration defects were found and fixed:

- The production CPU reader discarded `context.revision`. Auxiliary inventory/forward needs this identity, so a real compacted collection could raise `KeyError` before PPO, despite the earlier direct-data smoke passing. Ingress now retains the validated revision, including when rows enter the PPG cache. A real native game is written, admitted, compacted by the production reader and passed through auxiliary inventory/forward/backward in the regression.
- A conditional `from brass_ai.bank import sha` inside the collection planner made `sha` a function-local variable. Checking a frozen checkpoint before that import could raise `UnboundLocalError`. The duplicate import is removed. Historical-first planning is tested with the recorder both enabled and disabled.

The LR rule now raises only when the three-valid-iteration median is **strictly below 0.50 × target_kl**, or **0.005** at the current target. Exactly 0.005 holds. Existing health/anchor gates, warmup, 1.25 raise factor, protective cuts, cooldowns and clamps are unchanged. The reviewed hotfix changes no saved multiplier/window/cooldown and does not force a raise. Reports retain the actual ratio and absolute threshold; fresh configuration generation uses 0.50, while old explicitly recorded configurations retain 0.45 until migrated. A threshold-only helper is available as `scripts/configure-ppo-lr-threshold.py`; the current unadopted g560 run uses the combined helper above instead.

All **35 focused regressions passed**, covering auxiliary capture/gradients, real compact-reader ingress, planner ordering, PPG joint rollback/cache expiry, initial adoption, exact Adam/RNG restoration, strict LR boundaries and narrow config/source reviews. The isolated CPU smoke completed eight reduced iterations with eight PPO steps each, a scheduled PPG phase retaining 14/14 steps, and full restart checks at iterations 4 and 8. Overall / age-0 probe KL were **1.479e-6 / 1.635e-6**, below both 0.002 caps. Its artifact is `artifacts/local-scoring-debug-g580-smoke/report.json` (the directory name predates the correction to the g560 entry point; this is an eight-iteration synthetic fixture, not a g580 production log). The live lab checkpoint remains untouched and unavailable here; actual production restore is verified by the locked helper/startup on the lab machine. No long training or evaluations were launched.

That debug pass's reviewed source was `c33ff1ba936905fe06c4b6436d316a5073c9049be4a8d8d0e9c734f87e071314`; core compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. Six narrow review routes were added, preserving the preceding 74,105 routes. The native recorder binary and core architecture did not change during that pass.

## Diagnostic lifetime crash repair, 2026-10-22

The first production iteration exposed a reporting integration defect: `league._train` called `phases.release()` before `local_scoring.diagnostics(model, phases.critic_dataset, ...)`. Release deliberately clears the dataset, so the later read raised `AttributeError: 'NoneType' object has no attribute 'rows'`. The earlier smoke exercised training and checkpoint restore but omitted this reporting/release boundary.

`finish_training_diagnostics` now materializes the auxiliary report and its bounded ordinary learner held-out metrics before releasing the datasets. Only the small report survives into checkpoint/report publication. Missing live data fails explicitly instead of hiding missing metrics. Disabled or absent companions still release datasets without running auxiliary diagnostics. The timing now includes this work in final diagnostics. No optimization, target, configuration, cache schema, architecture or checkpoint format changes were made.

All **22 focused regressions passed**: real PPO-to-report-to-release-to-checkpoint lifecycle in apply/shadow modes; disabled/absent handling; unchanged model, Adam, RNG and phase/companion payloads; labels, gradients, cache integrity, PPG rollback, adoption, reviewed-source restart and byte-preserving incomplete-iteration archival. The smoke now calls the same finalizer as the production loop. `artifacts/local-scoring-diagnostics-recovery-smoke/report.json` records eight reduced CPU iterations in **43.73 seconds**, eight PPO steps each, and **14 retained PPG steps** at iteration 8. Every iteration reported 48 eligible ordinary held-out rows from six games. Full checkpoint restores at 4 and 8 passed; overall/age-0 PPG KL remained **1.479e-6 / 1.635e-6**. This fixture does not represent production throughput or playing-strength evidence. The live lab run and checkpoints were not accessed or modified.

That repair's source is `72ec2bf892f91bfb645acdf2b741c47276403fc5b38c159afe1ddeb7eeccc2c4`, with unchanged core compatibility. Eight explicit review routes preserve the previous 74,111 entries and the already authorized adoption/mode/LR-threshold transitions. An already adopted run can keep its exact config. The reported crash precedes checkpoint save, so retry from the last committed g560 after archiving the failed collection with `recover-step5-iteration.py`; use the [lab recovery command](../docs/MLP_V11_LAB.md#continue-from-g560-auxiliary-heads-1-branches-and-50-lr-threshold). Do not repeat config generation or `--fresh-ppg-buffer`. The new source recollects the failed iteration; unsaved PPO work is not promoted to committed state.

## Source-fix restart after adoption, 2026-10-22

The next startup failed with `Local scoring adoption cannot repeat`. The source-review admission itself worked, but the commit path was incomplete: g560's full checkpoint still predates auxiliary adoption, so preparation correctly returns the already durable adoption receipt. `commit_transition` incorrectly treated any receipt argument as a new adoption, even when identical to the saved receipt. The prior restart tests stopped short of publishing this second source transition.

The commit now accepts an **exactly identical existing receipt** and preserves it unchanged. Any difference is rejected before backup/state writes. Pending or completed cleanup, the original g560 activation/PPG origin, first phase g568, archived metadata, full checkpoint and all training settings remain intact. A source transition still gets its own backup and checkpoint-bound signature record; this does not permit re-adopting or resetting the buffer.

All **13 focused regressions passed**. The new integration regression first reproduced the exact production exception, then passed with the fix. It covers initial adoption, completed cleanup, a second source transition, injected failure while publishing state, retry against the same backup, repeated startup from the original g560 checkpoint, deterministic companion initialization, preserved PPO/value Adam and RNG, and pending-cleanup handling. Changed origin, next-phase or cleanup status receipts fail without writing. Its bounded CPU PPO smoke crosses into g561, runs held-out diagnostics, saves the first companion-bearing checkpoint and restores it with age 1 and the original PPG clock. These are isolated authored fixtures; the live lab checkpoint was not accessed or modified.

The current source is `ef1cb36d346edd6ce306324370fe60f759b297b58282edff093b3f08aadf80be`; core compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. Ten explicit routes preserve all 74,119 preceding approvals and their existing configuration restrictions. This startup error occurs before collection: after syncing the fix, repeat only the existing g560 **train** command in the lab guide. Do not remove the receipt/backup or repeat the fresh-buffer/configuration helper.

The implementation and bounded tests establish capture, gradient, cache and resume behavior. They do **not** establish improved playing strength or better learning per wall-clock hour. Normal subsequent PL evidence must answer that question.

## Main changed files

- Native capture/runtime: `src/BrassLocalScoringHost/LocalScoring.cs`, `scripts/build-local-scoring-host.py`, `training/runtime/local-scoring/ordinary/`, recovery `host_runtime.py`, `execution.py`, `ordinary.py`, `branching.py`.
- Labels/storage: recovery `local_scoring_targets.py`, `replay.py`, `ingress.py`, `ppg_replay.py`.
- Heads/training: learning `local_scoring_heads.py`; recovery `local_scoring_auxiliary.py`, `updates.py`, `actor_value.py`, `ppg_auxiliary.py`, `ppg_selection.py`.
- Adoption/reporting: recovery `local_scoring_adoption.py`, `config.py`, `exploration.py`, `resume.py`, `league.py`, `retention.py`, `history.py`; step5 `league.py`; source-review registry; `scripts/configure-local-scoring-aux.py`, `scripts/export-training-reports.py`, lab guide.
- Validation: `tests/Recovery/test_local_scoring*.py`, `tests/Recovery/LocalScoringChecks.cs`, report-export regression, `scripts/smoke-local-scoring.py`.
