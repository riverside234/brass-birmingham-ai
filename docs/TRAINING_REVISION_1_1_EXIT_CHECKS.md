# Results 1.1 revision: local exit checks

30 September 2026. The [revision plan](../ai-research/TRAINING_REVISION_1_1_PLAN.md) is implemented as Step 5 recipe 2.1, evaluation recipe 1.2 and searchlab recipe 1.3. Follow the [conda lab guide](TRAINING_RECOVERY_LAB.md) for a fresh run. These checks establish local execution and data integrity; they do not qualify trained playing strength or RTX performance.

## Focused checks

| Check | Result |
| --- | --- |
| `scripts/verify-recovery.py` | 20 passed: ingress/privacy and mutation rejection, all-seat feature parity, nonforced learning, KL guards, central routing, BC resume, retention/pins/restore and private matched search workers |
| `scripts/verify-searchlab.py` | 18 passed: four-seat selection/backups, observation privacy, fresh matched confirmation, legal timeout incumbents, distillation provenance and unchanged promotion requirements |
| `scripts/verify-step5.py` | 35 passed: legacy recipe compatibility, frozen qualification checks, BC/PPO separation, Pottery support, seat rotation, evaluation resumption and source/checkpoint integrity |
| Versioned configurations | All 21 training/evaluation/search configurations accepted by their strict validators; invalid fields and CPU-cap violations remain rejected |
| Python compilation | Changed modules and CLI helpers compiled |
| Source formatting | CR-aware Git whitespace check passed; pinned JSON bytes retain their existing newline rules |

BC neutral-context arrays match the pinned R2 encoder bitwise across four seats, including nonzero teacher style/intent inputs. Actor normalization excludes forced choices; all-forced trajectories still update the four-seat critic. Valid-looking changes to cash, selected action or terminal metadata invalidate checked records. Imported replays and unsigned feature sidecars receive full validation; independent receipts bind the raw replay, arrays and validator version.

The historical migration test now explicitly uses the original recipe. Its optimizer-preserving behavior is still checked. Recipe 2.1 intentionally starts fresh optimizer moments when importing compatible weights into a new run.

## Real host workflows

An earlier revision workflow completed two fresh qualified-v3 demonstrations, independent transition audits, one BC epoch, four training games over two resumptions, a four-game scheduled monitor and a separate four-game evaluation: **14 complete games**. The original opponent bank remained unchanged. This workflow used implementation `6500ba1369e2c1d2f1291bb69f9d27eeb8a36a2a29c1e928b2aa1cce575c483c`; its local records are under `artifacts/recovery/exit-8c3613bff0/`.

After the final attested-ingress and dependency-graph changes, a separate workflow completed **six fresh neural training games over three resumptions and four scheduled evaluation games**. Its records are under `artifacts/recovery/revision11-26fd8e41a3/`, with source identity:

```text
c2c1a0ea6f0295f00e5edcd4552958d6f31e2c8062b429a67c252090b33356b5
```

The final workflow verified:

- The monitor ran once at four ordinary games; resume neither repeated it nor added evaluation games to training counters.
- Three weight-only opponents and two full resume checkpoints remained hash-verified and loadable.
- Three retention transactions preserved current dependencies and quarantined retired full checkpoints/replays. The focused disposable-run check also restored quarantined files and rejected cleanup/restoration while active.
- The original frozen bank and its qualification evidence remained unchanged; source qualification still admits the original six identities.
- PPO performed 10, 10 and 6 optimizer steps. Preparation took 0.76, 0.72 and 0.42 seconds; replay loading took 2.58, 3.13 and 1.72 seconds on this local CPU.

These are small pipeline checks with unqualified smoke weights, not strength tournaments or proof of production-scale speedup.

## Replay-loading measurement

The same two raw games, containing 599 records, were copied into a disposable run. No supplied `results/` files were changed.

| Operation | Local elapsed time |
| --- | ---: |
| Full unsigned ingress validation and encoding | 29.16 s |
| Loading with independently recorded current validation/feature receipts | 2.89 s |
| Preparing separate neutral BC contexts from the checked arrays | 0.86 s |

Decoded authoritative records were identical. This is approximately a tenfold improvement in this specific replay-load comparison. The comparison includes local schema/cache warm-up effects; it is not an RTX, full-training or server-utilization benchmark. Old-version receipts trigger validation and cache regeneration rather than silently changing their trust status.

## Parallel search and runtime transfer

The authored lab worker compiled against the **pinned research rules runtime**. Child bank `bank-b880ac1fd5f1dfe8bcc2` contains five training-profile derivatives; original bank `bank-af17b1aadbe7f9b21fa7`, its adapters and its three-second profiles remain intact.

Four private v3 replicas completed a legal two-action turn with four semantic decisions and no reconstruction failures. The local run visited 46 reconstruction tokens and exposed one unique retained plan. Elapsed whole-turn time was **1.344 s**, above the one-second target. This is wiring evidence only; it does not show the required work/coverage advantage and does not admit the new teacher.

The quick S2/R4/S3 experiment ran one legal decision per variant with four private replicas each:

| Variant | Local elapsed time | Positive-confidence teacher |
| --- | ---: | --- |
| S2 | 1.547 s | No |
| OpenSpiel R4 | 1.219 s | No |
| S3 | 1.625 s | No |

Timeouts preserved legal incumbents. No teacher labels or timing/strength promotion were inferred from this smoke. The machine was also running other verification work, so these figures are not an uncontended algorithm comparison. Results are stored in `artifacts/recovery/search-smoke-revision11.json`.

`step5.py assets` exported six Git-ready authored model/worker assets and a **339-file private runtime bundle**. The new authored worker is `training/runtime/v3-lab/BrassLegacyAI.exe`. Child sources, profiles, mapping and worker are not ignored. Proprietary DLLs and `artifacts/step5-transfer/private-rules-runtime.zip` remain ignored. All transfer runtime fingerprints passed; no live game installation was changed.

## Lab acceptance still required

Run the guide's 25-game lab calibration and paired early/late Canal/Rail probes before using the one-second child. Its source/hardware/CPU-budget identity is distinct from the original parent's admission. Until it passes, `lab-reference.json` explicitly uses the qualified three-second control.

Measure fresh games/hour, meaningful decisions/second, load/update time, queue tails and CUDA profiler traces on the RTX lab machine. The one-second target can be exceeded by a slow or cold RPC; overshoot is recorded and fails timing qualification. More replicas do not by themselves establish equivalent work or better play.

Renew R4/S3 correctness on the actual execution host before development comparisons. Promotion still requires at least 30 independent paired setups, one uncontended concurrent game, equal wall time and equal within-turn CPU allowance, positive paired advantage, and the existing privacy/timing/Pottery gates. Small monitors and development screens cannot promote. Substantive BC curves, economic-play improvement, useful confirmed teachers and final playing strength remain lab measurements.
