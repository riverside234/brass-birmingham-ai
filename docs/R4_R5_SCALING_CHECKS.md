# R4 / R5 scaling checks

29 September 2026, Windows CPU / Python 3.12 / PyTorch 2.8.0+cpu / OpenSpiel 2.0.2.
This is the execution/config update to the [additional search experiments](R4_R5_SEARCH_LAB.md).

| Check | Result |
| --- | --- |
| Focused search tests | **14 passed**: the original ten algorithm/contract checks plus four scaling tests. |
| Larger config and scheduling | 64 setups produce exactly 768 unique `(variant, setup, seat)` jobs, with stable shuffled scheduling. Previous small defaults remain loadable. Invalid execution settings are rejected. |
| Timing qualification | A synthetic positive paired advantage can qualify in serial mode; the same result cannot qualify a parallel screening run. Existing incomplete-rotation, over-budget, test-partition and Pottery gates still pass their tests. |
| Match caching and failure evidence | Completed replay/metadata records are reused; changed spec identities fail; earlier failure files survive a later successful retry. Tested with authored trajectories and stub policies. |
| S3 update wiring and resume | Two authored teacher fixtures use separate sampling seeds, minibatch 512, candidate-cell cap 131072 and eight configured update steps. With no confirmed teacher targets the actual update function returns `updated: false`, model weights stay unchanged, and resuming a finished experiment does not update twice. |
| Real process/host smoke | **Three legal decisions**, one each from S2/R4/S3, across **two distinct spawned worker processes**. Every selected choice was accepted by the real R1 rules host; the source checkpoint file stayed unchanged. |

Commands executed locally, with `--root` pointing to this checkout:

```text
python scripts/verify-searchlab.py --root <checkout>
python scripts/smoke-searchlab.py --root <checkout> --device cpu --seconds 1 --workers 2
```

The process smoke completed in about 8.4 seconds including startup. Observed decision times were S2 **1.234 s**, R4 **1.093 s**, and S3 **1.250 s**. All three searches were incomplete at the one-second allowance; R4 completed one simulation, and no variant produced a confirmed teacher label. Legal fallbacks and recorded overruns are expected smoke outcomes, not a search advantage or latency qualification.

Tested search implementation identity: `ac30f2f199598a5e01c6ab63b5668b0c9d84ea4d65ec87279b689ff5cb2708d3`.
Step 5's identity stayed `9be910b103d9fd744555b2203d073e7bbf4be4fd1b6d36745fe4591fb12fe841` before and after this update. No training-run migration is introduced by the R4/R5 changes.

No full-game tournament or full privacy/chance integration rerun was performed for this small scaling update. The [original R4/R5 exit evidence](R4_R5_EXIT_CHECKS.md) remains historical evidence for its recorded source version. A fresh machine/device correctness gate is required before longer comparisons or teacher runs. Linux/Mono execution, CUDA behavior, eight-worker throughput, confirmed-teacher quality and trained playing strength still require lab measurement; no promotion was created.
