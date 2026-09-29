# R2 exit checks — passed locally

27 September 2026. **R2's scaffold gate passes on Windows CPU.** The
[machine-readable summary](R2_EXIT_CHECKS.json) records source fingerprints,
compatibility metadata, dependencies and measured scope. This is not a claim of
stronger play, completed R3 search, a lab GPU run, or a deployed game opponent.

## Implemented and checked

| Area | Evidence |
| --- | --- |
| Daiyu adaptation | Pinned model/batching/schema references; separate R2 provenance record; AGPL text/notices included; no pretrained weights imported |
| Model sizes | Fresh 473,229 / 1,897,389 / 7,797,901 parameter networks; medium is the default |
| Policy/value shapes | Variable legal-choice scorer; four stable-seat win-share, competition-rank and VP heads; single-choice/padded batches and candidate permutations checked |
| Seat identity | Four observer rotations and all 24 spending-order permutations; no two-player sign alternation or shared bot reward |
| Unit/CLI checks | **23 tests pass**, including initialization, training, exact resume, export, stale/misrouted responses, input snapshots, cancellation and deadlines |
| Real-game collection | **Eight complete four-player games; 2,455 decisions**: four legal-random and four fresh-model self-play games |
| R1 observation coverage | **41 nonterminal restored positions** across both eras and all five decision phases, with all four observing seats represented |
| Privacy | **82 counterfactual prediction comparisons**: resampled opponent hands/deck leave allowed features, candidates and network outputs unchanged |
| Legal choice encoding | All **14** semantic choice kinds occurred in validated full-game data; an additional **seven-decision** fixture explicitly checks two sales, merchant beer, development/skip bonuses and completion through the inference protocol |
| Learning | Finite gradients on real replay data; five value updates; one terminal on-policy update from four complete games / 16 separate seat trajectories; no scalar redistribution or stock imitation |
| Resume | Optimizer state and RNG restoration reproduce the next supervised update's weights exactly, with authored fixtures and real-game data |
| Teacher targets | Confirmed-choice coverage excludes unexamined actions from the loss denominator; raw scheduler visit targets are rejected; legal losing trajectories remain available |
| Compatibility | Reject mismatched rules, schemas, features, reference pins, old scalar heads, wrong player counts, corrupted checkpoints, duplicate/invalid replay records and malformed numeric data |
| Export | Fresh and trained medium models pass ONNX checker and CPU Runtime parity at three dynamic batch/candidate shapes each |
| Regressions | All **37 R0 tests**, all **22 R1 tests**, 16 pinned reference files and the installed patch-1.1.0 binary pair still pass checks |
| Repository boundary | Replay shards, weights, ONNX graphs, downloaded Python packages and proprietary game dependencies remain ignored by Git |

The maximum absolute PyTorch/ONNX difference was `6.68e-6` for the fresh medium
model and `9.06e-6` after training, below the `2e-4` absolute parity tolerance.
These checks cover float32 CPU export; other precisions/providers need their own
checks before release.

An isolated warmed forward measurement on this computer was approximately
**2.32 ms median / 2.88 ms p95** for one observation and 340 candidates, using four
PyTorch CPU threads. It excludes encoding, validation, IPC, model loading, search,
Unity and turn execution. It is not a promised in-game thinking time. The complete
integration verification took about **967 seconds**; repeated validation of full
public histories and in-memory replay preparation dominate this scaffold. Do not
extrapolate that verification run into GPU training throughput.

## Reproduce

Install the dependencies using [R2_TRAINING.md](R2_TRAINING.md). Unit/contract/export
checks require no game files:

```sh
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/verify-r2.py
```

The full integration suite additionally requires the fingerprinted R1 runtime
and generated R1 restoration traces under the same project root:

```sh
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/verify-r2.py --integration
```

On Windows, supply an absolute local checkout path with `--project-root`; this is
the explicit override used for the recorded run. The commands default to the
user's lab root, not an inferred current directory. On Linux, qualify the host's
Mono behavior before treating generated matches as production data. Offline
training on compatible transferred R2 shards does not require the game engine.

Generated evidence is under ignored `artifacts/r2/exit-b2447dd7d79c/`, with the
supplemental inference fixture in `artifacts/r2/special-phases.json`. The public
JSON summary combines the final unit suite, full integration result and the
supplemental check. Re-running `verify-r2.py` writes a new local summary;
`--integration` also creates a new isolated artifact directory.

## Remaining gates

R3/H1–H3 must implement the qualified legacy bank, complete-turn planner and
asynchronous live-game bridge. Later work adds search-generated teacher targets,
population training, opponent-family holdouts, substantial strength evaluation,
personality/difficulty calibration and human playtests. R2's fresh-model games
are weak diagnostic games; their self-play win shares cannot establish skill.

The lab server, CUDA device, Linux host, Intel integrated GPU and Unity-plus-AI
waiting time have **not** been tested. The replay reader is currently in-memory,
and batching is synchronous with deadlines checked before and after native
inference. A running native kernel is not interruptible by the Python cancellation
flag. The initial features compress public history and mats, and current
collection uses zero style/intent inputs. These limits are explicit in the
[training guide](R2_TRAINING.md) and remain outside this milestone's exit claim.

No game AI replacement was installed and no repository changes were pushed as
part of R2. The existing game patch remains unchanged.
