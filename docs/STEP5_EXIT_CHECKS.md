# Step 5 local exit checks

Implemented and checked on **29 September 2026, Windows CPU**, using Python 3.12
and PyTorch 2.8.0+cpu. The [machine-readable record](STEP5_EXIT_CHECKS.json) binds
the implementation, selected weights and private evidence hashes. The
[lab guide](STEP5_LAB_TRAINING.md) contains reproducible training/evaluation and
runtime-transfer commands.

| Check | Observed result |
| --- | --- |
| Focused Step 5 unit checks | 17 passed: independent seat context, equal industry exploration, Pottery coverage, self-play quotas, split isolation, stale-likelihood rejection, learner-only PPO gradients, complete-plan teacher coverage, Reptile interpolation and frozen snapshot checks |
| R2 regressions | All 24 tests passed, including checkpoint/export compatibility; R2 architecture/encoder and R3-bound adapter files were not changed |
| Runtime preflight | Two complete four-player random-policy games recorded portable semantic fingerprints; private dependencies and frozen v3 worker verified |
| Real-engine search privacy | Two different hidden assignments produced identical permitted root views, legal choices, fixed-work plans and confirmation values |
| Neural search distillation | One complete four-player game; one scheduled search, eight complete turn proposals, two fresh-confirmed finalists, 93 reply choices, no search timeout; an actual covered teacher row produced two gradient updates |
| Population PPO and restart | Eight full self-play games in two separate CLI invocations; 32 learner seat trajectories and 2,436 decisions; optimizer/checkpoint resume produced a third distinct identity after initialization and the two updates |
| Pottery participation | Twelve Pottery builds across the eight self-play games; both batches included a Pottery learning context and neither triggered the zero-use warning |
| Ordinary evaluation | One fresh validation setup, all four candidate seats, no failed games; promotion correctly rejected |
| Meta-RL training | One frozen neural opponent-lineup task, two fresh support updates, separate before/after query games; no query gradients |
| Meta-RL evaluation | An unseen lineup, two disposable support updates, separate before/after query games; initialization unchanged and adapted weights not published |
| Private runtime transfer | Clean import of all 226 manifest-listed files, idempotent re-import and rejection of a conflicting file in an isolated test directory |
| Git distribution | All 53 required authored asset paths are present and unignored; pinned asset hashes and the released checkpoint load passed; the private runtime ZIP remains ignored |
| Original Hard preservation | A complete four-original-Hard game through the transferred frozen runtime: 64 turns, 124 ordinary actions |

The actual first-batch model placed **2nd, 3rd, 3rd and 4th** against three fresh
neural reference seats. It did not win any of those four games. This is an
evaluation-path smoke check, not evidence of improved strength. The second-batch
weights are preserved as
[`step5-selfplay-medium-smoke.pt`](../training/models/releases/step5-selfplay-medium-smoke.pt)
with a [manifest](../training/models/releases/step5-selfplay-medium-smoke.json).
They contain 1,897,389 parameters and are explicitly **not promoted**. The tiny
meta-RL query samples likewise do not demonstrate adaptation benefit.

The two four-game PPO batches took about 443 and 455 seconds, including collection,
full replay validation and updates, while other CPU checks were running. These
are laptop workflow timings, not GPU training throughput estimates. The lab
must measure its own CPU simulation, IPC/encoding, GPU inference and update cost.

The Git distribution contains the three existing R2 bootstrap/scaffold weights,
the selected Step 5 smoke weights, current v3 original source/profiles/manifests,
and the project-authored R1/v3 executables. The original Hard implementation and
game rules DLLs are in the separately ignored, fingerprint-checked private ZIP.
`scripts/check-step5-distribution.py` checks the required paths against Git ignore
rules; unignored files still need to be committed/pushed before a server clone
can receive them.

Remaining acceptance work is explicit: Linux/Mono parity on the lab, CUDA
qualification, substantial self-play training, broad neural-population strength,
equal-budget search ablations, measured meta-adaptation benefit, calibrated
personality/difficulty, human enjoyment and trained-model integration in Unity.
The current game patch and frozen R3 policies were not replaced by this work.

## Qualification transfer repair

The first Git upload normalized the current v3 qualification report to LF, while
the pool retained its original CRLF byte hash. This was reproduced directly from
the committed Git blob. Main training rejected that report before creating a run;
pure-neural smoke training and diagnostic preflight did not exercise admission.

The follow-up preserves report bytes through Git and adds
`scripts/check_qualification_reports.py --repair-line-endings`, which restores
only newline variants matching the existing SHA-256 exactly. The distribution
check now covers the report and all six admission checks. Frozen bank code,
qualification pins, model weights and trainer signatures remain unchanged.

All **23 tests** passed (17 existing plus six transfer regressions), including
conversion in both directions, repeatable repair, rejection of edited content
and conflicting fingerprints, and Git clean-filter checks with `core.autocrlf`
set to `true`, `false` and `input`. All **54 distribution paths** passed checks.
These are local tests reproducing Linux checkout bytes, not a claim that this
agent ran the repair on the lab. The user reports that conda installation,
runtime tests and CUDA smoke training passed there. Updated conda commands,
repair instructions and staged training budgets are in the lab guide.
