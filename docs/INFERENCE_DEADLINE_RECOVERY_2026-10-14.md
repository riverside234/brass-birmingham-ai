# Teacher inference deadline failure and generation 170 recovery

The supplied lab traceback ends in `ValueError: Request no longer live` during
the first teacher collection after generation 170 (54,400 ordinary games).
The next 320 ordinary games and their in-memory PPO update had completed, but
generation 171 had not reached checkpoint publication. The lab files have not
been inspected locally; the recovery command verifies the actual committed
generation before changing any directory.

## Cause and fix

`NeuralPolicy.prediction_outputs` checks request liveness before calling
`accept_response`. The absolute whole-turn deadline can pass between these
checks. The second check previously raised an ordinary `ValueError`, bypassing
the search loop's normal `SearchIncomplete` handling and failing the actor.

The final acceptance gate now raises a specific `RequestStopped` subtype of
`ValueError`. Search converts only that subtype to `SearchIncomplete`, discards
the late result and retires its pending paths. Request/model/revision/seat,
candidate order, input digest and finite probability checks remain strict.
Non-search inference still rejects stopped requests. The identical race in
personality search is also repaired.

No training settings changed. Teacher phases remain 12/12/16/24 seconds per
shared whole turn, with four private search workers. Existing reward, model,
feature, replay and checkpoint formats remain unchanged.

## Continue the supplied lab run

Sync the changed source, the new recovery script/module and both files under
`configs/learning/*-sources.json` to the lab together. Run these commands in the
existing `ai-training` conda environment after the failed training process has
exited.

First inspect the recovery plan; this verifies the checkpoint, Adam state,
source receipt and live opponent hashes, and hashes the partial evidence:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/recover-step5-iteration.py \
  --root /data/not_backed_up/yxu209/ai-training \
  --run artifacts/step5/v4-selfplay-joint \
  --expected-iteration 170
```

Apply the same verified operation:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/recover-step5-iteration.py \
  --root /data/not_backed_up/yxu209/ai-training \
  --run artifacts/step5/v4-selfplay-joint \
  --expected-iteration 170 --apply
```

The command moves `iteration-0170` into
`artifacts/step5/v4-selfplay-joint/.interrupted-iterations/iteration-0170-<state>-<id>/`,
alongside a byte-exact `state-before.json` and a manifest of preserved file
hashes. It leaves `state.json`, committed checkpoints, champion/population,
optimizer/RNG and accounting intact. It refuses active runs, changed
checkpoints, an unexpected generation, links, an uncommitted generation-171
checkpoint (including a temporary checkpoint), or an iteration report that
could indicate a partially committed update. Such failures require inspection;
do not alter signatures or remove files to bypass them.

Resume the original run/config/IL origin with the source-review flag. This
one-iteration command verifies recovery through generation 171:

```sh
CUDA_VISIBLE_DEVICES=2 python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --root /data/not_backed_up/yxu209/ai-training \
  --config configs/step5/joint7-g150-gate72.json \
  --device cuda \
  train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-selfplay-joint \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source \
  --iterations 1
```

The unfinished iteration is recollected and updated from generation 170;
its old-source partial replays are preserved but are not mixed into the new
collection. No repeat IL pass or new run is needed. Normal reviewed-source
utility calibration still applies. After generation 171 succeeds, continue
with the desired **additional** iteration count. Reusing `--iterations 90`
directly from generation 170 targets generation 260; it does not automatically
subtract iterations completed during the failed invocation. If the original
target was generation 240, use 70 additional iterations from 170, or 69 after
the one-iteration check.

## Verification and compatibility

- Deterministic reproduction forced expiry between the two checks and produced
  the reported error before the fix. All nine deadline regression tests pass
  afterward, covering grouped multi-model inference, cancellation, personality
  search, invalid predictions and pending-path retirement without late backup.
- 49 recovery/search/source-receipt tests, 14 inference/personality tests and
  five further continuation/IL/combined-utility inference tests passed:
  **68 tests, no failures or skips**. Actual queue processes run on CPU; the
  lab Linux/CUDA training run has not been executed here.
- Recovery tests use disposable real checkpoints to verify exact checkpoint
  bytes, Adam moments, RNG, counters, population/champion and ordinary reviewed
  resume, plus dry run, repeat invocation and failed rename behavior.
- All 117 other configuration JSON files retain their original hashes.
  The 1,035 existing self-play approvals and 25 IL approvals are preserved.
  The new registry carries forward 101 existing explicit routes with identical
  configuration restrictions and adds the immediately preceding source route.

Previous implementation:
`56f229de63ee148b7bd9e41245ae4d0b8953d59d85c16440d7b4e00c6d52f4ac`.
Fixed implementation:
`6446834c236ef81c720c1f166a5f281798c64a2a4c7cba9419382e18531dc4ad`.
Learning compatibility:
`c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`.

Local reproduction, test logs and preservation receipts are under
`artifacts/inference-deadline-fix-20261014/`. No lab data or installed game
binaries were changed, and no training campaign was launched.
