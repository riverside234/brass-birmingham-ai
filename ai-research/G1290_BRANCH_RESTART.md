# Restart from g1290 in a new run folder

Date: 2026-11-10. Source run: `artifacts/step5/joint7-v11-2-g1290-trained-restart` (read only). New run: `artifacts/step5/joint7-v11-2-g1290-restart-g1290`.

## What the new run starts from

The g1290 full checkpoint is gone: the run keeps only its last two. Two retained pieces still describe g1290 exactly:

- **State:** the committed g1290 `state.json` from the results export `joint7-v11-2-g1290-trained-restart-g1290-20261109T041246972332Z`. Its bytes are checked against the export receipt, and its history must be the prefix of the source run's current history.
- **Weights:** the g1290 rating snapshot `population/weights-501ac43b…-aux-efd2fad16ee4fdca.pt`. It is the g1290 checkpoint with the optimizer, the PPG state and two diagnostic payloads removed, and it is checked against the state's signature and pool digests.

The new run keeps the generation count (the first iteration commits g1291), games, seed streams, opponent pool, ratings, the qualified Rust backend, the finished entropy anneal, the GAE lambda and per-table gates, the local-scoring heads, and the search head with its Stage A state.

## What starts fresh

| Part | New run |
| --- | --- |
| PPO | New AdamW; the 100-step LR warmup and the LR controller restart (multiplier 1.0) |
| PPG | New Adam, empty two-age replay, beta at the recipe floor 0.25 |
| GAE | Evidence windows refill; lambda 0.95 and table gates are carried from the g1289 report |
| Teacher | Empty label buffer; beta is measured again (below) |
| Search head | Empty value buffer; Stage A counts paused for the first 8 iterations |
| Opponents | Members whose weights are gone are dropped; missing probe caches are rebuilt from the IL probe |

## Recipe

This is the g1307 recipe (PPG keep-at-KL-cap, 64 search-head steps, 4.0e-6 PPO LR floor, share target 0.1, c_scale 1 on the CPU path), with two changes:

1. **Entropy:** no raise, so PPO entropy stays at 0.001 from the finished anneal.
2. **Search head:** Stage A lasts 30 iterations instead of 10. The g1290 counter (4) is kept, so Stage A ends after 26 more counted iterations.

A test checks that, once teaching is on, the recipe equals the g1343 recorded recipe minus the raise, with Stage A 30.

## Schedule

| Iterations | Commits | Teacher | Search head |
| --- | --- | --- | --- |
| 1–4 | g1291–g1294 | Shadow: labels and gradient measurements only, no teacher updates; measures beta for share 0.1 | Not trained; value buffer fills |
| 5–8 | g1295–g1298 | On, with the measured beta | Paused while the buffer refills |
| 9 on | g1299 on | On | Stage A head-only training, counts 5 to 30, then Stage B |

The fifth iteration needs the usual reviewed switch from shadow to on: run `continue-rust-teacher.py` at g1294 (command below). It refuses unless four consecutive full shadow iterations qualified the beta.

## Commands

Stop the trainer, commit and pull on the lab. The CPU workers are unchanged, so no prepare step is needed.

```bash
python scripts/prepare-rust-branch-restart.py
```

```bash
python scripts/continue-rust-shadow.py --run artifacts/step5/joint7-v11-2-g1290-restart-g1290 --expected-generation 1290 --iterations 4 --share-target 0.1 --inference cpu
```

```bash
python scripts/continue-rust-teacher.py --run artifacts/step5/joint7-v11-2-g1290-restart-g1290 --expected-generation 1294 --iterations 10 --share-target 0.1 --inference cpu
```

The first command prints `dropped_members` and `regenerated_probes`. Expect g1045 and g1145 to be dropped if the source run's retention deleted their weights.

**Share target.** Any value above 0 and at most 1 works, for example `--share-target 0.4`. Pass the same value to all three commands: the teacher beta measured during the four shadow iterations belongs to that share, and `continue-rust-teacher.py` refuses to switch on with a different one.

**Superseded files.** The source run's retention deletes superseded files such as an older compacted rating archive (`ratings/history-<digest>.json`). When a file the g1290 state names is gone from the source run, the script takes it from the g1290 results export and checks its bytes against the state's receipt. It stops only if neither place has it.

## Verification (laptop)

- 12 tests in `tests/Recovery/test_branch_restart.py`, including an end-to-end build of a small source run made from the exported g1290 state. The built run passes the real resume loader, PPO optimizer restore, exploration restore (entropy 0.001), the PPG phase (beta 0.25, lambda 0.95, historical gate kept), the teacher phase (empty beta, ready), the search head (Stage A, count 4 of 30, paused) and the backend state check.
- The shadow→on switch passes the reviewed transition check, and Stage A length can't be changed by a transition.
- Not run on the lab. The IL probe import and probe rebuilding were stubbed locally.
