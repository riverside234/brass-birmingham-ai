# g1490 restart: long-search labels, 32 tables, one-iteration beta

Date: 2026-11-10. Source run: `artifacts/step5/joint7-v11-2-g1290-restart-g1290` (read only). New run: `artifacts/step5/joint7-v11-2-g1490-restart`.

The g1491 revision ([G1491_TEACHER_REVISION.md](G1491_TEACHER_REVISION.md)) did not train well over g1492–g1500. This restart goes back to the g1490 weights with a smaller teacher recipe and keeps the exploration.

## What it starts from

The source run kept only its last two full checkpoints and deletes the rest at once (`quarantine_hours: 0`), so `generation-1490.pt` and `generation-1491.pt` are gone. What survives:

- **Weights:** the retained g1490 rating snapshot, `population/weights-4a1f58e5…-aux-602cd59664e409fd.pt`. It holds the model, its companion heads and the search-head optimizer, but no PPO optimizer.
- **State:** the committed g1491 state in `results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z`. It is the last state before the revision. No g1490 state was ever exported, and it can't be rebuilt exactly (the opponent pool's results move every iteration).

So the new run takes the exact g1491 state (pool, ratings, counters, games) and puts the g1490 weights in as the learner, counted as generation 1491. The g1490→g1491 update is dropped along with g1492–g1500. The history lists the g1490 model at both 1490 and 1491. The first checkpoint trained in the new run is g1492.

## Recipe

| Setting | Before g1491 | g1491 revision | g1490 restart |
| --- | --- | --- | --- |
| Teacher tables | 28 (+4 held out) | 48 (+4) | **32 (+4)** |
| Teacher labels | long searches | long and short | **long searches only** |
| Teacher rows per PPO step | 64 | 2048 | **72** |
| Label buffer | 4 iterations, 4:3:2:1 | 2 iterations, 4:3 | **4 iterations, 4:3:2:1** |
| PPO epochs | 2 | 1 | **2** |
| Short search | n=96, K=8 | n=128, K=8 | **n=96, K=8** (`--teacher-fast 96`) |
| Beta re-calibration | – | 2 shadow iterations | **1 shadow iteration** |
| Exploration | none | visit counts + cotton | **visit counts + cotton** (unchanged) |

Kept from the g1491 revision: the played action is sampled from the root visit counts for each seat's first 7 actions of each era and for every long-search action; cotton exploration on training tables; the PPO LR controller holds during calibration; every revised table checks model-logit parity once per table (long-only tables included). Long searches stay n=640, K=16, and the CPU path uses 64 workers.

In code this is the `g1490-branch-restart` variant of `training-revision-g1491/1.0.0`. It is adopted only by this restart and can't be switched to inside a run. The in-run g1491 marker is unchanged.

## What the restart resets

- **PPO:** new AdamW. The 100-step warmup restarts from 0.1× the base LR 6e-6. The LR multiplier g1491 committed (1.25, from the g1490→g1491 report) is carried, so after warmup the LR is 7.5e-6 again. The controller's KL window starts empty.
- **PPG:** new Adam and an empty replay; the GAE λ and per-table gates are kept.
- **Teacher:** empty label buffer and no beta. The first iteration (g1491→g1492) runs in shadow mode on all 32+4 tables and fixes beta = share_target / median raw share. Teaching is on from g1492.
- **Search head:** Stage C (w=1, trunk scale 1) is kept, with the learner's own head optimizer. Its value buffer starts empty, so fitting (and the Stage C trunk pass) pauses until eight fresh ages exist: it resumes in the g1499→g1500 iteration.
- **Companion heads:** the learner's own, moved to generation 1491 with no fitting step recorded.
- **Ratings:** the source compacted rating slot 1400 into a new archive at g1500 and deleted its files. The restart applies that same compaction, checked against the g1500 export: the archive must be the g1491 archive plus exactly that slot, with the same receipts and content hash.
- **Opponent pool:** probe caches the source has since deleted are regenerated from the members' weights (`regenerated_probes` in the summary); a member whose weights are gone would be dropped (`dropped_members`).

## Commands

Stop the source run's trainer, pull the code on the lab, then:

```bash
python scripts/prepare-rust-g1490-restart.py --share-target 0.75
```

```bash
python scripts/continue-rust-g1490-restart.py --expected-generation 1491 --iterations 100 --share-target 0.75
```

0.75 is the share target the g1491 run last used; pass any value above 0 and at most 1, the same in both commands. The second command defaults to `--inference cpu --workers 64 --teacher-fast 96 --teacher-fast-k 8 --teacher-full 640`. It runs the calibration iteration, then teaching for the rest (here to g1591). After a stop, rerun it with `--expected-generation` set to the last committed generation. No Rust code changed, so the CPU teacher needs no rebuild; the command rebuilds it only if the lab's build is stale.

The prepare step prints a summary (also saved as `branch-restart.json` in the new run) with the learner, compacted slots, carried LR multiplier and schedule.

## Verification (laptop)

- 19 new tests in `tests/Recovery/test_g1490_restart.py`:
  - **Recipe:** applied to the real g1491 recipe, it changes exactly the listed fields and validates. Other values are refused. Jobs carry `short_labels: false`, and short rows are refused at admission.
  - **Transitions:** the variant can't be adopted inside a run. The shadow→on switch is reviewed one way.
  - **Calibration:** one synthetic 32+4 shadow iteration qualifies a beta that authorizes teaching at Stage C. A 48+4 iteration doesn't count.
  - **Cotton receipts:** these allow allowance spent by unlabelled short searches between labels, and never a label going back.
  - **Logit check:** long rows of a revised table check logits once per table.
  - **Build:** a run tree built from the real g1491 state, with the real g1500 rating archive, prepares and then loads. That covers the checkpoint, optimizers with the 1.25 multiplier, exploration, PPG, companion heads, teacher phase at Stage C with the paused buffer, and backend receipts. Wrong weights, a missing later export, or an archive that doesn't extend the g1491 one are refused.
- Native: a CPU teacher built locally from the current overlays (26 crate and 2 cotton-core tests pass) ran a real long-only revised table. Every label is long and explored. Short searches still explore, leaving no label. The cotton receipt check passes, and logits are checked once per table. The g1491 short-label table test still passes.
- 24 existing suites touching the changed code (406 tests): 15 fail, and all 15 fail identically on the unmodified code (old fixtures, "Backend rollback manifest changed", a stale local CPU binary).
- Not run on the lab.
