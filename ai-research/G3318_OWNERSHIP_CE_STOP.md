# g3318 ownership stop: a held-out CE early stop for the era heads

Adopted with `--value-revision ownership-stop` at a committed generation (planned: g3318). It needs `heads`, which the run adopted at g3275 together with `targets`. The other g3140 items (`weight=0.15`, `weight=0.5`, `vp`, `rail-end`) are not part of this change and can still be adopted later.

## Why the heads always kept 16 steps

The 3.0.0 era heads (`recovery/local_scoring_v3.py`) were built to keep their steps when the value phase rejects an interval: the heads are not exported, so only the trunk was restored. They therefore had no early stop of their own. They trained for as long as the value phase ran, and the value phase's utility-loss check, which stops at the first rejected interval, decided how long that was. Every value phase attempts at least one 16-step interval, so the heads kept at least 16 steps even when the value phase kept none.

From g3275 to g3313, in the retention history:

| Value phase kept | Iterations | Head steps kept |
| --- | --- | --- |
| 0 steps (first interval rejected) | 17 of 39 | 16 |
| 16 | 11 | 32 |
| 32–54 | 11 | 48–54 |

Held-out owner CE (the `heldout` check run every fifth iteration: one row per game, era and quarter, 80 games) fell at every check: 1.230 (g3279), 0.932, 0.760, 0.646, 0.588, 0.549, 0.524 (g3309). The copy-the-board reference is 0.61–0.63, so the heads passed it at about g3299 and were still improving at g3309. They have been underfitted rather than overfitted so far, so the new stop is expected to keep most steps for now. It becomes the guard once more steps or the larger weights (`weight=0.15`, `0.5`) start to overfit them.

## What the item changes

The setting is `value_revision.ownership_stop = {contract: ownership-ce-early-stop/1.0.0, interval_steps: 16, criterion: validation-owner-ce, after_value_stop: heads-only}`. At every 16-step boundary of the value phase:

1. **The value check decides the trunk first**, unchanged. The trunk is kept, or restored on a held-out value-loss increase.
2. **The stop then scores the heads' owner cross-entropy** on the resulting state. It uses the value phase's own validation rows: every held-out row with a scoring map (about 31,000 rows from 80 games), weighted by the critic weights. Each row counts its mean CE over valid slots plus its mean CE over valid links, with the era weights relative to Canal. This is the owner-CE part of the training loss.
3. **If that CE is above the last kept boundary's**, the heads, their PPG Adam and the kept-work counters go back to that boundary. The interval keeps 0 head steps, and head fitting ends for the phase. From then on the auxiliary loss no longer reaches the trunk either.

**The utility check no longer limits the heads.** When the value check ends the trunk's phase, the heads keep fitting on the remaining planned batches with the trunk frozen. These heads-only steps use the same loss and learning-rate schedule, still checked every 16 steps. Head steps never exceed the phase's planned steps (54 at the current buffer size). With one epoch, this draws no extra random numbers.

**Unchanged:**
- the value check, its rows and its rule;
- the trunk's updates while the heads are still fitting;
- the KL guard;
- the PPO preservation steps;
- every exported output.

Without the item, behaviour is exactly as before.

**Cost: about 2–6 s per iteration.**
- Each CE check is one forward pass over the 31,000 validation rows, about 0.3 s, like the value check's. There are at most five per phase.
- The heads-only steps cost about 0.1 s each. They run only after the value check stops early, which leaves up to 38 steps.

## What to read in the next reports

| Field | Meaning |
| --- | --- |
| `value_phase.ownership_stop` (also `local_scoring_auxiliary.ownership_stop`) | `before`/`after` owner CE (overall, Canal, Rail), `kept_head_steps` / `attempted_head_steps`, `head_only_steps`, `stop_reason` (`heldout_ce_increased`, `phase_steps_exhausted`, `no_heldout_targets`), and per interval `trunk` (`kept`, `restored`, `frozen`), CE before and attempted, kept steps |
| `local_scoring_auxiliary.companion_steps.ppg`, `warmup.retained_fitting_steps` | Now the kept head steps, and can be 0 |
| `local_scoring_auxiliary.head_steps_kept_on_rejection` | `false` once the item is on |
| log line | `aux_ce=before->after aux_kept=kept/attempted heads_only=N aux_stop=reason` after `head_steps=` |

If `aux_stop=heldout_ce_increased` at the first boundary becomes common while the every-fifth-iteration `heldout` owner CE stops falling, the heads have reached what this weight and data support. That is the point to consider `weight=0.15`.

## Checks run (laptop, fixture models)

`tests/Recovery/test_ownership_stop_g3318.py`, 6 tests:
- **Recipe:** the item needs `heads`, is idempotent, only moves forward, and needs 16-step value-phase intervals.
- **Joint and heads-only loss:** they match.
- **An increase at the first boundary:** keeps 0 steps, restores the heads exactly and ends fitting (no loss in either path).
- **Decreases followed by an increase:** the decreases are kept until the increase.
- **Through `ppg_g210.fit`:** the counters and the warm-up agree with the stop's record. When the value check rejects at once, the heads fit alone for the rest of the planned steps (both epochs of the fixture) and the trunk stays bit-identical.

The existing g3140, local-scoring, PPG g210 and g3270 suites still pass.

## Code

| File | Change |
| --- | --- |
| `recovery/ownership_stop.py` (new) | The stop: validation-row cache, CE criterion, boundary decision, heads-only continuation, report |
| `recovery/training_revision_g3140.py` | Item `ownership-stop`, optional `ownership_stop` block (validated, forward-only, tag `cestop16`) |
| `recovery/local_scoring_v3.py` | Loss shared by the joint and heads-only paths (`head_loss`), `stop_snapshot`/`stop_restore`, `stopped`, report fields |
| `recovery/value_phase.py` | Boundary hook after the trunk decision; heads-only continuation after the loop; `ownership_stop` in the report |
| `recovery/ppg_g210.py`, `recovery/league.py` | Pass the setting; log line |
| `scripts/continue-rust-g1491.py`, `scripts/continue-rust-g1190-restart-2.py` | Help text |

Execution identity re-recorded; `7215217a…` (g3270–g3318) is kept as a predecessor so the g3318 checkpoint resumes. No Rust changed: the CPU workers need no rebuild.

## Continue command

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3318 --iterations 12 --share-target 0.1 --value-revision ownership-stop
```

Twelve iterations end on the g3330 rating. The item is stored with the run, so later continuations do not pass it again.
