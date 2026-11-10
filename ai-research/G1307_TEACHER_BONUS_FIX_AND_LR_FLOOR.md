# g1307: teacher bonus-state crash, collection waits, PPO LR floor

Date: 2026-11-10. Evidence: `results/joint7-v11-2-g1290-trained-restart-g1307-20261110T020738908085Z` (g1303–g1306) and the training log in `results/rust-lab-qualification/20261110T020701731467Z-…-on-training-f25a2c01-cc1d8b97`.

## 1. The crash (fixed)

`Native teacher closed output: teacher: invalid request: bonus state mismatch`, during the collection for g1308.

A multi-beer sale to a merchant with a develop bonus can reach a legal state where the bonus choice comes while beer is still owed. The rules engine plays through it. But the teacher rebuilds every decision root from the player's view through the frozen importer, which has a stricter rule for that phase and rejects it. Ordinary lanes never rebuild roots, which is why only a teacher table failed. Both the importer (sealed core) and the qualified `lab-native` teacher stay untouched.

Fix, in the CPU-path overlay: when the importer would reject the current state (`unimportable_bonus`), the teacher builds the root directly from the true state (`direct_root`: same view, recall, sorted legal choices, context and ledger). The policy network then picks the bonus, as for any decision outside a search; there is no label because labels exist only for action decisions. The turn's search session simply isn't asked at that one decision, and it searches the next payment normally.

Rust tests: the direct root equals the search root wherever search can build one; the mid-sale bonus is detected and search fails on it exactly as on the lab; a real turn session continues after the skipped decision. The GPU-path teacher has the same importer limit but isn't used by this run.

## 2. Collection waits (fixed)

Planning is now 0.7 s, but collection stayed at 84–92 s:

| Per teacher table (s) | g1291 alone | g1293 partial overlap | g1305–g1306 full overlap |
| --- | ---: | ---: | ---: |
| Elapsed | 48.4 | 58.1 | 71.7–78.7 |
| Search | 37.5 | 39.9 | 43.7–46.5 |
| Waiting for the per-turn ack | 9.6 | 16.6 | 23.2–33.2 |

Ordinary games cost the same CPU as before (2.5–2.8 s each) and teacher inference per request rose only 15%. The time was waiting: after every turn the Rust teacher blocks until the trainer's Python process verifies the turn and acknowledges it, and it keeps its CPU ticket while blocked. Under full overlap the trainer's Python is shared with the ordinary pool, the waits grew, and 28 of the 64 tickets sat idle while ordinary games queued for them.

The ack carries no decision (verification returns "ok" or raises), and fixed-count searches don't read the clock. So count-budget tables are now acknowledged first and verified while the worker searches its next turn. Events are still handled in order, so a failed check stops the table before any later turn or result is accepted. Timed tables keep the old order. Expected: tables near their search time (about 50 s) and collection well under 90 s. That's an estimate; the next export will show it.

## 3. Other observations, not changed

- PPG now keeps work: g1306 kept 59 steps with the KL-cap interval retained (48 steps before the hit, so beta held). The value phase costs 25–40 s instead of 7–8.
- The search head is training in Stage B at 64 steps per iteration (about 2 s). Reading its 8-age buffer takes 1–9 s.
- Entropy is at step 13 of 100 (0.00152 at g1307).
- Retention cleanup every fifth iteration takes about 60 s (14 GB rehashed for plan, move and delete). That is by design and was left alone.

## 4. PPO LR floor 4.0e-6 (implemented)

The controller's floor is a minimum multiplier on the 6.0e-6 base rate: it was 0.5 (3.0e-6) and is now 4.0e-6 / 6.0e-6 (exact in floating point). The current multiplier (1.0, so 6.0e-6), its windows and its cooldown are kept, so the next PPO step uses the same rate. Cuts now stop at 4.0e-6: 6.0 → 4.8 → 4.0. It is adopted once at a committed boundary from g1307 by `continue-rust-teacher.py`, passes the same reviewed-transition checks, and can't be undone. A persisted multiplier below the new floor would be refused at resume; g1307's is 1.0.

## 5. Continue

```bash
python scripts/prepare-rust-cpu-inference.py --expected-generation 1307
```

```bash
python scripts/qualify-rust-cpu-inference.py --expected-generation 1307
```

```bash
python scripts/continue-rust-teacher.py --expected-generation 1307 --iterations 10 --share-target 0.1 --inference cpu
```

The CPU worker sources changed, so the prepare step is required. The crashed g1308 collection is archived losslessly by the continue script before the new recipe starts.
