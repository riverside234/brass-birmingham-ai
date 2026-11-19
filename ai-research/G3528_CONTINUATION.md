# g3528: cotton retiming, Rail-end heads removed, soft policy weight 12

Adopted together at the committed g3528 boundary, with one continuation:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3528 --iterations 22 --share-target 0.1 --value-revision drop-rail-end --soft-policy-weight 12 --update-cotton-timing
```

- It reaches the g3550 rating; the g3530 rating runs on the way. Later continuations take no adoption flags.
- The continuation rebuilds the CPU teacher and collector (`ensure_cpu`), whose overlay changed. `lab-native/` did not,
  so the teacher job fingerprint is unchanged.
- Execution identity `7f8431f5…` (predecessor `e4afd4e2…`).

## 1. Cotton exploration retimed (`--update-cotton-timing`)

The flag now adopts the newest reviewed rule (`training_revision_g3528`): ordinary `ordinary-cotton-exploration/6.0.0`,
teacher `teacher-cotton-exploration/10.0.0`. It works from the g3471 rule (this run) or straight from the g2654 recipe.

| seat's action in the era | Canal (1–15) | Rail (1–16) |
| --- | --- | --- |
| no cotton exploration | 15 | 1, 2, 16 |
| no cotton develop | 15 | 15, 16 |
| no cotton-city network | 14, 15 | 15, 16 |

- **Against g3471:** Canal action 1 explores every cotton target again, and Canal action 14 targets cotton develops
  as well as builds (still no network). The Rail era is unchanged.
- **Where the rule lives:** the same rule is in Python and in its Rust twins:
  - Python: `step5.policy.cotton_permissions` with `COTTON_TIMINGS['g3528']`, used by the PPO actors, the replay audit and the teacher labels;
  - Rust: `cpu_cotton::retiming` for the teacher and `cpu_autoplay` for ordinary games.
- **Unchanged:** seats, quotas, rates, the three build-and-develop targets per era, the one network per era and the
  Rail-era build level 3. Cotton seats are drawn from the new contracts' streams.
- **Older data:** g3471 labels and rows keep the rule they were collected with.

## 2. Rail-end heads removed (`--value-revision drop-rail-end`)

- **What is removed:** the era heads lose their Rail-end owner and VP outputs (the Canal rows' game-end map) at g3528.
  - These trained at weight 0 for the whole run: no loss or preservation term reached them, so they are still at their
    zero initialisation.
  - The checkpoint therefore loses only those zero layers, 1,746 parameters, and their slots in both head Adams. The
    canal and rail layers and their Adam state carry over exactly.
  - The report records the migration at g3528 (`local_scoring.rail_end_heads`, with `max_abs` of the removed weights,
    expected 0).
- **Afterwards:** no Rail-end outputs are computed, frozen before PPO, preserved or checked. `rail-end` is refused.
- **Recipe:** `local_scoring_auxiliary.rail_end_removed = 3528`, and the tag gains `-norailend`.
- **Not used in this run:** the Rail-end λ-return targets built earlier today. A run that switches `rail-end` on
  with search-rows would get them, with 18-channel `local-bootstrap.npy` sidecars and a four-iteration warm-up on the
  actual map.

## 3. Soft policy weight 6 → 12 (`--soft-policy-weight 12`)

- The flag takes 2, 4, 6, 8, 10, 12 or 16 (× beta) at any committed boundary after adoption.
- `soft_policy.weight_change` records the latest change and nests the earlier one:
  `{generation: 3528, before: 6, previous: {generation: 3471, before: 8}}`.
- The head, its Adam and its warm-up carry over. Its state, stamped under the weight-6 settings, is accepted once at
  g3528.
- **Risk:** this doubles the soft head's pull on the policy trunk (its own gradient call, clip 1.0).

## Cost (estimates; not measured on the lab)

| item | per iteration |
| --- | --- |
| cotton retiming | none; the same allowances, spent one action earlier in Canal round 1 and on develops at Canal 14 |
| Rail-end heads removed | saves the frozen Rail-end outputs (about 0.5 GB during PPO), a third of the head output layers and their preservation terms: about −1 s |
| soft weight 12 | none |

## Checks after the first iteration (g3529)

- `teacher_phase.collection.cotton_exploration.contract` = `teacher-cotton-exploration/10.0.0`.
- The PPO cotton tables run `ordinary-cotton-exploration/6.0.0`.
- `local_scoring.rail_end_heads.migration` = `{parameters: 1746, max_abs: 0, generation: 3528}`.
- The local-scoring weights are canal = rail = 0.15, rail_end 0.
- `soft_policy.weight` = 12.

## Tests

- `tests/Recovery/test_cotton_retiming_g3528.py`:
  - the rule table, with only Canal 1 and Canal 14 changed;
  - PPO targets under 6.0.0 (g3471 tables keep their rule);
  - teacher labels and settings under 10.0.0 against 9.0.0;
  - the recipe from g3471 and from g2654, the jobs, the forward-only transition and the flag.
- `tests/Recovery/test_rail_end_g3528.py`:
  - the two-era heads equal the full heads without Rail-end;
  - the checkpoint and both Adams migrated once (later loads plain, never undone, only at its generation);
  - the item (untrained heads only, idempotent, `rail-end` refused, forward-only transition);
  - a full value phase without Rail-end outputs;
  - the Rail-end λ targets and the soft weight's nested changes.
- Rust: `cpu_cotton` and `cpu_autoplay` retimed cases (all 43 `cpu_` tests pass).
- Native: one CPU-teacher table under 10.0.0 passes the Python labels and receipt. Its cotton labels fell at Canal
  actions 4 and 5, so the changed actions are covered by the unit tests above.
