# g6547 soft weight 4 and soft held-out before PPO

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, committed at g6547 in the newest export
(`joint7-v11-2-g4970-rail025-soft8-beta0.02-g6547-20261127T203905330022Z`).

## Request (2026-11-27)

1. Set the soft policy weight to 4 (from 8).
2. Fix: record the held-out soft KL before and after PPO, as before g6230.

## 1. Soft weight 8 -> 4

This uses g3423's reviewed weight change (`--soft-policy-weight`, as at g3471, g3528 and g4970). It is not a new
revision block:
- `soft_policy.weight` becomes 4;
- `weight_change` becomes `{generation 6547, before 8}`, with the earlier changes nested under it (8 -> 6 at g3471,
  6 -> 12 at g3528, 12 -> 8 at g4970);
- the head, its Adam state and its warm-up carry over. The committed head was stamped under the weight-8 settings and
  is accepted at g6547 only.

`scripts/continue-rust-g4970.py` used to pass a fixed `--soft-policy-weight 8`. It now takes `--soft-policy-weight`
and otherwise passes the weight already in the run. Later continuations without the flag therefore keep 4.

What changes is the soft loss's scale in the trunk gradient: `soft.weight x beta`, from about 0.193 to about 0.097 at
the current beta. The head's own Adam step uses the unweighted, clipped soft gradient, so the head itself learns as
before. In the last 4 committed iterations (g6543-g6546), the soft trunk share (8 beta x soft trunk norm / PPO actor
norm) had a median of 0.08-0.10. At 4 it is expected near half that.

The teacher term's beta, its share target (0.065), the labels and the g6230 settings do not change: visited-only
targets from the current logits, weight-0 rows, and PPG clip 2.5. The soft head's rate has been the PPO step rate
(6e-6) since g6330, and it stays.

Dry run on the g6547 export:
- only `soft_policy` changes;
- the signature transition is forward True, back False, same True;
- `previous_settings` of the new block equals the committed block, so the head's stamp is accepted;
- the tag goes from `soft-t4-w8-s1` to `soft-t4-w4-s1`.

## 2. Soft held-out check before PPO

From g6230 to g6546, the visited-only soft check ran once, after PPO. Now `TeacherPhase.prepare` also runs
`SoftPolicy.heldout_masked` on the validation-table long labels before PPO. The policy's pre-update logits come from
that same forward, as they did with the old check. There is still no check after PPG.

`teacher_phase.heldout.soft` now holds:
- `before` and `after_ppo`, each with `rows`, `skipped_rows`, `kl`, `by_phase` and `dropped_unvisited_mass`;
- `ppo_delta` = after_ppo - before;
- `after_ppo_seconds`.

The log prints `soft_heldout=<before>-><after>(ppo) soft_rate=...`. This is reporting only, and nothing that trains
changes. The g6230 settings stamp (`soft_heldout='after-ppo-only, ...'`) stays as adopted.

## Cost

- The weight change has no compute cost. Its effect: the soft head's pull on the shared scorer and trunk is halved.
- The extra held-out check costs about 0.03 s per iteration, the median of the after-PPO check in g6543-g6546.

## Tests

- `tests/Recovery/test_soft_weight_g6547.py` (2 tests):
  - 8 -> 4 at g6547 on the run's weight history: kept by later continuations, forward-only, the head's stamp;
  - the wrapper's choices, and its default being the run's weight.
- `test_soft_head_revision_g6230.Phase`: two masked checks (before and after PPO), both reports, and `ppo_delta`.

8 focused modules ran (86 tests): these two, g3423 soft, g4475 teacher training, g3528, g3471, g5685 and g3270. All
pass.

## Execution identity

`6e2152b136a7f94bb4aa75c68e66970f6513c6031df0321bd75360fe4f058006`. Its predecessor is `310796d7...`, which the
uncommitted g6547 value-memory and exit change recorded on top of `4aaa5fba...`. The entry is
`g6547_soft_weight_and_heldout` in `configs/learning/b07-replay-location-review.json`. This identity covers the
working tree as it is now, with both changes.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 6547 --iterations 10 --share-target 0.065 --soft-policy-weight 4
```

`--soft-head-revision` is already adopted and is not needed again.
