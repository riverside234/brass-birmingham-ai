# g5685: IL replay dataset unlinked

On 2026-11-25 the user deleted `artifacts/il-replay/` on the lab. The next continuation of
`joint7-v11-2-g4970-rail025-soft8-beta0.02` (the `--clip-revision` command at g5685) then failed when the training
process started:

```
league._train -> demonstrations.completed_imitation(artifacts/step5/features4-v11-2-batch7-il)
  -> critic_upgrade.completed -> completed_imitation(parent BC run)
  -> sha(artifacts/il-replay/features4-compact-v8-batch7/manifest.json)   FileNotFoundError
```

The continuation script's own preparation had already passed. Only the trainer's startup read the dataset.

## What the run used the dataset for

| Use | When | Needed now? |
| --- | --- | --- |
| IL weights (from the BC run) | seeded generation 0 | no; the run resumes from its own checkpoints |
| validation positions | built the fixed IL-KL probe | no; the probe is kept in the run (`il-kl/manifest.json` + `probe.npz`, bound by `state.il_kl_limit.probe`) |
| rehearsal rows | rehearsal | no; `rehearsal_weight` is 0.0, so `rehearsal.schedule` is disabled for good |

At every startup the trainer still re-verified all three: the full IL lineage (the upgraded BC run, its parent BC
run, and the dataset manifest's hash), and the dataset manifest behind the committed probe.

## Changes (code-only, no training change)

- `league.imitation_handoff`: when `state.json` exists, the trainer binds the run's committed
  `state.signature.imitation_source`. It checks that origin against `--imitation-run`, and the existing check against
  the config's `imitation_checkpoint` stays. It no longer calls `completed_imitation`, so a resume needs neither the BC
  runs nor the dataset. Only a new run still walks the lineage, and a new run from this IL is impossible now anyway.
- `il_kl_probe.committed_source`: a committed probe's recorded source descriptor (manifest path and hash, IL model
  id, IL checkpoint hash, learning compatibility) must equal the run's signed origin. Nothing is read from
  `artifacts/il-replay`. It is used in three places:
  - `il_kl_probe.load` at resume;
  - `lab_continuation`'s dependency listing (rollback and diagnostic forks), which also stops listing the dataset
    manifest as a dependency;
  - `lab_continuation.restore_probe`.

  An uncommitted probe (a new run restarting before its first commit) still verifies its build inputs as before.
- The cleanup's reference scan already skipped missing manifests, so it needed no change.

Not changed: IL tools that build or evaluate IL itself and need the dataset by nature (`feature-engineering.py`,
`il_replay.py`, `il_head_to_head.py`, `joint7_preflight.py`, the machine-transfer script `step5-transfer.py`). None of
them runs in training.

## What a resumed run still reads outside its directory

These are live dependencies. Keep them on the lab.

- **Rating slots:** each slot plays the IL reference (`generation-0`), the trained-start baseline, the v4 legacy AI and
  historical checkpoints. 409 of the 445 rating agents keep their weights in
  `artifacts/step5/joint7-v11-2-g1190-restart-3/population/`, and the baseline is in that run's `seed/`. The g5670
  slot played 3 of its 12 agents from there.
- **Legacy AI runtime:** the v4 rating reference's pinned runtime under `artifacts/legacy-ai/search/`, plus
  `configs/legacy-ai`.
- **Native backend pins:** checked at every startup (`lab_backend.validate_state`), under
  `artifacts/rust-backend/20261107T122742698740Z-b07-continue-537723a8/`
  (`remaining/qualification.json` and `adopt/rollback/diagnostic-fork.json`).
- **The opponent registry:** `training/opponents/`.

## Tests

`tests/Recovery/test_il_replay_unlinked_g5685.py` (4) covers:
- a committed probe loading with the dataset deleted, refused for a changed IL identity; an uncommitted probe still
  needs the dataset;
- the diagnostic-fork probe restore without the dataset;
- `imitation_handoff`: a new run walks the lineage, a resumed run binds its committed origin, and a mismatch is
  refused;
- rehearsal staying off.

`test_b03.Continuation.test_probe_inventory_excludes_preparation_scratch_and_requires_sources` now expects the IL
dataset manifest to be absent from the continuation dependencies, with the IL checkpoint still present, and a preflight
with the manifest deleted to list the same dependencies. The error texts for a changed probe source are unchanged.

27 related modules ran (271 tests): the IL, IL-KL, b03/b07 continuation, branch-restart, fresh-seed, rehearsal-window,
trained-startup and resume suites. 55 tests fail. Each of them also fails on HEAD `fa0857c8` with the changes stashed,
except a `test_rehearsal_window` reader-reuse test that failed once and passes 3 of 3 when run alone (timing), and a
`test_relocation` module that does not import. The 46 not yet listed were added to
`docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md`.

## Execution identity

`d73df28c746fd7bbd38a623563a3d12608dcbcd9916b5947769d12458178d545`, with predecessor `991862ca...`. The entry is
`g5685_il_replay_unlinked` in `configs/learning/b07-replay-location-review.json`.

## Command

The `--clip-revision` command is unchanged, since g5685 was not committed again:

```
python scripts/continue-rust-g4970.py --expected-generation 5685 --iterations 10 --share-target 0.065 --clip-revision
```
