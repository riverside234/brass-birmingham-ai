# g4770: Rail-end heads back (PPG only), actual PPO-game targets, search pool sampled from 8 iterations

The user's requests from 2026-11-22, built as one recipe revision for the continuation from the committed g4770:

1. Add the Rail-end heads back (removed at g3528), starting at weight 0.05, with an option to raise it to 0.25.
2. For speed:
   - Train the Rail-end heads in the same value batches and updates as the Canal/Rail ownership and VP heads.
   - Train Rail-end only during PPG, on the actual Rail-era settlement map for PPO games. Skip its PPO snapshots,
     transfers, preservation loss and λ-return caching. Search games keep λ-returns (0.95).
   - Every Canal/Rail ownership and VP head also trains PPO games on the actual map only (λ = 1). Search games keep
     λ = 0.95.
3. Draw the era heads' search games as a random sample, 4 iterations in size, from the last 8 iterations' search
   games (was: the most recent 4 iterations).

The flag is `--era-head-revision` (module `recovery/training_revision_g4770.py`, top-level block `era_head_revision`).
It is adopted once at a committed boundary and then kept. `--rail-end-weight 0.25` raises the weight at any later
boundary.

Execution identity: `configs/learning/b07-replay-location-review.json` (`g4770_rail_end_restored`,
`g4770_ordinary_actual_targets`, `g4770_search_pool_sample`; `3d95823a…`, predecessor `7e52ba1d…`).

## Continue commands

From the newest export (g4770) to the g4790 rating:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 4770 --iterations 20 --share-target 0.1 --era-head-revision
```

To raise the Rail-end weight to 0.25 at a later committed boundary (for example g4790):

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 4790 --iterations 20 --share-target 0.1 --rail-end-weight 0.25
```

The collection, the CPU workers and every table output are unchanged, so nothing is rebuilt.

## 1. The Rail-end heads, restored

**What they are.** The Canal rows' prediction of the game-end (Rail-era) scoring map: owner and VP outputs for each
of the 49 slots and 69 links. They sit on the same shared, slice and hidden layers as the Canal and Rail outputs.

**Restoration** (`local_scoring_v3`; recipe `local_scoring_auxiliary.rail_end_restored = 4770`, weight 0.05):
- The g4770 checkpoint has the two-era heads. At g4770, once, it gains fresh Rail-end layers, zero-initialised as
  the heads' own initialisation (1,746 parameters), with no Adam state (`add_rail_end`).
- Every other parameter and both head Adams carry over exactly. Later loads are plain.
- The report records the restoration (`local_scoring.rail_end_heads.restoration`), beside the g3528 removal.
- A restoration is never undone, and the weight only moves up (0.05 → 0.25, `--rail-end-weight 0.25`).

**Training, PPG only, grouped with Canal/Rail.** One forward of the era heads gives all three horizons. The
Rail-end loss joins the Canal/Rail loss in the same batch, backward pass and step:
- joint PPG steps, heads-only steps and the search rows mixed into each batch;
- weight 0.05 against Canal/Rail 0.5.

There is no separate Rail-end phase or pass.

**Nothing in PPO.** `preservation_weights` and `value_preservation.outputs` leave the restored outputs out:
- no frozen snapshot rows (the frozen critic outputs stay at 2,140 floats per row, not 3,202);
- no GPU-to-CPU transfer;
- no preservation term;
- no PPO head step on them.

**Targets:**
- PPO-game Canal rows use the actual game-end map.
- Search-table Canal rows use λ 0.95 returns of the head's own frozen predictions through the Rail-era chain, from
  g4774 (`value_revision.search_rows.auxiliary.rail_end`). Until then they use the actual map: the reviewed g3528
  four-iteration warm-up, so the returns bootstrap from a trained head rather than a zero one.

**Ownership stop.** The stop measures and reports the Rail-end held-out CE (`rail_end_owner_ce`, and `railend_ce` in
the log line), but only Canal and Rail decide it.

Why: a fresh head's CE falls fast from ln 5. At its weight it would move the criterion by about 0.01 per interval,
against the 0.0003–0.001 the Canal/Rail heads move. The stop would then accept steps that cost the trained heads.

## 2. PPO games on actual targets (λ = 1) for every era head

With the revision:
- PPO keeps no frozen era-head outputs for returns (`frozen_local` is not kept).
- PPG computes no ordinary corrections and stages no `generation-N.local-bootstrap.npy`.
- An older age's sidecar is not read.
- The value-phase buffer report says `era_head_targets: actual-map`.

Search rows keep their λ 0.95 Canal/Rail returns.

Exactness: a full `ppg_g210.fit` with the revision gives the same era-head loss as the g3471 rule at λ = 1, to four
places (tested).

## 3. Search pool: a 4-iteration sample of the last 8

`search_auxiliary.tables(..., pool=...)`:
- **Candidates:** the search-value training tables admitted in the last 8 iterations. These are the train tables
  collected at G−7..G, plus the held-out tables that left the latest 32 in those iterations.
- **Sample size:** as many tables as the last 4 iterations admitted, the size of the 2.0.0 pool: 200 + 16 = 216 at
  g4769, from about 432 candidates.
- **Draw:** uniform, without replacement, by a generator of its own seeded with the generation. A fresh sample each
  iteration, and no other random stream moves.
- **Unchanged:**
  - the held-out set (the latest 32);
  - the mixing into every PPG batch;
  - the losses and the pooled stop criterion.
- **Report:** `value_phase.search_auxiliary.selection.sampled`, with the window, the candidates, the size and
  `by_generation`.

The shards are already in memory (the search-value buffer keeps 16 iterations), so reading 8 iterations costs nothing.

## Cost (estimates; g4768–g4769 timings, not measured with the change)

| item | per iteration |
| --- | --- |
| no ordinary corrections | −2.7 s (`buffer.local_bootstrap.seconds`) |
| no local-bootstrap sidecar | −265 MB written and hashed per age (about −1 s), and no sidecar reads in PPG |
| no `frozen_local` | about 1 GB less RAM held from the end of PPO to PPG staging |
| Rail-end in PPG | one more output pair per entity kind on Canal rows, and its targets (cached per map): about +0.5–1 s |
| Rail-end in the stop and the search λ (from g4774) | about +0.5 s |
| Rail-end in PPO | none (skipped) |
| search pool sample | none (same size; rows already cached) |

Net: about −2 to −3 s per iteration, plus the RAM and disk above.

## Checks after the first iteration (g4771)

- `local_scoring.rail_end_heads` shows `restored_at: 4770`, `weight: 0.05`, and
  `restoration: {parameters: 1746, max_abs: 0, generation: 4770}`.
- `local_scoring.weights` is `{canal: 0.5, rail: 0.5, rail_end: 0.05}`.
- `actor.value_preservation.floats_per_row` stays 2,140.
- `value_phase.buffer.era_head_targets` is `actual-map`, and there is no `local_bootstrap` entry.
- The log line has `railend_ce=…->…`. `ownership_stop.before.rail_end_owner_ce` is about 1.6 at first and falls.
- `local_scoring.by_era['CanalEra->RailEnd']` reports the Rail-end training sums.
- `value_phase.search_auxiliary.selection.sampled` is about `{candidates: 432, size: 216}`, with `by_generation` across
  G−15..G.
- From g4775: `search_auxiliary.lambda_targets.rail_end` has rows, and its `mean_owner_l1` is above 0.

## Tests

- `tests/Recovery/test_era_head_revision_g4770.py` covers:
  - the recipe: restored at 0.05, search returns from G + 4, idempotent, forward-only transitions, and the 0.25
    option (only up; a later `weight=` item leaves it);
  - the checkpoint migration: two-era → full heads with zero Rail-end layers and fresh Adam slots, once at its
    generation, never undone;
  - frozen PPO outputs without Rail-end;
  - a full `ppg_g210.fit`:
    - Rail-end trains in PPG with nothing frozen, cached or staged for PPO games, and search λ targets;
    - actual targets equal λ = 1;
    - the decisive stop CE is the Canal/Rail mean;
  - the pool sample: size, window, reproducibility, a fresh sample per generation, the unchanged held-out set.
- The g3528 Rail-end, g3423 search-aux, g3471 search-rows, g3140, g3318 stop, PPG and soft-policy suites still pass.
  Their only failures are the known ones in `docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md`, which now also lists
  `test_domino.Training.test_new_config_only_diversity_and_off_no_mutation` (it fails the same way on HEAD).
- Dry run on the export's g4770 state:
  - changed keys: `era_head_revision`, `local_scoring_auxiliary`, `value_revision`;
  - forward transitions accepted, backward refused, for adoption and for the 0.25 step;
  - labels and beta kept;
  - tag `…-eraheads-w05-railend005-…-searchrows-railendlam`.
