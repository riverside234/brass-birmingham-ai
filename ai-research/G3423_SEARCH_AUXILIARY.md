# g3423 search auxiliary: search-teacher tables train the ownership and local-VP heads

- **Date:** 2026-11-18. **Run:** `joint7-v11-2-g1190-restart-3`, newest committed generation g3423.
- **Adopted with** `--value-revision search-aux` at a committed boundary (planned: g3423, together with `--soft-policy`).
- **Needs** three things the run already has: the era heads (`heads`, g3275), the ownership stop (`ownership-stop`, g3318) and the g3270 teacher revision.
- **Revision 2.0.0** (same day, before any lab run). It replaces 1.0.0 because of 1.0.0's risk: one batch at the first PPG step, and a separate check on the 4 current search tables that could end head training on 256 noisy roots. In 2.0.0:
  - search rows from four iterations of tables are mixed into every PPG batch;
  - the held-out pool is the whole search-value validation set;
  - one pooled criterion replaces the separate rule.

## What it does

### 1. Each table sends its era-end scoring maps once, with its finish message

- **The maps were already computed.** The CPU teacher passes every transition through `ordinary_after` with a native scoring recorder (`lab-native/src/teacher.rs`), the same recorder ordinary games use. It records both era-end maps (owner and VP of every slot and link, with the official scoring audit) but never sent them. The `finished` message now carries `"scoring": recorder.finish(&state.export())`, exactly what an ordinary game's final message sends.
- **Where the change lives.** The change is an exact-text patch in `rust_backend/lab_cpu_overlay.py`; `lab-native/` itself is unchanged.
  - The patch compiles to the same binary a direct edit would, so there is no performance difference.
  - Teacher jobs are fingerprinted from `lab-native/*.rs`, so a direct edit would have relabelled every teacher job and the qualified producer. With the patch, only the CPU worker is rebuilt, and the continuation does that by itself (`ensure_cpu`).
- **Admission (with the item on).** The receipt goes through the same era audit as an ordinary game's (`local_scoring_targets.audit_eras`) and is bound to the table's own result (final VP). It is then sealed and packed onto the table's first root in its search-value shard.
  - The map is stored once per table, about 4 KB.
  - A held-out table still has its map when it leaves the held-out set eight iterations later.

### 2. Each played root is matched to its era and seat mapping

Each root of an all-seat table becomes one row (64 per table), built from the root's search-value row:
- **Input:** the acting seat's encoded state.
- **Seat mapping:** the root's `mapping`, stable seat → relative. This equals the encoder's `stable_to_relative`; both are `(s − seat) mod 4`, and admission checks it.
- **Era:** `turn_key[0]`.
- **Target:** that era's end map, plus the Rail-end map for Canal rows once `rail-end` is on.
- **Unchanged from ordinary rows:** labels, masks, the VP scale (/5) and the loss weights (0.15 per era now). The rows go through the companion's own `_auxiliary`.
- **Seat descriptors:** search rows have none, so they read as unknown seats, as for the search-value head.

### 3. Four iterations of search-value training tables, mixed into every PPG batch

- **The pool is the search-value phase's own training data,** read from its buffer without changing it. It holds the search-value training tables admitted in the last four iterations (`train_iterations` 4; the buffer keeps 16):
  - the train tables collected at G−3 … G (36 each);
  - the held-out tables that left the latest 32 in those four iterations (4 each, collected at G−11 … G−8). `rotating_holdout` moves them into the search-value training set as they leave.
  - In full that is 160 tables, 10,240 roots.
- **Mixing:**
  - Every PPG epoch draws a fresh permutation of the pool and spreads it evenly over that epoch's ordinary batches. Each batch gets `pool / batches` roots: about 190 at full size, with g3422's 220,062 ordinary rows in 54 batches.
  - Each search root is thus presented once per epoch, the same rate as an ordinary row.
  - The share goes to every step the heads train in: the joint steps, and the heads-only steps after the value check stops the trunk. In heads-only steps the search rows, like the ordinary rows, give the trunk no gradient.
- **Losses:** search rows add only the era heads' ownership and local-VP losses (and Rail-end when on), with no value, utility or policy term. Each search row weighs as one ordinary row of its step: the step's loss is divided by its ordinary batch size, so the ordinary rows' weights do not change.
- **The ordinary rows are untouched.**
  - The same rows, in the same order and in the same batches: the search permutation has its own generator, seeded by generation and epoch, so the ordinary shuffles and clone draws are unchanged.
  - The ordinary attempted, retained and fitting counters, the warm-up counters and `by_era` count ordinary rows only. The search rows' own era sums are in `search_auxiliary.by_era`.
- **Existing machinery:** every step keeps the heads' trunk-gradient scale, the PPG KL checks, the value check with its interval rollback, and the ownership stop.
- **Implementation note.** The search rows of a step run as a second small forward and backward pass before that step's optimizer update. This equals one mixed batch, because the model has no layers that depend on the batch.

### 4. The search-value validation set, pooled into the ownership stop's criterion

- **The held-out pool** is the search-value phase's validation set (the latest 32 held-out tables), keeping the tables that have maps.
  - The criterion reads 4 tables (256 roots) in the first iteration after adoption, 8 in the second, and 32 (2,048 roots) from the eighth on.
  - These tables are never in the training pool: they train only after they leave the latest 32.
- **One criterion.**
  - At every 16-step boundary the stop scores the owner CE of the ordinary validation rows (critic weights) and of the search roots (weight 1, as an ordinary row).
  - It pools the two set by set, row for row. The pooled CE is `(ce_ord · W_ord + ce_search · W_search) / (W_ord + W_search)`.
  - An increase restores the heads and their Adam to the last kept boundary, and ends head fitting (reason `heldout_ce_increased`). This is the existing rollback, with no separate search rule.
- **Weight of the search tables.** The pooled criterion follows row counts. With g3423's 31,489 ordinary validation rows, the search roots are about 0.8% of the criterion at 4 tables and about 6% at 32.
- **What the report keeps:** `before` and `after` stay the ordinary rows' CE, so they compare with earlier iterations. The decisive values are in `criterion` and in `intervals[*].criterion_*`.

### How the pools fill after adoption at g3423

| State (iteration) | Train tables (roots) | Search held-out tables |
| --- | --- | --- |
| g3423 | 36 (2,304): only g3423's tables have maps | 4 |
| g3424 | 72 (4,608) | 8 |
| g3425 | 108 (6,912) | 12 |
| g3426 – g3430 | 144 (9,216) | 16 → 32 (+4 per iteration) |
| g3431 – g3433 | 148 → 156 (+4 per iteration): the leaving held-out tables with maps (g3423's first) | 32 |
| g3434 on | 160 (10,240) | 32 |

Tables collected before the rebuild have no map. They are counted in `selection.without_maps` and left out.

### Without the item

- Admission keeps no map, PPG gets no search rows, and the ownership stop is exactly as before.
- The rebuilt binary still sends the maps, and the Python side ignores them.

## Cost

| Item | Cost (*estimate* except where read) |
| --- | --- |
| CPU worker | One rebuild, done by the continuation with its focused regressions. The laptop compile of the patched teacher took 50 s (incremental). |
| Native work | None new: the recorder already ran. The finish message grows by about 15 KB per table. |
| Admission | Under 0.1 s for 40 tables (one audit and seal each). About 4 KB per table in the shards. |
| Reading the pools | About 225 shard reads per iteration. Each is a hash plus cached rows, the work the search-value phase already does on the same buffer. |
| PPG | About +0.4–1.0 s per iteration:<br>• one extra small forward and backward pass (about 45–190 roots) per head step, up to 54 steps;<br>• 2,048 held-out roots scored at each of up to five boundaries;<br>• about 150 MB of states staged on the GPU, transient.<br>For scale, at g3422 the whole value phase took 20 s, 9.2 s of it in training. |
| Search, collection, held-out data | None: the tables already exist in the search-value buffer |

**Risk and what it means.** The search rows are about 1% of the auxiliary presentations at first and 4.7% at full size (10,240 against 220,062 ordinary rows). Each table trains in four consecutive iterations. The pooled criterion weighs them by the same row count (0.8% → 6%).
- Neither can end head training on its own: ordinary rows still decide most of the criterion. That is the intended low-risk default.
- The flip side is a small effect. If the search tables' owner CE (`heldout_owner_ce`) does not fall over 20–30 iterations, their share is too small to matter. The next lever is their row weight (say ×4), and it raises their influence on both training and the criterion.

## What to read

| Field | Meaning |
| --- | --- |
| `value_phase.search_auxiliary.selection` | Train tables (`train_generations`, should be G−3 … G), rotated (`rotated_generations`, should be G−11 … G−8 once they have maps), held-out (`heldout_generations`); `without_maps` |
| `value_phase.search_auxiliary.train` / `.heldout` | Rows, games and rows by era of the pool and of the validation tables |
| `value_phase.search_auxiliary.presented` | Rows and steps that carried search rows, joint and heads-only, rows per step, batches per epoch |
| `value_phase.search_auxiliary.kept` | Rows in intervals the ownership stop kept (`heads_rows`), and in joint steps the value check kept (`trunk_rows`) |
| `value_phase.search_auxiliary.weighted_loss`, `.by_era` | The search rows' auxiliary loss, and their owner CE and VP error by era, kind and open/settled |
| `value_phase.search_auxiliary.heldout_owner_ce` | The search validation tables' CE, before and after the phase |
| `value_phase.search_auxiliary.criterion`, `ownership_stop.criterion` | The pooled CE before and after, and the search share of its weight |
| `value_phase.ownership_stop.intervals[*]` | `owner_ce_*` (ordinary), `search_*`, `criterion_*` (decisive) at every boundary |
| `search_value.scoring_maps` | Tables admitted with a map |
| log line | `search_aux=Nrows/Ngames search_held=Mgames search_ce=before->after aux_crit=before->after search_kept=kept/presented`, after `aux_stop=` |

## Checks run (laptop, fixture models)

`tests/Recovery/test_search_auxiliary_g3423.py`, 14 tests:
- **Maps.**
  - The receipt is audited, sealed and bound to its table. A changed total, a missing era, the wrong producer and a final VP that disagrees with the result are all refused.
  - The ordinary receipt check behaves as before.
  - Admission stores the map once, on the first root, and only with the item on.
- **Selection.**
  - The pool is the train tables of the last `iterations` generations plus exactly the held-out tables that left the latest 32 in those iterations, checked for 1, 2 and the recipe's 4 (G−4's train tables are left out).
  - The held-out set is the latest 32 and never overlaps the pool. After adoption it grows 4, 8, … 32.
  - Tables without maps are counted and skipped.
  - The held-out order is `rotating_holdout`'s own.
- **Mixing.**
  - Each epoch presents every search root exactly once, in even shares (sizes differ by at most one), with a fresh permutation each epoch.
  - The torch generator that orders the ordinary batches is untouched.
- **Rows.** Each root reads its era's end map through its own seat mapping (the classes are checked one by one), and a wrong mapping is refused.
- **Companion.**
  - Search rows add only era-head losses: the outcome, utility and policy parameters get no gradient.
  - The trunk follows the heads' scale, and gets nothing in a heads-only step.
  - The ordinary counters and era sums are unchanged.
  - A stopped head returns nothing.
- **Stop.**
  - `before`/`after` equal the plain stop's, and without search the criterion is the ordinary CE.
  - The pooled CE is the row-weighted mean of the two sets. An ordinary rise that the search fall outweighs keeps the interval; the reverse stops the heads.
- **Through `ppg_g210.fit`.**
  - Every step the heads attempt carries its share.
  - The ordinary presentations, attempted steps and ordinary head accounting equal those of the same phase without search rows.
  - With the value check forced to reject its first interval, the heads-only steps carry search rows and no trunk rows are kept.
  - Without maps the rows are skipped.
- **Recipe.** The item needs the teacher revision and the ownership stop, is idempotent and forward-only, and `is_on` still recognizes the recipe.
- **Native.** The overlay adds the field once, and `lab-native/src/teacher.rs` is untouched. The patched CPU teacher compiles (`cargo build --features teacher --bin brass-lab-teacher`).

Also run with this change: g3318 ownership stop, g3140, soft policy S1, g3270, `test_local_scoring_ppg` and `test_ppg_g210` (74 tests in all). All pass except two `test_ppg_g210.Restored` tests, which need an old g0660 export that is not in the local results repository.

No teacher table was played locally. The first lab iteration is the check that the rebuilt teacher's maps pass admission, in `search_value.scoring_maps`.

## Code

| File | Change |
| --- | --- |
| `rust_backend/lab_cpu_overlay.py` | The finish message carries `scoring` (CPU teacher only) |
| `rust_backend/lab_teacher.py` | `collect` returns it with the table |
| `recovery/local_scoring_targets.py` | `audit_eras` (shared by ordinary games and tables), `validate_table_receipt` |
| `recovery/search_auxiliary.py` | Table envelopes; the pools (`tables`, `latest`); `SearchRows`; the per-batch share (`share`, `add`) and the report |
| `recovery/search_value_phase.py`, `search_value_iteration.py` | Keep the map on the table's shard; hand the pools to PPG in `prepare_ppg` |
| `recovery/local_scoring_v3.py` | `search_loss` (era-head losses only, counted apart; `frozen` for heads-only steps), `search_report` |
| `recovery/ownership_stop.py` | The search validation tables pooled into the criterion (`criterion`), reported at every boundary |
| `recovery/value_phase.py`, `ppg_g210.py` | The search share in every joint and heads-only step; the report |
| `recovery/training_revision_g3140.py` | Item `search-aux` (block `search_auxiliary`, contract 2.0.0, `train_iterations` 4, forward-only, tag `searchaux`) |
| `recovery/league.py`, `scripts/continue-rust-g1491.py`, `continue-rust-g1190-restart-2.py` | Log line, help text |

The execution identity is re-recorded as `259d39df…`, with `e62799b4…` (the committed 1.0.0 and soft policy S1) kept as a predecessor.

## Continue command

You have not run the soft-policy command yet. This one adopts soft policy S1 and the search auxiliary (2.0.0) together at g3423 and runs to the g3430 rating:

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3423 --iterations 7 --share-target 0.1 --soft-policy --value-revision search-aux
```

- **Before the first iteration** the continuation rebuilds the CPU teacher, which takes a few minutes once.
- **After the first iteration (g3424), check:**
  - `search_value.scoring_maps` is 40 of 40;
  - the log shows `search_aux=2304rows/36games search_held=4games`, a `search_ce` before → after and an `aux_crit`;
  - `presented.steps` equals `ownership_stop.attempted_head_steps`;
  - the soft-policy checks in [G3423_SOFT_POLICY_S1.md](G3423_SOFT_POLICY_S1.md).
- **Then 20 iterations to the g3450 rating.** Both settings are stored with the run:

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3430 --iterations 20 --share-target 0.1
```

- **The pools fill as in the table above:** 144 tables from g3426, 32 held-out tables by g3430, 160 tables from g3434.
- **Adopting both at once** means the first iterations change two things. They act in different places, and each is reported separately:
  - the soft head trains on the teacher term inside PPO (`teacher_phase.term.soft`);
  - the search rows train the era heads in PPG (`value_phase.search_auxiliary`).
