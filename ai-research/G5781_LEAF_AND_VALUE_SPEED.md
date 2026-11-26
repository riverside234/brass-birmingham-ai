# g5781: leaf decision building and value-phase speed

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, committed at g5781 in the newest export
(`joint7-v11-2-g4970-rail025-soft8-beta0.02-g5781-20261126T011308721406Z`). Code-only, no training change: every
search, label, value row and value-phase number is identical. No flag.

## Request (2026-11-25)

1. Implement the performance improvement for decision materialization and view building (the search leaf's legal
   set and string view, the largest engine cost left after the g4138 decision reuse).
2. Improve the value phase's performance; find where its bottleneck is.
3. Follow-up: remove the SHA-256 hash from the per-row descriptor-dropout path, and remove or improve its other
   per-row parts (bisect, record wrapper, structured-row read).
4. Follow-up: finish the two parts left open, the era heads' per-row Python (per-phase device tables) and the view's
   text copies (outside the five teacher core files).

## 1. Leaf decision building (Rust core)

### Where the time went

The engine harness (`tmp/engine-perf/search_profile.rs` with `crate::prof` spans on the teacher's core layout,
`tmp/leaf-perf/`): one greedy four-seat game, teacher whole-turn searches (96/640, k 8), a stub network, 14,848
simulations. A new leaf decision cost 87-99 µs of the 176-207 µs per simulation (laptop, thermal state varies):

| part | µs per simulation |
| --- | --- |
| legal set: staged generation | 29-37 |
| legal set: export (choice IDs and fields) | 12-13 |
| legal set: sort and duplicate check | 5-6 |
| view: slots, links, merchants | 15-18 |
| view: history export | 10-12 |
| view: players (discards, mats) | 8-10 |

The staged generation re-ran `can_build` for every distinct card in hand (up to 8) on every slot and type. Each call
rescanned all 49 slots two or three times and allocated: `resource_slots` vectors, `format!("{industry:?}")`, and a
`to_owned` merchant name per merchant inside `market_connected`. Only the card's own permission depends on the card.

### Change (`native/brass-ai-core/src/state/choices.rs`, `state/mod.rs`)

Both files are in `lab_cpu_overlay.TEACHER_CORE_FILES`, so training and the local game take the change from one
tree; the overlay fingerprint makes the next continuation rebuild the CPU teacher (`ensure_cpu`).

- `buildable()`: the card-independent build rules (tile, era, overbuild, slot types, Canal one tile per region, coal,
  cash) once per Action decision, with per-region own-tile counts, empty single-type slots and memoized coal cubes and
  market reach. Per card, only `card_allows` remains: its rule (`location:`, `wild:location`, industries) is read once
  per card from a static table.
- `sale_targets()`: the (slot, merchant) sales once per decision instead of once per card.
- Merchant regions from a static table (no string allocation per merchant per call).
- `export_choice`: each semantic ID written once into an exactly sized string; industry names from a static table.
- The sort is unstable: IDs are unique (duplicates are still refused), so the order is the same.
- Work checkpoints: one per slot in the build pass and one per card, instead of one per card and slot. The teacher
  runs without a work scope, where a checkpoint returns at once; the counts are diagnostics only.

### Results

Engine harness, the same games: both digests unchanged (`ae912578…` seed 3, `448eb8bf…` seed 4).

| µs per simulation | HEAD | new |
| --- | --- | --- |
| staged generation | 32.0 / 37.0 | 7.0 / 7.1 |
| export | 13.2 / 12.0 | 10.8 / 9.9 |
| history export | 11.6 / 11.5 | 8.6 / 8.0 |
| sort and check | 5.9 / 5.3 | 4.2 / 3.8 |
| whole leaf decision | 97.9 / 99.2 | 65.0 / 61.4 |
| whole simulation | 206.9 / 205.0 | 166.1 / 157.5 |

Interleaved, twice per seed (the laptop was also running tests): a simulation 221-230 µs on HEAD against 164-178 µs
new (23-28% less), the leaf decision 107-110 µs against 64-69 µs; the digests were unchanged in every run.

The real CPU teacher, built exactly as `compile_worker` builds it, from HEAD's engine files and from the new ones
(`tmp/leaf-perf/build_teacher_head.py`, `tmp/leaf-perf/compare_teacher.py`), with the fixture network and the lab
budget (640/96, k 8), each table twice, interleaved. Outputs are identical: labels, value rows, search rows, the
search metrics without timings, the result and the network positions. The laptop was also running the test suite, so
the timings are noisy:

| table | HEAD non-network µs per simulation | new | HEAD table | new table |
| --- | --- | --- | --- | --- |
| standard (104 searches, 24,672 simulations) | 544, 499 | 508, 466 | 50.1, 49.1 s | 49.8, 46.8 s |
| cotton (86 searches, 18,048 simulations) | 569, 464 | 481, 404 | 44.3, 35.7 s | 40.6, 32.1 s |

Non-network time per simulation fell 7-15%. The network forward is about two thirds of a real table's search (24-33 s
of 34-49 s), so a table falls by about 3-10% (load noise is large here).
The real teacher spends about 470-570 µs of non-network time per simulation, against about 160-200 µs in the stub
harness; the leaf build is therefore a smaller share there.

At g5780 a lab table took 41 s (median of 54; the slowest 75 s): about 24.7 s forward, 1.5 s encoding and 11 s engine
(450 µs per simulation).

### Follow-up: catalog names borrowed, not copied (`state/dto.rs` `Name`)

Every view and exported choice cloned its catalog text: slot, link, tile, merchant, region and card names, and the
pending fields. That was about 35 µs per simulation, allocation-bound. They are now `Name = Cow<'static, str>`
(slice fields `Cow<'static, [_]>`). The catalog leaks its names once per process into `&'static str`, so a copy is a
pointer copy; names parsed from input stay owned. Serialization, equality, ordering and hashing are those of the text,
so every message and every search key is unchanged. Choice IDs stay `String`; merchant beer targets
(`merchant:<id>`) come from a static table.

- Readers adjusted: `state/belief.rs`, `chance.rs`, `initial.rs`, `phases.rs`, `simulation.rs`, `indexes.rs`,
  `catalog.rs`, `features/mod.rs`, `board.rs` and `derived.rs`. `TEACHER_CORE_FILES` now copies these 16 files
  (`lab_cpu_overlay`), so the training teacher and the local game take the same change.
- Sealed `lab-native` compiles unchanged. Three CPU overlay files the repo owns needed type-neutral edits (they still
  compile against the sealed core's `String`s, which the ordinary collector uses): `lab-native-cpu/src/cpu_autoplay.rs`,
  `cpu_cotton.rs` and `core-overlay/cotton.rs`.

Harness, interleaved, twice per seed (same digests in every run):

| µs per simulation | HEAD | legal-set change | + borrowed names |
| --- | --- | --- | --- |
| whole leaf decision | 86-92 | 59-62 | 36-37 |
| view | 37-38 | 35-37 | 14-15 |
| legal set | 46-50 | 20-21 | 16.5-16.7 |
| whole simulation | 178-185 | 148-155 | 108-113 |

The real CPU teacher from HEAD's engine against this one, the same lab-budget tables as above, each twice,
interleaved (a quieter laptop this time). Outputs are identical: 31 labels, 64 value rows and 24,654 positions
(standard); 32 labels, 64 value rows and 17,998 positions (cotton).

| table | HEAD non-network µs per simulation | new | HEAD table | new table |
| --- | --- | --- | --- | --- |
| standard | 362, 306 | 207, 210 | 34.0, 25.8 s | 22.6, 23.0 s |
| cotton | 262, 261 | 166, 163 | 18.5, 18.3 s | 15.0, 14.7 s |

Non-network time per simulation fell 31-43%, and a table by about 11-19% (the first HEAD standard table ran cold).

### What remains in the engine (not done)

- `brass.step` (about 26 µs per simulation before this work): each step clones the whole state and rebuilds its
  indexes.
- The real teacher's non-network time is about 3 times the stub harness's. The likely cause is the forward
  streaming 17 MB of weights through the caches between engine calls; profiling it needs the lab.

## 2. Value phase

### Where the time goes (g5686-g5780, one trainer process since the g5685 restart)

The value phase took 12-27 s of a 155-185 s iteration:

| part | seconds |
| --- | --- |
| cache stage (`buffer.stage`: serialize 2.4, hash 0.8, flush 0.4) | 4.2 |
| replay open and index | 1.05 |
| ownership stop setup (search lambda targets 0.9, rows 0.3) | 1.8 |
| joint PPG steps (16-54) | 0.17-0.35 each |
| head-only era-head steps after the value check stops the trunk (0-38) | 0.21 each |
| held-out acceptance scans (one per 16-step interval) | 0.27 each |

A step is 4,096 value rows in 8 microbatches of 512, 512 clone rows and about 680 search rows. The model is a
512-wide MLP (8 residual blocks), about 100 GFLOP per step, a few milliseconds on the lab GPU. The step is host
bound. Its time follows the era heads' host work: steps while the heads fit cost about 0.3 s, steps after their
ownership stop about 0.17 s, and `local_scoring_auxiliary.training.forward_seconds` is 0.12 s per active step.

Per microbatch, the host ran (measured on the g4770 PPG fixture's real critic replay, 512 rows):

| part | ms |
| --- | --- |
| descriptor dropout: a replay-row read (bisect, record wrapper, structured row) and a SHA-256 per row | 1.9 |
| era-head scoring metadata | 1.05 |
| era-head target arrays (slots, links; twice with Rail-end) | 0.6-1.3 |
| Rail-end metadata | 0.4 |
| fitting-token validity loop | 0.4 |

There were also about 18 device synchronizations: the rows read back from the device, the critic weights read back,
and pageable uploads of every target array, each waiting for all queued kernels. Interval bookkeeping read every
kept row's game ID through the same row reads: 65,536 rows per accepted interval and 217k twice at the end.

### Change (code-only)

- `seat_descriptors.dropout_mask`: the dropout decision (`drop_game`, unchanged rule) is made once per game and
  epoch and kept on the dataset. No row hashes or reads a replay row. The PPG critic replay maps rows to games with one
  gather of its sealed game column (`CriticDataset.game_codes`, `game_names`); other datasets memoize per game id.
  SHA-256 now runs once per game per epoch, a few hundred times a phase instead of per row.
- `critic_inputs`: the replay's descriptors are read with the batch's host row list, which hits the batch's
  prepared column, so there's no device round trip. Device-resident descriptors use the batch's own index (`index=`).
  Dropped rows are set with `torch.where`, giving the same values without a boolean-index read-back.
- `value_phase.branch_rows`: branch-family counts from the game column (`CriticDataset.branch_games`).
- The era heads (`local_scoring_v3` forward and `head_loss`, the 2.0.0 `Companion.forward`, `value_output`) take the
  batch's host rows (`value_phase.host_ids`) instead of reading the index back. The critic weights come from one host
  copy per weights tensor (`host_weights`). Target arrays and flags upload through pinned non-blocking copies
  (`upload`).
- PPO's critic fitting (`updates`) uses the same dropout and host-row paths.
- `value_phase.step_profile` (report only): host seconds per part of the joint steps (`value_rows`, `search_rows`,
  `clone_rows`, `clip_and_device_wait`, `optimizer`). The clip's norm read is each step's one device sync, so the
  next export shows whether the device or the host sets the step.

Every tensor operation is unchanged.

### Results

- Exactness: `tests/Recovery/test_value_phase_speed_g5781.py` runs the g4770 PPG phase (ownership stop, search rows,
  Rail-end heads) with the new paths and with every replaced path restored. Every reported number is identical. On
  the real critic replay inside the phase, dropout masks and descriptors equal the per-row rule over four dropout
  settings, and branch counts and host weights equal the row reads.
- Host time (laptop, under load): dropout per 512-row microbatch 3.3-4.7 ms → 0.5-0.7 ms; branch count over 65,536
  rows 174 ms → 4.6 ms.
- Expected per iteration (an estimate): about 1-2 s less in the value phase from dropout (424 joint microbatches,
  plus up to 304 head-only), up to about 1.5 s from the branch counts when intervals are kept, about 0.5 s in PPO's
  critic fitting, plus whatever the removed synchronizations save on the GPU.

### Follow-up: era-head target tables (`local_scoring_v3.EraTargets`)

The era heads' remaining per-row Python (scoring metadata, target arrays, Rail-end metadata, validity loop) and their
uploads are gone from the batches. Per phase, each dataset's targets are built once and kept on the device. That
covers the PPG critic replay, from its sealed game, context, seat and mapping columns, and the search rows, from the
metadata they keep:

- **Per key:** each distinct (map, era, seat, seat mapping) gets its classes, local VP and mask, computed by the code
  `target_arrays` uses (`target_value`), plus its Rail-end map.
- **Per row:** indices into those tables, and the Canal, map and Rail-end flags.

A batch gathers its rows' targets on the device. On the host, only the valid rows' fitting tokens and the map support
are built. The target cache's hit and miss counters and its order are replayed exactly (`_count_targets`). Other
datasets keep the per-batch path.

- Exactness: the same speed test now runs the reference phase with the tables off too. Every reported number,
  target-cache counters included, is identical, and the new run's report shows the tables were built
  (`local_scoring_auxiliary.training.target_table_seconds`).
- Host time per 512-row microbatch (laptop): 6.3 ms of per-row target building → 1.8 ms of table gathers. About
  3 s less per value phase across joint and head-only steps, and no target uploads.

### What remains in the value phase (not done)

- The fixed parts: the cache stage's 4.2 s (serialize and SHA-256 seal of 1.8 GB), the 1.8 s ownership setup.
- `step_profile` in the next export tells whether the joint step is now device or host bound.

## Checks

- Core: the new `state::choices::reference_tests` compares the restructured generation with the original per-card
  enumeration on every state of six played games (1,982 states, 744 Action states, 11 continued sales), and every
  exported choice and history ID with `Choice::semantic_id`. The full `brass-ai-core` suite passes, including the
  decision-reuse and cooperative-cancellation tests.
- CPU teacher (built as on the lab): it builds; the 44 `cpu_` tests, the cotton core tests and the decision-cache
  tests pass; real tables are identical to HEAD's (above), with and without the borrowed names.
- Ordinary CPU collector (built as on the lab, on the sealed core with the CPU overlay): it builds and its 41 `cpu_`
  tests pass, so the edited overlay files compile against both string types.
- Workspace: `brass-ai-core`, `brass-ai-worker` and `brass-ai-inference` type-check with all targets; the full core
  suite and the worker's 25 `local` tests pass.
- Python: the focused suites (critic descriptors, ownership stop, search auxiliary, search rows, Rail-end, era-head
  and auxiliary revisions, the new speed test) pass. The two `test_ppg_g210.Restored` g660 tests fail at HEAD too: their
  export is missing (`docs/KNOWN_TEST_FAILURES`).

### Lab fix: architecture source equivalence (2026-11-25)

The first lab continuation stopped in `ensure_cpu`: the focused regression
`test_training_revision_g1491.Recipe.test_new_labels_retire_the_old_labels_and_beta` failed (evidence
`results/rust-lab-qualification/20261126T025946701442Z-cpu-inference-…`). `seat_descriptors.py` is one of the six
files `contracts.architecture_fingerprint` hashes, so the dropout edit moved the source digest from `f9f5cbb8…` to
`a0c2124a…`, which no equivalence receipt mapped. That changed `compatibility()['architecture_sha256']`, and with it
every checkpoint signature check and the teacher buffer identity. On the g5781 export, the committed signature
no longer matched, and the teacher buffer (`0bdd3bb6…`) would have been retired together with its beta.

`configs/learning/reward-source-equivalence.json` now names `a0c2124a…` as the exact equivalent source. This is the
same receipt every earlier exact model-source edit updated, and the descriptor tensors are identical (the speed test
above). The architecture identity stays `05368f93…`. On the g5781 export, the signature compatibility and the
teacher buffer identity match again, and the teacher weight is kept. The focused regression list the lab runs passes
locally (the `NativeWorkers` cases need the built binaries; they passed on the lab). The one other failure in the set
run, `test_table_revision_g4551…vp_report_and_chart…`, is a listed known failure and fails the same way at the
previous commit.

## Cost

None: no extra CPU, games or held-out data. Memory: a few hundred booleans per dropout setting, one host copy of the
critic weights, the era-head tables (a few thousand map entries and an index per row, tens of MB on the GPU), and the
catalog's names leaked once per process (kilobytes). The next continuation rebuilds the CPU teacher (a few minutes in `ensure_cpu`).

## Execution identity

`cb7721f75684bec7119e95cb9229c75f626e460285b13b9b552efcf87910d0d3`, with predecessor `d73df28c…` (the committed
g5781 checkpoint's). It replaces `6a725b62…`, which never trained. The entry is `g5781_leaf_and_value_speed` in
`configs/learning/b07-replay-location-review.json`.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 5781 --iterations 10 --share-target 0.065
```

Use the run's committed generation in `--expected-generation`. The clip, optimizer and auxiliary revisions are already
adopted.

## Local game

The local packages build from the same `native/` tree, so the next local release picks the engine change up with a
rebuild (`scripts/local-game/search-stage/build-search-stage.py binaries`). The local search is network-bound (4-7 ms
per simulation), so the gain there is about 1-2%. Nothing local was rebuilt or installed.
