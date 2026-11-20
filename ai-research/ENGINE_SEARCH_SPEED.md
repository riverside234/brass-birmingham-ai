# Core engine speed: exact decision reuse in the teacher search

Code-only. No recipe flag, no run state and no training change: searches, labels, value rows, search rows and every
recorded output are bit-identical. It was first recorded alone as execution identity `e606cedf…` (predecessor
`7f8399ea…`). It now ships in one identity, `25c6c3ad…`, with the g4138 teacher search revision
(`ai-research/G4138_TEACHER_SEARCH.md`), which is a recipe change. The next continuation rebuilds the CPU workers by
itself (`ensure_cpu`: the overlay fingerprint covers the copied core files).

From the newest export (g4138) to the g4150 rating, with the g4138 revision; use the run's current generation if it
has moved:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 4138 --iterations 12 --share-target 0.1 --teacher-search-revision
```

The expected timings below are for the engine change alone. The revision adds about 5–8 s per iteration.

## Why simulations were slow (g4137 export)

The 40 CPU teacher tables set the collection time:

- They run in parallel, one per CPU, for 43–49 s each, and the last search ends at 50 s of a 63 s collection.
- The 320 ordinary games need only about 1 s of native work each, about 11 s per ordinary lane.
- A table runs 64 searches and 14,848 simulations.

| per table (median of 40) | seconds | per simulation |
| --- | --- | --- |
| search | 45.0 | 2.98 ms |
| network forward (ONNX, 13.8–17.1k positions, batch 1) | 18.0 | 1.18 ms |
| feature encoding | 1.7 | 0.11 ms |
| **engine and search bookkeeping** | **25.3** | **1.69 ms (57%)** |

A local profile of the same whole-turn search shows where the engine time goes. It used `Session::decide` with the
teacher's 96/640 budgets, k 8, a stub network and one four-seat game, on the laptop: 1.1 ms per simulation, with no
network time.

| part | share | cause |
| --- | --- | --- |
| hashing the search key | ~40% | `Key` hashed its whole `InformationState`: the full string observation including ~300 history entries, the recall, every legal choice and the context. That is about 38 µs per hash, repeated in the tree lookup (about 5 per simulation), tree insert, network-cache lookup, insert and eviction; each hit also deep-compared the keys twice |
| rebuilding the observation | ~25% | every simulated step rebuilt the public observation as strings (`refresh_checked`), then cloned it again for the acting seat |
| legal moves | ~15% | every simulated step regenerated and string-exported the legal set, also at nodes already in the tree |
| the rest | ~20% | state copies, chance steps, sampling, backup |

Only the leaf of a simulation (one per simulation) needs a fresh decision. The other ~4.6 steps per simulation revisit
decisions the search has already built, and paid the full cost again.

## Change (`native/brass-ai-core`, copied into the CPU teacher core only)

1. **Decision reuse** (`search/brass.rs`). Each search root (one `Brass`) keeps the decisions it has built, keyed by
   exact bytes of everything the acting seat can see (`NativeState::visible_identity`):
   - every owned field except other seats' hands, the deck order and the hidden discard prefixes;
   - the seat's recall (after the root seat's common prefix);
   - the insolvency ledger.

   These bytes refine the key: equal bytes imply an equal permitted view, recall, legal set and ledger. A repeated
   decision returns the same shared `Arc<Key>`, so the tree, the network cache and the root check compare pointers.
   Sampled hidden cards are never read, so every determinization of the same information set shares one entry. Each
   root keeps at most 64 MiB of identity bytes; beyond that a decision is rebuilt.
2. **Cheap key hash.** `Key` hashes a subset of the fields its equality compares: seat, revision, phase, own hand, the
   last history entry, cash, VP, pending action, recall length and the legal-set size. Equal keys still hash equally,
   and equality still decides every lookup.
3. **No per-step observation** (`state/observation.rs`). `ObservationMemory::refresh_recall` keeps the hands and
   recall exactly as `refresh_checked` does, but drops the public projection. A new leaf builds its view directly
   from the state (`information_at`), equal to the published projection.
4. **Ledger from the state** (`search/reward.rs`). The insolvency ledger reads the same public accounts (era, round,
   phase, debts, cash, income, tiles on the board) off the state (`Accounts`). At a continuation Finish it keeps
   those accounts instead of a full view copy.
5. **Build** (`lab_cpu_overlay.TEACHER_CORE_FILES`). The teacher build also copies `state/observation.rs` and
   `search/reward.rs`. A zero-insolvency teacher core keeps its exact `reward.rs` patch on the copied file. The
   ordinary collector, the sealed core and the teacher job fingerprint are unchanged.

Unchanged: the per-leaf string view and legal set (the network encoder reads them), the rules, the search
algorithm, budgets, RNG streams and every output format.

## Measured locally

The harness is `tmp/engine-perf/search_profile.rs`: the whole-turn search above, the old and new core, the same
seeds. The digest covers every search's visit counts, returns, plans, network-cache hits and played actions.

| game | old µs per simulation | new µs per simulation | ratio | digest (both) |
| --- | --- | --- | --- | --- |
| seed 4, search picks the moves | 1,135 | 199 | 5.7× | `812106e6…` |
| seed 5, search picks the moves | 1,070 | 191 | 5.6× | `fc658863…` |
| seed 3, greedy build-first moves (14,848 simulations, as a lab table) | 857 | 179 | 4.8× | `ae912578…` |
| seed 4, greedy build-first moves | 874 | 182 | 4.8× | `448eb8bf…` |

What remains per simulation (greedy game, about 180 µs):
- the leaf's view and legal set, built once: 86 µs (legal moves 46, string view 36);
- applying moves: 30 µs;
- sampling the world: 9 µs;
- the network cache: 11 µs, of which the stub is 4;
- backup: 7 µs;
- decision lookups (identity bytes, cache and tree): about 10 µs in all.

Real CPU teacher, built exactly as `compile_worker` builds it, with the old and new core. It ran with the test fixture
network and the lab budget (640/96 simulations, k 8, c .20/.70), one all-seat table each, interleaved twice. The laptop
was under load, so compare ratios rather than absolute seconds.

| table | simulations | old table (two runs) | new table (two runs) | old engine per simulation | new engine per simulation |
| --- | --- | --- | --- | --- | --- |
| standard, 105 searches | 25,856 | 99.8 s, 105.0 s | 50.8 s, 53.0 s | 2,137, 2,222 µs | 476, 490 µs |
| cotton (g3528 rule), 93 searches | 20,896 | 77.4 s, 78.8 s | 37.8 s, 40.3 s | 2,105, 2,149 µs | 439, 458 µs |

Engine time per simulation fell 4.5–4.8× and the table about 2×. The fixture network's forward (1.3–1.6 ms per
position under this load) sets the rest. Labels, value rows, search rows, the search metrics without timings, the game
result and the network positions are identical between the old and new teacher and between repeats.

## Expected on the lab (estimate)

At g4137 a table's search took 45.0 s: 18.0 s forward, 1.7 s encoding and 25.3 s engine.
- With the engine 4.5× cheaper, about 5.6 s, a table takes about 26–27 s instead of 43–49 s.
- The last search would end at about 29 s instead of 50 s.
- The roughly 13 s teacher finalization after the last search is Python-side and does not change.
- The collection would take about 40–45 s instead of 63 s, saving about 20 s of a 190 s iteration (about 10%).
- Ordinary games (about 11 s per lane) stay well inside that.

## Checks

- New core tests (`search::brass::decision_cache_tests`, release and inside the real teacher core):
  - every reused decision is rebuilt and compared in full;
  - the same searches with reuse off give identical reports: selected action, every node's statistics and every
    trajectory;
  - sampled worlds of one root share the root key;
  - native accounts and lazily kept recall equal the published projection on every state of a game.
- The full core suite passes: 85 tests, including the session and reuse tests against their recorded fixtures.
- The CPU teacher build's 43 `cpu_` tests and the cotton core tests pass.
- Harness digests are identical to the old engine on the four games above, and on seven more during development.
  The digest covers every search's visit counts, returns, plans, network-cache hits and the played actions.
- Old and new CPU teachers give identical lab-budget tables, standard and cotton, with the Python verifier
  accepting both (`tmp/engine-perf/scripts/compare_teacher.py`).
- Native teacher regressions pass on the new binary: `test_teacher_count_search_is_deterministic`,
  `test_retuned_budget_reaches_long_search_and_its_labels`, and the g3471 search-rows admission test.
- Focused Python tests pass:
  - `test_teacher_cotton` with the `test_cpu_inference` configuration, startup, bundles, training-entry and CPU-pool
    classes (35 tests);
  - the 3 `test_cpu_inference` overlay tests.

## Cost and risk

- No extra CPU, memory of note (a few MB per search root), tables, games or held-out data.
- The lab rebuild takes the usual few minutes inside `ensure_cpu`.
- Risk: a decision cache that missed a field the key depends on would silently change searches. The identity
  therefore covers every owned field except the three hidden zones. The tests rebuild and compare every reuse.

## What to check after the first iteration

- `teacher_phase.collection.collection_metrics[*].elapsed_ns`: about 26–27 s per table instead of 43–49 s.
- `forward_ns` and `positions` unchanged in kind (same searches).
- `timings.collection_seconds` falls by about the same amount as the slowest table.
- Labels, value rows and search rows are admitted as before (identical outputs).

## What remains

After this change the network forward is about two thirds of a table. In order of size:
1. ONNX forward: 1.18 ms per position at batch 1 on one thread.
2. The teacher finalization after the last search: collate 8 s and check 10 s at g4137, partly overlapped. It is
   Python-side.
3. Leaf legal moves: about 25% of the remaining engine time. `can_build` rescans every slot for each card and
   industry, so an exact restructure could save about 1 s per table.
4. The leaf's string view: about 20%. Removing it needs a native-state feature encoder.
