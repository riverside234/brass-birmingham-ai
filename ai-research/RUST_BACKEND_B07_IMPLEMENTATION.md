# B07: cheaper native collection and bounded adoption

Updated on 2026-11-07. **R4 ordinary qualification is Pending. R3 completed
collection but failed the matched probability tolerance; useful performance,
canaries, production adoption and the next normal commit/restart remain Pending.** Run the
[new g1250 bundle](../distribution/rust-lab-b07-g1250/README.md); do not rerun
passing B01–B06 qualifications. The installed game is unchanged.

## R4: V4 tables, retained features and reusable lanes

Pulled **inside `results/`**, from `f4652ed` to `e71dc25`. Latest failure:
`rust-lab-qualification/20261107T101615796829Z-20261107T101336169559Z-g1250-b07-2fcbfcca-0192d294`.
Both arms completed 3 families, 1,161 decisions and 675 forced rows. Native
collection took 30.066s versus reference 24.149s; accepted work was 32.684s
versus 27.212s. Native first read was 2.071s versus 2.467s. Per-family full
validation/assembly cost 2.119–2.444s. These measurements remain failed-run
evidence, not a current qualification pass.

The comparison stopped on probability difference **4.68192948e-7**, relative
**3.22833888e-6**, just outside `rtol=3e-6, atol=1e-8`. Different queue/batch
shapes can change float32 GPU reductions; the discrete rows inspected before
the assertion matched. This is consistent with batching roundoff, not proof
that all later rows matched. R4 checks *all prepared features and targets first*,
then uses the existing float32 head-check scale (`rtol=1e-5, atol=1e-7`) for
GPU-derived behavior/branch probabilities and reports maximum errors. Actions,
candidate order, observations, results, opponent identity, contexts and RNG
components remain exact; forced/external probabilities must equal one exactly.
Independently, publication reconstructs each row's probability from its own
raw evidence with **1e-12 absolute tolerance** (previously 3e-6), and replays
every sampling draw. Benchmark tolerance never changes stored training data.

R4 retains the completed R3 compact requests, binary features, forced batches,
batched records and two CPU-admitted publishers. The remaining changes are:

- **V4-containing ordinary tables stay in Rust.** Use the original `LegacyPolicy`
  / `LabLegacyPolicy` and its C# decision worker on Rust's public observation
  and ordered legal IDs. Rust alone advances the authoritative state. The C#
  worker's private planning simulations, plan queues, intent, seed draws,
  per-turn work/budget accounting and policy IDs are unchanged. Only neural
  seats use forced batching, matching the reference actor. Searched/nonordinary
  policies retain their explicit reference route; archived V4 fallback replay
  remains readable with its original C# producer identity.
- **Keep training arrays.** Rust captures the exact features at every accepted
  revision, including forced and V4 rows, and exports bounded binary batches.
  The Python actor retains `CheckedGame` arrays/seals and its input-validation
  proof. A single transient NPZ file carries compressed JSON, binary arrays
  and bindings; the scheduler holds its hash/proof separately. Publication still
  performs full behavior/RNG/game validation, reusing those arrays. The final
  cache remains `derived-features/4.0.0`, compatible with existing readers.
  Archived reference collectors retain their existing CheckedGame metadata too.
  Missing/obsolete caches still use full validation and regeneration.
- **Cheaper verification and fewer copies.** Forced verification checks the
  singleton/evidence/context/counter/probability and consumes exactly one
  `random()` for v2, two for v3 including its component draw, without NumPy
  distribution construction. Other sampling math is unchanged. Rust hashes
  record bytes incrementally and drains exported buffers in order; stale cursors
  fail and completion seals survive draining. Readers hash the same bytes they
  decode, verify row/context/seat bindings, and reuse the already verified replay
  hash. Cache version is pinned once per immutable process; byte hashes are never
  replaced with stat/mtime trust. Private assembly transfers validated seals;
  public conversion still detaches data and later mutations remain detectable.
- **Reuse per-lane processes and policies.** A bounded 12-entry cache retains
  original policy objects/C# workers and the collector between games in the
  ordinary pool. Each game gets a new token and per-seat RNG; queues, intents,
  counters, contexts, branch opening overrides and feature bindings reset.
  Cancellation retires the lane and private C# workers; failures never reuse a
  partially completed game. Existing CPU admission, model cohorts, callbacks,
  branch scheduling and whole-family recovery stay in place.

Short local checks: the 24-stage original C# V4 worker smoke matched every
observation, action, queued plan, intent, RNG and work-budget field, covering
13 forced stages and 18 queued continuations. C# state progression took 1.344s
versus Rust 0.415s; the unchanged V4 decision worker itself took about 11.3s per
arm and remains a bottleneck. Two alternating publication pairs on the same
three small authored families matched games/features/targets: median publication
plus first read **3.014s → 2.427s (19.5% lower)**; validation/assembly
**0.451s → 0.274s**, warm read **0.885s → 0.619s**. An intermediate separate-file
handoff regressed and was replaced with the single-file design; its measurement
is retained. These are component smokes, not g1250 GPU throughput evidence.

Regressions cover retained arrays on native and archived fallback paths,
independent receipt/cache/row tampering, no Python re-encoding, exact forced RNG,
V4 queue continuity/reset, process/token reuse, stale and canceled work, draining,
publisher failure and cold recovery. Run `--ordinary-only` for the new bundle.
Local validation passed 48 distinct focused Python checks and eight Rust checks;
the complete two-decision lane smoke also reused the real collector and original
NeuralPolicy objects across games with a deterministic test inference reply.
The actual g1250 matched batch still includes self-play, historical NN and
historical V4 strata: selection now uses V4 presence, so changing the backend
route does not accidentally drop that third table. Keep the original checkpoint,
config, 60-worker cap, teacher-off default and search_v1 behavior. No B01–B06
qualification or long session is repeated. Canary/adoption work remains separate.

## B07 R3 qualification repair

Pulled the results repository at `f4652ed`; failed attempt:
`results/rust-lab-qualification/20261107T090047266944Z-20261107T085645018439Z-g1250-b07-d0097dac-faa1535f`.
Ordinary work matched exactly (3 families, 1,161 decisions, 675 forced rows).
Native accepted work fell from the historical B06 55.265s to 40.246s, but the
current reference was 28.833s. These are bounded samples, not a statistical or
whole-iteration speedup claim. The negative useful-gain gate remains in force.

The first shadow table stopped after 52 verified turns: Rust reported
`stale/model/epoch GPU reply` for a correctly echoed request. A regression test
reproduces the exact defect: the default serde_json parser decodes
`0.015000000000000001` as the adjacent f64 `0.015`. This budget occurs naturally
as `0.025 - 0.01`. Python preserves its original value; the Rust parser caused
the mismatch. A genuine stale reply was not admitted.

Enable `serde_json/float_roundtrip` for both adapters: exact teacher budgets and ordinary sampling probabilities.
Do not round budgets, add tolerance, omit fields or weaken exact binding checks.
No search equations, NN weights, feature definitions or shared-core source change. Native tests cover the exact failed decimal, every spending
step and adjacent floats, deterministic exponent/mantissa cases, subnormals,
signed zero and identity/budget mutations. The new teacher binary is explicitly
built/tested in a detached directory; the ordinary transport adapter is also rebuilt
against the unchanged B01 core. Original C# bootstrap pins are reused. No installed-game update occurs.

R3 seals the original successful ordinary phase as historical evidence and an
exact source review. Its benchmark scripts stay byte-identical, but native
publication/reads changed, so the short matched ordinary scope must be measured
again. The retry also reruns the unfinished canaries and continuation; it never
repeats successful B01–B06 qualification.
The repaired build receipt can be supplied with `--teacher-build` on a further
retry; producer, implementation, core, binary/test-log and C# pins must match.

The B07 lab GPU forward total was 7.898s versus reference 8.523s, so the remaining
measured excess is outside GPU forward work. The serial publisher spent
4.222/4.357/4.182s on three families; native first read was 5.247s versus 2.455s.
The native tables themselves took 16.691s/17.210s; one historical V4 table still
uses the exact whole-table reference route. These overlapping times must not
be summed as an end-to-end attribution.

R3 removes the second per-row `check_game` walk during native assembly after
complete family validation, preserving the already computed row/label seals
and independently checked scoring envelope. Public conversion still detaches
caller input. Private publication/read paths avoid a second deep copy of
freshly decoded boards/histories, while checking exact replay/cache bytes on
every read. Branch attachment refreshes only the changed header seal. No source,
behavior/RNG, scoring, critic, context or mutation check is waived.

At most two isolated publishers can work on different complete families, with
the same shared CPU admission and bounded family queue. Callback delivery remains
serialized; results retain frozen-job order. Crash/first-error/cancellation and
atomic whole-family publication checks remain. New receipts separate part-read,
family-seal, full validation/assembly, JSON encoding, replay write/hash and
feature-cache time, so the next lab result can identify the remaining tail.

The short R2/R3 CPU comparison (`rust_lab_b07_r3_publication.py`) uses three
alternating batches of the same three eight-decision authored families. Median
publication was 1.219s to 1.124s, first read 0.500s to 0.452s, and warm read
0.608s to 0.475s. Combined publication/first-read median was 1.719s to 1.577s
(8.3% lower). An intermediate file-read optimization regressed by allocating
the full maximum read size; the final version uses the opened file size plus
one byte, validates the bound and exact length, then hashes those same bytes.
Games, features and prepared targets matched. This small serial component test
does not measure GPU throughput or the two-publisher overlap; lab acceptance
remains Pending and must use current-source matched work.

## What the completed B06 record shows

Pulled from the results repository at `f06c0a1`:
`results/rust-lab-qualification/20261107T080012822445Z-20261107T075601452661Z-g1250-b06-c1cd94b2-2c29d5e3/evidence/qualification.json`.
B06 passed correctness and bounded measurements on g1250. Both ordinary arms
produced three identical parent families, 1,161 decisions and 675 forced rows.
No natural branch occurred in this small sample.

| Work, seconds | Reference | B06 native adapter |
| --- | ---: | ---: |
| Collection, including startup/publication | 25.485 | 36.226 |
| First admission/read | 2.507 | 18.457 |
| First preparation | 0.604 | 0.582 |
| Accepted work total | 28.596 | 55.265 |
| Warm replay read | 3.008 | 8.179 |

The native transition, projection and encoding components together consumed
less than one second per native family. Python lanes and repeated full
admission were the dominant avoidable costs. Native lanes previously shared
the learner's process/GIL; inference-forward totals were 19.677s versus 8.657s
on the same 446 positions. Contention is a hypothesis to test with the new
process layout, not a measured isolated GIL speedup. One historical table used
the qualified whole-table reference fallback; its cost remains in comparisons.

The eight matched 4s/K16 teacher turns passed with maximum post-work 0.297s.
Frozen-row batch verification took 0.514s versus 1.638s for streamed checks;
one-use reuse took another 0.331s. Streaming reduces the final pause but does
not establish lower total verification cost. B07 preserves the existing bounded
teacher verifier; it does not weaken checks to force a passing timing result.
B06 therefore passed measurement/correctness, **not useful performance adoption**.

## Collection and validation changes

`lab_pool.NativePool` retains the existing frozen planner, ready-branch scheduler,
CPU admission, model cohorts, inference service and reference fallback. Actors
now run in spawned processes with the configured lanes per process; up to two
separate writers own publication. Actor processes match the reference pool's active width.
The saved 60-slot cap is shared by actors, writer and preparation readers.
Board histories pass as bounded file/hash handles, not learner-process IPC
payloads. Completed families continue streaming into existing preparation.
Branch checkpoints remain immediate, bounded messages; no sibling waits in a lane.

Publication calls the complete native family/admission validator once. It no
longer writes and validates an intermediate `FamilyStore` archive before
repeating the same admission. Input frames also reuse the reference actor's
immutable `BOUNDARY.admit` path for sampling/encoding. No policy, RNG, feature,
reward, auxiliary-target or Rust engine equation changes occur.

The independent writer adds a validation receipt bound to current implementation,
replay hash, full admission/job binding, family seal and feature-cache hashes.
Normal ingestion verifies those exact bytes and bindings, then reconstructs
checked games without replaying behavioral/RNG validation. All in-memory row
and metadata seals remain enforced. No extra duplicate board cache is written.
The loaded implementation fingerprint is computed once per process; deployments
start fresh workers/readers. Replay and feature contents are hashed on every
read, never accepted through an mtime/size shortcut.
Old-source receipts or missing derived features use full validation. Changed
bytes/receipts fail. A restart's on-disk `.record.json` is **not** independent
authority: `Publisher.load` performs full validation against the scheduler's
frozen job before returning it. Diagnostic data remains excluded from training.

Failures preserve the original error and abort shared CPU admission before
joining workers. Duplicate/stale branch messages also cancel before executor
shutdown. Partial/unsealed parts never train; a replay published immediately
before a receipt-write crash can be revalidated and recovered. Retention handles
the nested native replay directory and pins current v4 feature sidecars plus
backend rollback dependencies.

The local fixture comparison is reproducible with
`python scripts/rust_lab_b07_publication.py --out artifacts/b07-publication.json`.
It compares three alternating batches of three authored families, including
historical and V4 fallback, and checks equal prepared tensors/targets. Its
receipt in the bundle is a CPU component measurement, not a GPU throughput claim.
The first check exposed repeated source hashing; caching only the process's
loaded-source identity removed that overhead. The final small-fixture medians
were 0.778s baseline versus 0.821s for publication plus first read, and 0.274s
versus 0.231s for warm reads. Full validation passes fell from three to one per
family (warm reads: one to zero), with equal outputs. This tiny fixture shows
the reduced validation work and a faster warm read, but **does not establish
end-to-end gain**. The completed matched lab comparison failed the useful-gain adoption gate.

## Compact ordinary transport (R3)

The existing typed Rust state, candidate list and feature encoder remain the
sources of truth. `append_compact` sends the selected semantic action ID,
revision/token, feature context, sampling behavior and evidence. Rust reconstructs
the row from its held decision; it rejects illegal IDs, stale revisions, foreign
models/contexts and invalid forced probabilities. The full ingress validator
still reproduces policy probabilities/RNG and verifies complete training data.
The old wire operations remain diagnostic controls only.

`feature_arrays` sends all 3,128 state floats, N×327 action floats and four i64
seat indices in one little-endian binary reply. Metadata binds token, revision,
model/context, ordered action IDs, schema/layout and exact sizes. Python views
these arrays without float-list construction and exposes them read-only. Frames
are bounded (4 MiB JSON, 16 MiB binary); truncated, oversized or foreign replies
retire the lane. Singleton decisions skip unused actor tensor transfer; the
existing publication validator still produces/checks their training features.

After an action Rust previews up to 16 consecutive singleton decisions, bounded
by 2 MiB and the reference collector's actor, era, round, turn, turn-owner and
non-debt guards. It stops before a meaningful choice, preserving every branch
opportunity. Python runs the existing sampler on each preview in original order,
including both v3 RNG draws and action counters, then acknowledges the complete
batch. Staged native transitions are reused. Pending records cannot checkpoint,
finish or export; cancellation retires them, and restart uses the normal frozen
job/revalidated-family recovery. In-flight IPC polls cancellation every 50 ms.
Oracle-per-step diagnostics disable staging so their live-state comparisons stay
valid; production and matched ordinary benchmarks enable it.

Rust serializes each accepted record/evidence once, retains those exact bytes,
hashes the final arrays incrementally, and exports at most 64 rows / about 2 MiB
per binary batch. Python verifies offsets, identity, lengths and complete-array
hashes before decoding. No escaped JSON strings or per-row round trips are
needed. Temporary parts and replay reads hash the same compressed bytes that
are decoded, avoiding extra file passes. Full first admission, atomic family
publication, cache hashes and cold recovery validation remain in place.

`rust_lab_b07_transport.py` is a short wire-only comparison with identical frozen
origins and the real versioned Python sampler, including a 413-candidate state
and a late-rail completion. It checks every record, RNG state, eligible branch
checkpoint, transferred feature tensor, scoring result and export hash. This is
not a checkpoint inference benchmark. The separate lab ordinary gate uses the
same actual g1250 checkpoint/jobs in both reference/native arms. Use
`--ordinary-only` to measure that scope without unrelated teacher/update work.

Local wire medians (two alternating matched samples, 24 decisions including
12 forced rows, starting at 413 legal candidates):

| Component | Previous wire | Compact/binary wire |
| --- | ---: | ---: |
| Startup | 0.108s | 0.106s |
| Collection loop | 1.213s | 0.401s |
| Feature transfer/conversion | 0.715s | 0.057s |
| Action acknowledgements | 0.409s | 0.260s |
| Request serialization | 0.040s | 0.005s |
| Reply JSON decoding | 0.453s | 0.056s |
| Pipe calls | 119 | 50 |
| Request bytes | 1,946,375 | 98,745 |
| Reply bytes | 10,169,707 | 8,198,752 |

Timings overlap. These are small CPU transport samples, not NN strength or
whole-training speed measurements. A separate two-row late-rail completion
checks export/reward seals; the Rust complete-game regression crosses multiple
64-row export batches. Remaining costs include rich decision-view JSON,
full independent publication/feature-cache validation, process startup and
GPU-service orchestration. The lab report must determine their current relative
cost and the total accepted-work gain. g1250 weights are not present locally;
no different checkpoint is substituted for that qualification.

## Bounded lab qualification

The bundle pins the original g1250 checkpoint
`de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`
and exact config `895d866502425dad30f2277b822dba2a15247f9a701d2dd5dd10b7e396de3ca8`.
Its source review records every changed learning dependency relative to B06.
Existing restricted config/source transitions are retained, not broadened.
R3 rebuilds the ordinary adapter against the unchanged B01 core and the teacher
with exact float decoding against its unchanged overlay. Original C# pins and
B01-B06 evidence are reused. New ordinary producer/build pins bind the new data.

1. Short CPU regressions cover admission equality, branch targets, cache/source
   invalidation, worker/writer/reader errors, stale/duplicate/canceled teacher
   replies, full-state adoption failure/recovery and rollback integrity.
2. Remeasure the affected ordinary scope against reference: at most three frozen parents plus
   natural branches, current execution configuration and neutral recipe, no
   forced fork. Measure startup, publication, reads, preparation, GPU queues,
   process CPU/memory and accepted work separately. Watchdog: 360s.
3. Two independent shadow passes, at most two all-seat tables per pass (train
   and held-out), frozen g1250 and w=0. Full turns use search_v1, 4s/K16, two
   independent one-thread sessions; the existing all-seat fast turns remain
   0.8s/K4. Original 1s/0.2s post caps include verification and sealing. The
   slower table's policy-sealing cost is conservatively charged to each final
   turn before its value sealing. Reload both buffers and a shadow checkpoint.
   No teacher, search-head, PPO or PPG update in these passes. Watchdog: 210s each.
4. Full diagnostic checkpoint/Adam/RNG/dependency restore and the existing short
   authored-input current-recipe PPO/PPG component commit/reload. Zero new games
   here; this is not a production iteration. Watchdog: 180s.

Each attempt exports partial failures or success under
`results/rust-lab-qualification/`. The two shadow passes have fresh verifiers,
processes and epochs; frozen ordinary data is reused. The exact g1250 null
utility receipt is allowed only by the user's existing shadow authorization.
It stays null in production; other missing readiness defers the teacher scope.

`passed` records bounded correctness. `adoption_ready` additionally requires
equal accepted work and measured improvement versus the current reference arm.
No whole-iteration speedup, saturated-worker scaling, strength or full 28+4
teacher schedule is inferred. Unobserved natural branches stay a performance
coverage gap; earlier native branch correctness remains valid.

## Explicit backend adoption and rollback

`manage-rust-lab-b07.py adopt` operates under the existing run lock. It requires
the passing current-source B07 receipt, useful matched gain, and exact currently
committed generation/checkpoint/config. It copies the complete rollback state
and dependencies and restores PPO, PPG, companion Adams and RNG before touching
production state. The existing reviewed source-transition mechanism records the
source change. A separate operational `native_backend` selection routes ordinary
training only; champion/evaluation jobs remain on their existing reference pool.
Teacher mode must remain off. Defaults stay reference until this command runs.

The next normally authorized iteration writes backend identity into its full
checkpoint and report. `verify-next` checks actual native-family use, exactly
G+1, the state/checkpoint binding, and full isolated restart. This command does
not launch another training campaign. Interrupted collection can reuse fully
revalidated families; a cached family still counts as native work.

`reference` switches at the latest clean committed boundary and retains every
ordinary checkpoint, counter and compatible cache. It records the switch so a
pre-next-checkpoint mismatch cannot silently select a backend. A missing native
binary cannot block this rollback. `verify-rollback` separately restores the
saved full pre-adoption point and its exact source/config/dependencies without
rewinding current training. That byte-preserved point and source-transition
backups remain available for disaster recovery; it is not a weights-only reset.
Backend-only selection is stored in state and subsequent checkpoints, never
smuggled into a learning-config exception. Later teacher enablement remains a
separate explicit readiness/beta/PPO/PPG decision. DOMiNO/AZ_DB remain off.
