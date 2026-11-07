# B07: cheaper native collection and bounded adoption

Updated on 2026-11-07. **R8 lab qualification, both shadow canaries, diagnostic
continuation, Rust adoption and full rollback restoration passed on g1250;
the real g1250→g1251 commit and full restart now passed on the lab.** Use the
[normal g1251 continuation command](../distribution/rust-lab-b07-g1250/README.md#continue-from-g1251-after-qualificationadoption-finishes);
do not repeat passed qualifications. The installed game is unchanged.

## Lab g1250 → g1251 acceptance

Pulled inside `results/` to `c51c7d9`, export
`20261107T134344148855Z-20261107T133341969715Z-b07-continue-cf42c992-3ed81ebb`.
Its [production receipt](../results/rust-lab-qualification/20261107T134344148855Z-20261107T133341969715Z-b07-continue-cf42c992-3ed81ebb/evidence/qualification.json)
reports `passed=true`, Rust active, teacher off and generation 1251. The real
trainer prepared 320 shards, performed 52 PPO steps and 48 accepted value steps,
and committed checkpoint
`a6ccace46d17b14751086d1715a70543355b9a6e6a89fa55eb080917882adf27`.
The next-checkpoint verifier checked actual native collection and persisted
backend identity, then restored the complete new checkpoint and dependencies
in isolation, including all four optimizers, CUDA RNG and PPG ages 1250/1251.
The saved pre-adoption g1250 rollback also passed its separate full restore.

All eight exported files match their recorded byte counts and SHA256 bindings,
including the decompressed diagnostic fork. This latest export contains no
failure traceback. It closes B07's teacher-off production backend migration and
first normal commit/restart gate. Production teaching and later strength or
throughput campaigns retain their separate gates. These are evidence and status
updates only: execution source remains
`94a0e5d1d03210ad93759f2aedd6f7f15cca4bb826300df7b6de04f033199ff2`.
No local tests, benchmarks, training or new archive were needed for this review.

## Automatic training startup follow-up

Pulled results inside `results/` to `a679f02`, export
`20261107T131709467058Z-20261107T131647644898Z-b07-continue-953a2157-2b90dbc8`.
The retry passed rollback verification, then stopped in the automatic reward
scheduler's source precheck before entering the inner trainer. The preceding
repair applied the exact source-compatibility check only inside that trainer;
it missed the earlier check in `automatic_training.run`.

Apply the same `training_signature` helper before the scheduler's existing
`check_signature`. The full execution-source review includes these two added
lines; its qualified source, same-config restriction and frozen job/receipt
bindings stay unchanged. No general `--resume-reviewed-source` override is
needed, and unknown execution changes or recipe edits still fail. The scheduler
retains its normal automatic-stage validation. This attempt collected no new
games or updates; the earlier pending iteration remains available for retry.

Regression coverage now follows the continuation launcher's command through the
real CLI, config validation, run lock, generation guard and automatic scheduler,
stopping before model loading/collection. It also exercises repeated startup,
stale generations, changed recipes, unknown execution (including an attempted
override), missing backend selection and reference startup, checking that state
and pending replay bytes stay unchanged and the lock is released. Run the same
g1250 continuation command after syncing Git; do not repeat passing qualifications
or extract an older ZIP. That retry subsequently passed; see the acceptance above.

All **11 affected debug checks passed in 28.291s** (seven startup checks plus
four source/reload binding checks). No models, games, updates or benchmarks ran.

## Git-only PPG replay-location fix

Results pulled inside `results/` to `f08a704`, export
`20261107T125446333545Z-20261107T124611526805Z-b07-continue-e84bd233-c1155826`.
The imitation dependency fix worked. The real run prepared all 320 shards, then
stopped before PPO at `PPG cache needs current ordinary replays`. The PPG source
builder accepted only files directly under `iteration-1250`; Rust's publisher
uses `iteration-1250/native-ordinary/replay/<frozen-job-id>.jsonl.gz`. Earlier
critic-cache tests used the old direct layout and missed this integration gap.

The shared PPG source builder now accepts that exact nested native layout with
ordinary admission, matching generation, model, reward, job, replay hash, feature
receipt and frozen-job filename. Other nested directories and stale generations
remain rejected. Project confinement and link rejection stay in place. The older
disk buffer's cold rebuild uses the same path checks. Both PPG cache types retain
the complete original native admission and encoded-feature receipts, without
moving replays or rereading/encoding families at source-receipt creation.

An explicit `b07-replay-location-review.json` binds the complete new execution
fingerprint to the previously qualified source. Only that exact code revision,
the same teacher-off config/data/opponents and an existing native-backend state
may preserve the frozen collection signature on retry. Unknown source changes
fail closed. This keeps all collected job IDs, native receipts and checkpoint
bindings unchanged; the normal cold admission still checks their bytes. The full
execution fingerprint and review digest are recorded separately in launcher
evidence and the next native checkpoint/report, and checked at reload. The
ordinary/search implementation, sampling, features, rewards and optimizer are
unchanged; the review covers the path repair and its retry integration only.

Retry with the same `continue-rust-lab-b07.py --expected-generation 1250` command
after pulling Git. Keep the pending iteration and original qualification/build
directories; do not delete, rename or rewrite its collected shards. Passing
qualification/adoption is reused. No archive is created. This repair and the
startup follow-up subsequently passed the real commit/restart gate above.

Thirty-nine focused debug checks passed across the affected suites, including
actual native-layout PPG staging with old/new critic ages, cold feature/target
rebuild and byte-tamper rejection, path/generation/model/reward guards, full
adoption/rollback restoration, unchanged frozen-signature retry, unknown-source
rejection and launcher/receipt reuse. Fixture-only failures during development
were corrected; only those failing checks were repeated. No benchmark, complete
game, training iteration or unrelated lab qualification ran locally.

## Git-only continuation fix: preserve the completed imitation dependency

Pulled inside `results/` to `e92a29e`, export
`20261107T123415579666Z-20261107T122742698740Z-b07-continue-537723a8-89005841`.
The remaining qualification returned `passed=true` and `adoption_ready=true`.
Production selected Rust with teaching off and verified the complete rollback
copy. The ordinary trainer then stopped before collection with
`Latest-model recipe requires the pinned completed --imitation-run`.
The launcher carried the committed config but omitted that CLI dependency.

The launcher now derives `--imitation-run` from the committed signature's
`imitation_source.bc_run`, checks the required recipe/source binding before any
child process, and preserves it across adoption. It uses the existing project
relocation resolver. The real trainer retains full IL lineage validation and
strict self-play checkpoint/optimizer loading; the launcher adds no model load,
initialization or source override. It records the exact training command.

Twelve focused debug regressions passed, including the actual saved g1250 recipe
through the real trainer argument parser, retry after adoption, verification-only
after g1251, malformed/missing/conflicting dependencies, relocation and failure
without a new commit. No benchmarks, games or training ran locally.

This changes the launcher, tests, documentation and their tracked manifest hashes.
The training implementation remains
`9efcf34f853d75ab844366b6819da4b0921be38d3e2dfb806cf282ace76e7a30`;
native binaries and passing qualification receipts remain valid. There is **no
new archive**. Pull the code repository and run the same continuation command;
the selected backend skips qualification/adoption and retries the one real
iteration. Do not extract an old ZIP over the updated checkout. Subsequent
g1251 training commands must also carry the pinned imitation dependency.

## R8: one snapshot per checkpoint commit

Pulled inside `results/` to `0a86890`. Export
`20261107T121012269722Z-20261107T121006613134Z-g1250-b07-70687251-d0dc643a`
failed in fixtures, before any native build, game or training. R7 strict checkpoint
loading succeeded; the fixture then passed its checkpoint directory to ValueBuffer,
which correctly rejected the unrelated `shadow.pt`. Keep the unknown-file check;
use a dedicated values subdirectory and test both successful reload and rejection
of a foreign file in that directory.

Inspection found that the trainer already saves `search_values.payload(G+1)` in
a local variable, but `committed(state)` called `payload()` again. That repeated
the full-history JSON detachment and companion/receipt copying. R8 retains the
first detached payload and its integrity stamp privately in SearchValueIteration.
The existing trainer saves that object; state commit verification rehashes that
same object against both its original stamp and the durable state binding. It
never reconstructs labels from live state during verification. Repeated payload
requests return the same object after integrity verification. This removes the
second JSON conversion and companion snapshot; compact JSON also avoids unused
whitespace in the one required conversion. Integrity hashing is deliberately
retained, including the initial mutation-detection binding; no wall-time speedup
is claimed without measurement.

The payload is detached from live age metadata, immutable label trees, companion
Adam state and buffer receipts. NaN and both infinities still fail conversion.
A failed snapshot conversion leaves no prepared publication; a failed save or
state check retains the exact snapshot for retry. Admit, prepare-PPG and phase
updates cannot run after publication begins. Successful commit advances generation
and clears the snapshot; duplicate/stale commits fail. A restarted process uses
the unchanged strict checkpoint/state identity checks and constructs a fresh
snapshot for the next generation. Checkpoint format, labels, weights_only=True,
search_v1, deadlines, rewards, recipe, shard integrity and rollback rules are
unchanged. The companion optimizer remains owned by the existing lifecycle.

**34 distinct focused debugging checks passed** (6 targeted checks, then the
33-check lab fixture set; five overlap). No benchmarks ran. Coverage includes
real strict save/load, conversion count exactly one,
object identity, repeated calls, failed-save retry without partial files, next
commit/restart, three-generation head/Adam restoration, nonfinite values, payload
and live-state mutation isolation, forged state stamps, stage/reward/receipt
mutation, pending-update rejection and foreign-file rejection. New tests are in
`test_b07_checkpoint.py`; lab fixture selection includes them. No local benchmark,
full game/training run or unrelated successful qualification is repeated.

The R8 source review changes only search-value phase/iteration publication relative
to R7. Keep explicit same-config continuation approvals and exact native receipt
pins. Reuse the R6 teacher binary/tests and R4 ordinary result; the R7 fixture
failure provides no successful canary to reuse. The [R8 bundle commands](../distribution/rust-lab-b07-g1250/README.md)
run only remaining gates, with adoption and g1251 continuation commands retained.
Those lab qualification/adoption gates and the real next checkpoint/restart
subsequently passed; see the acceptance above.

## R7: strict checkpoint reload repair

Pulled inside `results/` to `06baece`. Latest export:
`20261107T120105221052Z-20261107T115808701063Z-b07-continue-1128185f-142111ac`.
Both tables and their sealing completed. Train policy/value sealing measured
0.088712s + 0.066669s; heldout 0.048026s + 0.038263s. The next step, loading
`shadow.pt`, failed with `Unsupported global: GLOBAL dict` under weights-only
loading. R6's immutable rows survived `deepcopy` into `label_ages`; their custom
pickle reducers emitted constructor globals. No production training started.

R7 converts only the label-age JSON subtree to detached plain containers in
`SearchValuePhase.payload()`. This common checkpoint boundary covers shadow and
future production search-value payloads. It preserves row values and rejects
nonfinite JSON; companion tensors/optimizer state keep their existing serialization.
No changes to strict `weights_only=True`, no allowlisting, no arbitrary-pickle
fallback. Live immutable reuse and per-turn deadlines remain unchanged. Conversion
is checkpoint publication work and does not repeat admission or search.

New lab fixtures cover real checkpoint save/strict reload with frozen rows,
plain nested container types, identical labels, restored phase payloads and
mutation detachment; native-build reuse tests cover original receipt pinning and
tampering. **No local tests or benchmarks run, per user request.** These new checks
remain unexecuted locally. The first lab fixture stage runs them before games.

Only one runtime source file changed from R6; explicit source/resume reviews
preserve the same saved configuration and all prior restrictions. Reuse the exact
R6 native teacher executable/test/build receipts and passing R4 ordinary evidence.
The failed canary is not counted as passed; rerun pending canaries and continuation.
New immutable R7 bundle/commands are in `distribution/rust-lab-b07-g1250/README.md`.
Lab qualification, adoption and real next-generation commit remain Pending.

## R6: fast post-work and sealing repair

Pulled inside `results/` to `d9bb26c`. The latest export is
`20261107T112657686949Z-20261107T112405895418Z-b07-continue-d8c78a96-64e2b7c9`.
R5 completed both canary tables, but first-table value sealing took 0.258618s,
policy sealing 0.226983s, and native post-work 0.016149s: 0.501750s against the
old 0.2s fast cap. No training or production adoption started. Raising the cap
alone to 0.5s would still miss that recorded total, so R6 also removes the
measured repeated-history-copy bottleneck.

- Detach validated raw and admitted label trees once using the existing immutable
  JSON containers inside the original per-turn verification deadline. Reuse them
  through policy and value publication. Feature arrays still get detached copies;
  all sealed hashes, cold-reader validation, target and reward checks remain.
- Allow the exact live immutable objects to cross the one-use cache by identity.
  Plain reloaded/copied objects retain full canonical-byte comparison, including
  numeric type differences. Model/job identity, completeness and cancellation
  remain checked. A late/canceled final cache publication is retired.
- Charge each canary table's own policy sealing before its value sealing; do not
  charge another table's larger cost. Missing, negative, nonfinite or boolean
  accounting entries are rejected. Late publication still fails qualification.
- Fast turns now permit **0.8s/K4 search + at most 0.5s cumulative post-work**,
  per user instruction. Python config/verifier/sealing and Rust job/deadline
  validation agree. Full turns remain **4s/K16 + 1s**. Search admission, action-two
  absolute cutoffs, RNG and search_v1 equations are unchanged.

Two AB/BA pairs replayed the exact two failed lab tables through real encoders,
policy/value publication and cold readers, without new games or training:

| Local component measurement, sum of two tables | R5 median | R6 median |
| --- | ---: | ---: |
| Snapshot detachment during turns | 0.367s | 0.696s |
| Deferred finalization | 1.734s | 0.811s |
| Detachment + finalization | 2.101s | 1.507s |

Deferred finalization fell **53.2%**; including the additional detachment cost,
these measured components fell **28.3%**. All policy/value rows, targets and
cold-read features matched. Both R6 pairs fit their revised caps in this local
component probe; R5 overruns are retained. Model fingerprinting and independent
NN verification were excluded from both arms, and cold-reader correctness was
checked outside the timed finalization. This is neither an end-to-end lab deadline
pass nor a whole-training throughput claim. Remaining cost includes immutable
snapshot construction, required encoding and durable shard compression/integrity.
The probe is `scripts/rust_lab_b07_sealing.py`; shipped evidence is `r6-sealing.json`.

R6 reuses only the original byte-pinned passing R4 ordinary executable and result.
Its source review enumerates exactly the changed teacher Python files and Rust
teacher-only module/tests; ordinary compiled inputs and execution are unchanged.
Rebound receipts retain the original producer/source and receipt pins. The old
teacher cannot be reused because its fast deadline differs: rebuild and run its
short Rust regressions before pending canaries. R5 has no successful completion
phase to reuse; any later R6 retry can reuse byte-bound successful phases of the
same workflow. No B01-B06 or passing ordinary benchmark is repeated.

Local validation: 32 distinct focused Python regressions and 34 Rust teacher/wire
regressions, including absolute deadline edges, narrowing, overflow, cancellation,
immutable detachment, strict reload fallback, cold replay equality, stale/tampered
source/build/phase receipts, teacher rebuild selection, and exactly-once training
orchestration. Source approval covers the unchanged g1250 recipe and rejects an
unrelated recipe change. The g1251 continuation CLI is parsed by a regression and
its g1247 config is verified equal to the saved config. The user then stopped local testing. Fresh-extraction execution is Pending on
the lab; archive construction records input hashes without running more tests.

The [R6 commands](../distribution/rust-lab-b07-g1250/README.md) finish remaining
qualification/adoption and g1250→g1251; then the normal trainer continues g1251→g1252.
Production teaching stays off. Full-scale teacher publication and Linux/CUDA
canary/adoption/next-commit success remain lab gates, not locally claimed passes.

## R5: finish qualification and continue the real run

Pulled **inside `results/`**, from `e71dc25` to `9e219a2`. Passing ordinary export:
`rust-lab-qualification/20261107T110307031706Z-20261107T110018084733Z-g1250-b07-49ed6ce1-fc892c78`.
All three frozen strata (self-play, historical NN, historical V4) matched:
1,161 decisions, 675 forced rows, exact discrete trajectories and prepared
features/targets. GPU-derived probability maximum error was 8.234e-7 absolute,
3.703e-6 relative, within the documented benchmark tolerance. No natural branch
occurred in this sample; branch correctness retains its earlier focused evidence.

| Bounded matched measurement | Reference | R4 Rust |
| --- | ---: | ---: |
| Collection including publication | 27.616s | 23.614s |
| First replay read | 2.430s | 2.613s |
| First preparation | 0.586s | 0.592s |
| Total accepted work | 30.632s | 26.820s |
| Warm replay read | 2.907s | 1.998s |
| Sampled process-tree CPU | 102.97s | 85.25s |
| Sampled peak process-tree RSS | 4.917GB | 7.479GB |

Accepted work is **12.4% lower wall time (1.142× throughput)** in this one
bounded comparison; collection is 14.5% lower. This closes the useful-gain gate
for this scope, not a statistical or whole-iteration performance claim. First
read and peak host memory remain tradeoffs. Do not repeat the passing ordinary
batch merely to complete adoption.

R5 leaves every `implementation_sources()` byte, runtime/NN/config/optimizer,
native binary input, benchmark equation, C# V4 policy and search_v1 unchanged
from R4 (`12d996a03debe816d1c3661872faeafa9e79c6ea97646d83cf19393dd57bd5ba`).
Only qualification/management scripts, their focused regressions and documentation
change. The immutable R4 archive and passing evidence are pinned inside R5.

`qualify-rust-lab-b07.py --complete-remaining` verifies the original g1250
state/checkpoint/config/source, population/runtime pins and ordinary build,
reuses its passing fixtures/wire/V4/ordinary scope, and runs only two existing
two-table shadow passes plus the bounded full-state continuation/update/reload
check. The repaired teacher build is reused if supplied and exact; otherwise it
is built/tested once. Shadow teaching stays at w=0 with its g1250-only null-receipt
authorization. Production teaching stays off. Search remains 4s/K16 plus <=1s
post-work, with two independent one-thread sessions. Each canary retains its
210s watchdog; continuation retains 180s. `--resume-from` accepts only this exact
workflow, committed-state identity and byte-pinned successful phase receipts;
an interrupted later phase never authorizes incomplete scope or reruns an earlier pass.

`continue-rust-lab-b07.py --expected-generation 1250` performs those remaining
gates, calls the existing locked backend adoption, verifies the full rollback
copy, writes the **actual committed config** to the new evidence directory,
and invokes the existing `step5.py train --iterations 1 --expected-generation 1250`.
It uses the same restart run, full model/Adam/RNG/counters/PPG ages/local-scoring
state and saved 60-worker cap. No latest-config substitution, fresh initialization,
data-recipe change or new source waiver occurs. Rust becomes the persisted
ordinary training backend; evaluation remains on its established path.

After the real iteration, `verify-next` checks g1251 generation/model, Rust
selection in the checkpoint and report, qualification identity, actual native
family use and full isolated checkpoint/optimizer/dependency restart. The command
then exports the combined result under `results/rust-lab-qualification/`.
This one iteration is real training, not a shortened smoke or a benchmark.
It can take the normal iteration time; local development does not run it.
Re-running the same G command after G+1 committed performs verification only;
it cannot launch G+2. Interruption after adoption resumes the selected Rust
backend through the normal trainer's pending-work recovery. A failed qualification
never adopts or starts training. `reference` remains the explicit progress-preserving
fallback; the wrapper will not override a deliberate reference switch.

Local regressions cover source/state/build/phase tampering, failed-phase reuse,
the CLI's actual phase selection, exact committed-config forwarding, failure
before adoption/during training, interrupted adoption, and no duplicate iteration.
The existing real full-state adoption/restore/reference/rollback fixture checks
are also run. These are development checks, not claims of production lab adoption.

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
   0.8s/K4. Current 1s/0.5s post caps include verification and sealing (R6 fast-cap revision). The
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
