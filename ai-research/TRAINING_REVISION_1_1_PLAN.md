# Training fixes after the results 1.1 test

30 September 2026. This review covers learning quality, demonstration generation, CPU and GPU throughput, artifact retention, and evaluation cost for Step 5 and the additional R4/R5 experiments. **The C0–C5 code/config revision is implemented locally; lab calibration, throughput and playing-strength acceptance remain external measurements.** The supplied 1.1 run validates the CUDA integration path but is too small to judge strength. See the [conda guide](../docs/TRAINING_RECOVERY_LAB.md) for recipe 2.1, new search identities, fresh-run commands and safe cleanup, and [exit checks](../docs/TRAINING_REVISION_1_1_EXIT_CHECKS.md) for evidence and limits. The review below preserves the measured diagnosis and proposed acceptance criteria.

Use the existing conda environment `ai-training`. The laboratory project root remains `/data/not_backed_up/yxu209/ai-training/`. Preserve `results/1.0` and `results/1.1` as review evidence. Retention should operate on explicitly owned lab run directories under that root, with dependency checks before deletion. Following the user's additional request, use a **one-second lab v3 search target with more CPU resources per search**, separately calibrated from laptop play. The work and quality comparison below determines whether that shorter clock can replace the three-second reference.

This plan extends the implemented [learning recovery plan](TRAINING_RECOVERY_PLAN.md), [throughput plan](TRAINING_THROUGHPUT_PLAN.md), and [recovery lab guide](../docs/TRAINING_RECOVERY_LAB.md). Four-player rules, complete legal choices, strict private observations, the qualified current-v3 bank, and separate search promotion gates remain requirements.

## What the supplied run actually tested

The [integration runner](../tests/Recovery/integration.py) intentionally overrides the production settings. Its workload matches [exit.json](../results/1.1/exit.json) and the saved [league state](../results/1.1/league/state.json):

| Measurement | Results 1.1 | Production default at the time of results 1.1 |
| --- | --- | --- |
| Network | Small, 473,229 parameters | Medium, approximately 1.90M parameters |
| Demonstrations | One training game and one validation game | Initial budget of 1,000 games with setup-based partitions |
| Imitation training | One epoch, six optimizer steps | At least three epochs; up to ten with early stopping |
| PPO collection | Two rounds of two games; four training games total | 320 games per iteration |
| PPO effective batch and microbatch | 64 and 8 | 2,048 and 512, bounded by candidate cells |
| Ordinary live game capacity | Two processes with one host each | Eight processes with four hosts each, up to 32 games |
| Legacy live game capacity | One private lane | Eight private lanes |
| Evaluation | One setup, four seat rotations, one neural lineup | Development and full panels are considerably larger |
| Search teachers | Disabled in this test | S2 after 640 games, then every 1,600 games |

The test reports a successful restart at iteration two, 20 then 14 PPO optimizer steps, no failed evaluation games, no expired/cancelled inference requests, and an unchanged source bank. These findings support pipeline correctness. They do not establish useful playing strength, successful S2/S3 distillation, or production-scale utilization.

The four evaluation games use the final checkpoint against the **fresh random initial network**, not against v3. All four share one setup seed. They are correlated seat rotations, not four independent strength trials.

## Training quality still needs work

The policy remains economically weak in this smoke run. The following values are from the [first collection report](../results/1.1/league/iteration-0000/report.json), [second collection report](../results/1.1/league/iteration-0001/report.json), and [post-update evaluation](../results/1.1/evaluation/report.json).

| Finding | Recorded evidence | Interpretation |
| --- | --- | --- |
| First collection | Eight learner seat trajectories averaged 11.625 VP | Incoming BC weights, before the first PPO update |
| Second collection | Four self-play seats scored zero; the learner against three v3 seats scored 9 VP versus 107–116 VP | Different setups and compositions; this alone does not prove an update caused collapse |
| Final evaluation | Mean 1.5 VP, zero wins, zero selected sales despite 11 recorded feasible-sale decisions | Very weak observed behavior, even against an untrained reference |
| Productive construction | First collection made 29 builds and four sell choices; second made 11 builds and no sell choices across its learner seats | Construction, financing and conversion still need joint evaluation |
| Forced choices | 362/639 and 217/390 learner decisions, about 57% and 56% | Much of the recorded work has no policy choice |
| Pottery | One learner Pottery build in the first collection and one in final evaluation | No evidence of a structural ban; far too little evidence of competent Pottery play |

The audited demonstration training game scored 83–125 VP across its four seats and contained 19 sell choices. The validation demonstration scored 87–117 VP and contained 18. That is useful economic behavior to learn, although v3 remains an imperfect teacher. Both tiny demo games contain only four of the five admitted profiles; Supplier is absent. A larger, balanced collection is necessary before judging the imitation approach.

### What the losses establish

The [BC report](../results/1.1/bc/report.json) has only one epoch. Meaningful-choice top-one agreement is 11.66% during training and 13.92% on the single validation game. Natural policy loss is 1.702 during training versus 1.664 on validation; balanced policy loss is 2.276 versus 2.267. Natural and balanced numbers use different weighting and must be compared within their own columns. Training metrics also combine predictions made during successive updates, while validation uses the resulting frozen weights.

There is no held-out learning curve and no evaluation of the BC checkpoint before PPO. We cannot diagnose overfitting or measure how much that single imitation epoch helped. Exact-choice agreement also penalizes legitimate alternatives; add action-family and build/sale completion diagnostics alongside it.

PPO now makes real effective-minibatch updates. Across its two epochs, combined value loss changes from 1.399 to 1.375 in round zero and 1.418 to 1.379 in round one. Final recorded KL changes are approximately `2.70e-5` and `2.19e-5`, below the `0.01` guard. Gradients remain finite, no gradient clipping is recorded, and the clipping fraction is zero. These logs show small policy movement, not an exploding-gradient failure. The near-zero actor-loss mean is not, by itself, evidence that the actor cannot learn.

The actor uses terminal win share minus its current value estimate. Much of this tiny collection is weak agents playing each other, where one seat still wins an unproductive game. Also, the current normalization and reported KL include forced choices: those choices have probability one and no policy gradient. This dilutes policy metrics and the actor contribution relative to the critic. It is a design issue worth addressing, but does not establish the cause of the low scores after four games.

### Learning revision

1. **Use a substantive warm start before judging self-play.** Start with a 200-game demonstration pilot, verify coverage and audited partitions, then extend to the existing 1,000-game budget if the data are useful. Use separate collection directories/specs and nonoverlapping setup seeds for additional data, then audit a fresh combined manifest: the current generator rejects a changed game count in an existing specification. Use the medium model for the primary run, three to five BC epochs initially, and the existing early-stopping mechanism. Reserve validation/test setups before training. Do not train on the four evaluation traces or qualification-only games.
2. **Check the BC checkpoint separately.** Run a small comparison against its unchanged initialization and a separate admitted-v3 probe. Require an improvement in productive builds, sales/flips, cash/income, and score margins before committing to a long run. Retain natural/balanced loss curves and meaningful-choice metrics by phase and era. Do not select only for exact imitation of a weak teacher.
3. **Keep the implemented curriculum initially.** The v3 table share remains 40% → 20% → 10% → 5%; historical neural and current self-play dominate later. Keep all five qualified training profiles and rotate seats. Retire very weak historical generations from the active opponent pool once better frozen generations exist, while preserving a few as diagnostic controls.
4. **Separate actor and critic sampling.** Normalize actor advantages and policy losses over nonforced learner choices. Report nonforced KL and entropy, root-action versus continuation metrics, and actual actor/critic sample counts. Keep four-seat value supervision at forced states, using a separate bounded critic sample or cached features. This changes update weighting and needs a versioned recipe and fresh optimizer moments.
5. **Measure before increasing learning rate.** Start with PPO LR `3e-5`, two epochs and the existing KL guard. After a meaningful BC warm start and several full rounds, compare one change at a time: effective batch 1,024 versus 2,048, then LR `1e-4` versus `3e-5` if policy movement remains tiny. Increasing the memory microbatch does not need to reduce optimizer-step frequency.
6. **Prevent forgetting only if observed.** The current `rehearsal_manifest` is null, so rehearsal is off even though the reports display a coefficient of 0.1. Log `rehearsal_active`, rows and steps explicitly. If post-BC economic competence decays, test the existing separate, decaying BC rehearsal loss against a no-rehearsal control. Expert actions must never enter PPO importance ratios.

Retain win share as the primary evaluation objective. If adequate BC and several thousand ordinary games still lead to unproductive self-play, run a separate, small reward ablation with a bounded VP/score-margin or causal progress term. The current VP head is auxiliary value supervision; it is not an actor reward for economic progress. Any new actor reward needs an explicit contract, weight, scale and exploit checks. Avoid automatic rewards for merely building a tile: they can encourage exactly the unflipped construction the user wants to eliminate.

Pottery Flex remains held out as an **opponent identity**. Pottery actions, opportunities and learner contexts remain in training. Check completed Pottery sales and their actual VP contribution, rather than imposing an industry build quota. Exact era-end tile scoring/waste is not yet reported; existing last-seen unflipped counts are proxies and must stay labeled as such.

## Where CPU and GPU time goes

### Measured bottlenecks

| Operation | Results 1.1 timing | Consequence |
| --- | --- | --- |
| Round zero collection | 61.34 seconds for two games | Includes startup, replay publication and strict replay ingestion |
| Round zero PPO | 25.71 seconds; preparation 24.40 seconds | Preparation is 94.9% of this update interval |
| Round one collection | 226.67 seconds for one ordinary and one v3 table | Ordinary and heavy collection currently run sequentially |
| Round one PPO | 14.99 seconds; preparation 14.41 seconds | Preparation is 96.1% of this update interval |
| PPO preparation per learner row | Approximately 38.2 and 37.0 ms | CPU preprocessing dwarfs the logged training work |
| PPO forward/backward totals | 0.486 and 0.221 seconds | Host elapsed intervals, not isolated CUDA kernel durations |
| Demonstration legacy planning | 187.76 and 187.70 seconds per game | Pure v3 planning is CPU work, with a frozen three-second whole-turn budget |
| Demonstration replay publication | 16.07 and 15.46 seconds per game | Includes full validation/encoding before compression, not just disk writing |
| Training replay publication | 11.71–15.09 seconds per game | A substantial repeat of earlier work |
| Final evaluation | 118.53 seconds for four games on two live hosts | Smoke execution settings; not the 32-host production preset |

These timers overlap or nest. Do not add `host.rpc.*`, stream waits, outer host calls and policy times together as if they were independent wall time. Likewise, asynchronous replay publication can overlap another game's execution. The reported PPO interval includes preparation; BC's `supervised.prepare` currently excludes dataset loading and encoding performed before its epoch function.

The final evaluation service processed 1,260 positions in 1,244 model batches: **1.013 positions per batch**, with a maximum of two. The logged CUDA forward interval averaged approximately 0.473 ms per batch, while queue waiting averaged 3.52 ms per request. Peak CUDA allocation was approximately 48.3 MiB and reservation 68 MiB. These are allocator statistics, with process-lifetime peaks; they are not GPU utilization percentages or a complete device-memory measurement.

Only two games were live, and a single game's next decision depends on its previous one. A configured batch cap of eight cannot create eight independent pending positions here. The traces do not measure the production model, 32-host capacity, CPU utilization percentage, or steady-state GPU activity. There is no supplied `nvidia-smi`/per-process CPU timeline, and host resident memory is null because the optional measurement dependency was absent.

### Repeated validation is the first fix

The current path revalidates or re-encodes a decision in [remote inference](../src/brass_ai/recovery/inference.py), [replay publication and ingestion](../src/brass_ai/learning/replay.py), and [PPO/BC preparation](../src/brass_ai/recovery/updates.py). Public history is repeated in every view. The training demonstration has 354 decisions with histories growing to 353 events, producing approximately 28 MiB of row JSON before compression. Repeatedly walking all earlier history makes the work grow with the square of game length.

A local Windows check on 16 evenly spaced rows from the first training replay supports this diagnosis. A profiler attributed approximately 98% of default encoder time to schema validation. An additional unprofiled check took approximately 1.550 seconds with default validation and 0.026 seconds using the existing `validated=True` path on the same already-checked rows, with matching state/action arrays. This is a small local CPU check, **not a lab or whole-pipeline speedup claim**. It does not justify bypassing validation of new inputs.

Implement a trusted, immutable record boundary:

- Fully validate each new observation and its complete choices before granting a trusted handle. Bind it to source/runtime, game, seat, revision and input digest. Raw `RulesHost.observe()` currently returns decoded data; it does not itself validate the schema.
- Preserve full validation for foreign/imported replays and changed hashes. A source being local is insufficient to trust an arbitrary mutable dictionary.
- Encode a validated snapshot once and reuse those exact arrays for inference, replay verification and training. Use the prepared encoder internally after that boundary. Keep shape, finite-value, stable-seat mapping, choice-ID and response-attribution checks.
- Introduce a versioned, derived feature cache keyed by replay hash, game/decision identity, feature/encoder fingerprints and style/intent. BC uses neutral context; its cache must not silently reuse legacy post-choice context or a learner's differently conditioned features.
- Validate incrementally accumulated history only when its prefix is demonstrably unchanged. New/restored game branches need independent history state. Do not use a cache from another determinization or opponent observer.
- Preserve the current original replay as authoritative evidence. Cached features are derived data, can be regenerated, and must never contain privileged engine state.

This removes repeated work without weakening legal-choice or hidden-information contracts. After it lands, profile again before addressing smaller tensor or serialization costs.

### Demonstration generation and audit

The v3 planner dominates the measured demonstration time. Generating these games does not require neural GPU inference. Introduce a separately identified **one-second lab v3 teacher** using the budget calibration below. Preserve the original three-second bank and its reports as a control; admit the changed lab teacher using new evidence rather than overwriting its existing qualification. Retain v3's cash, conversion, merchant-beer and personality behavior.

Benchmark eight private game lanes with up to four search workers per active turn against sixteen game lanes with up to two search workers. Both use a global cap of 32 search workers on the 64-CPU allocation, reserving headroom for live rules hosts, feature preparation, IPC and the inference coordinator. Keep one tensor thread per actor, measure per-game plan work and deadlines, and reject configurations where contention materially reduces useful search. Report aggregate process counts as well as active CPU time: each lane may own several idle private planner hosts. Use whole-game parallelism first if the within-turn implementation has not landed; it does not automatically increase a particular turn's search work.

Parallelize the currently sequential transition audit across four to eight independent rules hosts. Each worker restores and checks a whole game; the coordinator merges results in deterministic order, verifies global duplicates and partitions, and commits one immutable manifest. Preserve exact observation/choice/terminal agreement. Auditing has little useful GPU work, so optimize audited games per hour rather than device occupancy.

With sufficient headroom, submit ordinary and legacy jobs to their separate persistent pools concurrently during RL collection. Freeze one incoming policy for the whole round and join both pools before updating it. This can feed the GPU while v3 thinks and eliminate the current idle gap; benchmark it against sequential collection because timed planner contention can alter choices. Fair search promotion remains isolated and uncontended.

### Separate lab and laptop search budgets

Separate **wall time**, **completed search work**, **CPU resources per turn**, and **concurrent games** in configuration and telemetry. More simulations/work allowance alone does not make computation faster. Additional cores help a turn when work is parallelized; additional game workers mainly increase dataset throughput. The supplied logs identify 64 available CPU processing units and the server GPU, but contain no measured comparison of server and laptop search speed.

The current [v3 planner](../src/BrassLegacyAI/Planner.cs) runs its proposal/beam expansion serially. Its profile stops at the clock or `work_limit`; the balanced profile permits up to 240 reconstructed worlds at 3,000 ms. The [legacy reply search](../src/BrassLegacyAI/ReplySearch.cs), neural S2 and R4/S3 simulation loops also perform their inner work sequentially. Their existing `workers` settings chiefly distribute different games. Setting a larger worker count or node cap therefore cannot, by itself, guarantee the same decisions in one second.

Implement explicit execution roles with separate identities:

| Role | Initial wall-time proposal per whole turn | CPU and work proposal | Purpose |
| --- | --- | --- | --- |
| Lab v3 demonstrations and occasional opponents | **1 second** | Start with 1, 2 and 4 private search workers; preserve the reference's useful candidate/beam/work coverage | Faster economic labels and curriculum opponents |
| Lab neural S2/R4/S3 generation | Test 1 second, then 3 seconds if required for complete confirmation | Up to 4 search workers; preserve proposal breadth, rollout horizon, sample coverage and fresh confirmation | Faster teachers and additional experiments; no fabricated labels on timeout |
| Laptop v3 play | Retain the 3-second reference initially | 1 worker initially; test 2 if Unity remains responsive | Player-facing behavior and latency calibrated on the actual laptop |
| Laptop neural play | Separately calibrated 1/3/10-second difficulty/thinking settings | Small measured CPU worker count; CPU or Intel GPU inference as previously planned | Runtime difficulty, strength and user waiting time |
| Formal search comparison | A declared equal wall-time and CPU-resource cap for all variants | One concurrent game; equal permitted within-turn resources | Distinguish algorithm gains from extra hardware allocation |

These are proposed role defaults and experiments, not measured equivalences. The current 30-second neural S2 teacher setting was chosen to investigate completed labels on the slower path. Reduce it after preprocessing and search execution improve, using the shortest lab cap that produces the required useful confirmations. Do not assume that moving this setting directly to one second preserves its teacher work.

Calibrate with a small shared set of early/late Canal and Rail positions, including setup actions, sales/merchant beer, cash pressure and two-action combinations. Capture the reference's **completed** worlds/plans, candidate-family coverage, reply choices, simulations, pilot/fresh samples, selected complete plan, and fallbacks. A node is not a fixed unit of CPU cost; rebuilding one world and simulating another can require very different work.

First test the existing serial planner at one second on the lab after the validation fixes. A stronger per-core CPU may already reach the useful reference work. Then test bounded parallel search where needed:

1. **v3 complete-turn planning:** parallelize independent first-action/continuation branches with private reconstructed hosts and a common incumbent. Start with separate processes so engine/global state cannot leak across branches. Merge complete legal plans by canonical signature. Preserve mandatory action families and enabling combinations. Parallelism must not let whichever branch finishes first monopolize the work budget or change style selection accidentally.
2. **S2 replies and finalist confirmation:** distribute independent plan/world evaluations with common scenario seeds. Admit a matched sample only when all compared finalists have completed it. Keep confirmation worlds fresh and separate from proposal/pilot work; aggregation must retain four independent seat utilities.
3. **R4/S3 trees:** start with private root searches/particles and merge compatible root statistics under the same information-set/action key. This is a separately identified parallel search variant, not an assertion that several independent trees equal one serial MCTS tree. Do not merge hidden information across observers or silently treat R4 visit counts as confirmed teachers.

Use persistent search workers to avoid launch time consuming the one-second turn. Give each worker a deterministic branch/scenario seed, one allowed observation context and a private rules host. A coordinator grants a bounded number of work tokens and enforces **one deadline covering both actions**, reconstruction, queue waits, inference, merging and validation. Reserve time to return a legal incumbent; action two receives the remaining allowance. Do not enable the existing `fixed_work=True` mode as a way to ignore the one-second clock: use it only for isolated work calibration with an external watchdog.

Compare one-second lab runs against the existing three-second bank on the lab as the teacher-quality control, on the same positions and then a compact four-player gameplay probe. Compare the deployment work/latency against three-second laptop traces separately when available. Exact move equality is not required when alternatives are equally credible, but productive builds/sales, cash safety, style behavior, useful plan coverage and teacher-confirmation rates must not deteriorate materially. Set the acceptance margin before comparison. Use the five admitted tactics for development comparisons and reserve Pottery Flex's opponent identity for a final transfer check after selecting the budget. Training still includes every industry, including Pottery.

A changed clock, parallel execution or work-allocation rule creates a new source/profile/bank identity and new demonstration provenance. Once accepted, make that lab identity the generation/curriculum default; retain the original three-second identity for historical benchmarks and laptop compatibility. If one second cannot deliver the useful reference coverage, choose between more per-search CPU resources with fewer simultaneous games and a slightly longer measured lab cap. Record the tradeoff openly rather than claiming equivalent search from the clock setting alone.

### RL execution and GPU feeding

Skip inference for a **plain neural policy's** single legal choice when only action selection is needed. Record probability one, the correct policy identity/context, and the complete original choice list. Preserve random-stream semantics or version the changed sampling behavior. Encode those states lazily for critic training. Do not apply this shortcut indiscriminately to legacy intent updates, search leaves, or cached search continuations that need their existing bookkeeping.

The current eight Python processes with four host threads each can contend on Python validation/encoding work. Compare **32 processes with one host each** against the existing eight-by-four arrangement at the same 32 live games, after removing duplicate work. Then test 16 and 48 ordinary live games only if the measured throughput curve warrants it. Private state and routing must remain per host/seat; extra processes are not permission to share an engine with policies.

Once preparation is smaller, improve batch packing and transfers: reuse bounded staging buffers, prefetch cached feature batches, and avoid repeatedly building the same target/index arrays. Benchmark larger microbatches at unchanged effective batch and optimizer semantics. Keep padded candidate cells bounded and measure useful versus padded work; never truncate legal choices to make a batch fit.

| Setting | Starting proposal | Comparison or limit |
| --- | --- | --- |
| Main model and games | Medium; 320 ordinary games per iteration | Keep the learning workload stable while measuring execution |
| Ordinary actors | Existing 8 × 4 versus 32 × 1 | Same seeds/population and 32 live games |
| Heavy actors | 8 games × 4 search workers versus 16 games × 2 | 32 global search-worker cap; new one-second lab identity after calibration |
| Inference batch cap | 64 initially | Try 128 only after enough independent requests exist |
| Flush delay | Test 0.5, 1 and 2 ms | Prefer lower latency for small pools; retain whole-turn deadlines |
| PPO effective batch | 2,048 initially | Separate learning comparison at 1,024 |
| PPO memory microbatch | 512, then 1,024 | Effective batch remains unchanged; candidate-cell cap still applies |
| Candidate-cell cap | 131,072 initially; compare 262,144 | Only after packing/preparation fixes and actual memory measurement |
| Precision | FP32 control | Existing BF16 option only as a measured capability/parity experiment |

Do not enlarge the network simply to fill a 90 GB GPU. Select settings using fresh games/hour, meaningful decisions/second, optimizer steps/hour, end-to-end iteration time, and search tail latency. Batch/worker scaling can increase throughput even when the small network still uses little VRAM.

Log dataset load, audit, schema checks, feature encoding, IPC wait, packing/transfer, model work, replay serialization/compression, collection tails and checkpoint commit separately. Add phase-scoped memory peaks, CPU process-tree sampling, inference-batch histograms, and a short CUDA profiler window. Warm-up and maximum call times must be visible: BC records 7.33 seconds over six update batches but a 0.023-second median, indicating a large outlier that the current percentile method hides. Its cause is unmeasured; initialization is a possibility, not an established result.

After preparation is fixed, review KL-probe cost. In the first smoke update, policy measurement took 0.789 seconds versus 0.486 for forward/backward. The existing fixed probe includes every row in these small rounds and runs after each optimizer step. Test a smaller, phase-balanced **nonforced** probe while retaining complete final policy-change measurement and a reliable stop guard. Do not simply disable the guard to improve throughput.

## Artifact retention and fewer files

The supplied folder contains **30 files, seven checkpoints and approximately 45.6 MiB**. Checkpoints occupy 30.9 MiB; replays 14.6 MiB. `bc/best.pt`, `bc/last.pt` and `bc/epoch-0001.pt` are byte-identical, so this single BC epoch already has three copies of one full checkpoint. Production currently retains every generation and every game shard indefinitely.

The four supplied training shards average approximately 1.438 MiB compressed. At that observed size, a 320-game round would create roughly 460 MiB of replay files before checkpoints and teachers. This is an illustration from small-model smoke games, not a forecast for trained play. With one shard per game, 100 such rounds would create at least 32,000 ordinary replay files and roughly 45 GiB at the same size.

The delivered retention manager is shared by BC, Step 5, evaluation and searchlab. It offers automatic cleanup **after a successful durable commit**, with an inspectable dry-run mode, a machine-readable quarantine manifest, exact restoration and a switch to disable pruning. These facilities were absent from the results 1.1 implementation.

### Keep files according to dependencies

| Artifact | Proposed default retention | Required protection |
| --- | --- | --- |
| Full RL resume checkpoints | Latest and one previous successful generation | Model, optimizer, RNG, counters and signature; checkpoint/state commit is complete |
| Historical neural opponents | Eight recent distinct weights plus four diverse/milestone anchors | Bounded active pool, selected BC origin and references used by unfinished jobs |
| Best and selected release models | Keep until explicitly retired | Evaluation/release links and qualification evidence |
| BC epochs | Latest full resume state and best candidate; remove other unpinned epochs | BC state pointers and downstream runs that use the chosen warm start |
| Raw ordinary PPO replays | Last four completed rounds, plus explicitly pinned datasets/teachers/evidence | No cleanup of an incomplete round; current PPO is on-policy and does not reuse all earlier shards |
| Demonstration dataset | Keep all shards referenced by the admitted manifest | Exact source hashes, setup partitions and transition-audit provenance |
| Derived encoded data | Active BC dataset and current/recent RL rounds, bounded by disk budget | Regenerable from retained immutable source; fingerprinted |
| Routine evaluation replays | Retain up to four representative complete-game shards and all failures; compact outcomes for all games | Mark pruned screens non-promotable and non-resumable from raw evidence |
| Formal promotion/search evidence | Keep full referenced raw records, source weights, gates and reports | Existing promotion rehashes raw replay evidence; aggregate scores are insufficient |
| Profiling and aborted temporary files | Keep the latest two traces; reclaim inactive old temporary files after a grace interval | Never remove a file owned by a live run/job |

Keep small reports, source/config/runtime fingerprints, seeds, parent model IDs, lineage, retention manifests and summary curves throughout a run. They must clearly distinguish archived metadata from weights/replays that remain available. The current `state.history` is also the sampling pool: deleting a historical checkpoint without changing that logic will break training. Split immutable lineage from a bounded active-opponent index first, and evict inactive CPU model copies, GPU cache entries and actor policy objects only after all jobs release them. The current GPU cache is bounded, but CPU model and per-lane policy registries are not.

Use full checkpoints for exact resume and **compatible weights-only snapshots** for most frozen opponents. A checkpoint file hash changes when optimizer data are removed even if the model ID is unchanged; write new path/hash entries atomically rather than modifying an existing pinned checkpoint. Keep at least two distinct PPO-trained references available for formal Step 5 qualification. A BC or random checkpoint cannot be relabeled as PPO-trained.

Reduce BC alias duplication. Resolve immutable best/last state pointers in new CLI helpers, or use read-only references to immutable files on a supported filesystem. Preserve compatibility for existing alias consumers until they migrate; downstream runs with a pinned `best.pt` must keep that exact file available.

### Group complete games into bounded shards

Replace one-file-per-game publication with approximately **eight complete games per shard**, bounded by 256 MiB of serialized source or a short flush interval. A 320-game round can then use roughly 40 replay files when size limits permit, instead of 320. Keep partitions separate where practical and index individual game IDs, setup seeds, provenance, hashes and shard membership.

The replay format already allows multiple whole games, but actor caching, audit records and result routing currently assume one game per actor path. Update those assumptions together. A job is acknowledged only when its containing shard and index are durable. A crash may require recollecting an uncommitted group; it must never create duplicate games, fake terminal targets, or reuse a partial shard. Bounded writer queues must apply backpressure and report serialization, validation and compression separately.

### Cleanup transaction

Resolve the run root and every proposed target inside `/data/not_backed_up/yxu209/ai-training/artifacts/` or its explicitly selected owned subdirectory. Reject symlink/reparse escapes, active locks and unresolved references. Never use a broad recursive pattern such as deleting every `.pt` file.

After a round commits, compute the reference graph, list deletions with reasons and bytes, and atomically write the retention decision. Initially move reclaimable files into a per-run quarantine for 24 hours; permanently remove them only after another successful commit or explicit cleanup. Configure the grace interval and account for quarantine in disk usage. Preserve exact paths for restoration. The first production cleanup should run in dry-run mode before enabling automatic application.

Do not prune `training/opponents`, shipped R2/R3 models, the private runtime, installed game files, user-provided `results`, or another run simply because they are large. Their ownership and qualification dependencies differ from temporary artifacts.

## Smaller and less frequent evaluation

The results 1.1 training loop did **not** automatically run evaluation after every iteration. Recipe 2.1 adds a small persistent scheduler and lighter named panels. Evaluation remains available as a separate command; it never contributes training samples or advances the ordinary-game curriculum.

| Purpose | Proposed games | Frequency at 320 ordinary games per iteration | Selection limits |
| --- | ---: | --- | --- |
| Existing collection diagnostics | Zero additional games | Every iteration | Incoming-policy diagnostics; all-current win share is mechanically 25% |
| BC acceptance | 16 neural games, then a separate 8-game admitted-v3 probe | Once per shortlisted BC candidate | Economic bootstrap check, not qualification |
| Routine RL monitor | **8 games:** two development setups × four seat rotations × one neural lineup | **Every five iterations**, or 1,600 new ordinary games | One stable reference, search off, neutral focus; cannot promote |
| Broader progress probe | **16 games:** two setups × four rotations × mixed-neural and mixed-v3 lineups | **Every twenty iterations**, or after a monitor warning | Admitted v3 only; no repeatedly tuned Pottery Flex holdout |
| Formal Step 5 shortlist | **480 games:** 30 independent setups × four rotations × four lineups | Only a selected milestone that merits qualification | Keep two trained references, uncertainty and Pottery gates |
| Extra confirmation | Current 64-setup panel, 1,024 games with four lineups | Only if the 30-setup result is uncertain or final validation warrants it | Final test stays untouched during recipe tuning |

This cuts ordinary development panels from the current 64–96 games to eight. It also reduces the proposed standard formal panel from 64 to 30 setups while retaining the existing minimum independent-setup requirement. These are different evidence levels: an eight-game monitor can detect obvious failures, but it cannot reliably rank close candidates. Compare against a stable anchor across several monitor windows and expand only when a decision needs more confidence.

At present, two neural references automatically add the `mixed-neural` lineup, and `include_legacy` adds another lineup. Passing `setups=2` alone does not guarantee eight games. Add an explicit lineup selector and compute/display the expected game count before execution. For routine monitoring, pass exactly one frozen reference and exclude legacy/search. For the 16-game broader probe, the new selector must omit the otherwise automatic extra neural lineup.

Persist the scheduler's last completed ordinary-game count, selected checkpoint hash, panel/seed specification and output reference. Resume must not repeat a completed evaluation, count evaluation games as training, or use a partial result for selection. Retire temporary evaluation outputs using the retention manager. Development screens must never become promotion-eligible merely because a reduced full-panel command happens to meet part of the old gate.

### Manual monitor

The following uses CLI overrides independently of the automatic scheduler. `MONITOR_CANDIDATE` must name an available completed checkpoint that differs from the reference; use `scripts/resolve_checkpoint.py` if retention has retired the full generation. Choose a fresh output directory each time.

```bash
conda activate ai-training
MONITOR_CANDIDATE=/data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-main/generation-0005.pt
MONITOR_REFERENCE=/data/not_backed_up/yxu209/ai-training/artifacts/step5/recovery-bc/best.pt

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate \
  --candidate "$MONITOR_CANDIDATE" --references "$MONITOR_REFERENCE" \
  --setups 2 --no-include-legacy --no-search \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/monitor-g0005
```

One reference, two setups, four rotations, development panel, no legacy and no search gives exactly eight games under the current evaluator. Initially the BC anchor is useful for checking whether PPO forgets its economic warm start; replace it with a qualified neural anchor when available, recording the change in the monitor series.

## S2 and the additional R4 and R5 experiments

Results 1.1 contain no search comparison or search-trained labels. Neither the OpenSpiel adapter nor S3 can be credited with these outcomes. Their shared preprocessing can benefit from the same immutable-cache fixes; changing the search framework alone does not resolve repeated CPU validation or create GPU work during pure legacy generation.

Keep both additional experiments implemented and scheduled separately from the main optimizer. Use the same competent frozen BC/PPO checkpoint, source/device gate, opponent references, complete legal choices and whole-turn clock when comparing S2/R4/S3.

Apply the separate lab/play CPU and clock roles to all three search variants. At equal resources, measure completed simulations and fresh confirmation as well as elapsed time. With the new within-turn parallel variants, record `game_workers` and `search_workers` separately; a serial qualification means one uncontended game, with the same declared within-turn CPU allowance for every compared variant. Changing that allowance requires a renewed gate and explicitly identified comparison.

- Start each shortlisted checkpoint with the existing one-decision-per-variant smoke and correctness gate. Run a **two-setup development screen: 2 × 4 rotations × 3 variants = 24 games**, rather than the default 768-game screen. Schedule it no more often than every twenty main iterations and only when selecting a new checkpoint or search setting. Parallel results remain screening evidence.
- Start S3 distillation with eight fresh **training-only** games and inspect completed-confirmation labels. Scale to the existing 64-game budget only after teacher coverage and a fresh post-distillation comparison justify it. Keep parents, teachers and selected variants pinned during this work; never train on search comparison/evaluation games.
- Retain the main S2 schedule initially, but reserve at most roughly 10% of measured training wall time for useful teachers. Add a readiness/budget decision at the due boundary. If confirmation produces no labels, record that outcome and defer escalation rather than repeatedly buying empty teacher games. Search games must not advance the ordinary-game curriculum counter.
- For a promising search variant, keep the **serial one-worker comparison with at least 30 independent paired setups**. Three variants at that size mean 360 games. Respect the equal-budget, privacy, correctness and paired-confidence gates; small screens cannot waive them.

Apply retention to search screens, failed teacher runs and profiler traces as well as Step 5. Preserve full source/gate/report/raw evidence for a promoted search result. Main training state and source checkpoints remain independently resumable.

## Implementation order and exit checks

| Work package | Main files | Focused exit check |
| --- | --- | --- |
| C0 Accurate phase reporting and lightweight monitors | `recovery/telemetry.py`, `recovery/evaluation.py`, `step5/cli.py`, evaluation configs | Display actual lineup/game counts; eight-game screen cannot promote; show preparation/load separately and rehearsal active state |
| C1 Validate and encode once | `recovery/encoding.py`, `recovery/inference.py`, `learning/replay.py`, `recovery/updates.py`, demonstration loader | Same tensors, legal IDs, seat targets and behavior likelihoods; reject mutated/hash-mismatched records; imported inputs still fully checked |
| C2 Faster data generation and collection | `recovery/demonstrations.py`, `recovery/execution.py`, `recovery/league.py` | Small fresh collection/audit restores all games; compare equal-size pools and preserve original bank evidence |
| C2 Search budget roles and parallel work | `BrassLegacyAI/Planner.cs`, `BrassLegacyAI/ReplySearch.cs`, neural/searchlab executors and versioned profiles | One-second lab target compared with three-second reference work/quality; bounded global CPU allocation, legal incumbents and new qualified lab bank |
| C3 Useful learning batches | `recovery/updates.py`, strict training config and update recipe | Forced choices excluded only from actor statistics/sampling; critic targets and PPO ratios correct; fresh BC and post-update monitors |
| C4 Bounded storage and population | New retention manager; checkpoint, league, BC and searchlab integration | Dry-run reference graph, protected best/resume/dataset/report files, crash-safe commit and restore, no dangling population paths |
| C5 Evaluation scheduling and search budgets | League scheduler, named evaluation/search configs, lab guides | Game-count schedule survives resume; holdout/evaluation traces excluded from gradients; serial promotion gate preserved |

Implement C0 and C1 first, then benchmark C2. Introduce storage bounds before a long training run. Keep C3's changed learning weighting in a distinct recipe; execution-only performance comparisons should keep the old weighting until they have a clean timing comparison. Update the main implementation plan and lab guides when each package is actually delivered.

Strict schemas currently reject unknown fields and runs require exact config/source signatures. Add versioned retention/scheduling/lineup fields instead of editing a running 2.0.0 configuration in place. For learning changes, warm-start a new run with compatible weights, fresh optimizer moments and explicit parent provenance. A future execution-only migration must validate and record preserved optimizer/counter semantics; never bypass the source-signature check.

Use focused checks and a small integration run rather than another large test suite or full tournament. Required evidence before scaling is:

1. Unchanged original reports and frozen bank hashes; replay/feature parity and mutation rejection across action, payment, continuation and debt phases.
2. Demonstration audit with independent games and correct global duplicate/partition handling; BC train/validation curves from a substantive sample.
3. A retention dry run listing every protected/deletable file, then cleanup/resume in a disposable run; current checkpoints, active opponents and pinned dataset/evidence remain loadable.
4. One compact fresh GPU benchmark per shortlisted execution setting, with enough jobs to occupy its pool. Report actual CPU/GPU timelines, memory, queue tails, collection/PPO wall time and meaningful throughput; exclude cached games and separate cold start.
5. Exact eight-game monitor and persistent cadence; a promising BC/PPO candidate improves economic completion and score margins on development probes. Formal strength/search qualification is separate and runs only for a shortlist.

Acceptance targets are reduced repeated preparation and replay-publication work, higher fresh games/updates per hour, bounded storage after cleanup, and improving economic play. Do not claim a utilization percentage or playing-strength gain until the corresponding lab measurements exist.

## Delivered revision, 30 September 2026

The analysis above describes the supplied results 1.1 run. The following now exists in code; its recipe changes need a fresh run as documented in the [conda guide](../docs/TRAINING_RECOVERY_LAB.md).

| Package | Delivered components | Acceptance still measured on the lab |
| --- | --- | --- |
| C0 | Load/preparation/update timers, first/steady/max call statistics, rehearsal activity, explicit panel counts | Production CPU/GPU timelines and profiler memory attribution |
| C1 | Incremental strict ingress, row/context seals, immutable feature arrays and independently committed cache receipts; neutral BC derivation with encoder parity | Fresh production throughput and packing/transfer balance |
| C2 collection | 32 ordinary private actors; overlapping 8 × 4 heavy pool; grouped atomic shards, bounded writer queue and eight-process transition audit | Compare 32 × 1 ordinary lanes and 8 × 4 / 16 × 2 heavy work on 64 allocated CPUs |
| C2 search | New lab child bank with deterministic root partitions, shared 1-second/720-token turn allowance and original scoring; private S2/R4/S3 replicas | Run 25-game lab calibration and same-hardware work/quality probes; child remains unadmitted until they pass |
| C3 | Nonforced BC/PPO actor normalization, independently bounded four-seat critic samples, phase-balanced nonforced KL probes and final full KL; unchanged terminal reward | Substantive imitation curves and fresh economic-play comparisons; tune one hyperparameter at a time |
| C4 | Bounded weight-only opponent index, two resume checkpoints, dependency graph/pins, quarantine/restore, durable move journals, registry/cache eviction | Inspect the first production cleanup dry run and available disk headroom |
| C5 | Resume-safe game-count scheduler, 8-/16-game development panels, 480-game formal shortlist, bounded main S2 scheduling and independent 24-game R4/S3 screens | Renewal of search correctness on the execution host, useful fresh teacher labels and equal-resource advantage |

Current configurations are `step5-training/2.1.0`, `step5-evaluation-config/1.2.0` and `searchlab-experiment/1.3.0`. Archived `recovery-v2-*` configurations preserve the prior recipe for separately identified controls; changed source signatures still require a fresh run. Local checks and unqualified timing evidence are recorded in [the revision exit report](../docs/TRAINING_REVISION_1_1_EXIT_CHECKS.md). None of these changes declares a stronger trained model, qualifies the new lab teacher, or adds the later personality/goal-head stage before Step 5 has finished.
