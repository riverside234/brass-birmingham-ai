# B02.5 / B03 / B04 AZ_DB implementation

Date: 2026-10-30 (America/Chicago).

Implemented the opt-in `teacher-phase/2.0.0` profile from [the AZ_DB revision plan](AZ_DB_TEACHER_DIVERSITY_REVISION_PLAN.md). The current transfer is [distribution/rust-lab-g1330](../distribution/rust-lab-g1330/README.md); it supersedes older source-update extraction instructions while preserving their ZIP bytes. `search_v1` remains selected and V205 remains disabled. Production teacher mode stays off. Linux/CUDA qualification, B05 updates and B07 adoption remain **Pending**.

## Current training reference

The supplied g1330 report binds the existing `joint7-v11-2-grouped-lr-critic` lineage at 425,600 games:

| Identity | Value |
| --- | --- |
| Checkpoint | `artifacts/step5/joint7-v11-2-grouped-lr-critic/generation-1330.pt` |
| SHA-256 | `d7e4e9473472251289b20991f423908c30a55a548a66e7be6afbc0c1f0c236b8` |
| Model | `e85ba41819c4550d5c27ca60ba5f52d8fd4ea127574b63478e953487dd559521` |
| Saved trainer | `299f58b01ffcc5551207c16cf74f0f7c8e8aa7f87c4e957ee37ab8d2880e5f42` |
| Reference report | `results/joint7-v11-2-grouped-lr-critic-g1330-20261031T024940047516Z` |
| Ordinary diversity | DOMiNO `apply`, adopted at g1326 |
| Teacher readiness | Missing retained intrinsic-fit receipt; v2 collection deferred |

The supplement preserves the user's g1330 bulk snapshot-weight implementation and corrected local/intrinsic loss, attempted/retained fitting and unmeasured KL telemetry. It does not regenerate the ordinary recipe. Normal continuation preserves model, all Adams/RNG, PPG cache, population/PFSP, active probe/anchor, DOMiNO controller and progress. The original 75,852 source-review entries remain intact; new source-only and narrow teacher adoption/toggle routes are appended. No prior DOMiNO/reward migration is replayed.

No retained intrinsic fit is evidenced in the supplied g1327–g1330 logs: g1327's retained PPG was task-only; subsequent intervals rolled back. The old checkpoint's missing optional metadata is a normal deferral, not checkpoint corruption. After ordinary accepted PPG retains actual intrinsic steps, the new hook records the precise fitted direction, generation, head/encoder IDs, actual strategy/phase/game/descriptor support and normalization evidence. A rejected or zero-step interval leaves the prior receipt unchanged. No extra PPG or synthetic readiness is scheduled.

## Implemented boundaries

**B02.5.** New config/adoption defaults off and disables dormant teacher value. One independently drawn context per physical seat-game is frozen on the teacher-only deterministic stream; retries reuse the same assignment. The evaluator freezes current coefficients and the actual retained-fit direction, with current head/model and fingerprint identities. Latest task readiness and intrinsic readiness must both pass. Conditional labels validate job/seat/context/bundle, legal order, all three return channels and the existing float64 improved-policy equation. Admission and disk reload encode the recorded context. Four ages and 4:3:2:1 sampling remain; dynamic weights/coefficients do not change static compatibility. Objective/producer/regime changes retire the old window and archive beta calibration. Cleanup retains rollback dependencies.

The existing policy-only KL term still shares PPO clipping/Adam/KL safeguards. It creates no task/descriptor/utility/intrinsic-head gradient and no teacher critic dataset. Reporting includes support by split/strategy/phase/kind, unique label games versus seat-games, raw Q scales, coefficient contributions, fit/direction lag, held-out error/bias/zero baseline, descriptor drift, age/strategy policy/objective drift and presentations. Sparse strategy support is reported, not silently rejected. Retained held-out shards are descriptive only.

**B03.** The diagnostic fork now constructs live `Diversity` and compares the actual intrinsic head, both PPO/PPG Adams, dual Adam/logits, centroids, support, quality, regime/archive and optional fit payload with the sealed source. Fingerprint files and teacher evaluator/checkpoint dependencies are inventoried. Exact checkpoint bytes, origin mapping, ordinary cache and progress remain protected. Complete B02 ordinary replay proves 7/8/12/4 causal main-action normalization against the existing Python occupancy extractor before B04 can use it. Teacher admission stays separate from ordinary replay and never advances DOMiNO/PFSP evidence.

**B04.** The original sealed Rust core is copied to a detached build and exact checked hooks install the opt-in objective. This extension has its own real source/build identity; it is no longer described as export-only. Rust ports the exact public predecision 112-coordinate event and 448-coordinate phase reward, adds root-only signed intrinsic prefixes, and combines only the root owner's task component. Other actors select their own task utility, without sign flips. Initial node values remain frozen. Task, intrinsic and combined sums share counts and publish atomically; canceled insertion/publication restores all channels.

The scalar head is used only at root-owned observations. At another actor's leaf, a separate T1 RNG stream continues the same sampled world to the first root-owned decision or terminal, bounded by the original 4-second clock and total depth 256. It adds no main-tree visits, labels, rows or opponent task cost. Incomplete intrinsic work discards the whole pending trajectory. Pure neural values may be cached; stochastic continuation results are not. Actual action-two validation binds the full information/context/task ledger plus objective bundle and intrinsic prefix.

The inference service registers one frozen companion and shares encoded state/trunk work. Task descriptors remain unknown; only intrinsic output uses known learner/lag-zero/current-context self descriptors. Heterogeneous contexts can batch under the same base model. The existing shared-slot owner, epoch, cancellation, deadline and reply-retirement rules remain in force. Diagnostic controls cannot supply live collection calibration or enter training buffers.

## Short local evidence

Development used authored fixtures and finite count probes only, with no full teacher games, training iterations or installed-game changes. The sealed `local-development.json` records exact final source/producer identities and log hashes. Logs retain initial failures and the focused reruns that resolved them.

- Python B02.5/B03/B04, g1330 DOMiNO review and result-export regressions cover policy-only gradients, off/shadow preservation, mode/adoption crash gap, buffer ownership, strict replay isolation, stale/canceled inference and report portability. The Windows symlink fixture is skipped because unprivileged symlink creation is unavailable. CUDA checks remain for the lab.
- New conditional tests cover IID contexts, missing/partial/rollback fit receipts, old fitted directions versus advanced centroids, corrupted identities/channels/normalization, four-age reload, retained rollback shards, live DOMiNO/Adam restore, read-only scalar inference and candidate sizes 1/772 with batches 1/2.
- A staged PPG cache regression performs two tiny intrinsic updates, restores the first accepted snapshot, then records only those retained rows/steps. Cached CPU metadata and dense tensor metadata produce identical flat descriptor receipts without changing weights, either intrinsic Adam, the PPG Adam, controller or RNG. The cache path reads small CPU columns in bounded batches, with no state/action reconstruction or per-row device transfers.
- Native tests: 5 ordinary-lane cases and 30 session/teacher cases pass. The finite authored corpus covers four phases × seven contexts with two count-budget trajectories per control, including zero-contribution parity. Action-two intrinsic prefix mismatch forces revalidation; expired continuation preserves only complete evidence. These roots are authored mechanics fixtures, not reachable-game or strength samples.
- Two core objective unit tests pass: signed root-only mixing and hand-calculated action-two suffix/causal insolvency accounting with rollback of atomic channel publication.
- Python/Rust descriptor equality passes for four phases × seven main-action families, plus populated public descriptor fixtures for all four seats (flipped assets, exhausted stacks, links and clipped financing). Real native/Python inference probes verify task-only and intrinsic label targets/logits, root-owner continuation and exclusion of diagnostic probes from production admission.

The suite and final audit caught an unpropagated session objective, ordering of source-review alternatives, malformed Rail fixture card counts, fixture reward/cache mismatches and the cached descriptor column retaining its batch dimension for a scalar row index. These were corrected; no integrity checks were weakened. Source-only review routes no longer attempt unrelated historical recipe upgrades before teacher adoption.

## Lab qualification and limits

From the updated project root:

```bash
unzip -o distribution/rust-lab-g1330/rust-lab-az-db-g1330.zip
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0
```

Defaults bind g1330 and its exact checkpoint; [the bundle README](../distribution/rust-lab-g1330/README.md) gives explicit generation/hash overrides, individual-stage commands and config-only adoption. The launcher always exports a fresh Git-visible `results/rust-lab-qualification/` directory. Original lab artifacts remain available for subsequent gate checks.

Once readiness passes, B04 predeclares two normal eight-trajectory calibration roots and two eight-trajectory K1 continuation probes. It requires actual continuation work, then runs two roots × three counterbalanced **4s/K16** controls and exactly two shadow tables. Neutral/context-only calibrations remain separately identified. A failed finite calibration does not invent a CPU estimate or grow into a campaign. Reports include quota, main/continuation work, incomplete work, waits, channel scales and `matched-4s-controls.json`.

The actual lab checkpoint has not been restored here, CUDA parity/cost is unmeasured, and complete ordinary normalization is still a lab B03 gate. The g1330 teacher readiness deferral is expected. This implementation does not establish seven distinct learned strategies, stronger play, faster throughput or safe production teacher updates. Those require the stated lab checks and later B05/B07 gates. Ordinary training can continue through the reviewed route while teaching remains off.
