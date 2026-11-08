# Stage C retention and replay-free machine transfer

User revision, 8 November 2026. Latest pulled results commit: `a21a13d`, export `results/joint7-v11-2-g1290-trained-restart-g1284-20261108T193427364056Z`. Training remains teacher-on, fixed policy-teacher beta `0.16110139609854138`, PPG beta floor `0.25`, search_v1, V205 off, branching 0%.

## Latest evidence

g1280–g1283 collection phases committed g1284 with revised Stage A at 4/10. Each performed 32 head-only steps. Whole value-phase time was 26.03 / 27.03 / 26.38 / 27.30 seconds; head update time was 1.95 / 1.92 / 1.74 / 1.86 seconds. PPG rejected its first 16-step interval on increased held-out loss in all four phases. These are actual lab observations, not a new matched benchmark. Head-buffer reading/target preparation remained variable (0.80–6.72 seconds). Stage C has not run in this export.

## Stage C

`search_value_retention.reserve_games` independently draws eight complete 64-position teacher games from the live eight-age training return buffer. It binds the seed to the generation, companion seed and sorted candidate game IDs. The report records the exact selected games, ages and digest. It excludes every row of those games from this phase's fitting, including both 16-step intervals. Input receipt/row bindings remain unchanged; the transient holdout retains its original split attribution.

These are within-phase validation games and may have contributed to earlier phases. The separate four-game periodic diagnostic remains separate. No extra games/search work is requested for this retention check.

The loss is MSE of current main utility plus the search residual against the same lambda-return targets used for fitting. Pack 512 holdout inputs once; recompute predictions at the post-PPG baseline and after each interval. A strict mean-loss increase stops the pass and restores the rejected interval's model weights, ordinary value Adam, search-head Adam and companion step counter. Exact ties remain accepted, as in the current PPG retention rule. Previously accepted intervals stay committed. Nonfinite losses and exceptions fail transactionally. The existing joint KL cap remains 0.003 and can reject an interval independently. Stage progression/warm-up stays unconditional outside explicit transfer-buffer rebuilding.

The low-level `unconditional_pass` name remains for compatibility with the unconditional stage schedule and its old focused callers; production supplies the eight-game holdout. If there are too few complete games, production defers the update rather than bypassing retention.

## Transfer without replay or full IL

See [the operational guide](../docs/STEP5_MACHINE_TRANSFER.md) for export/restore/preflight/continuation commands.

Current native exports are compact and include the fixed `il-kl/probe.npz` plus its original manifest, required completed-IL model/lineage/dataset manifests, full learner checkpoint and live opponents, ratings/pool metadata, original/active native runtime files and their pinned evidence, bootstrap/rules DLLs and source-transition backups. Native executable permissions are preserved on Linux. They exclude replay buffers and original IL shards/caches. Active original-IL rehearsal and native full-mode exports are rejected explicitly.

Recipe 4 shares the full stdlib source inventory with the trainer, including Rust inputs and bootstrap/C# sources. It records actual execution identity while preserving the reviewed qualified source alias in state. Legacy recipes 1–3 remain available for old archives. Native pins keep their historical path spelling; actual reads/process launches resolve through the existing confined relocation map. Historical checkpoint and receipt bytes remain unchanged.

The fixed IL probe's committed manifest, cache SHA-256, semantic tensor hash, source/model/config/compatibility bindings and `allow_pickle=False` loading remain checked. Only a committed probe may load without its original validation shards; original-source validation remains mandatory for bootstrap/build. No IL dataset is reconstructed or resampled during continuation.

Restoration creates a separate `machine-transfer-resume.json` and original-state evidence. The first continuation resets only PPG/teacher/return buffer references at its bound checkpoint. No weights, Adam/RNG state, beta, trip history or stage counters are reset. The next checkpoint/state persist the transfer receipt. Pre-commit retries authorize the same reset; later committed resumes retain fresh buffers and require the exact receipt. In-place source archive verification does not install a reset instruction.

Auxiliary training and its counter pause until the eight-age return buffer is filled with consecutive fresh collection ages. At transferred g1284, Stage A remains 4/10 until eight consecutive new-reward teacher ages exist; the eighth allows the next head update. Old utility-readiness is invalidated, so fresh ordinary PPG readiness can delay teacher collection and the calendar date of Stage A resumption. Missing ages delay readiness. A/B/C counters otherwise retain the existing schedule. PPG uses current then current+previous fresh ages; teacher-policy fitting uses available fresh ages without importing old targets.

The original B07 rollback manifest is retained only as pinned evidence. Its old diagnostic replay payload is omitted; the transfer archive is the destination restore point.

## Validation

Focused authored-tensor/archive regressions cover game-level partitioning, RNG neutrality, independently computed holdout MSE, first/second-interval retention, both Adam/companion rollback, KL/cancellation, paused counters and cold phase restoration, fixed-probe reuse without source shards, shared source identity, omission of replay/IL caches, native runtime path mapping, checkpoint/state byte preservation and one-time reset/retry bindings. No local gameplay, benchmark or training campaign is required. Destination preflight and actual moved-run continuation remain lab checks.

Debug evidence: `artifacts/stage-c-machine-transfer/`. **58 distinct affected Python debug checks and four filtered Rust numeric tests passed**, plus workflow actionlint, 44 changed-Python AST checks and `git diff --check`. Earlier fixture/dependency failures were corrected and their affected checks rerun. The original pinned zstd dependency was reused from `artifacts/step5-transfer-deps`; no package installation, local gameplay or benchmark was needed. The Git-visible receipt is [stage-c-machine-transfer.json](../distribution/rust-collector-performance/stage-c-machine-transfer.json).


## Destination-only reward revision (latest user correction)

Use **placement = 0.05**, not the earlier 10% proposal; win remains 1.0 and VP remains .001 with cap180. The untied non-VP vector is 1.05 / .033333 / .016667 / 0. Insolvency event reward charge becomes exactly zero. The positive .025 input-budget scale is kept so existing NN tensors and normalization remain compatible, while the ledger subtracts nothing. Official game rules, VP/income/cash tie ordering and shared-win/pairwise-tie handling stay unchanged.

Export queues this exact config without mutating source state/checkpoint/reward. A destination restoration instruction and exact source checkpoint authorize its normal reviewed same-run adoption. Preserve weights, all Adam/RNG state, fixed teacher beta, counters and progress; invalidate old-reward readiness and all old reward-bound buffers. Preserve the actual old checkpoint reward even if a config transition was committed just before moving. Search-head fitting/counters remain paused until eight fresh return ages, with no old targets relabeled. The historical beta receipt remains historical; this change does not rerun its completed qualification.

The original Rust core rejected event_cost0. A separate teacher-only adapter applies a bound validation extension in a detached sealed core and runs four tiny numeric tests (zero charge/new placement, old positive-cost behavior, invalid-budget rejection and capped VP payments). No search/traversal/backup equation changes. Original core8d59ec4b... becomes e6bedd50... only in this separate variant. The ordinary Rust worker and installed game are unchanged. Prepare on the source before exporting, or run the restore JSON's prepare_reward command on the destination. A matching shipped variant is reused. Do not send compiler caches, original IL or replay buffers.

Four filtered Rust debug tests passed locally (0.01s test execution after compilation); no gameplay or benchmark ran. The 11 new Python transfer/reward checks include strict checkpoint loading and a pre-commit retry with unchanged checkpoint bytes, parameters, Adam state and RNG. Seven VP-timing checks cover boundary/cap arithmetic, inherited rail-branch credits, frozen-tensor reconstruction, prepared metadata and GAE. The remaining 40 affected checks cover Stage-C retention, replay-free transfer, the committed IL probe and the prior entropy-clock behavior. Debug fixture failures were repaired and only affected checks rerun.


## Canal and final VP payments

The latest user request extends the destination reward to `brass-combined-return/3.0.0`: `canal = .001 * min(canal_VP,180)` and `final_delta = .001 * min(final_VP,180) - canal`. Payment follows canal scoring exactly once, at the first authoritative rail observation; the second payment accompanies terminal win/placement. Do not apply the 180 cap independently to a raw rail VP increment or pay the canal amount twice. If final VP is lower, retain the signed correction so total reward is exact.

`reward_timing` derives payments and paid-to-date credits from complete bound public replays. Rail-only branches must carry their parent canal credit; missing/reversed/late boundaries fail closed. Compact ingress retains the credit. The raw NN/PPG and search outputs remain full-game values; subtract the known credit analytically for future returns and GAE values. That is an exact change of value coordinate, with the same model tensors/optimizer state and no need to infer past canal scoring. Existing undiscounted search ranking/backup remains valid. Auxiliary target fitting likewise keeps the full-game coordinate; each teacher table records both payments and per-root future utility using its already-bound first rail root and frozen observer-relative VP indices. No extra simulation, feature transfer or inference is introduced. Old v1/v2 recipes keep terminal-only payout.
