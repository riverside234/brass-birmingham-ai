# R3 H3–H5: tactical bank, reply search and integration

This extends the [H0–H2 foundation](R3_LEGACY_BANK.md). All research opponents use four players, the pinned patch 1.1.0 runtime, R1 contract 0.2.0, CPU heuristics and complete semantic legal actions. No LLM or neural weights are required. See [the exit record](R3_EXIT_CHECKS.md) for actual results and promotion decisions. The [v3 update](LOCAL_AI_V3_CONVERSION.md) records current game/research integration and exact verification status. The earlier [compatibility checks](UI_PATCH_RESEARCH_COMPATIBILITY.md) explain the preserved rules-runtime boundary.

## H3: six conditional styles

| Profile | Preferred opportunities | Recovery |
| --- | --- | --- |
| Balanced | Highest estimated complete-turn result | Re-evaluate the available alternatives |
| Rail Tempo | Useful Rail links; brewery/iron support | Complete investments or fall back when the plan is infeasible |
| Cotton | Feasible cotton investments and sales | Repair missing beer with an affordable own brewery, or abandon the sale plan |
| Goods/Beer | Manufactured goods and brewery support | Shorter feasible investments and the common evaluator |
| Supplier | Iron/coal when visible supply and expected demand justify them | Stop preferring supply when completion likelihood is low |
| Pottery Flex | Affordable pottery with matching cards, sites and completion access | Balanced opportunities when any prerequisite disappears |

`Tactics.cs` uses the public board and that seat's hand. Its versioned intent records contain focus, target, milestone, required cards/resources, available cash and remaining actions. A feasible target is retained across decisions; changed opportunities and stalled plans produce explicit abandonment reasons. Intent is a proposal/selection preference, not a legality rule or a forced script. Resource demand and financing forecasts remain approximate.

Preference affects proposal ordering and then chooses within a bounded credible set: normally 1.5 heuristic points from the best complete plan, with zero window for Balanced and terminal outcomes. A style therefore may make the same move as Balanced when its preferred move is poor. There is no shared bot objective or exchange of private intent. Search diagnostics record industry choices, intent milestones, openings and recovery; the tests and tournament measure diversity rather than assuming six labels imply it.

The optional game bridge persists one detached intent per seat using `IntentJournal`. It pins game ID, rules, profile fingerprints and schema, hashes records, saves by atomic replacement, selects only records at or before the loaded revision, and prunes abandoned future records after undo. Cross-seat/profile/game records and corrupt sidecars are rejected. The R2 adapter keeps state in separate weakly held per-game/per-seat contexts; restoring a revision clears cached plans.

## H4: mandatory S2 implementation

`ReplySearch.cs` performs the following sequence:

1. Spend the declared proposal fraction on S1, publishing a complete legal incumbent first.
2. Retain several fixed complete current-turn plans, including that incumbent. The same two-action plan is applied in every sampled world; it cannot adapt to an unseen opponent hand or future draw.
3. Reconstruct each world from allowed information and a separate sampling seed. Apply the current turn, then advance the actual scheduler through draws, income, debt and era changes until the root's next ordinary turn or game end. There can be zero to six intervening ordinary opponent turns.
4. Give each simulated opponent its own strict observation and legal choices. An independently selected tactical rollout policy chooses its own actions. The rollout interface receives no host, root hand or root intent. This is a fixed-policy opponent model, not recursive adversarial search or a three-player coalition.
5. Compare all candidates on matched pilot scenarios. Select finalists, then compare them on disjoint fresh samples. Only complete matched batches count. Report four-seat estimates, sample variance when at least two samples exist, horizons and sample IDs.
6. Use a bounded style preference among credible confirmed finalists. If time, work, cancellation or a rollout failure prevents confirmation, keep the valid S1 incumbent and report the reason. Rollout failures remain visible and prevent qualification; they do not become strategic wins.

The initial S2 settings use up to four plans, two pilot batches, two fresh batches, 40% proposal budget, 240 proposal-work units before allocation, and 2,048 reply-choice units. They are small heuristic estimates, not calibrated confidence in beating a person. Nonterminal values use the common heuristic evaluator; terminal rollout utility is official win share plus `0.05 × (4 − rank)`, an explicit secondary placement preference. Tournament results always report actual official win shares, without that bonus. The S1 ledger describes the immediate afterstate; S2's separate `replies.estimates` describes its sampled horizon. Do not mix those units.

The six current `*-v3` S1 profiles have 3,000 ms / 240-node whole-turn allowances. In the historical v2 study, six `*-v2` profiles have 3,000 ms whole-turn allowances; six `*-s2-v2` profiles have 10,000 ms. `balanced-s1-10s-v2` is the equal-time S1 control. Both action executions share the original deadline. The headless match timer includes observation capture, search, validation and commit; process IPC and rendered presentation are outside it. A valid incumbent/commit may exceed the nominal cap. These settings are not calibrated Easy/Medium/Hard levels or strict real-time guarantees.

## H5: immutable bank and evidence gates

The original H5 v2 snapshot is `bank-804c85204da65824cb9b`, under `training/opponents/bank/`. Its manifest hashes original C# sources, the permitted-input Python adapter, contracts, profiles, compiler and all 112 runtime files. Original source snapshots retain their exact bytes in Git. Game binaries remain under ignored `artifacts/legacy-ai/bank/`; they are never part of the public bank. A changed source/profile/runtime produces another identity, and a frozen identity cannot be overwritten. The [v3 conversion update](LOCAL_AI_V3_CONVERSION.md) records its separate cash/deadline/merchant policy and qualification; the commands below target the new v3 bank. The v2 sources and qualification remain immutable historical comparisons.

`FrozenOpponent` verifies the content address, snapshot files, active adapter/contracts and exact runtime before starting a worker. This prevents a later rebuild or mutable profile file from silently changing a training opponent. For an older adapter version, use an isolated checkout with its pinned adapter files; do not bypass these checks.

The historical `qualification-d9db1fbda13fcfcb173d` prespecifies 156 matches: eight main configurations against three guarded stock bots on four fresh setup groups with all four seat assignments, sixteen diverse tables including reversed clockwise arrangements and seat rotations, and identical-profile tables for every S1/S2 style. Pottery Flex is reserved as a held-out family. All jobs run once; interrupted runs resume missing jobs only. Reports include failures, official win shares/ties, rank, score margin, headless latency, reply confirmation/fallback counts, and setup-clustered uncertainty. Failed games are never credited as wins.

This is a small initial qualification, not the larger general-strength release study. The preregistered niche gate requires zero observed failures, all required table categories, and mean guarded-stock rank better than 2.5. The general improved-Hard gate additionally requires at least 30 setup groups and a lower win-share uncertainty bound above 25%. The present four-group study cannot pass that general gate. Human enjoyment, calibrated difficulty, rendered deployment and lab/Linux validation are separate checks.

`pool.json` contains only admitted immutable identities and the hashes of their qualification reports. Training and evaluation pools are disjoint. The admission tool refuses to retarget an existing identity. S2 promotion is separate: implementation and legal completions alone do not establish benefit over S1 at equal time. Specialist S2 configurations have limited initial coverage and are not automatically admitted.

The v3 [limited validation amendment](../training/opponents/evaluations/qualification-c487cd9f12a201a5690c/spec.json) retains all 28 games completed when the user stopped the original 118-game study, accepting a 25-game limit. Coverage is 24 guarded-stock games, three mixed tables and one identical-profile table, with at least two guarded games for each style. Zero execution failures, zero observed zero-VP borrowing-floor stalls and guarded mean rank better than 2.5 remain required. Per-style coverage is smaller and uneven; this admits niche opponents for the requested use, not calibrated strength. The [v3 report](LOCAL_AI_V3_CONVERSION.md) owns final admission/installation status.

## Commands and lab use

All Python entry points default to `/data/not_backed_up/yxu209/ai-training/`. On the laptop, explicitly pass `--root .`. Runtime libraries must come from your separately supplied matching game installation. The existing R1 host is also needed for R2 collection. First repeat compatibility and runtime checks on Linux/Mono; this Windows result does not certify the lab platform.

```bash
cd /data/not_backed_up/yxu209/ai-training/
# Rebuild from frozen source using private dependencies already below the lab root.
python scripts/prepare-legacy-bank.py --bank bank-af17b1aadbe7f9b21fa7 \
  --managed artifacts/private-managed --mono mono

python scripts/check-frozen-bank.py --bank bank-af17b1aadbe7f9b21fa7 \
  --mono mono --output artifacts/legacy-ai/lab-platform-check.json

# Inspect/reproduce the exact preregistered tournament, without changing its profile settings.
python scripts/qualify-legacy-bank.py run \
  --spec training/opponents/evaluations/qualification-c487cd9f12a201a5690c/spec.json \
  --output artifacts/legacy-ai/lab-qualification --mono mono

# Select four frozen versions from the admitted training pool.
python scripts/sample-legacy-pool.py --seed 71 --purpose training

# Load each admitted version, commit a legal turn and check rejected pool uses.
python scripts/check-legacy-admission.py --mono mono \
  --output artifacts/legacy-ai/lab-admission-check.json
```

Pass the returned four full `bank/profile` identities to `scripts/collect-legacy-bank.py --policies ... --seeds 41001 41002 --workers 2 --purpose training --output artifacts/r3/replay.jsonl.gz --mono mono`. It rechecks admission before collection, freezes assignments within each game, cyclically rotates seats across games, and writes complete R2-compatible shards with four-seat targets, behavior provenance, style and intent. Supply `model:artifacts/r2/checkpoint.pt` in any policy slot to mix a frozen neural checkpoint with bank opponents. `random` is available for diagnostics. `--device cuda` affects neural inference only.

`--purpose diagnostic` permits testing unqualified frozen versions and explicitly marks the output. It must not be presented as qualification. Evaluation uses the separate held-out pool. The R2 training reader rejects evaluation-only shards even if their setup hashes would otherwise fall into its training partition. The original guarded stock remains a historical benchmark outside the strict-observation policy adapter.

For example, R2 can learn outcome values from the collected shard:

```bash
python scripts/r2.py train --shards artifacts/r3/replay.jsonl.gz \
  --mode value --steps 100 --checkpoint artifacts/r3/value-bootstrap.pt
```

Legacy decisions are marked `external`, and no heuristic choice is labeled an optimal teacher. The R2 on-policy update rejects these trajectories. Search distillation and population actor training are later work; initial bank replay integration supplies valid outcomes and opponents, not a completed neural training campaign.

The historical research overlay remains uninstalled; the separate [v3 game package](LOCAL_AI_V3_CONVERSION.md) supplies the in-game strategy selector. The research Unity checks described here exercise routing/Update with a mock view router and scripted human metadata. They do not establish rendered human-play acceptance, entertainment value or laptop foreground latency.
