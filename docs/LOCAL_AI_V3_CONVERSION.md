# V3 industry completion and merchant beer (package 1.4.1)

**Installed and verified:** package 1.4.1, 28 September 2026 at 20:07 America/Chicago (29 September 01:07 UTC). This updates the six v3 Custom Game strategies and ports both the earlier cash repair and the new conversion policy into a separately frozen four-player research bank.

## Behavior

The evaluator distinguishes an industry that has flipped from one that could still flip. An accessible cotton, goods or pottery sale no longer receives 97% of its points in advance. The planner counts each seat's actual remaining actions, including its position in the current round. A sellable industry has no current-era completion value after its owner's last action, even while other players still have turns. Level-one Canal tiles have no Rail carry-over value; level-two and higher tiles retain a separate discounted Rail estimate.

Brewery, coal and iron tiles flip through resource consumption, not through Sell. Their forecast now uses remaining actions, public reachable sale demand, accessible supply and discounted opponent demand. Merchant beer reduces expected brewery demand. Speculative future beer demand is capped before it is divided among competing barrels; an empty market cannot make several unsupported breweries look almost certain to flip. Unused resource tiles also receive a more cautious Canal carry-over estimate. With no actions left for any seat, unused resources have zero forecast completion value. The forecast is still approximate: it cannot guarantee that opponents will consume a resource, or that every investment will flip.

Sales and double-rail actions compare complete legal continuations under multiple beer-source preferences. These include merchant beer, draining an owned brewery, and conserving owned beer. The engine applies the actual merchant reward before the resulting position is evaluated. The planner can therefore prefer a merchant bonus, or prefer a brewery flip when that produces the better result. Merchant VP already credited to the player's score is counted once. Merchant beer remains restricted to the eligible sale; it cannot pay for rail links or be consumed when no beer is needed.

The Gloucester free-development choice compares the available upgrades and can skip an unhelpful development. The quick incumbent batches additional legal sales into the same action. Its bounded cash-recovery fallback retains distinct merchants and tries different beer sources even at zero thinking budget: draining a brewery or collecting merchant money can avoid liquidation. Exact terminal outcomes remain authoritative; among equal win-share/rank outcomes, v3 prefers the actual VP, income and cash result before style preferences. Personalities cannot override the existing immediate-income-payment protection.

Completion also needs financing. Among equally payment-safe complete turns, v3 preserves future borrowing access when an alternative keeps it available. This prevents short search from stacking loans early, spending the proceeds on unfinished industries, and becoming unable to fund their completion. An emergency loan remains preferable to an immediate shortfall; a turn that raises income back above the borrowing floor is eligible, and the preference ends when there is no future turn to finance. This is a deliberately conservative policy, not a change to legal borrowing rules.

A public-board route estimate counts missing links to compatible merchants in the current era, plus remaining actions and financing. A one-canal connection can justify building production; a remote or unaffordable route receives much less credit. Network proposals prioritize completing those routes. Brewery proposal ordering uses the same demand forecast as final evaluation, preventing cheap unsupported breweries from crowding out useful production. Deadline checks still override route potential.

The anytime search first completes diverse opening actions before spending the remaining budget widening one branch. It retains a candidate for each explored industry type. This lets a build-and-sell sequence compete with a loan or a resource investment early, while every published incumbent remains a complete legal turn.

All six style weights and normal game thinking settings remain intact. The live game uses a whole-turn allowance of at least 10 seconds / 1,024 search nodes. The research bank uses its separately declared 3-second / 240-node S1 profiles. Equal inputs and equal fixed work are compared explicitly; different time budgets are not described as identical decisions.

## Versioning and saved games

New game assignments use `brass-local-ai/1.4.1`. The package embeds the original v2 profiles and the original 1.4.0 v3 profiles alongside the new v3 profiles. Old assignments and source hashes remain validated against their original bytes. The live bridge enables the updated policies on a detached profile while preserving saved assignment and intent identities.

The original frozen v2 bank, `bank-804c85204da65824cb9b`, is preserved. The new bank is `bank-af17b1aadbe7f9b21fa7`; its worker SHA-256 is `5d0238095df57de6e9bc593469139e8e402908f07dea052c3b75f2747d2a79be`. Its [manifest](../training/opponents/bank/bank-af17b1aadbe7f9b21fa7/manifest.json) pins sources, all six v3 profiles, compiler, R1 rules dependencies and runtime files. The new flags are `cash_policy: 1` and `conversion_policy: 1`.

The shared `Conversion`, `CashManagement` and `Evaluator` sources are identical in research and deployment. Their planners differ only in supported player counts/search modes. Research still uses the preserved rules-1.1.0 dependencies and observation contract 0.2.0. The installed rules/helper assemblies remain separate from that dependency set. S2 remains available as the historical diagnostic experiment; this v3 admission concerns the six S1 profiles.

## Verification

Binary-bound checks passed for the frozen policy and package. The original tournament was reduced at the user's request; its limited admission scope is stated below.

- **435 assertions in each runtime, 54 matching scenario decisions:** final-turn sales for every style in Canal and Rail; all five merchant bonus locations for every style; a combined two-industry sale that drains an expiring brewery; and a final brewery flip that is better than taking merchant VP. Additional checks cover remaining-seat order, zero opportunity after the era deadline, level-two carry-over, legal continuations and isolated planning.
- **34 frozen-worker tests:** existing R3 contracts plus explicit v3 flags, cash protection under an exhausted budget, and fixed-work hidden-world invariance for all four seats.
- **1,028 core runtime/save assertions, 294 cash regressions, 68 tactical assertions and 1,441 search/settlement checks** cover the deployment policy.
- **60 cash/beer assertions in each runtime, 12 matching decisions:** every style must retain a legal payment-safe sale through its own brewery or a merchant money bonus, including every published incumbent at zero thinking budget.
- **72 credit-access assertions in each runtime, 12 matching decisions:** all six styles preserve borrowing access when a payment-safe alternative exists, including at zero budget, while still taking an emergency loan when needed.
- **19 demand assertions in each runtime:** a legally replayed opening receives conservative unsupported-beer forecasts. At 240 nodes it chooses the original non-brewery regression solution; at 64 nodes it avoids accumulating multiple unsupported breweries. A single early brewery remains possible. Both runtimes must choose the same turn at each budget. The original 240-node requirement is retained; the 64-node extension does not require every early brewery investment to be rejected.
- **96 old-save compatibility assertions** pass across generated saves from packages 1.3.2 and 1.4.0. No real user save is used for these checks.
- **36 native UI turns / 17,471 assertions** pass across two-, three- and four-player full/introductory games. Maximum observed turn time was 10.965 seconds with a 10-second thinking allowance, during concurrent validation work. The unchanged 35-second watchdog is retained.
- **24 retained R2 scaffold regressions** cover external-policy provenance, evaluation holdout, model shapes, checkpoint/resume and export compatibility. Their recorded replay-reader and test-source fingerprints are unchanged; fresh integration with this frozen v3 worker is recorded below.
- **Frozen-source rebuild:** the original snapshot produces a byte-identical worker in a different output directory on Windows.

The original [118-game specification](../training/opponents/evaluations/qualification-97c7a2f1add8fc7aa336/spec.json) was stopped at the user's request that 25 qualification games were sufficient. **28 games had completed when the workers stopped; all 28 are retained, with no outcome filtering.** The [scope amendment](../training/opponents/evaluations/qualification-c487cd9f12a201a5690c/spec.json) preserves the original job settings, source-record hashes and original admission thresholds for audit. This is a post-start reduction in coverage, not a completed preregistered 118-game study.

The retained sample contains 24 guarded-stock games, three mixed tables and one identical-profile table. Every style has at least two guarded games. Limited niche-opponent admission still requires matching correctness evidence, zero execution failures, zero observed zero-VP/zero-cash borrowing-floor stalls, and guarded mean rank better than 2.5; it does not require the original per-style mixed/identical coverage or 16 guarded games. Pottery Flex remains held out for evaluation. No general strength, calibrated difficulty or Linux qualification is claimed. No additional qualification games were started after the user's stop request.

Five candidates were superseded before admission or installation: `bank-c891530a53c0ae07765b` missed a payment-safe beer choice in the exhausted-budget fixture; `bank-b8e1a4d9bf7bdc059c50` exposed unsupported brewery investment; `bank-a190d581edf4cff68698` exposed an additional borrowing-floor stall; `bank-1166d8854d8af3997cc3` exposed insufficient complete-turn breadth at short thinking times. `bank-b5988f07c96eb193f3f4` completed all 118 games legally but an additional economic audit found stalled zero-VP seats; it was never admitted or installed. Their studies and immutable snapshots remain for provenance. The final candidate uses fresh evaluation seeds. Seeds 69001, 74001 and 78001 are development regressions, not held-out strength results.

The new merchant-route regression reproduces seed 82002 through legal actions at 64 and 240 nodes, checking reachable production, financing, unchanged live state and the deadline override (24 assertions in each runtime). Full 64-node development replays of the previously stalled setups now finish at 96/86/102/71 VP (seed 82002) and 97/108/106/103 VP (seed 83001), with positive income for every seat. These setups are excluded from the new qualification.

Before the final merchant-route revision, the earlier candidate's seed-69001 diagnostic also completed at 64 nodes, scoring 81/118/84/91 with positive income for every seat. The revised breadth policy completed the mixed seed-78001 diagnostic at a fixed 64-node budget with scores 107/94/89/91 and positive income for every seat. Its separate 3-second/240-node replay scored 94/97/78/89 and completed all 124 actions. These are regression results, not a claim of general strength. Wall-time qualification is run separately from the large fixed-work regression jobs to avoid starving its thinking budget.

### Fixed-work complete-game diagnostics

Four isolated games compare package 1.4.0 with this revision on two common setup seeds. Every game completes all 124 actions, with each of the four players taking 16 turns, at 240 nodes per turn. The revised games have no projected payment shortfalls or liquidations. Concurrent wall times are not foreground UI latency.

| Setup | Policy | Unflipped Canal level-one tiles | Unflipped Rail sale industries | Merchant beers used |
| --- | --- | ---: | ---: | ---: |
| 72001 | 1.4.0 | 1 | 4 | 7 |
| 72001 | 1.4.1 | 0 | 0 | 10 |
| 72002 | 1.4.0 | 2 | 5 | 6 |
| 72002 | 1.4.1 | 0 | 0 | 7 |

Across these two setups, unflipped Canal level-one tiles fall from 3 to 0 and unflipped Rail sale industries from 9 to 0. Merchant-beer use rises from 13 to 17. These all-old/all-new tables are small behavior diagnostics, not a controlled playing-strength result or a guarantee that all industries flip. The new tables finish at 114/112/126/99 and 89/110/81/101 VP, with positive income for every seat. One unflipped level-two manufacturer survives the first setup's Canal transition; one iron works remains unflipped at the second setup's Rail end.

### Normal-budget gameplay and training integration

The 10,000-ms / 1,024-node all-AI game (seed 72003) completed all 124 actions and all four players' 16 turns, scoring 88/110/122/93. Every seat finished with positive income. There were no projected payment shortfalls or liquidations. The table used six merchant beers; a player can still prefer its own brewery when the flip is worth more. This game retains one unflipped Canal level-one manufacturer and one unflipped Rail manufacturer, demonstrating the limits of the heuristic. The six separate mixed human-model/AI lifecycle games also passed across two-, three- and four-player full/introductory games (2,033 assertions).

The [fresh R2 integration record](../training/opponents/evaluations/qualification-c487cd9f12a201a5690c/r2-integration.json) covers two complete games and 685 decisions from this frozen bank. Two value-training updates changed the small model; four-seat outcomes and style/intent context were preserved. The checks rejected external legacy trajectories as neural on-policy data, isolated separate game contexts and invalidated a stale queued choice after restoration. These are scaffold checks, not trained neural playing strength.

The [limited qualification report](../training/opponents/evaluations/qualification-c487cd9f12a201a5690c/report.json) retains 28 completed games with zero execution failures and zero observed zero-VP/zero-cash borrowing-floor stalls. All six profiles satisfy the amended niche-use gate. Guarded-stock coverage is Balanced 4 games, Rail Tempo 5, Cotton 5, Goods/Beer 2, Supplier 2 and Pottery Flex 6; mean ranks range from 1.0 to 1.5. These small, uneven samples are not comparative strength ratings.

The active [pool](../training/opponents/pool.json) now uses only the new v3 bank: Balanced, Rail Tempo, Cotton, Goods/Beer and Supplier for training; Pottery Flex for held-out evaluation. The [admission check](../training/opponents/evaluations/qualification-c487cd9f12a201a5690c/admission-check.json) loaded all six frozen versions, committed legal opening turns and rejected seven invalid pool uses. The sampler returns v3 identities. Earlier v2 pools remain archived.

**Installation passed.** The final package qualifier and 79 isolated upgrade/rollback packaging assertions passed. The installed player passed 57 startup/menu/game-creation smoke assertions. Installed fingerprints match the candidate, all 116 previous hooks and six merchant-bonus rewrites remain present, profile provenance passes, and the untouched original backup was verified. Prior rules/helper/city-rendering evidence is retained only for byte-identical engine/helper DLLs. The previous 1.4.0 package remains archived; real user saves were not opened by validation.

Installed AI DLL SHA-256: `53213F92D3F403AA39026400AF37EBD61278D37E6F53863FA568D96FC4D77897`. The engine/helper hashes and exact installation time are in ignored `artifacts/patch-installation.json`; installed-player evidence is in `artifacts/patch-build/local-ai-installed.json` and `installation-verification.json`.

## Lab reproduction

All default lab paths begin at `/data/not_backed_up/yxu209/ai-training/`. Private game DLLs are not in Git. Supply the exact preserved rules-1.1.0 dependency set rather than copying the newly installed game DLLs into the R1 host.

```bash
cd /data/not_backed_up/yxu209/ai-training/
python scripts/prepare-legacy-bank.py --bank bank-af17b1aadbe7f9b21fa7 \
  --managed artifacts/private-managed --mono mono
python scripts/check-frozen-bank.py --bank bank-af17b1aadbe7f9b21fa7 \
  --mono mono --output artifacts/legacy-ai/v3-lab-check.json
python scripts/sample-legacy-pool.py --seed 71 --purpose training
```

The sampler returns only v3 identities from the active pool. Old pool snapshots and frozen v2 sources/runtimes remain available for historical comparisons. Use the sampled identities with `scripts/collect-legacy-bank.py`; external-policy provenance, four-seat targets and the rejection of legacy replay as on-policy neural data remain part of the R2 contract.

## Local release reproduction

Use the Windows game checkout for these commands. Keep the preserved R1 rules dependencies at `artifacts/r1-host/guarded`; the newly installed game assemblies are a different runtime profile.

```powershell
./scripts/build-patch.ps1
./scripts/build-legacy.ps1 -ManagedPath artifacts/r1-host/guarded
./tests/BrassLocalAI/run-runtime.ps1
./tests/BrassLocalAI/run-cash.ps1 -Mode regression
./tests/BrassLocalAI/run-tactical.ps1
./tests/BrassLocalAI/run-search-gameplay.ps1
./tests/BrassLocalAI/run-v2-save-compatibility.ps1 -PreviousRelease 1.3.2
./tests/BrassLocalAI/run-v2-save-compatibility.ps1 -PreviousRelease 1.4.0
./tests/BrassLocalAI/run-unity.ps1
./tests/BrassLocalAI/run-gameplay-live.ps1
```

Run each of `run-conversion.ps1`, `run-cash-beer.ps1`, `run-credit.ps1`, `run-demand.ps1` and `run-route.ps1` in `tests/BrassLocalAI/` both normally and with `-Research`. Run `run-conversion-matches.ps1 -Mode single -Seed 72001` and seed 72002, each with and without `-Revised`, followed by `merge-conversion-matches.ps1`; run its `-Mode live-budget` separately. These are complete games, not quick smoke tests.

The frozen-bank suite, `verify-v3-bank.py`, declared qualification report, R2 integration and admission check must refer to the same immutable bank. Existing evaluation records are create-once: write new output paths when repeating a platform check. After the active pool is admitted, the final package sequence is:

```powershell
./tests/BrassLocalAI/qualify-release.ps1
./tests/BrassPatch/packaging-regression.ps1 -PreviousRelease 1.4.0
./scripts/install-patch.ps1
./scripts/test-installed-local-ai.ps1 -Smoke
./tests/BrassLocalAI/verify-h5-provenance.ps1 -Installed
./tests/BrassPatch/CityLocator/verify-release.ps1
```

The installer requires the game to be closed and refuses missing or mismatched validation. Existing stock and prior-release backups remain available. No qualification gate is skipped by the installation sequence.
