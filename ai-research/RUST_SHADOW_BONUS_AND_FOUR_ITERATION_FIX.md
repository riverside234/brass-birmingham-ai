# Ordinary bonus transition and four-iteration shadow qualification

2026-11-07. The supplied log shows **g1258 committed**, with 320 ordinary games,
46 PPO steps, 512 teacher labels and one pending gradient observation. The next
ordinary collection failed in `append_compact` with `bonus state mismatch`.
Results were pulled inside `results/` to `d23483f`; its exported failures predate
this failure, so the attached terminal traceback is the evidence for this repair.

## Cause and repair

During a two-beer sale, drinking Gloucester merchant beer can enter the Develop
bonus while one beer is still owed. The sealed core's typed transition supports
that intermediate phase. Its external-state importer instead requires a bonus
to have no unpaid resources. `ordinary_after` exported and re-imported every
legal transition just to normalize its revision, hitting that disagreement.

Retain the owned typed transition when its revision is already correct. Preserve
the strict import of external origins, legal action/revision checks, transactional
rollback, scoring, hidden-deck/chance normalization and one revision per stage.
The same helper serves ordinary collection, forced previews and non-search
teacher progression. Both adapters must be rebuilt; the sealed core, search_v1
equations, budgets and selected actions are unchanged. Original qualification
and binaries remain historical evidence, with exact new adapter receipts.

## Four full measurements

`teacher_phase.term.shadow_iterations` now defaults to **4** and is 4 in the
current shadow config. Require four consecutive full **28 train + 4 held-out**
iterations with all four phases and finite nonzero shared-encoder gradient
observations. Persist `required_iterations=4`, the raw observations and fixed
`beta=clip(share_target/median(raw_ratio),0.01,10)` in the sealed receipt.
Targets 0.25, 0.4 and 0.5 remain available; subsequent monitoring does not adapt
the fixed beta. Shadow never applies teacher gradients or enables on-mode.

Restore the completed g1258 observation from the full checkpoint. Historical
two-iteration receipts stay readable; extending their requirement preserves
measurements but makes any two-iteration beta provisional until four qualify.
The exact 2→4 diagnostic config transition retains pending ordinary families,
teacher tables, label/value buffers and their original identities. Only the
ordinary collection binding projects this diagnostic setting to its historical
value; the saved training signature and native config binding record the actual
four-iteration config. No other config edit inherits that projection.

## Lab continuation

Sync these changes and fully exit the failed trainer. At the committed g1258
boundary, in the existing lab environment:

```bash
python scripts/prepare-rust-teacher-verification.py --expected-generation 1258 --offline && \
python scripts/prepare-rust-collector-performance.py --expected-generation 1258 --offline && \
python scripts/continue-rust-shadow.py --expected-generation 1257 --iterations 4 --share-target 0.25
```

The preparation commands rebuild only the affected adapters and run their short
wire regressions. They do not repeat B01–B07 qualification or reset the run. The
continuation guard remains **1257**, so the existing g1258 commit counts toward
the four-iteration boundary: run only **three remaining iterations**, through
g1261. Repeating the continuation command runs only unfinished generations.
If a later checkpoint has already committed, use its actual generation for
preparation. Keep pending cache files and original build evidence; the launcher
exports success/failure logs and qualification receipts to the Git-visible
`results/rust-lab-qualification/` directory.

## Debug evidence

The focused native reproducer covers the interrupted sale with both Develop and
Skip, exact typed-core successor/feature equivalence, compact/full paths, forced
acknowledgements and stale replies. Python regressions cover the 2→4 migration,
preserved frozen jobs/table seeds/cache files, the actual native router binding,
strict checkpoint restart, four-iteration gating and fixed/manual-target beta.
**40 distinct Python and 4 Rust focused debug checks passed.** The two new Rust
cases reproduce `bonus state mismatch` on the prior adapter and pass after the
repair; their completed-sale prefixes take 0.04s in total (a regression runtime,
not a throughput benchmark). See
`distribution/rust-collector-performance/shadow-bonus-fix.json` for the final
counts, source/build identities and raw debug log hashes. No local benchmarks,
live training sessions or unrelated successful qualifications are run for this
repair. The actual Linux/CUDA continuation remains Pending on lab.
