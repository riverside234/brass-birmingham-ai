# Rule AI v5: funded higher-tier industries

V5 is a new candidate descended from the official v4
`bank-3dc65253a610191d66a5`. Its six profiles are named
`<style>-v5-masked-industry`; `training/opponents/candidates/industry-v5.json`
identifies the frozen candidate and executable. Neither Network+Scout nor the
Bounded Network candidate is its strategy parent. The dormant NetworkOptions
helper has no callers. Existing search/cache/deadline improvements are retained.

Active version restored on 2026-10-18: **`bank-357f50e6bcbdd119ea8c`**,
`funded-industry-v5/1.0.0`. The later scoring revision
`bank-ebd555a2e9794596dc11` is withdrawn and retained only as historical evidence.
The active source and six profile files match the original frozen snapshot.

## Strategy

| Industry | V5 preferred tier | Conditions |
| --- | --- | --- |
| Cotton | IV in both eras | Fall back to an economically useful, achievable tier when development, iron or financing is too costly. |
| Iron | Canal III; Rail IV | Retain market-shortage / income-recovery checks and actual iron prices. |
| Manufacturer | IV early; VI in Rail or mature economy | Mature means income >= 5 and at least seven remaining actions. |
| Coal | Canal II; Rail III | Retain the original shortage / recovery checks. |
| Pottery | Canal III; Rail V | Retain non-developable barriers; build the intervening mandatory pottery first. |
| Brewery | Existing supply logic | No speculative beer-demand or Scout additions. |

These are preferences in proposal/intent selection. Legal lower-tier builds
remain available to search. Higher level alone does not earn value or override
the full-turn outcome. Each action re-evaluates the opportunity.

`FundedDevelopment.Quote` checks the target card/site, era, buyer access, up to
two necessary connections, existing beer commitments, funding of necessary
breweries, total cash, one feasible loan, negative-income runway, and remaining
actions. Paid development retains a Ã‚Â£4 buffer. It examines up to six removals
(normally three develop actions); an already-open second removal is counted
correctly. Non-developable tiles cannot be skipped. Further levels of the same
tier do not justify unnecessary removals.

All removals and subsequent industry/brewery builds share **one** public iron
stock and market price ladder. The previous independent quotes could count the
same free cube or cheap market cube repeatedly. Plans use current public supply;
other players may consume it before execution, so the host's legal mask and
actual payment remain authoritative.

Development proposals put more weight on the feasible replacement's
value (0.95 versus 0.75), reduce the extra develop-action penalty from 1.5 to 1,
and raise the base value cap from 9 to 12 before priority multipliers. A positive upgrade gets a 20%
priority increase only when it needs no loan and the marginal iron bill is at
most Ã‚Â£2 per removed tile. The named tier ends the development intent bonus.
Mat options also recognize a funded Rail replacement behind an unusable
Canal-only tile; they still disappear at terminal states.

The scoring revision's income-eight pivot, extra coal discounts, stronger tier
multipliers, revised loans and barrel-allocation changes are no longer active.
Existing search/cache/deadline improvements and the corrected reachable-goal
regression fixture are retained. The source rollback receipt is
`artifacts/v5-development/rollback-original-20261018/rollback.json`.

## Build and check

From the project root, using the pinned Roslyn toolchain and private rules DLLs:

```sh
python scripts/build-industry-v5.py --root .
python tests/MaskedISMCTS/check_industry_v5.py --output artifacts/v5-development/validation --smoke
```

To check shared source before freezing a candidate, add `--current-source` to
the validation command. For six bounded, replayed supplier positions:

```sh
python tests/MaskedISMCTS/check_v5_scoring_positions.py \
  --runtime artifacts/legacy-ai/bank/<v5-bank-id>/runtime \
  --output artifacts/v5-development/supplier-scoring-check
```

Use a new output directory for each validation. Compilation is isolated under
`artifacts/v5-development`, then frozen under a new content-addressed bank.
The v4 bank, installed game DLL, selected opponent pool, completed IL data and
existing training checkpoints are not replaced by either command.

The game and lab compile the same `FundedDevelopment.cs`,
`InvestmentOpportunity.cs` and `TournamentOpenings.cs`. Therefore **a new
shared-runtime build uses v5 logic even if its profile ancestry is the v4 bank**;
the shared source receipt identifies that change. Old frozen-v4 executables
continue to reproduce v4. Existing shared runtimes must pass their usual source
freshness checks, rather than silently using new code under an old receipt.
Selecting the v5 bank as a training-pool default or installing it into the game
is a separate deployment action; the candidate has no strength qualification.

For a portable authored executable, after obtaining the new bank ID:

```sh
python scripts/build-shared-search.py --root . --bank <v5-bank-id> --compile --publish
```

The matching `BrassLegacyAI.exe` and `prebuilt.json` under
`training/runtime/shared-search/<v5-bank-id>` can be transferred to the lab with
the source and frozen bank. The lab requires the pinned private rules DLLs and
Mono; it can consume the matching prebuilt without Roslyn.

## Validation scope

Focused tests cover cash and credit, increasing iron costs, shared iron stock,
cards/buyers, non-developable pottery, finite development/action limits,
unchanged input observations, terminal behavior, and ending development bias
at the chosen tier. Existing industry, economy, game-runtime, parallel/cache/deadline
and shared game/IL/self-play integration checks are also exercised. The smoke
uses four real early/late Canal/Rail positions with a 2-second shared whole-turn
budget and one search worker, replaying all selected actions through the rules
host. This verifies implementation and legal execution, not stronger play or
a measured increase in average built/sold industry level over full games.

## Original v5 validation (2026-10-18, before scoring revision)

Previous candidate: `bank-357f50e6bcbdd119ea8c`. Its original validation
recorded 160 passing native assertions (21 v5, 17 existing industry, 99 game
runtime, 23 parallel/cache/deadline), 21 shared-search/prebuilt Python tests and
two isolated-bank tests. The [full report](../artifacts/v5-development/checks-04/report.json)
contains all selected turns and focused test results.

| 2-second / 1-worker smoke position | Completed simulations | Interrupted | Native search ms |
| --- | ---: | ---: | ---: |
| Canal round 2 | 11 | 0 | 1985.9 |
| Canal round 8 | 31 | 0 | 1954.9 |
| Rail round 4 | 5 | 0 | 1960.4 |
| Rail round 8 | 12 | 0 | 1985.8 |

All four complete turns replayed legally, with zero search failures. The first
end-to-end IPC call took 2.297 seconds including startup; later calls took
1.938Ã¢â‚¬â€œ2.000 seconds. Extra investment checks can reduce completed search work
under a fixed clock; this small smoke is not a throughput comparison to v4.

Portable shared executables were compiled for both the explicit v5 bank
(`search-f15761358204bd2531999837`) and the existing v4 **profile ancestry**
(`search-a3605a51ae5ec22ffebcc3ae`). Both bind the new shared v5 source. The
historical frozen v4 executable, selected pool and installed game DLL were
verified unchanged. No training or full-game tournament was launched.

The unselected intermediate draft `bank-f762959bfd1da972a5dc` and tested prior
v5 `bank-357f50e6bcbdd119ea8c` are retained. The candidate selector identifies
the restored original v5; neither frozen bank is overwritten.

## Withdrawn scoring revision validation (historical, 2026-10-18)

Withdrawn candidate: **`bank-ebd555a2e9794596dc11`**, strategy
`funded-industry-v5/1.1.0`. The
[compact receipt](../artifacts/v5-development/rollback-original-20261018/withdrawn-revision/training/opponents/candidates/industry-v5-validation.json)
recorded **231 native assertions** (40 new scoring, 21 v5, 37 economy,
17 industry, 93 game-runtime and 23 parallel/cache/deadline) and **23 Python
tests**, all passing. The frozen source matches the tested source. The new
shared policy files occur exactly once in the game/IL/self-play source list.

Four balanced 2-second / 1-worker roots completed 10, 32, 4 and 11 simulations
(Canal rounds 2/8, Rail rounds 4/8). All selected turns replayed legally with
zero failures/interrupted simulations. Native search times were 1.898Ã¢â‚¬â€œ1.975s;
the first end-to-end call took 2.266s including process/JIT startup.
[Full check report](../artifacts/v5-development/scoring-revision-20261018/final-checks/report.json).

Six additional supplier positions were restored from the previous tournament;
replaying the original moves reproduced the original result exactly. Each
position then received one 2-second / 1-worker old/new comparison:

| Version | Completed simulations | Interrupted | Total end-to-end seconds |
| --- | ---: | ---: | ---: |
| Previous v5 | 51 | 2 | 12.094 |
| Revised v5 | 47 | 2 | 11.827 |

All 12 compared turns were legal with zero search failures. At Canal round 6,
the revised policy selected Cotton development instead of the earlier unsold
Pottery commitment. At Rail round 4/income 11 it selected financing instead of
another coal mine. Two other tested positions retained coal, as the preference
is not a prohibition. The completed-work count was about 8% lower in this tiny
sample; there is no speedup claim. Changed search trees, startup and clock noise
also affect this measurement. No full-game playing-strength claim follows.
[Paired supplier report](../artifacts/v5-development/scoring-revision-20261018/final-supplier-probes/report.json).

An older economy test required Cotton IV despite a mat needing eight removals,
above the six-removal cap. It failed against the frozen previous v5 too. The
test now checks an actual funded, reachable high-tier goal while preserving its
intent-progress and stall checks.

Portable shared bundles were rebuilt for the new v5 bank
(`search-be15045801937d56277a31e9`) and the existing v4 profile ancestry
(`search-3cec4897175e532fd8012fe7`). These were built for the withdrawn revised source;
the original frozen executables remain unchanged. All 330 protected files,
including the previous banks, selected pool and installed game DLL, matched
their pre-change hashes. This revision is frozen and compiled, but has not
been installed into the live game or promoted to a training-pool default.

The scoring-revision section above describes the withdrawn experiment. Current
shared bundles for the original v5 and v4 profile ancestry have been restored
to the original v5 source. No archived frozen bank is modified by the rollback.
