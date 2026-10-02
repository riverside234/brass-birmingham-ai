# Masked v3: era economics and parallel execution

This revision implements separate canal and rail economic priorities, shared
Masked ISMCTS with **two game workers/four lab workers**, and the **24-table** lab configuration. The user
explicitly requested installation after the tied comparison, replacing the
earlier win-only condition. The final game update is patch **1.4.16**; it changes
the game allowance from four to two while preserving the 1.4.15 strategy. The UI
and rules binaries are preserved byte-for-byte. Existing Easy/Mid/Hard clocks remain 2/5/12 seconds;
older saved ten-second assignments remain readable.

## Why the first candidate scored 40.7 VP

The first eight-game screen tied 4–4 against the previous frozen Masked bank.
The candidate averaged 40.6875 VP; the control averaged 42.125. In the 16
candidate seat-games, 150 of 256 rail-era actions were development, with only
27 builds and four sales. Candidate seats finished with mean cash £28.50.
The control also wasted actions: 88 developments, 21 builds and 49 passes.

This evidence identifies a proposal/evaluation problem. The old heuristic
always gave cotton/brewery development a positive ranking, treated passing as
approximately -100, and generally favored borrowing only below £15 cash.
It could discard useful stock repeatedly, or fail to fund an expensive
investment despite having more than £15. Short search inherited those priors
and complete-turn incumbents. Increasing search workers did not repair their
economic meaning. These findings do not establish a rules-engine defect or
prove that ISMCTS itself is the cause of weak play.

The second candidate replaced those rules and screened on new seeds. It built
117 rail-era industries across 16 seat-games, but lost 3–5, averaging 76.75
versus 80.375 VP. These are different setups from the first screen: the raw
score increase is **not** a matched measurement of improvement. The final
candidate also removes a duplicated link discount and restores Rail Tempo's
soft preference. Read the immutable reports under
`training/opponents/evaluations/masked-era-*/report.json` for outcomes;
all failed/tied candidates remain separate evidence.

The third screen tied 4–4, averaging 77.25 versus 73.625 VP. The final
free-development revision was screened again on the same two setups and four
seat rotations: **78.75 versus 76.0 VP, four wins each**, average rank 2.375
versus 2.625. Both versions used one second/four workers. The candidate built
7.0625 rail-era industries per seat-game versus 2.4375 for the control.
Its mean measured whole-turn time was 1.097 seconds, maximum 1.438 seconds:
completion/worker retirement can overrun the nominal search clock.

The final candidate's mean VP is 1.5 above the third candidate's screen, but
the control's score also changed and the VP advantage narrowed. Timed parallel
search is not bit-identical across runs. There is **no demonstrated win-rate
increase or statistically established superiority** from the bonus revision.
The final bank is `bank-3f7221cadcd472aafdb9`; its immutable receipt is
`training/opponents/evaluations/masked-era-1.3.0/report.json`. It is active for
training and shares search/evaluation sources with game patch 1.4.16.

## Economic changes

- Canal: retain income, sale/completion deadlines and viable level-two
  carry-over investments.
- Rail: rank construction with the declared engine's coal/iron quotes and
  actual single/double-rail base charges. Compare complete industry plans
  alongside complete rail plans, and price scarce beer as an opportunity.
- Keep cash valuable when it can fund a credible investment; price likely
  additional financing actions separately from money already spent.
- Industry and adjacent link VP use the **same** uncertain flip event.
  Conversion forecasts already discount competition and deadlines; link
  scoring does not apply that uncertainty a second time. Secured points are
  recorded separately from forecast points.
- Development must unlock a reachable, fundable replacement. Up to two low
  tiles may be cleared in one action. Identical-level copies and the final
  useful tile receive no automatic development reward.
- Borrowing proposals consider the quoted funding gap of an identified
  investment. Passing is a legal fallback when every available expenditure
  is wasteful. Complete simulated afterstates still determine the result.
- Free merchant development ranks the change in reachable investment value,
  including cards, a suitable buyer, funding and time for construction/sale.
  It has no iron/action charge and adds no bonus twice to the final score.
  A legal skip preserves stock when the upgrade is unusable. The bonus fixture
  now includes a genuine pottery card, funding and a pottery buyer; the previous
  fixture offered an upgrade that could not be sold. Styles may select another
  useful upgrade; the balanced fixture also verifies the level-three pottery.

The industry/cash prospects are conservative heuristics, not exact future
profits. Card use, payment, market transactions, sale paths and final scoring
are executed through the existing declared rules host. All opponents retain
their own permitted observations; no opponent hand or future draw is exposed
to the real decision. The six styles retain preferences rather than hard
industry/rail quotas. Pottery Flex's opponent identity remains evaluation-only;
learner Pottery actions and exploration are unrestricted.

## Independent game/lab workers and bounded queues

Lab heuristic v3 uses one worker process per policy with four private C# root
owners; the game coordinator uses two. Each owner has its own hypothetical
states, RNG and tree. The final
merge combines only completed raw four-seat returns and complete-plan
evidence. It does not merge opponent nodes or pretend interrupted work was
completed. The owners retire before snapshots or UI resource scopes release.

A shared CPU heuristic queue coalesces identical permitted inputs and drains
bounded batches outside its lock. It is a CPU scheduling/cache mechanism,
not GPU neural inference. It includes observer, hand and public state in its
keys, returns independent arrays, and reports cache/coalescing/queue counts.

NN search also uses four private owners, with private reply channels feeding
one bounded central CPU/GPU inference queue. Pending paths carry a small
**selection-only virtual loss of 0.02**; completed visits/returns are unchanged.
Reservations release on completion, expiry or cancellation. Node ownership
is checked; shared channels/model registries and final decision access are
locked. No tree locks or virtual loss across separate roots are required.
Partial batches flush within 1 ms or the earliest request deadline. The
additional Masked NN probe uses this same central service.

The lab can queue 24 heavy tables, each requesting four workers. A FIFO,
spawn-safe admission gate allows at most **60 active logical work slots**,
leaving four of the 64-core allocation for the learner/services. Ordinary
turns request one slot; a search forest requests all four atomically. A
whole-turn search clock starts **after admission**, so queue waiting cannot
consume its thinking time. Every lease releases in `finally`. This is a
planning allowance, not OS affinity: monitor actual contention and timing.

## Training and evaluation

Imitation generation retains searched Masked v3: one second per whole turn,
four roots, 24 queued tables. Audit complete trajectories before BC. Neural
weights, optimizers and populations start fresh; an already audited entirely
heuristic dataset can be reused with its archived provenance.

Before 10,000 completed RL games, v3 opponents use a separately identified
**search-free incumbent** variant. These games run in the ordinary actor pool;
they provide external opponent actions, never PPO likelihoods or search visit
labels. At 10,000 games, the curriculum reaches 5% v3 tables and switches
those opponents to searched one-second/four-worker v3. Imitation and manual
v3 evaluation always use the searched variant.

Ordinary self-play uses a bounded semantic action-feature cache with exact R2
tensor parity, lower-cost gzip replay compression, persistent hosts and
asynchronous shard writing. Integrity seals and transition audits remain.
NN distillation retains Masked ISMCTS + the fresh 1.84M MLP-ResNet, four roots,
K 16/8, raw Q and the supported information frontier. It does not inherit
heuristic Q normalization or reply/depth caps. Search labels require completed
support; insufficient work reports fallback and creates no fabricated label.

Routine evaluation remains eight games every 1,600 training games, with a
16-game broader screen every 6,400. Evaluation never enters training replay.
Retention and same-run resumption remain enabled. Use a **new run directory**
after this source/config change. See [the conda lab guide](TRAINING_RECOVERY_LAB.md).

## Exit evidence and installed release

The final two-worker package passed 94 Masked runtime checks, 22 era/forest checks,
10 execution checks, 442 conversion/merchant checks, 72 credit checks,
60 cash/beer checks, and 1,240 native checks. Genuine generated saves from
1.3.2/1.4.0/1.4.3 passed 54 checks each; 1.4.9/1.4.11/1.4.15 passed 252 each,
covering all six profiles and three clocks. Artificial fixtures do not modify
user saves. The installed binaries are verified against the qualified package;
the prior 1.4.15 and 1.4.13 binaries are retained for recovery. Saved strategy
identities remain `brass-local-ai/1.4.15`; 1.4.16 changes only game worker count.

Python's nine targeted checks passed. The fresh 1.84M NN integration exercised
imitation, PPO, exact resume, evaluation and actual search distillation; the
final heuristic bank separately generated and audited a 332-row game with
four workers and rejected the held-out Pottery opponent. CPU preflight passed
on the refreshed distribution. These local checks do not establish CUDA
throughput or contention at the lab's full 64-core/24-table scale.

A fresh untrained NN four-worker probe verified central batching and actual
completed evidence on repeated decisions. With two in-flight paths per owner
and a ten-second CPU budget, the warm repetition completed six simulations
and produced a supported search target; the cold repetition completed none
and produced no teacher. Eight paths per owner completed none in the tested
three-/ten-second cold CPU probes. This is throughput/fallback evidence, not
a strength result or a measured CUDA recommendation. The optional probe
flags `--repeat 2 --pending-paths 2` expose this comparison without changing
the main lab recipe. Owner trees remain private and forests fully retire
before the next request.

The private transfer bundle now contains **565 runtime files**, including the
new bank and previous Masked/S1 controls. Its SHA-256 is
`dc71a796a2dda932175e795ddfadd5441807d16bec7d8b2e679009a84bc14954`.
Authored workers/profiles/source remain tracked; proprietary DLLs stay in the
private bundle. Neural training still starts from random weights.

Parallel MCTS references: [original ISMCTS paper](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)
and [Cazenave's Batch MCTS paper](https://ludii.games/citations/ARXIV2021-1.pdf).
These motivate private-root information-set search and batched evaluation;
they do not establish the strength or speed of this project's implementation.
