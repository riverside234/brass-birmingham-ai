# Bounded network improvements to v4

Candidate **`bank-29b8fdbbd14ff00c7c88`**, based on selected v4 **`bank-3dc65253a610191d66a5`**. The game and training bank remain on the selected v4 until this candidate passes the two-control screen and deployment checks.

## Implemented scope

1. **Expansion depends on the current plan.** Network intent uses a small preference for city size and brewery-slot capacity. Industry intent uses compatible vacant slots, the observing player's supporting cards and merchant types reachable through the proposed connection. A prepared region/component map is shared across card variants within the proposal call. The total extra prior is capped at **0.75**. It changes proposal ranking, not official industry/link VP.
2. **Double rail is a complete alternative.** A private worker can construct at most **one** extra pair, using the legal host's second-link and resource/payment choices. It is scored with the same complete-turn economics as industries and sales; there is no double-rail score bonus. Both fees, coal payments and brewery beer are replayed. The completed plan must retain round-payment and investment working capital under the existing cash projection.
3. **Additional early canals have an opportunity cost.** An unsupported Canal connection costs **0.75 heuristic units**, capped at 1.5 per complete turn, only in **rounds 1–3**. The first owned canal is exempt. A connection opening actual later industry-card access, resolving an unavailable coal route for a later build, or improving a useful sale route is exempt. From **round 4 onward**, this extra penalty is zero; actual scoring and merchant-access links remain available. Terminal VP is unchanged.

The six strategy books, profile JSON, intent timeout/progress fixes, conversion, borrowing and rail-cost logic are pinned to v4. `NetworkOptions.cs` does not include Scout incentives, hypothetical wild hands, speculative brewery-demand forecasts or brewery-building candidates. Current legal actions remain intact.

## Search time and workers

Ordinary incumbents, industries, cash rescue, development/setup and sales retain their existing allocation. The optional pair is considered **after** those candidates, starts only before 45% of the clock, and stops at 50%; the normal completed fallback remains available. Fixed-work checks use the same bounded one-pair rule without a wall-clock cutoff. Timings and attempted/completed/rejected pair counts are merged from the private workers.

Game deployment retains **two** workers and **2/5/12 seconds**. Lab teachers retain **one second / four workers**, with **24 queued heavy tables** and the existing shared 60+4 CPU allowance. Neural Masked ISMCTS, inference batching, the fresh MLP-ResNet and the RL curriculum are unchanged.

## Focused checks and comparison

Thirty focused checks passed, including actual two-link rules replay: with a full coal market the pair paid **17 cash (15 in rail fees + 2 coal)**, reduced market coal by two and consumed one real brewery barrel. Checks cover bounds, supporting cards, compatible merchants, empty-slot VP, immutability, the first-canal and round-3/round-4 boundaries, enabling-build ordering, missing beer, working capital and expired clocks.

The frozen candidate uses the same six profiles and the same strategy/intent/economic source hashes as selected v4. Focused evidence: `artifacts/v4-network/focused-3.json`.

The prescribed screen is **24 full four-player games**, three predetermined setups/style pairs and four seat rotations against each control:

- Current selected v4: `bank-3dc65253a610191d66a5`.
- Economic v3: `bank-f5107c644af2fbf4ac99`.

All seats receive one second and four workers. Local execution uses two concurrent tables to limit CPU contention. Lab comparison can use 24 queued tables. Games never enter training replay. Promotion requires **strictly more official wins/first places, higher mean VP and a lower mean numeric rank against each control**; equality fails. This resolves the repeated VP wording in the request consistently with the preceding win/VP/rank rule.

```sh
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-network-v4.py --root /data/not_backed_up/yxu209/ai-training/ --bank bank-29b8fdbbd14ff00c7c88 --tables 24 --seed 844000 --output /data/not_backed_up/yxu209/ai-training/artifacts/v4-network/lab-screen
```

Local evidence is saved in `artifacts/v4-network/smoke-1/`. The report includes VP, official win share, first places, mean rank, completed MCTS simulations per decision/second, nodes, latency and optional-pair work. Throughput comparisons reflect the games' changing states as well as implementation overhead; they are not an isolated causal timing benchmark. Final results and deployment disposition will be recorded after the screen completes.
