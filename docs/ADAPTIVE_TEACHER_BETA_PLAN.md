# Adaptive Teacher Beta: Implementation Plan

Status: proposal only; no training behavior changes implemented.

Adapt the search-teacher policy-distillation weight to keep its shared-encoder gradient ratio near `share_target` (currently `0.05`). This is ongoing adjustment during teacher-on training, separate from the existing shadow beta qualification/re-measurement phase.

1. **Add an opt-in mode.** Keep fixed beta as the default. Initialize adaptive beta from the currently accepted beta. Adaptive updates must not trigger shadow training, beta re-measurement, or utility qualification. Keep existing re-measurement workflows separate.

2. **Reuse existing gradient measurements.** For each completed training iteration, average PPO actor and unweighted teacher KL gradient norms over the same valid sampled steps. Measure only shared state/entity encoders, before gradient clipping and Adam; exclude entropy and critic losses. Existing `teacher_norm` includes beta: record the raw norm directly or divide by that step's `beta_used`. Do not add backward passes or model copies.

3. **Update once per iteration.** Keep beta fixed within the iteration; apply the following update to the next iteration:

   $$
   G_{P,t}=\rho G_{P,t-1}+(1-\rho)\bar n_{P,t},\qquad
   G_{T,t}=\rho G_{T,t-1}+(1-\rho)\bar n_{T,t}
   $$

   $$
   \beta_t^*=\operatorname{clip}\left(s_{\mathrm{target}}\frac{G_{P,t}}{G_{T,t}+\epsilon},\,0.01,\,10\right),\qquad
   \beta_{t+1}=(1-\eta)\beta_t+\eta\beta_t^*
   $$

   Start with `rho=0.9`, `eta=0.1` per training iteration; these are trial settings. Initialize EMAs from the first valid paired measurements. With no valid positive measurements, retain beta and EMA state; missing data must not become zero gradients.

4. **Persist and report controller state.** Checkpoint active beta, both EMAs, and update count separately from the qualification receipt. Prevent fixed-beta restore/finalization from overwriting adaptive beta. After a separate re-measurement, initialize from its newly accepted beta and reset EMAs. Log beta used/next, raw norms, and the EMA ratio alongside the existing median ratio; those two ratio statistics are not identical.

5. **Validate before enabling.** Test convergence toward a constant target ratio, bounds, missing measurements, exact resume, and unchanged fixed mode. Run a short comparison against fixed beta, checking runtime overhead, held-out teacher KL, search improvement, and playing strength. A steadier ratio alone is not evidence of stronger play.
