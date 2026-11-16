# KLENT regularization for search-teacher training

Implementation proposal prepared on 2026-11-14. The initial entropy setting and experiment lengths below are proposed trial values.

Apply KLENT's reverse-KL and entropy policy-improvement objective to the targets supplied by the existing search teacher. Search supplies action-value estimates, and the existing teacher term fits the policy to the resulting distribution during PPO. This adaptation focuses on policy-target construction and its integration with replay, gradient balancing, and continuation.

The intended benefit is better learning from the teacher's search budget through less concentrated policy targets. Establish that benefit through playing-strength and elapsed-time comparisons. The retained search workload still determines collection cost.

The source review used the attached paper, identified by its metadata as arXiv 2602.10894v2, and the complete released `main.py`, `resnet.py`, `util.py`, and dependency list at commit `b10ff63d3d22422d0eadaa8000261b6b314060b9`. Local integration points were inspected in the working tree at HEAD `9fd5841dc38896431865e127a5be2eba9fe1cac3`. [Attached paper](<C:/Users/BobXu/Desktop/ke.pdf>), [published paper](https://arxiv.org/abs/2602.10894v2), [pinned reference code](https://github.com/KazukiOhta/klent/tree/b10ff63d3d22422d0eadaa8000261b6b314060b9).

The paper and implementation establish the following details relevant to this revision.

| Detail | Source finding | Application to this plan |
| --- | --- | --- |
| Policy improvement | Equations 2 and 3 maximize expected action value minus reverse KL to the prior, plus entropy. | Use search estimates and the policy logits recorded with each search row. |
| Fitting | Equation 4 and `main.py` fit the complete improved distribution by cross-entropy. Q regression uses the action actually taken. | Apply the complete target through the existing policy-teacher KL loss, whose policy gradient equals that cross-entropy gradient. |
| Collection | The reference computes targets with a fixed parameter snapshot and samples training actions from those targets. | Freeze targets from each row's collection snapshot. Our hybrid obtains its training observations from the configured search action selector. |
| Returns | The reference uses lambda returns with `lambda = exp(-1 / tau)` and `tau = 8`. Its `gamma = -1` handles alternating two-player value perspectives. | Use the existing four-seat utility mapping and return implementation for Brass. |
| Network | The reference uses a shared ResNet with separate policy and action-value heads, zero-initialized final kernels, and tanh Q outputs. | Search already supplies the action-value vector required for this target update. The local utility and search-value heads provide state evaluations for that search. |
| Evaluation | Main learning curves use the greedy policy network. Additional experiments evaluate test-time search. | Evaluate the learned policy and the deployed search agent with explicitly matched inference budgets. |

These findings come from paper pages 3-8, Appendix A, and Appendices E-H, and from the reference's collection, return, loss, and evaluation code. [Paper](https://arxiv.org/html/2602.10894v2#S4), [reference training code](https://github.com/KazukiOhta/klent/blob/b10ff63d3d22422d0eadaa8000261b6b314060b9/main.py), [reference network](https://github.com/KazukiOhta/klent/blob/b10ff63d3d22422d0eadaa8000261b6b314060b9/resnet.py).

The paper's strongest theoretical results concern exact policy-update dynamics under stated game assumptions. Appendix C.10 also analyzes two-player general-sum games, and Remark C.13 discusses a possible multiplayer extension. Four-player Brass combines partial information, approximate search values, function approximation, replay, and PPO. Treat stability and strength in this hybrid as experimental outcomes. The paper's fourfold simulator-efficiency example compares 75 million with 300 million evaluations averaged across five games. It measures the complete KLENT algorithm. [Paper, Section 6 and Appendix C](https://arxiv.org/html/2602.10894v2).

For an admitted teacher row, let the legal actions be indexed by `legal_ids`. Let `z` be the recorded policy logits, `n` the completed visit counts, and `Q_search` the acting seat's search return for each visited action. Complete unvisited actions with the existing `v_mix` estimator.

$$
Q_{\mathrm{complete}}(a)=
\begin{cases}
Q_{\mathrm{search}}(a), & n(a)>0,\\
v_{\mathrm{mix}}, & n(a)=0.
\end{cases}
$$

The current target uses the recorded search scale.

$$
\kappa=c_2\bigl(c_1+\max_a n(a)\bigr),\qquad
p_{\mathrm{search}}=\operatorname{softmax}\left(z+\kappa Q_{\mathrm{complete}}\right).
$$

This is the KLENT closed-form update with entropy coefficient zero and KL coefficient `beta_KL = 1 / kappa`. Both the Rust search implementation and the Python admission oracle implement this expression. [Rust search mathematics](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/native/brass-ai-core/src/search/math.rs:78>), [Python oracle](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/teacher_labels.py:73>).

Introduce one dimensionless parameter `entropy_ratio = alpha_ENT / beta_KL`. Define the frozen prior over the legal actions as `pi_collection = softmax(z)`. The proposed target solves

$$
\max_p\ \sum_a p(a)Q_{\mathrm{complete}}(a)
-\beta_{\mathrm{KL}}D_{\mathrm{KL}}(p\|\pi_{\mathrm{collection}})
+\alpha_{\mathrm{ENT}}H(p).
$$

With `beta_KL = 1 / kappa` and `alpha_ENT = entropy_ratio / kappa`, its solution is

$$
p_{\mathrm{train}}=
\operatorname{softmax}\left(
\frac{\beta_{\mathrm{KL}}z+Q_{\mathrm{complete}}}
{\beta_{\mathrm{KL}}+\alpha_{\mathrm{ENT}}}
\right)
=\operatorname{softmax}\left(
\frac{z+\kappa Q_{\mathrm{complete}}}{1+\mathrm{entropy\_ratio}}
\right).
$$

Using logits in place of log probabilities is valid because their difference is a state-wise additive constant. This mapping retains the existing count-dependent search scaling and makes the new control a target temperature of `1 + entropy_ratio`. Its state-dependent regularization coefficients are a deliberate adaptation of the paper's constant coefficients.

Propose `entropy_ratio = 0.30` for the first candidate, giving target temperature `1.30`. This borrows the paper's ratio `0.03 / 0.1` while expressing regularization in the local search-value scale. Use `0.00` as the compatibility control and `0.10` as a smaller candidate if the initial analysis shows excessive spreading. The paper's absolute coefficients depend on its reward scale. [Paper, Section 6.1 and Appendix G](https://arxiv.org/html/2602.10894v2#S6.SS1).

| Quantity | Meaning in this implementation |
| --- | --- |
| `beta_KL = 1 / kappa` | Prior-policy regularization inside target construction. |
| `entropy_ratio` | Strength of target entropy relative to that prior regularization. |
| `TeacherTerm.beta` | Multiplier on the teacher gradient inside the PPO optimizer step. |
| `share_target` | Requested ratio of teacher to PPO actor gradient norms on the measured shared parameters. |
| PPG cloning beta | Policy-preservation coefficient in the PPG phase. |
| Search-value lambda | Mixing parameter for existing search-value return targets. |

Implement the revision through the following work items.

1. **Add the target setting and persistent continuation support.**

   Add `teacher_phase.target_regularization.entropy_ratio`, with a finite, nonnegative numeric value. Interpret a missing block as zero. Add the proposed continuation option `--teacher-target-entropy-ratio`; its parser default should be `None`, so omission uses the run's committed value. Explicit zero restores the compatibility target. The option is part of this proposed implementation.

   Extend the existing configuration validation and recipe comparisons to recognize the setting. Carry it through the `continue-rust-g1190-restart-2.py` delegation into `continue-rust-g1491.py` and `continue-rust-shadow.py`, including the configuration file those scripts generate. Validate the setting once at configuration boundaries.

   Include the normalized target recipe in the training signature and beta-calibration basis. Interpret missing legacy recipe metadata as zero. Collection identity should continue to describe the model, observation, search, reward, and action-selection recipe. Project out this learner-only target setting in collection comparisons so already sealed raw search work can be reused.

   Integrate this with the repository's existing source-transition mechanism, including teacher-on recipe recognition. Test the ordinary continuation entry point end to end, because the current validators compare complete recipe structures. This implementation should handle its authorized configuration transition automatically.

2. **Prepare stable targets from admitted search evidence.**

   Add a small pure target-construction function beside the existing oracle in `teacher_labels.py`. For a positive ratio, use the row's recorded `logits`, `counts`, `mean_returns`, `v_mix`, `c1`, and `c2`. Existing admission establishes their order and validity. Compute `kappa` from that row, including when processing an older buffer age.

   Use the existing stable float64 softmax for preparation and store the resulting detached vector in float32. Reconstruct scores from the recorded logits and completed values, which preserve information about legal actions whose exported float32 target probability underflowed. For a zero ratio, use the existing exported target directly to preserve baseline values exactly.

   The target covers the full ordered legal set. A zero visit count selects `v_mix`; the placeholder zero in `mean_returns` has no value-estimate meaning for that action. At batching time, padded actions receive zero target mass and the same legal mask used by policy scoring. Forced, cancelled, and evidence-free rows follow existing admission semantics.

   Keep the collection logits and completed search returns fixed throughout fitting and replay. Build the target from those recorded inputs on each cold load. Use the acting-seat return already exported in the row, with the current utility units and insolvency accounting.

   Extend the existing `LabelRow` representation with an optional prepared target and an accessor for the policy loss. Shallow row views can share the raw dictionary and encoded features. Cache one prepared vector per resident row and active target recipe, and evict it with the row. This gives linear preparation cost in the total number of legal candidates and approximately four bytes of persistent target storage per candidate.

3. **Use the prepared target in the existing teacher loss.**

   Both the standard `TeacherTerm.apply` path and the chunked `weighted_gradients` path should consume the prepared target. The teacher objective is

   $$
   L_{\mathrm{teacher}}=
   \mathbb{E}_{\mathrm{rows}}\left[
   D_{\mathrm{KL}}(p_{\mathrm{train}}\|\pi_\theta)
   \right].
   $$

   For fixed targets, this has the same gradient as cross-entropy fitting in KLENT. Its contribution to the existing combined update is `TeacherTerm.beta * grad(L_teacher)`. Compute the target outside autograd. The shared actor encoders and policy scorer receive this gradient through the current optimizer integration.

   Use the existing row sampler, age weights, long/short weighting where configured, and teacher rows per step. The inspected g1190 search-refresh variant uses 96 teacher rows per PPO step and long-search policy labels. Short searches still participate in game collection and search-value data. Apply the helper to a short policy label only in a recipe that already admits that profile.

   The experiment uses the current ordinary PPO objective and recorded behavior probabilities as controls. Teacher targets are supervised distributions on teacher observations. Collection actions, cotton mixtures, and conditional second choices retain their actual behavior provenance when forming ordinary PPO examples.

4. **Preserve search evidence and define buffer reuse precisely.**

   Store the original exported `teacher.target` and other search statistics in the existing sealed shards. Keep prepared learning targets in memory and reproduce them from raw evidence plus the committed target setting. This allows a ratio change to reuse the valid four-age policy-label window and its existing source, reward, scale, and lineage checks.

   The current teacher staleness rule compares policy drift with `KL(p_search || pi_collection)`. Continue using that raw search quantity as the freshness reference in `TeacherTerm` and in the older `SearchValuePhase.stale_ages` path. Report `KL(p_train || pi_collection)` separately. This makes the first experiment isolate the target-shape change from changes in age eligibility.

   Use the original search-improvement statistic for historical qualification context. Use the prepared target for fitting and effective-target held-out diagnostics. Audit every direct read of `label['target']` according to these two uses.

   Policy-target entropy is absent from search returns, rewards, and bootstrap calculations. Existing ordinary replay, search-value training data, and dedicated held-out return games remain valid inputs under the same collection recipe. A target-setting change requires rebuilding the in-memory target cache. Search, reward, or producer changes continue to use their existing compatibility rules.

   Apply a requested setting at a reproducible learner boundary. Reuse pending raw collection that matches the frozen model and job. If an interrupted learner checkpoint contains optimizer work under the prior target recipe, recover that work with its recorded recipe before applying the change. Journal the effective target recipe with the existing configuration transition so retries apply it once and preserve iteration accounting.

5. **Remeasure the teacher gradient weight with teaching active.**

   A target-shape change changes the raw teacher gradient norm. At adoption, schedule the existing teacher-on beta remeasurement with the previous active `TeacherTerm.beta` as the first iteration's weight. Use the new targets for every teacher gradient in that iteration. The existing full-table measurement path supplies its required held-out games.

   Clear the running teacher-share observations, outside-target streak, and adaptive gradient EMAs for the new target recipe. Retain the historical activation receipt as provenance. Bind the new measurement to the target-recipe identity, collection model, and measured generation, then accept its beta for the next iteration.

   Include the target-recipe identity in `TeacherTerm.beta_basis()` and the existing adaptive restore path. A repeated continuation with the same setting should resume the current beta and EMA state. Subsequent `share_target` edits should use the existing retargeting behavior with measurements from the active recipe.

   Reuse the sampled PPO/teacher gradient norms already produced by the update. Missing positive pairs retain the pending measurement and previous usable beta while teaching proceeds. The continuation uses the run's committed utility-readiness preference and search-head stage.

   Implement this through `adaptive_teacher_beta.py`, `TeacherTerm`, and the existing continuation state handling. The search-retune helper contains a useful teacher-on measurement pattern, while target adoption needs its own configuration transition because its data compatibility differs. [Teacher gradient integration](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/teacher_term.py:131>), [adaptive beta](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/adaptive_teacher_beta.py:60>).

6. **Make diagnostics comparable and inexpensive.**

   For held-out policy rows, calculate before/after fit metrics against both `p_search` and `p_train`, using the same policy forward outputs. Expose explicit names such as `search_target_kl` and `training_target_kl`. Preserve the meaning of the existing historical KL field and add the effective-target field alongside it.

   During target preparation, accumulate entropy, normalized entropy `H(p) / log(number_of_legal_actions)`, maximum probability, target-to-prior KL, and probability mass on unvisited actions. Also accumulate the expected completed-search return under the prior and each target. Group the report by search profile, era phase, and cotton versus standard tables using existing row metadata.

   The regularized objective trades expected value against entropy and prior proximity. Evaluate the entire objective when checking mathematical correctness. A positive entropy ratio preserves the ordering of fixed target scores, so top-action agreement is only a compatibility diagnostic. Track the distribution metrics as well as teacher-gradient share, beta used/next, and missing measurement counts.

   Reuse the current JSON report and export path. Include the target setting and recipe identity in the exported report. Aggregate statistics in bulk and reuse prepared vectors and existing held-out forwards. Add preparation time and target-cache bytes to performance reporting so the incremental cost can be measured. [Teacher lifecycle and held-out metrics](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/teacher_phase.py:294>), [report exporter](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/scripts/reports/export-training-reports.py>).

7. **Keep value learning consistent with the game's reward perspective.**

   The reference trains a per-action Q head with lambda-return regression. This hybrid obtains its action values from search, whose leaf evaluations use the existing utility and search-value heads. Use the current four-seat reward vectors, observer mappings, and search-value lambda of 0.8 in the initial experiment.

   The reference's alternating-player sign convention belongs to its two-player return representation. Brass return construction should continue through `search_value_rows.targets` and the current reward-timing implementation, which explicitly account for each seat's future utility and insolvency costs. Entropy enters the policy-improvement objective at the target-preparation step.

   The shared-encoder teacher gradient can still affect value predictions indirectly. Include held-out utility/search-value prediction error and calibration in the experiment report. The first trial uses the current search-head stage, blend, PPG cloning schedule, and epoch caps as experimental controls. [Reference return implementation](https://github.com/KazukiOhta/klent/blob/b10ff63d3d22422d0eadaa8000261b6b314060b9/main.py#L71-L83), [Brass return construction](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/search_value_rows.py:66>).

8. **Verify the mathematics and integration through the normal entry points.**

   Add focused regressions for the specific failure modes below, reusing the repository's existing teacher and resume fixtures.

   | Check | Expected result |
   | --- | --- |
   | Missing setting and explicit zero | Existing target values, loss gradients, row sampling, and resumed controller state match the compatibility path. |
   | Closed-form equivalence | Direct KLENT coefficients and the ratio expression agree for varied logits, counts, search scales, and legal widths. |
   | Legal support and completed values | Unvisited actions use `v_mix`, candidate ordering is preserved, and padded actions have zero mass. |
   | Numerical stability | Extreme finite logits, tiny probabilities, one legal action, and mixed-width batches produce finite targets and losses wherever the existing admission permits them. |
   | Objective properties | Target normalization, additive logit/value shift invariance, and nondecreasing entropy with ratio hold. The regularized objective at its solution is at least its value at the prior. |
   | Gradient semantics | KL fitting matches detached-target cross-entropy gradients. Standard and chunked paths agree for the same weighted rows. |
   | Replay reproducibility | Warm and cold loads produce the same prepared target for each raw row and recipe. Changing the learner model leaves a stored target fixed. |
   | Metric semantics | Search freshness uses raw search targets, while fitting and effective held-out metrics use the prepared targets. |
   | Teacher-on transition | The first adopted iteration teaches with the previous active weight, measures the new gradient, and commits the next beta with fresh EMA state. |
   | Retry and interrupted work | Cold resume, repeated flags, a stop after collection, and a stop after learning restore the correct recipe and apply the transition once. |
   | Configuration and export | The g1190 continuation entry point accepts, persists, resumes, and reports the setting through its delegated scripts. |

   As a planning check, 256 synthetic cases were evaluated with the existing NumPy oracle functions extracted from `teacher_labels.py`. Across legal widths 2, 7, 64, and 513, search scales 0.1 and 0.7, and ratios 0, 0.1, 0.3, and 1, the maximum difference between the two target formulas was approximately `2.78e-15`. Entropy changes were nonnegative and the regularized objective improved relative to the prior. This establishes the algebraic mapping on those cases. Training integration and strength validation are implementation deliverables.

9. **Run a matched experiment before choosing a training default.**

   Start the compatibility and candidate arms from the same committed checkpoint, including optimizer and controller state. Use 32 teacher training games and the current four-game held-out schedule. Match ordinary table mix, search counts and scales, cotton settings, hardware allocation, training budget, and evaluation seeds. Run teacher-on remeasurement in both arms so calibration freshness is also matched.

   First replay a representative sealed label window through target preparation for ratios 0, 0.1, and 0.3. Measure target-cache size, preparation time, entropy, and mass on unvisited actions. Include candidate-heavy roots and continuation decisions. Use this analysis to choose the first positive-ratio trial value.

   Then run a proposed 20-iteration pilot for ratios 0 and 0.3, or the smaller selected candidate. Extend to at least 60 iterations for a strength comparison if the pilot is operationally sound. Independent training seeds improve confidence where resources permit. Treat these counts as experiment budgets.

   Evaluate both models on the same held-out search rows, fixed opponent checkpoints, setup seeds, and seat rotations. Use the existing original-g1290 PL anchor and a fixed evaluation pool for cross-arm comparisons. Report paired win/rank outcomes, VP, policy entropy, value errors, and confidence intervals clustered by game or setup seed. Teacher-table VP and easier target fitting provide diagnostics alongside playing strength.

   Compare policy-only play at the same inference settings and search play at the same rollout or elapsed-time budget. Compare learning progress against both iteration count and total training wall time. Target preparation should fit within approximately 1% of iteration wall time as an initial engineering budget, measured on the actual lab workload. The main hypothesis is stronger or more stable policies per unit of search training.

   Select the positive ratio when the matched results support improved playing strength or reduced degradation over continued training at acceptable runtime cost. Setting the ratio to zero provides an immediate configuration-level fallback using the same raw evidence. Recovering a pre-experiment model uses its saved checkpoint and optimizer state.

The implementation touches the following existing components.

| Component | Planned work |
| --- | --- |
| [teacher_labels.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/teacher_labels.py>) | Pure target preparation and the lightweight prepared-target row view. |
| [teacher_term.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/teacher_term.py>) | Prepared targets in both loss paths, explicit metric meanings, and target-aware beta basis. |
| [teacher_phase.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/teacher_phase.py>) | Target preparation/cache lifetime, held-out target reuse, report integration, and transition state. |
| [teacher_phase_config.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/teacher_phase_config.py>) | Optional setting, normalized recipe identity, and target-only configuration compatibility. |
| [adaptive_teacher_beta.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/adaptive_teacher_beta.py>) | Measurement provenance and restoration tied to the target recipe. |
| [lab_shadow_config.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/rust_backend/lab_shadow_config.py>) and [resume.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/src/brass_ai/recovery/resume.py>) | Teacher-on recognition, learner-only configuration transitions, and pending-work compatibility through the existing resume mechanism. |
| [continue-rust-g1491.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/scripts/continue-rust-g1491.py>) and [continue-rust-shadow.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/scripts/continue-rust-shadow.py>) | Parse, persist, and forward the proposed option and schedule one target-aware teacher-on measurement. |
| [continue-rust-g1190-restart-2.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/scripts/continue-rust-g1190-restart-2.py>) | Document the new continuation option and its persistent behavior. |
| [export-training-reports.py](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/scripts/reports/export-training-reports.py>) | Verify inclusion of the new settings and diagnostic fields in the standard export. |
| [Recovery tests](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/tests/Recovery>) and [Rust backend lab tests](<C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/tests/RustBackendLab>) | Focused numerical, gradient, buffer, continuation, and crash-resume regressions. |

Completion requires a working continuation option, deterministic cold resume, compatibility-mode regression results, teacher-on beta remeasurement evidence, and a report of the matched experiment's strength and runtime results. The English plan records the proposed design and its validation work.
