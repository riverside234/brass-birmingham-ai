# Feature engineering addition: lightweight hand matching

**Date:** 14 October 2026  
**Status:** Historical experiment; hand coverage rolled back on 15 October 2026. The active pipeline uses A–F only and five residual blocks. See [the rollback report](SIX_GROUP_FIVE_BLOCK_REPORT.md).  
**Prerequisite:** Review the completed [six-group feature revision](FEATURE_ENGINEERING_REVISION_PLAN.md) and reuse its contracts, derived context and caches.

## 1. Goal and scope

Expose how current hand cards support current legal building opportunities, including alternatives and dependence on wildcards. Use factual coverage relationships; do not assign strategic importance scores or predict future draws/network expansion.

Revise the subsequent **IL and Step 5 self-play pipelines**, not just the encoder. Preserve current reward/head semantics, architecture/width except necessary input integration, training settings, search budgets, worker/resource controls, evaluation cadence and retention. Report the resulting parameter count. Do not modify the installed game AI or launch long training.

## 2. Compact feature contract

Build one mapping per applicable observation: `building opportunity -> supporting cards in hand`. Derive it from the complete legal candidates already generated. Deduplicate equivalent construction outcomes using explicit keys, such as slot and industry plus any other necessary outcome fields. Preserve all original policy choices, IDs, labels and masks.

| Location | Required additions |
| --- | --- |
| State | Distinct current build opportunities supported by each hand-card type; number supported only by wildcards; applicability/validity masks. Reuse existing hand vocabulary/counts. |
| Action | For identified card consumption/discards: opportunities losing all support, and opportunities losing their last ordinary-card support while retaining wildcard support. |
| Build action | Whether alternative ordinary cards or wildcards support that same build; explicit validity where unresolved. |

Respect duplicate copies: consuming one copy differs from consuming the last. Evaluate multiple-card removal jointly without double counting. These deltas describe card coverage on the **current, fixed-board opportunity set**, not exact post-action legality. Completing the selected build must not count as losing an unused opportunity.

During payment/bonus/subchoice phases, no Build candidates does not mean no building opportunities. Mark the summary unavailable unless an exact, still-valid source exists. Unknown and genuinely zero must remain distinguishable.

Compute coverage once and reuse bounded set/bitset operations across candidates. No additional full legal enumeration, simulator clones or future search. Use only the observer's permitted information. State-dependent coverage must not enter a cache keyed only by card/action text; invalidate on relevant board, hand, phase and commitment changes.

## 3. Implementation order

- [x] **HM0 — Inspect and specify.** Trace the completed reference/optimized encoders, action representations, schema, caches and active IL/self-play entry points. Define deduplication, masks, normalization, fields and parameter impact. Audit the three existing-feature semantics in section 4. Record a small baseline.
- [x] **HM1 — Encode and version.** Implement coverage and candidate deltas, shared-context caching and reference/optimized parity. Correct confirmed defects from the semantic audit with minimal changes. Introduce compatible feature/model versions; update batching, shared-memory shapes and relevant export checks. Do not overwrite existing schema identities or silently pad/truncate old tensors.
- [x] **HM2 — Integrate training.** Complete the IL, handoff, self-play, teacher, evaluation and resume requirements below. Encoder-only completion is insufficient.
- [x] **HM3 — Verify and document.** Run focused regressions and a small end-to-end smoke; measure incremental overhead; update lab commands and this checklist with evidence.

### HM2: required downstream changes

**IL:** Update loading, feature preparation, batching, fresh model initialization, training and validation. Re-encode compatible existing raw v4 demonstrations; create new caches, manifests and receipts while preserving records, action labels and setup splits. New feature dimensions alone do not require new teacher games. Explicitly report insufficient raw inputs. Preserve old datasets/runs and reject incompatible encoded artifacts.

**IL -> self-play:** Support the new IL checkpoint as the initializer for a new self-play run. Apply the same feature semantics to rollout actors, dynamic inference batching, shared-memory transport, replay writing/loading, PPO updates, Masked ISMCTS NN teachers and evaluation. Preserve search deadlines and existing performance optimizations.

**Models and targets:** Admit only compatible neural historical checkpoints into the new run; preserve rule-based v4 opponent support. Follow existing bootstrap/fallback rules when compatible history is unavailable. Collect fresh PPO rollouts and NN teaching targets; do not silently reuse incompatible replay or teacher caches.

**Resume:** Fresh training does not require migration of old model weights/optimizers. Checkpoints produced by the revised pipeline must resume with their own weights, optimizer state, training counters and scheduling state intact. Reject old incompatible checkpoints clearly; do not relabel their identities or overwrite old runs.

## 4. Focused acceptance checks

### Required review of existing feature semantics

These checks verify the completed six-group implementation; they do not automatically authorize additional feature families or expensive future search. Check the actual encoded information as well as the underlying rule calculation. Report each check as passed, corrected or unresolved with evidence.

| Check | Required verification and focused regression |
| --- | --- |
| **Who benefits from action consequences?** | Verify that guaranteed flips and income changes are attributed to the affected player, including when consuming an opponent's resources. Immediate bonus VP and deferred industry-scoring potential must remain distinct. Compare own-resource and opponent-resource consumption against legal engine transitions, including depletion of the last resource and stable/observer-relative seat mapping. Ensure aggregate effects do not incorrectly credit the acting player or obscure the recipient where the feature contract promises attribution. |
| **Are resources usable for this particular action?** | Apply the actual rules for the selected sale, link or other action, including source ownership, required connectivity, merchant eligibility and previously committed consumption. Equal global resource totals must not imply equal action feasibility. Test accessible versus inaccessible resources, applicable merchant beer, and two opportunities competing for the same beer. Individual feasibility must not be reported as joint feasibility. Reuse existing calculations; do not add another full legal enumeration. |
| **Can a development target still be realized?** | Verify that stack unlock/era eligibility is distinguished from construction affordability and practical realization. Check that existing finance, remaining-action and sale/beer features retain the information needed to distinguish the same higher-tier target in an early affordable position from a late or constrained position. Cover incomplete development, era restrictions, insufficient remaining actions and unavailable sale resources where applicable. Do not encode guaranteed future construction/flipping from stack progress alone, or assume opponents leave resources available. Use rule facts and explicit unknowns, without a handcrafted profitability score or extra planning search. |

Fix reproducible accounting, attribution, applicability or cache errors within the existing groups. If a confirmed semantic gap requires a new field, document the smallest correction and version it through HM1/HM2; do not silently change tensor meanings. Tests establish correctness and distinguishable inputs, not that the trained policy will necessarily choose well.

### Hand matching and pipeline regressions

- Ordinary-card alternatives, ordinary-plus-wildcard and wildcard-only support; duplicate copies and last-copy removal; multi-card removal without double counting.
- Opportunity deduplication without changing choices/labels; completed build excluded from unused-opportunity loss; phase applicability and stale-cache invalidation.
- Reference/optimized and cached/uncached parity; observer privacy; raw demonstration re-encoding and label/split preservation; incompatible artifact/history rejection.
- A bounded smoke covering fresh IL, IL handoff, rollout/PPO update, NN teacher inference, evaluation and new-checkpoint save/resume. Report paths not exercised; no long campaign.
- Compare incremental encoder latency and memory against the current six-group pipeline on identical positions/candidates, including large candidate sets. Report any measured search-throughput regression with hardware/settings and startup/warm-cache distinctions. Do not infer playing strength from these checks.

Update the existing feature lab guide with verified commands for raw-data audit/re-encoding, fresh IL, IL evaluation, new self-play and subsequent resume. Inspect actual CLI options; do not invent flags. Use conda environment `ai-training` and lab root `/data/not_backed_up/yxu209/ai-training/`.

## 5. Execution prompt

```text
Implement HM0-HM3 in ai-research/FEATURE_ENGINEERING_ADDITION_PLAN.md.
Reuse the completed six-group revision. Include the downstream IL/self-play
integration, the three existing-feature semantic checks, focused regressions,
small smoke/benchmark and updated lab instructions.
Preserve current training semantics/settings and existing datasets/runs.
Do not modify the installed game AI or launch a long campaign.
```

## 6. Completion evidence

See [the HM contract](HAND_MATCHING_CONTRACT.md), [implementation/validation report](HAND_MATCHING_REPORT.md)
and [updated lab guide](../docs/FEATURE_ENGINEERING_LAB.md).

| Stage | Evidence |
| --- | --- |
| HM0 | Before-source/config pins and isolated six-group runtime; identical 15-position/1,361-candidate corpus. All three native/encoded semantic checks passed; no A–F correction indicated. |
| HM1 | Immutable Features 3 schema; dimensions 5,685/576; width 304 retained; base 2,643,597 parameters. A–F prefix parity, duplicate/joint removals, masks, stale-cache rejection, shared transport and dynamic export checks. |
| HM2 | Two existing raw heuristic games re-audited: 691 positions/27,866 candidates, labels/splits preserved. Fresh IL, IL/rehearsal config binding, PPO, v4 opponents, NN teacher probes, evaluation and new-version Adam/RNG resume exercised. Old artifacts/history rejected. Versioned cache retention pins/cleanup/restore fixed. |
| HM3 | 216 focused tests passed. Optimized six-game smoke 428.0s; final-source two-game smoke 226.375s; four diagnostic evaluation games and four teacher-turn probes in each. Final named ONNX parity passed. CPU persistent-miss median +42.8% and Python allocation peaks +9.4–13.7% in the final small paired benchmark. |

Machine-readable final receipt: `artifacts/feature-engineering/hm0-hm3/review.json`.
The six-game smoke preceded a retention-only source change; final-source revalidation
is recorded separately. Teacher probes generated four qualified diagnostic labels
per final smoke, used zero for training and had zero policy drift. Full scheduled
five-game teacher/distillation, CUDA/production concurrency and playing strength
remain unmeasured end to end; functional completion does not claim those outcomes.
