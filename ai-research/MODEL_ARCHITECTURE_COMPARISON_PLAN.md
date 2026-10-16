# Architecture Comparison Plan

**Scheme 3: Wider Whole-Group Projections vs. Scheme 1: Industry Subgroups with a Whole-Industry Bypass**

Date: 2026-10-15. Status: design review only; neither candidate is implemented by this document.

## 1. Decision requested

Choose the architecture with the stronger expected **eventual playing strength** for this Brass: Birmingham policy/value model, given adequate training. Implementation simplicity and inference latency are relevant constraints, but they are not substitutes for playing strength.

The two primary candidates are:

- **Scheme 3 / W352:** retain whole-group projections, widen the industry and map representations, widen the shared backbone from 304 to 352, and reduce its residual blocks from six to five.
- **Scheme 1 / I304:** add nonlinear per-industry branches alongside a learned whole-industry projection, widen the map representation, and retain the 304-wide, six-block backbone.

**W336** is a secondary width setting within Scheme 3. Its parameter count is almost identical to I304, allowing the reviewer to separate architectural preference from model size. It is not a third architecture family.

These are concrete comparison specifications, not a predetermined recommendation. No comparative training, playing-strength evaluation, or latency benchmark has been performed for these candidates. Parameter counts and dense multiply-accumulate (MAC) estimates below are analytical calculations from the inspected implementation.

The main question is where extra capacity is most useful: **industry-specific nonlinear processing before compression**, or **wider representations and a wider shared model for interactions across all feature groups**.

## 2. Verified current implementation

The current model is a grouped MLP residual network, not a Transformer or graph network. Both candidates retain this model family.

| Property | Current implementation |
| --- | --- |
| Architecture identity | `brass-four-seat-mlp-resnet/9.0.0` |
| Feature identity | `permitted-features/4.0.0` |
| Derived-feature identity | `permitted-derived-context/3.0.0` |
| Group layout identity | `grouped-state-layout/5.0.0` |
| State input | 3,128 dimensions |
| Candidate action input | 327 dimensions |
| Six projected group widths | 32 / 64 / 48 / 192 / 224 / 48 |
| Context passthrough | 12 dimensions |
| State fusion | 620 → 304 |
| Shared residual backbone | Width 304, six blocks |
| Base-model parameters | 2,005,917 |
| Parameters including utility residual head | 2,015,937 |

The former two-million-parameter restriction is not a selection criterion. Both proposals remain below the currently documented three-million-parameter ceiling.

### 2.1 Input groups

| Group | Input dimensions | Current projected dimensions |
| --- | ---: | ---: |
| `turn_global` | 31 | 32 |
| `cards_history` | 218 | 64 |
| `player_economy` | 109 | 48 |
| `industry_development` | 1,314 | 192 |
| `board_resources` | 1,247 | 224 |
| `current_action` | 197 | 48 |
| `policy_context` | 12 | 12, passed through |

The state vector combines categorical indicators, counts, scaled numerical values, and derived features. Many derived facts use a three-component value/known/applicable representation; selected uncertain facts retain additional bounds. Neither proposal adds embeddings, numerical buckets, or new raw features.

The inputs already encode substantial domain semantics, including development targets, resource and saleability information, costs, and differences in potential rewards. Extra local nonlinear processing could still help, but raw input dimensionality alone does not establish where the current model loses useful information.

### 2.2 Industry structure

The 1,314 industry dimensions can be partitioned into six disjoint, semantically defined groups of 219 dimensions each:

`Brewery`, `CoalMine`, `CottonMill`, `IronWorks`, `Manufacturer`, `Pottery`.

Each industry subgroup contains:

| Contents of one industry subgroup | Dimensions |
| --- | ---: |
| Current player's raw industry mat | 36 |
| Three opponents' raw mats, retaining separate relative seats | 108 |
| Current player's derived industry/development facts | 75 |
| **Total** | **219** |

A raw mat contains an eight-bin level histogram and two 14-dimensional tile descriptors. The 75 derived dimensions represent 25 facts with their known/applicable information. The proposed split is **six industry groups containing all four players' relevant mats**, not 24 independent player-by-industry groups.

### 2.3 Map structure

The 1,247 map/resource dimensions comprise:

| Contents | Dimensions |
| --- | ---: |
| 49 slots × 18 owner/industry/tile features | 882 |
| 69 links × 4 ownership features | 276 |
| Nine merchants × seven acceptance/beer features | 63 |
| Eight derived realization facts × three components | 24 |
| Coal and iron market values | 2 |
| **Total** | **1,247** |

Both candidates retain a **single whole-map projection** and increase its output to 288. Fixed slot, link, and merchant identities remain intact. Neither candidate introduces pooling, map subdivision, or a graph representation.

## 3. Common network conventions

Unless explicitly changed below, preserve the current architecture and training semantics.

- Ordinary group projection: `Linear(input_dim, output_dim) → GELU`.
- Fusion: concatenate the six group outputs and the 12 context dimensions, then apply `Linear(fusion_input, H) → GELU`.
- Residual block: `x + Linear(H,H)(GELU(Linear(H,H)(LayerNorm(H)(x))))`.
- Retain the current initialization that zeros the final linear layer in each residual branch.
- Apply the existing final `LayerNorm(H)` after the residual blocks.
- Action encoder: `327 → 128 → GELU → LayerNorm(128)`.
- Policy scorer: `(H + 128) → 128 → GELU → 1`, with existing legal-action masking.
- Official outcome head: `H → 128 → GELU → 12`, preserving four-seat win/rank/VP semantics and seat mapping.
- Utility residual head: `(H + 4) → 32 → GELU → 4`, retaining the insolvency-budget inputs, deterministic outcome-based prior, and existing output initialization.
- Retain probabilistic action sampling, feature scaling, unknown/applicable handling, observer-relative state ordering, and output seat conventions.
- No new attention, dropout, normalization scheme, learned gate, or feature encoding.

Changing `H` necessarily changes input widths of the policy and value heads. Their hidden widths, output definitions, and loss semantics remain unchanged.

The state encoder is evaluated once per position when reuse is available; the action encoder and scorer operate on candidate actions. Preserve existing caching and reuse rather than rebuilding a state encoder for each action.

## 4. Scheme 3: Wider whole-group projections

### 4.1 W352: primary specification

| Component | Specification |
| --- | --- |
| Group output widths | 32 / 64 / 48 / **256** / **288** / 48 |
| Industry projection | `Linear(1314,256) → GELU` |
| Map projection | `Linear(1247,288) → GELU` |
| Context passthrough | 12 |
| Fusion | `Linear(748,352) → GELU` |
| Shared backbone | **Five** residual blocks at width **352** |
| Final normalization | `LayerNorm(352)` |

This spends capacity on a larger direct industry representation and a wider shared latent space. There is no additional industry-specific hidden layer before the group output.

### 4.2 W336: near-equal-parameter variant

Keep the same group outputs and five-block structure, but use `H = 336`, fusion `748 → 336`, and corresponding head input dimensions.

W336 has only 2,144 fewer total parameters than I304. It is useful when asking whether an architectural preference survives a nearly equal parameter budget. Its full inference work is still higher because the wider shared state increases the per-candidate scorer cost.

### 4.3 Hypothesized advantages and risks

Potential advantages:

- The industry output has 256 dimensions rather than 192, and the shared state has 352 rather than 304.
- Every shared residual block can process information from all groups at the larger width, potentially benefiting interactions among industry, geography, economy, hand, and action context.
- The model can allocate its industry projection freely across the full input instead of assigning a fixed 64-dimensional local budget to each industry.

Risks and uncertainties:

- There is one fewer shared residual block. Whether width compensates for depth is task dependent.
- The industry front end still has only one nonlinear stage before fusion. Widening does not reproduce the additional nonlinear local processing of I304.
- A larger latent space does not demonstrate that the original latent space was the main limitation.
- Wider shared state increases candidate-scoring work as well as state-encoding work.

## 5. Scheme 1: Industry subgroups plus a whole-industry bypass

### 5.1 I304: primary specification

The whole-industry bypass is a **learned full-input projection**, not an identity connection. All 1,314 industry inputs remain available to that path.

Let `x_ind` be the full industry vector, and let `x_j` be the six disjoint 219-dimensional industry gathers in the fixed order listed above:

```text
z_j = GELU(Linear_j(219, 64)(x_j))          # six independent parameter sets
z   = concat(z_1, ..., z_6)                # 384 dimensions

full  = Linear_full(1314, 192)(x_ind)
local = Linear_mix(384, 192)(z)
h_ind = GELU(full + local)                # 192-dimensional industry output
```

The branch outputs are added **before** the final GELU. There is no additional GELU on `local` before that sum. This precise definition is needed to make the counts and comparison reproducible.

| Component | Specification |
| --- | --- |
| Group output widths | 32 / 64 / 48 / **192** / **288** / 48 |
| Industry local projections | Six independent `219 → 64 → GELU` layers |
| Local mixing projection | `384 → 192` |
| Whole-industry projection | `1314 → 192` |
| Industry combination | Add the two paths, then GELU |
| Map projection | `Linear(1247,288) → GELU` |
| Context passthrough | 12 |
| Fusion | `Linear(684,304) → GELU` |
| Shared backbone | **Six** residual blocks at width **304** |
| Final normalization | `LayerNorm(304)` |

Initialize the six local projections normally and initialize the weight and bias of `Linear_mix` to zero. This makes the industry module initially equal to its whole-industry path. Do not zero all local layers or introduce a simultaneously zero multiplicative gate.

With zero `Linear_mix` weights, the inner local projections normally receive zero gradient on the first backward pass; the mixing layer can learn first, enabling upstream gradients on subsequent updates. Tests must reflect this intended behavior.

### 5.2 Hypothesized advantages and risks

Potential advantages:

- Industry-specific nonlinear relationships can be formed before the final 192-dimensional compression.
- The whole-industry path retains direct access to interactions across industries and to features that the local branches might otherwise discard.
- The local mixing layer can combine representations from all six industries, while the six shared residual blocks retain their existing depth.
- Zero-initialized mixing introduces the additional branch without initially perturbing the industry module's direct output.

Risks and uncertainties:

- Both industry paths still converge into 192 dimensions; the fused state remains 304-dimensional. Additional upstream parameters do not eliminate either compression step.
- The six 64-dimensional local budgets may not match the relative complexity or importance of different industries.
- Independent local parameters provide a structural partition, not parameter sharing or automatic transfer of statistical strength across industries.
- Some added work may duplicate relationships already supplied by derived features or learnable in the shared backbone.
- The additional branch has a short initialization transient and more operators/gathers. Their practical effects need measurement.

When the local branch is zero, I304 contains the original industry's projection as a special case. This does **not** make the entire candidate checkpoint compatible with the current model: the map and fusion dimensions still change.

## 6. Exact parameter allocation

Counts include linear biases and affine LayerNorm parameters. Index buffers and other non-trainable metadata are excluded. The utility model is the main comparison because it includes the existing utility residual head used downstream.

| Component | W352: Scheme 3 | I304: Scheme 1 | W336: Scheme 3 variant |
| --- | ---: | ---: | ---: |
| Six group projections, including I304 local branch | 725,888 | 800,128 | 725,888 |
| Fusion | 263,648 | 208,240 | 251,664 |
| Residual blocks | 1,246,080 | 1,116,288 | 1,135,680 |
| Final LayerNorm | 704 | 608 | 672 |
| Action encoder, including LayerNorm | 42,240 | 42,240 | 42,240 |
| Policy scorer | 61,697 | 55,553 | 59,649 |
| Official outcome head | 46,732 | 40,588 | 44,684 |
| Utility residual head | 11,556 | 10,020 | 11,044 |
| **Total, utility model** | **2,398,545** | **2,273,665** | **2,271,521** |
| **Total, base model without utility residual head** | **2,386,989** | **2,263,645** | **2,260,477** |

The I304 local branch adds 158,400 parameters: 84,480 in the six local projections and 73,920 in the mixing projection. Its entire industry module has 410,880 parameters, including the 252,480-parameter whole-industry path.

For allocation comparisons, define the **front end** as group projections plus fusion, and the **shared trunk** as residual blocks plus final LayerNorm. Action processing and prediction heads are separate.

| Allocation | W352 | I304 | W336 |
| --- | ---: | ---: | ---: |
| Front-end parameters | 989,536 | 1,008,368 | 977,552 |
| Shared-trunk parameters | 1,246,784 | 1,116,896 | 1,136,352 |
| Front end / complete state encoder | 44.25% | 47.45% | 46.24% |
| Shared trunk / complete state encoder | 55.75% | 52.55% | 53.76% |
| Front end / complete utility model | 41.26% | 44.35% | 43.04% |
| Shared trunk / complete utility model | 51.98% | 49.12% | 50.03% |

W352 is approximately 5.49% larger overall than I304, despite having 18,832 fewer front-end parameters. Its shared trunk has 129,888 more parameters. W336 is approximately 0.094% smaller overall than I304.

These ratios do not establish an ideal MLP allocation or a winner. Parameters that process semantically useful local interactions are not inherently less valuable than shared parameters. Conversely, allocating more parameters before a fixed bottleneck is not evidence that more useful information reaches the heads.

## 7. Compute comparison and measurement limits

The following estimates count dense linear MACs, including the utility head. They exclude activations, normalization, gathers, masking, softmax, feature encoding, data transfer, and search overhead.

For a position with `K` scored candidate slots:

```text
MACs per position = fixed_state_and_value_MACs + K * per_candidate_MACs
```

With padded batches, `K` is the number of processed candidate slots, which may exceed the number of legal candidates in an individual position.

| Analytical work | W352 | I304 | W336 |
| --- | ---: | ---: | ---: |
| Fixed MACs per position | 2,285,600 | 2,166,240 | 2,160,992 |
| MACs per candidate | 103,424 | 97,280 | 101,376 |
| Total at K = 64 | 8,904,736 | 8,392,160 | 8,649,056 |
| Total at K = 256 | 28,762,144 | 27,069,920 | 28,113,248 |

W352 requires about 6.1–6.3% more dense MACs than I304 at those candidate counts. W336 is close in parameter count and fixed state work, but requires about 3.1–3.9% more total MACs in those examples. One MAC is commonly counted as two floating-point operations.

**No latency claim follows directly from these estimates.** I304 introduces several small local projections and gathers; W352 uses larger dense matrices. Kernel overhead, batching, device utilization, and encoder reuse can change their practical ordering.

A small future benchmark should use identical encoded positions, legal-candidate counts, batch sizes, precision, devices, and cache states. Report state-only latency and complete policy/value latency separately, including the existing `new_encoder` and `warm_hit` paths. Do not reconstruct models or encoders inside a timed hot path unless measuring that cost intentionally.

## 8. How to assess the designs from an MLP perspective

| Question | W352 emphasis | I304 emphasis |
| --- | --- | --- |
| Where is extra nonlinear processing placed? | Wider shared layers after fusion | An additional local industry path before fusion |
| Industry representation delivered to fusion | 256 dimensions | 192 dimensions |
| Map representation delivered to fusion | 288 dimensions | 288 dimensions |
| Shared representation | 352 dimensions | 304 dimensions |
| Shared residual depth | Five blocks | Six blocks |
| Structural assumption | Full groups can learn useful compression directly | Per-industry relations benefit from local nonlinear processing |
| Potential main failure mode | Extra width does not compensate for lost depth or missing local processing | Extra industry processing does not address the limiting bottleneck |

Neither candidate is established as a function-class superset of the other. Increasing width while reducing depth changes several expressive and optimization properties at once. Adding an industry bypass preserves a useful path inside that module, but it does not prove the full model is stronger than a wider alternative.

Please assess specifically whether the supplied industry features still demand substantial nonlinear preprocessing, whether interactions across industry and map are more likely to benefit from additional shared width, and whether the fifth-to-sixth shared block is likely to matter at this scale. State the assumptions behind the answer; the inspected dimensions cannot settle these questions on their own.

## 9. Training, data, and checkpoint constraints

Both candidates must use the same selected training recipe. Preserve the latest-model self-play behavior, eligible-seat collection, opponent policies, PPO behavior-probability checks, reward terms, teacher label admission, and source freshness safeguards.

The inspected defaults include learner exploration 0.08; five teacher games per collection; eight teacher search workers per game; shared turn budgets of 9 / 12 / 12 / 12 seconds for early Canal / late Canal / early Rail / late Rail; and `search_wall_fraction = null`. Keep the selected teacher cadence, distillation limits, and teacher-value configuration identical across candidates. The fresh-run teacher-value default is enabled; existing explicitly disabled settings must not be silently changed for a comparison.

The active teacher-value work uses existing prediction heads and does not add a separate parameter-count category to this comparison. Its losses may affect how shared capacity is used, so neither candidate should receive a different auxiliary training setup by accident.

Compatibility requirements for any later implementation:

1. Keep the 3,128-dimensional state features, 327-dimensional action features, semantic order, scaling, and known/applicable masks unchanged.
2. Reuse the existing raw IL dataset when it passes the current feature and source-admission checks. Rebuild derived caches when their architecture or source provenance requires it.
3. Introduce distinct architecture/layout identities and configuration validation for the selected topology. Preserve old manifests and identities.
4. Reject incompatible model checkpoints clearly. Do not silently reshape weights, discard a trained sixth block, reinterpret projections, or partially load a checkpoint as though it were an exact continuation.
5. Any weight migration is a separate, explicit design requiring validation of optimizer state and training provenance. It is not implied by zero-initializing the I304 branch.
6. A fair comparison should use fresh, separate output directories and the same admitted IL data. Do not compare one migrated candidate against one freshly trained candidate without treating initialization as an additional variable.
7. Never reuse incompatible cached on-policy trajectories across architectures or label them as freshly collected behavior.

This document authorizes no changes to the current training run, checkpoints, optimizer state, population, counters, or ongoing implementation work.

## 10. Proposed validation and evidence

### 10.1 Focused implementation checks, if a candidate is selected

- Verify all input/output dimensions and exact parameter totals.
- For I304, verify six disjoint 219-field gathers cover the complete industry group exactly once, preserve relative opponent seats, and retain every mask component.
- Verify that zero local mixing reproduces the whole-industry module output, that the mixing layer receives a gradient, and that local inner layers can receive gradients after the mixing weights update.
- Preserve legal-action masking, stable-seat outputs, utility inputs, and existing encoder-cache correctness.
- Validate architecture identities, checkpoint rejection, configuration consistency, and IL dataset admission.
- Check ONNX parity for supported batch/candidate dimensions and the production precision path.
- Run a bounded inference/training smoke test in a separate directory. Such a test checks operation and numerical behavior, not playing strength.

### 10.2 Evidence relevant to eventual strength

If both candidates are trained later, use identical IL data, game-level train/validation separation, comparable training coverage, and comparable tuning effort. Inspect learning curves and evidence of undertraining rather than assuming equal optimizer-step counts mean equal convergence.

Useful diagnostics include held-out policy loss, outcome/value errors and calibration, rare action-family performance, phase-specific errors, actual sample use, and numerical stability. They help explain results but are not replacements for game outcomes.

For playing strength, use paired seeds and seat-balanced games against a fixed independent opponent suite, supplemented by direct cross-play. Report uncertainty while accounting for paired or correlated games. Evaluate under two explicitly distinguished conditions:

- **Controlled search work:** comparable completed simulations or another verified search-work budget, to examine policy/value quality with search resources controlled.
- **Fixed wall time:** the practical deployment setting, where inference speed can affect the amount of completed search.

Do not mix the conclusions from these two conditions. Small smoke tests, parameter counts, higher action entropy, or lower training loss do not establish stronger play. No long training or evaluation campaign is part of creating this document.

## 11. Questions for the external reviewer

Please provide an independent recommendation using this specification, rather than assuming either candidate was previously preferred.

1. Which family has the stronger expected eventual playing strength: Scheme 3 or Scheme 1? Give a confidence level and the assumptions that matter most.
2. Does your answer change between W352 and the near-equal-parameter W336 comparison? Explain whether your preference comes from topology or simply from extra capacity.
3. For these already engineered industry features, is the extra per-industry nonlinear stage likely to be more useful than a larger industry output and wider shared representation?
4. How consequential is the change from six 304-wide residual blocks to five wider blocks? Avoid treating either depth or width as universally superior.
5. Does I304 place too much of its incremental capacity before the 192-dimensional industry output, or is that compression a reasonable way to construct better sufficient features?
6. How might joint policy, official outcome, utility, and teacher-value supervision change the preferred allocation of shared capacity?
7. What is the smallest informative validation that could overturn your recommendation? Distinguish implementation correctness, optimization evidence, and playing-strength evidence.

Please end with one concrete architecture recommendation, including exact widths and block count. If recommending a change to a candidate, identify it explicitly rather than silently comparing a different architecture. Do not invent Elo gains, measured latency, or a universal optimal projection-to-backbone parameter ratio.

## Appendix A. Reproducible counting formulas

Let `P(a,b) = a*b + b` for a biased linear layer, `H` be shared width, `R` residual block count, and `F` fusion input width. Let `P_groups` include every group projection and, for I304, the complete local industry branch.

```text
Residual block parameters      = 2*H^2 + 4*H
Final LayerNorm parameters     = 2*H
Action encoder parameters      = 42,240
Policy scorer parameters       = 128*H + 16,641
Official outcome head          = 128*H + 1,676
Utility residual head          = 32*H + 292

Utility model parameters
  = P_groups + 2*R*H^2 + (F + 4*R + 291)*H + 60,849

Fixed dense MACs per position
  = M_groups + 2*R*H^2 + (F + 160)*H + 1,792

Dense MACs per candidate
  = 58,368 + 128*H
```

| Formula inputs | W352 | I304 | W336 |
| --- | ---: | ---: | ---: |
| `H` | 352 | 304 | 336 |
| `R` | 5 | 6 | 5 |
| `F` | 748 | 684 | 748 |
| `P_groups` | 725,888 | 800,128 | 725,888 |
| `M_groups` | 725,152 | 798,880 | 725,152 |

## Appendix B. Repository evidence

The current implementation facts were read from the following files. Candidate layers and counts are proposed specifications derived from them, not implemented or benchmarked results.

- [Base model and heads](../src/brass_ai/learning/model.py)
- [Utility model](../src/brass_ai/learning/utility_model.py)
- [State grouping implementation](../src/brass_ai/learning/state_layout.py)
- [Feature encoding](../src/brass_ai/learning/features.py)
- [Feature contracts](../src/brass_ai/feature_engineering/contracts.py)
- [Industry-derived features](../src/brass_ai/feature_engineering/industry.py)
- [Feature manifest](../configs/learning/features-4.0.0.json)
- [State-group manifest](../configs/learning/state-groups-5.0.0.json)
- [Grouped-projection contract](GROUPED_PROJECTION_CONTRACT.md)
- [Current training configuration](../configs/step5/joint7-latest.json)
- [Feature engineering lab instructions](../docs/FEATURE_ENGINEERING_LAB.md)
- [Teacher auxiliary-value report](../docs/TEACHER_AUXILIARY_VALUE_REPORT.md)
