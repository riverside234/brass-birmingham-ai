# Transformer IL and search-free self-play implementation plan

Date: 16 October 2026

Planning revision: feature/token audit and architecture research, 16 October 2026. This revision supersedes the earlier unimplemented 165-token/GELU design. Sections 3–4 define the single proposed implementation; research alternatives are explicitly deferred, not additional required models.

Status: implementation plan only. The repository was inspected to prepare this document. The transformer, configuration changes, data expansion, tests, benchmarks, and training described below have not been implemented or run by this planning task.

## 1. Requested outcome and scope

Build one approximately 26–29 million parameter transformer for imitation learning (IL), ordinary self-play, and direct play. Reuse the current state and candidate-action feature engineering. Increase the number and diversity of distinct qualified training positions. During self-play, run no search and no teacher distillation.

The planned primary model is an eight-layer, width-512 entity transformer with eight attention heads, a **1,408-wide SwiGLU** feed-forward sublayer, **187 tokens**, graph-aware attention biases, and candidate queries conditioned on the state and referenced entities. The concrete design below has a calculated base count of **28,083,780 parameters** and a combined-return utility variant of **28,100,456 parameters**. These are design calculations, not counts from an implemented model.

Interpret “no search and teacher distill during self-play” as disabling both search and teacher distillation. Apply this to all seats in the new training recipe, including historical and legacy opponents. Also disable teacher-outcome auxiliary updates and supervised IL rehearsal inside self-play. Standalone IL remains a separate stage and can reuse previously audited demonstrations, including demonstrations originally produced by searched heuristics. New offline demonstration generation, if needed, is a separate explicit data-preparation job; self-play must never launch it.

Reuse the current permitted information, rule calculations, normalization, uncertainty semantics, action identities, legal-action masks, and targets. Do not add hand-coverage features, reveal opponents' private cards, change the rules, or invent randomized intermediate starting positions.

Introduce the transformer in new IL and self-play directories. Existing MLP checkpoints, datasets, populations, source-review receipts, and active training runs remain intact. This plan does not migrate an MLP's weights or optimizer into a transformer. Preserve the old implementation for compatibility; an architecture comparison campaign is not a required implementation deliverable.

The intended benefit is a stronger direct policy. Neither greater capacity, a successful smoke test, nor more varied actions establishes improved playing strength.

## 2. Findings from the current implementation

All source paths in this document are repository-relative.

| Area | Inspected behavior | Required consequence |
| --- | --- | --- |
| Main network | `learning/model.py` implements grouped v9, six width-304 residual blocks, 2,005,917 base parameters | Add a separately identified transformer rather than relabeling v9 |
| Inputs | Features-4 has 3,128 state inputs and 327 inputs per legal candidate | Keep these external tensor contracts and feature values unchanged |
| Layout | `learning/state_layout.py` records named segments, component offsets, ownership, and source | Generate token gathers from these declarations, with complete coverage checks |
| Candidate scorer | State contribution is computed once and reused across candidates | Preserve this principle; do not rerun the transformer per candidate |
| IL construction | `recovery/demonstrations.py` directly constructs `FourSeatNet(ModelConfig.preset('medium'))`; the CLI passes update settings but not a general model specification | Wire the architecture through the complete IL path; changing JSON alone is insufficient |
| IL storage | Standalone IL uses bounded prepared shards, small IPC receipts, lazy batches, and recoverable preparation | Preserve this repair; never store all token embeddings or all dense actions in parent RAM |
| Data admission | Audited heuristic data permits an architecture-source difference only; rules/features/encoder/targets must match | Reuse eligible Features-4 data without weakening checkpoint or neural-replay admission |
| Ordinary collection | Latest-model tables collect all four current-model seats; other table types collect one learner | Retain this attribution and the current learner exploration mixture |
| Curriculum | Current latest recipe is 15/25/60, then 45/25/30, then 60/25/15 self-play/historical/legacy | Preserve these actual current quotas, not the older 70/25/5 proposal |
| Teacher path | Collections, distillation, teacher value updates, and post-teaching drift measurement remain wired into the iteration | Add a genuinely disabled execution path, not just a zero loss multiplier |
| Legacy opponents | `searched_legacy()` can enable search through a boost window or game threshold | A new explicit mode must override both paths |
| Existing legacy “fast” mode | `incumbent_only` skips MCTS but `MaskedSearch.cs` still constructs, completes, and scores turn plans | Zero completed simulations alone does not prove direct action selection |
| Memory assumptions | The current execution config can keep 72 models and validates capacity against the retained population | Separate retained checkpoint count from resident-model memory for a 28M model |
| Active defaults | Learner exploration 0.08, temperature 1.0, entropy weight 0.005, value weight 0.5; insolvency 0.005 capped at 0.075 | Preserve these settings |

The additional feature audit used `learning/state_layout.py`, `learning/features.py`, `feature_engineering/contracts.py`, `feature_engineering/static_profile.py`, `configs/learning/features-4.0.0.json`, and `src/BrassHost/Observation.cs`. It identified these changes to the proposed transformer adapter:

| Finding | Improvement in this revision |
| --- | --- |
| A single hand token hides individual card identities inside one projection; public discards were separated into player tokens | Use 27 card-type tokens, each joining the observer's count with the four public discard counts; retain public known-wild/history fields in player tokens |
| The observer's industry mat and its development facts describe the same entity | Merge them into six observer-industry tokens; keep separate public opponent-industry tokens without inventing opponent development facts |
| The previous global token combined 88 inputs with unrelated roles | Split turn/order, resources/realization, and supplies/policy context into three tokens |
| Empty slots have no tile industry, but their allowed building types are fixed public information | Add the six allowed-type bits from the admitted static profile to each slot's model-side projection |
| Slot locations and merchant locations do not overlap in this catalogue | Replace the always-empty slot–merchant same-location relation with link–merchant incidence |
| Native observations expose `Region1`, `Region2`, and `Region3` for a link | Build incidence from every declared region; never assume two endpoints or parse connectivity from a link name |
| Feature industry ordering differs from policy-context industry ordering | Generate all mappings by industry name and freeze them in the token manifest |
| `industry:CottonMill+Manufacturer` is one card type supporting two industries | Encode both industry bits; preserve wild-card kinds and card multiplicity |
| The earlier candidate query depended only on action features | Condition its query on CLS and the encoded target, merchant, industry, and selected-card entities |

These are representation and implementation improvements. They retain the existing economic feature calculations and do not establish a stronger policy by themselves.

Relevant existing documentation: `ai-research/GROUPED_PROJECTION_CONTRACT.md`, `ai-research/SCORER_FACTORIZATION_REPORT.md`, `docs/FEATURE_ENGINEERING_LAB.md`, `docs/IL_LOADER_RECOVERY_2026-10-15.md`, and `docs/TEACHER_AUXILIARY_VALUE_REPORT.md`.

Recheck the source immediately before implementation because unrelated work may continue after this plan was written.

## 3. Model contract

### 3.1 Versioning and configuration

Proposed new identities, to be reserved and frozen during implementation:

- Architecture: `brass-four-seat-entity-transformer/1.0.0`.
- Token layout: `brass-entity-tokens/1.0.0`.
- Self-play recipe: `latest-transformer-direct-selfplay/1.0.0`.
- Training configuration format: `step5-training/3.0.0`.
- Preserve `permitted-features/4.0.0`, the current derived-feature contract, and the existing official outcome-target semantics.
- Set the new architecture's parameter ceiling to 29,000,000, covering the utility variant. Keep the MLP's existing ceiling unchanged.

Checkpoint identity must bind the architecture, tokenizer manifest/hash, feature manifest/hash, static board metadata, attention/head configuration, utility contract, and architecture-specific source fingerprint. Do not use the MLP scorer's source-equivalence receipt to admit a transformer.

Freeze the activation, `ffn_hidden=1408`, `norm_first=true`, LayerNorm epsilon `1e-5`, eight attention heads, twelve relation channels, nine distance buckets, reference-conditioning maps, bias exceptions, initialization specification, and `dropout=0.0` in the resolved model config. A same-shaped change to any of these semantics is still a model-contract change. Record actual attention backend/precision in execution provenance; implementation may use only backends that pass the required parity checks.

Add an explicit model factory/registry with separate strict config types for the supported MLP and transformer. Avoid making every historical config field optional. Unknown architectures, unexpected fields, changed token order, or a mismatched static relation matrix must fail before weights load.

### 3.2 Encoder

| Setting | Initial value |
| --- | --- |
| Feature-bearing tokens | 186 |
| Summary token | 1 learned CLS token |
| Total tokens | **187**, fixed for the supported board/profile |
| Width | 512 |
| Layers | 8 independent encoder blocks |
| Attention | 8 heads, 64 dimensions per head, bidirectional |
| Feed-forward network | **SwiGLU**, two 512 → 1,408 projections followed by 1,408 → 512 |
| Normalization | Pre-layer normalization in each block; final LayerNorm |
| Attention structure | Twelve binary relation channels plus nine era-specific static distance buckets; soft additive biases |
| Dropout | **0.0 everywhere**, including attention and candidate attention |
| Linear biases | Enabled except the five explicit action-reference conditioning maps |
| Initial numerical path | FP32; BF16 is an explicit later validated option |

Each block is defined explicitly:

```text
x1 = x + attention(LayerNorm1(x), relation_bias)
u  = LayerNorm2(x1)
x2 = x1 + W_down(SiLU(W_gate(u)) * W_up(u))
```

The three FFN maps have their declared biases; `SiLU(z) = z * sigmoid(z)`. The two 512 → 1,408 maps may share one packed linear operation without tying weights. The original GLU work reduces hidden width to two-thirds when replacing two FFN matrices with three. Here, 1,408 is an aligned rounding of 2,048 × 2/3, giving about 3.1% more FFN matrix multiply work per token than the earlier 2,048-wide GELU proposal. This is a capacity choice, not a promised speedup. Its research support comes from language tasks, not Brass or PPO. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

Use ordinary residual additions and independently initialize each block. Do not copy the MLP's zero-initialized residual-output convention automatically. Build an explicit block for SwiGLU and the relation bias; the stock encoder layer is not the complete intended architecture. If any stack-building utility clones layers, initialize each copy independently rather than leaving identical initial values. [PyTorch encoder initialization guidance](https://docs.pytorch.org/docs/2.8/generated/torch.nn.TransformerEncoder.html).

Use dense fixed-length tensors, with no requirement for NestedTensor or a particular fused kernel. Maintain a straightforward matmul/softmax reference attention path. Optimized scaled-dot-product attention must receive `dropout_p=0.0` and `is_causal=False` explicitly, including during training. Its boolean-mask meaning is “true participates”; do not reuse a padding mask with the opposite convention. Relation biases are floating additive scores, not legal-action masks. [PyTorch 2.8 scaled-dot-product attention](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html).

The absence of dropout is deliberate: collecting in inference mode and reconstructing PPO probabilities in training mode must not introduce an unrecorded random policy transformation.

### 3.3 Policy and value heads

Preserve the external input signature:

```text
states:             [B, 3128]
candidate_actions:  [B, A, 327]
legal_mask:         [B, A]
stable_to_relative: [B, 4]
```

The forward path is:

```text
existing state features
    -> deterministic entity gathers + admitted static metadata
    -> 187 tokens at width 512
    -> 8 Pre-LN / graph-biased attention / SwiGLU blocks
    -> encoded entities + CLS summary

existing candidate features [327]
    -> Linear(327, 256), GELU, LayerNorm -> a[256]
    -> add projected CLS / target / merchant / industry / selected-card context
    -> LayerNorm(256), GELU -> query projection 256 -> 512
    -> one cross-attention operation over encoded state tokens
    -> output projection 512 -> 256, LayerNorm
    -> concatenate a[256], attended context[256], CLS[512]
    -> Linear(1024, 256), GELU, Linear(256, 1)
    -> unchanged legal mask and policy probability construction

for each relative player p:
    concatenate CLS[512], player_token[p][512]
    -> shared Linear(1024, 128), GELU, Linear(128, 3)
    -> win logit, rank raw value, VP raw value
    -> existing four-seat output transformations and stable-seat mapping
```

The query combines the existing candidate embedding with explicit entity references:

```text
r_target   = encoded entity selected by the existing target encoding
r_merchant = encoded merchant selected by the existing merchant encoding
r_industry = observer-industry token selected by the existing industry encoding
r_cards    = sum(card_count[k] / 3 * encoded_card_type[k])

q_latent = GELU(LN(a + P_cls(CLS)
                    + P_target(r_target) + P_merchant(r_merchant)
                    + P_industry(r_industry) + P_cards(r_cards)))
q = Linear(256, 512)(q_latent)
```

Each `P_*` is a separate **bias-free** 512 → 256 linear map. A missing reference contributes an exact zero; do not route all-zero one-hots to entity zero. `card_count / 3` is already present in the action tensor: use it once, without a second division or conversion to a presence bit. This preserves both repeated physical cards and multi-card actions. The target map covers 49 slots, 69 links, nine merchants, and the market target; map the last to the resources/realization token. Merchant references and merchant targets are separate semantic roles even when they name the same entity.

Generate these gathers from the existing action schema, with explicit name-based industry mappings. Keep all 327 action inputs in the candidate encoder, including the bounded financial consequences. The reference branches supplement those inputs with contextualized entities; they do not replace action features or recompute legal actions. Candidate queries now depend on the overall state before choosing what state evidence to attend to. This is a project-specific design inference, not a published Brass result.

Candidate attention uses eight heads with 64-dimensional keys/queries/values. Compute state keys and values once per encoded position. Also project the relevant entity banks for each reference role once, then gather/weight their 256-dimensional results per candidate. Do not repeat the five dense context projections independently for every candidate. This is algebraically equivalent because the reference maps are linear and bias-free.

There is no candidate-to-candidate attention and no autoregressive action decoder. Candidate chunks can be scored independently and concatenated before one global legal-action softmax. Candidates never feed back into state encoding or value prediction.

Split the scorer's first affine computation so its CLS contribution is evaluated once per position. Keep both candidate-dependent contributions inside the per-candidate calculation and add them before GELU. This preserves the current optimization's purpose.

The shared per-player outcome head gives each player's public economy/history a direct route to the critic. Its raw result is `[B, 4, 3]`: take column 0 for the four win logits, column 1 for rank, and column 2 for VP. Apply win softmax across the player axis, rank transformation `1 + 3 * sigmoid(raw)`, VP transformation `100 * raw`, and the existing relative-to-stable remapping. Do not flatten player-major triples and feed them into the old win/rank/VP-major slices. Factor the shared CLS contribution once where practical. It does not receive other players' private observations. Keep value heads independent of candidate count, candidate order, and candidate padding.

The utility variant retains the existing combined-return construction and a residual utility head `(512 + 4) → 32 → 4`, with its final layer zero-initialized as today. Utility calibration uses fresh ordinary returns only; it is not a teacher update.

### 3.4 Calculated parameter budget

This table assumes the exact tokenizer and heads in sections 3–4, the stated bias exceptions, one shared tokenizer LayerNorm, one final encoder LayerNorm, and the specified relation-bias parameters. It counts learned parameters, not buffers, optimizer state, activations, or serialized metadata. Each block has `4d² + 3df + 9d + 2f` parameters, for d=512 and f=1,408: 3,218,688 per block.

| Component | Calculated parameters |
| --- | ---: |
| Eight SwiGLU encoder blocks | 25,749,504 |
| Twelve token-type input projections | 299,520 |
| Token/type/seat/industry embeddings | 107,520 |
| Learned CLS vector | 512 |
| Tokenizer LayerNorm | 1,024 |
| Final encoder LayerNorm | 1,024 |
| (Twelve relations + nine distance buckets) × eight heads × eight layers | 1,344 |
| Candidate encoder and its LayerNorm | 84,480 |
| Five bias-free reference maps and query-conditioning LayerNorm | 655,872 |
| Candidate cross-attention Q/K/V/output projections | 788,224 |
| Candidate cross-attention output LayerNorm | 512 |
| Policy scorer | 262,657 |
| Shared per-player official outcome head | 131,587 |
| **Base model** | **28,083,780** |
| Utility residual head | 16,676 |
| **Self-play utility model** | **28,100,456** |

Implementation must generate an actual module-by-module count and reconcile it with this table. If a correctness requirement changes the layout, revise the manifest and budget transparently while staying within 26–29M. Do not hide a difference by omitting parameters from the report.

### 3.5 Research decisions and limits

Research was reviewed on 16 October 2026 using primary papers and the project's installed PyTorch API generation. These sources motivate engineering choices; none demonstrates that this model will outperform the current Brass model.

| Choice | Evidence and applicability | Decision |
| --- | --- | --- |
| Entity tokens plus engineered numerical features | [FT-Transformer](https://arxiv.org/abs/2106.11959) supports learned feature embeddings on tabular tasks; it does not prescribe this game's entity granularity | Keep the validated economic calculations; organize related fields into entities and retain global summaries |
| Graph-aware attention | [Graphormer](https://arxiv.org/abs/2106.05234) uses structural and shortest-path encodings as attention biases on graph tasks | Adopt a small, explicit relation/distance bias adapted to the public board; retain global attention and existing rule-derived reachability facts |
| SwiGLU FFN | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) reports gains for gated FFNs in language tasks at comparable budgets | Adopt width 1,408 within the parameter budget; no claim of RL gains or lower latency |
| Pre-LN and stable PPO reconstruction | [GTrXL](https://arxiv.org/abs/1910.06764) studies transformer optimization and gated residual connections in memory-demanding RL | Keep Pre-LN and deterministic forward behavior. Defer full GRU-style residual gates, recurrent memory, and sequence replay; Pre-LN alone is not the complete GTrXL method |
| Q/K normalization | [Query-Key Normalization](https://arxiv.org/abs/2010.04245) changes attention scaling and reports results in low-resource translation | Defer. Add attention-logit/entropy diagnostics; reconsider only if actual optimization evidence justifies a separately versioned change |
| RMSNorm | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) proposes a simpler normalization and reports task-dependent efficiency | Retain LayerNorm; a speed benefit on the actual CPU/export/GPU paths needs measurement before changing normalization |
| Piecewise/periodic numeric embeddings | [On Embeddings for Numerical Features](https://arxiv.org/abs/2203.05556) finds improvements on tabular benchmarks | Defer scalar expansion and fitted bins. Current bounded features, masks, and scales already have an audited contract; changing them would require additional data-fitting/leakage safeguards |
| RoPE, causal decoding, GQA, MoE, linear attention | This is a fixed, short set of board entities and legal candidates, not a long autoregressive sequence | Do not add them in this revision. Use entity identities and graph structure; there is no measured bottleneck warranting these additional contracts |

Do not bundle a new optimizer, extra auxiliary losses, a reward change, or an adaptive exploration schedule with these architecture changes. Keep the original IL/PPO settings as explicit initial settings and report optimization diagnostics. This remains one implementation target, not a required multi-architecture experiment campaign.

## 4. Reuse all current feature engineering

### 4.1 Complete token mapping

Keep both the reference encoder and `PreparedEncoder` outputs unchanged. Tokenize their existing flat state tensor inside the model using versioned gather indices. Do not recompute economic facts in a second transformer-specific rule implementation.

The following allocation covers every current state component exactly once before learned projection. The extra columns contain only frozen public catalogue metadata, not new observed game-state fields:

| Token family | Count | Existing components each | Existing total | Static extras each | Projector input width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Board slots | 49 | 18 | 882 | 6 allowed-industry bits | 24 |
| Links | 69 | 4 | 276 | 2 era bits | 6 |
| Merchants | 9 | 7 | 63 | 4 bonus-kind bits | 11 |
| Observer's industry mat + development facts | 6 | 36 + 75 | 666 | 0 | 111 |
| Other players' public industry mats | 18 | 36 | 648 | 0 | 36 |
| Observer's economy + known wilds + history | 1 | 21 + 2 + 18 | 41 | 0 | 41 |
| Other players' economy + known wilds + history | 3 | 24 + 2 + 18 | 132 | 0 | 44 |
| Card type: own count + four public discard counts | 27 | 1 + 4 | 135 | 4 card-kind + 6 industry-scope bits | 15 |
| Turn and current order | 1 | 31 + 16 | 47 | 0 | 47 |
| Markets and board realization summaries | 1 | 2 + 24 | 26 | 0 | 26 |
| Card supplies and policy context | 1 | 3 + 12 | 15 | 0 | 15 |
| Current/pending action and opportunity counts | 1 | 197 | 197 | 0 | 197 |
| **Total feature-bearing tokens** | **186** | | **3,128** | **738 static values overall** | |

Add one learned CLS token, giving **187** tokens. All 3,128 original components survive exactly; no structural zero-padding inputs are needed. The 738 extra constants are explicit new model-side metadata derived from public facts already implicit in the fixed feature identities. Keep them separate in the manifest rather than claiming that the model receives only 3,128 projected scalars. They do not change the external Features-4 tensor or require new game demonstrations.

Freeze these details in the generated manifest:

- Use twelve type-specific linear projections with widths `24, 6, 11, 111, 36, 41, 44, 15, 47, 26, 15, 197`, each producing 512 values. The two 15-input families have separate weights. Apply `shared_LayerNorm(GELU(projected_input) + token_id + type_id + applicable_seat_id + applicable_industry_id)`.
- Use 187 token-identity embeddings, thirteen type embeddings including CLS, four relative-seat embeddings, and six industry embeddings. CLS receives its learned vector and its token/type embeddings before the shared tokenizer normalization, without an input projector. Apply seat/industry embeddings only to tokens with a single defined identity; other contributions are nonlearned zero. A card supporting multiple industries uses its explicit multi-hot scope rather than an arbitrarily selected industry embedding.
- Own/other player projectors differ because the observer's spending-difference triplet is absent in the existing contract. Do not create a new self-relative finance fact to make their shapes equal.
- A mat's 36 values are eight level counts and two existing tile-head encodings. The observer's 75 development values join that same industry's mat. Only the observer has those computed facts; opponents keep their public mat contents.
- Each card-type token holds the observer's hand count and the four relative players' public discard counts, all in their existing `/8` scale. There is no separate full-hand token, inferred opponent hand, or reconstructed hidden discard.
- Card-kind ordering is `industry, location, wild_industry, wild_location`. Industry scope uses the six feature-schema industry names. The combined CottonMill/Manufacturer card has both bits; wild-industry has all six rule-supported types. Location and wild-location cards have zero industry-scope bits. These encode card semantics, not whether the current hand can legally build somewhere.
- Static merchant bonus ordering is `Income, Develop, VP, Money`. Link era ordering is `Canal, Rail`. Derive values from the admitted catalogue/profile and bind them to the manifest; do not infer them from live hidden state.
- Keep all 327 candidate inputs: 184 existing raw fields and 143 derived components, including the five bounded portion-finance fields. Do not reduce candidates to action-kind IDs or recreate their consequences from CLS.

Generate a coverage ledger from `state_layout().segments`: source index, semantic name, token ID, token-local index, owner, scale/component meanings, and the provenance of every extra constant. Test exact gather/unpack of the original tensor before learned projection. Branches that later gather encoded entities for actions or values are intentional learned-context reuse, not duplicate primary feature allocation.

### 4.2 Preserve meanings and identities

Retain normalized values, signs, exact/unknown/inapplicable masks, and genuine interval bounds. Exact derived quantities retain their `[value / scale, known, applicable]` triplets. Bounded quantities retain all seven components, including each bound's known flag. Never convert an unknown quantity to an exact zero or replace a lower/upper interval by its midpoint. In particular, preserve the state card-supported-action bounds and all five action portion-finance bounds.

Keep observer-relative seat semantics and the independent mapping of output heads back to stable seats. Public discard columns inside a card token remain ordered by relative player. Do not sum them and lose who played the card.

Empty board slots, unbuilt links, and zero-count card types are real tokens. Missing tiles, an exhausted mat, or an inapplicable development target do not imply token padding. For this fixed board, all 187 tokens remain present. A future variable-board extension would require a new layout contract and padding rules.

No token may combine private observations from several seats. Each learner decision is encoded from that acting seat's permitted view, even when all four seats share the model.

Industry indices are especially easy to get wrong here. Features-4 orders industries as `Brewery, CoalMine, CottonMill, IronWorks, Manufacturer, Pottery`, whereas the policy-context/exploration order is `CottonMill, Manufacturer, Pottery, Brewery, IronWorks, CoalMine`. Resolve by names, including the context passthrough, action industry, mat token, slot type, and card scope. Test every industry rather than just index zero.

### 4.3 Board relations and topology

Compaction removed some constant board facts because the MLP could identify their meaning from fixed input positions. Preserve this information through explicit token identities and source-pinned static relations.

Use the following twelve binary relation channels. Relations are symmetric and can overlap; “self” is the only explicitly diagonal channel. For the other channels, connect different entities only.

| Index | Relation | Source |
| --- | --- | --- |
| 0 | Entity to itself | Fixed identity |
| 1 | Slot to incident link | Static location/region incidence |
| 2 | Links sharing any region | Static region sets, including three-region links |
| 3 | Slots at the same location | Static slot catalogue |
| 4 | Link to incident merchant | Static merchant location and link region set |
| 5 | Player token to that player's industry tokens | Fixed relative-seat identity |
| 6 | Industry tokens of the same industry across players | Fixed industry identity |
| 7 | Card-type semantic references | Location cards to slots at that location; industry cards to matching industry tokens; wild cards to their rule-supported categories |
| 8 | Occupied slot or owned link to its owner player token | Existing state owner one-hots |
| 9 | Occupied slot to matching industry tokens across players | Existing tile-industry one-hot |
| 10 | Merchant to industry tokens it currently accepts | Existing merchant type fields |
| 11 | Pending-action token to its referenced target/merchant | Existing pending target/merchant encodings, with market mapped to resources/realization |

Channels 0–7 are fixed buffers. Channels 8–11 are computed by tensor gathers/comparisons from the unchanged state vector, without another rules query. All-zero owner/industry fields on an empty slot create no ownership/industry relation. Merchant acceptance is dynamic; a static bonus kind must not be mistaken for its current accepted industries. Wild-card semantic references state only their type/location scope, not legal reach or current hand coverage.

Add a small static-distance bias inspired by [Graphormer's spatial encoding](https://arxiv.org/html/2106.05234v5). Precompute one possible-board topology per era, using **all** regions declared for each possible link. For this representation, traversing one link between any two of its incident regions costs one; this is a board-distance descriptor, not a rules-engine claim about legal resource movement. A slot/merchant anchors to its region; a link anchors to its full incident-region set. Pair distance is the minimum region distance between anchors. Use nine buckets: `0, 1, 2, 3, 4, 5, 6+, unreachable, nonspatial`. A spatial entity's self-distance is zero; a non-self pair involving a link from the other era uses `unreachable`; any pair involving a nonspatial token uses `nonspatial`.

The era is selected from the existing state features. Distances describe possible static topology, not currently built connections, ownership, affordability, coal/beer reach, or sale legality. Existing engineered economic/reachability features remain authoritative. Do not add per-decision graph search to compute this descriptor; its two matrices are frozen model buffers.

For each layer/head, learn twelve relation coefficients and nine distance-bucket coefficients, all initialized to zero. Attention scores are `QKᵀ / sqrt(64) + sum(active_relation_coefficients) + distance_coefficient`. Unlike the cited Graphormer formulation, this small adaptation uses coefficients per layer; the parameter ledger counts that choice. Keep these as soft biases: no relation or distance bucket blocks global attention.

Generate and validate the static metadata from an admitted public catalogue/native observation fixture and the pinned rules/static-profile identity. Use the native `regions` array and era rather than guessing endpoints from names. Check the current catalogue's 49 slots, 69 links, and nine merchants, and ensure each intended relation has valid examples. The old slot–merchant same-location channel is deliberately absent because those location sets do not overlap on this board.

Bind ordered IDs, source hashes, static extras, relation buffers, distance buffers, and action-reference mappings into the token-layout hash. No cached static matrix may be reused just because its shape matches. At runtime, bias construction must use the acting seat's permitted input and exact era; model-dependent bias/cache entries must respect update invalidation.

### 4.4 Feature-quality checks and deliberate limits

The current feature engineering already provides useful quantities that attention would otherwise need to approximate: economic constraints, realization summaries, development consequences, and candidate costs/results with uncertainty flags. Keep these alongside board entities. The proposed improvements expose relationships more directly; they do not increase the information available to the policy.

Require these targeted semantic fixtures before training:

1. Move a public tile's owner between two relative seats: only the appropriate original features, gathered owner fields, and ownership relation change; no private hand changes or unrelated static metadata changes.
2. Check an empty slot, exhausted mat, absent merchant tile, unknown bound, and known exact zero separately. None is equivalent to dropping a token.
3. Exercise CottonMill/Manufacturer cards, location cards, both wild kinds, repeated card choices, and empty selected-card sets. Preserve the distinction between hand/discard `/8` and action-card `/3` scales.
4. Check the observer's current and next-era development facts against the same industry's mat, without copying those facts to another player.
5. Check a three-region link and a merchant connection, both eras, disconnected pairs, and inactive-era links. Static distance must stay unchanged when a link changes owner; the ownership channel must change.
6. Check all 128 action targets, absent target/merchant/industry fields, and each industry/context mapping. Chunking or candidate reordering must not alter what entity a choice references.

These tests assert encoder semantics and algebra, not monotonic behavior of a randomly initialized network. Do not claim a larger cash value must improve an untrained model's score, or interpret attention weights as proof of strategic reasoning.

Keep the existing aggregate history rather than introducing recurrent memory or hidden-card inference. Do not reintroduce hand-coverage features, duplicate all 3,128 inputs into an extra full-state token, add new rule-derived finance estimates, or introduce fitted quantile/log transforms. Those would enlarge the scope and compatibility obligations. New static metadata must be recoverable from the admitted catalogue for existing datasets; if a required fact cannot be verified there, report a contract gap before implementing it.

The current engineered summaries remain alongside entities. Numeric/categorical feature embeddings are compatible with transformer processing, as illustrated by [FT-Transformer](https://arxiv.org/abs/2106.11959); that result does not establish the best token layout or playing strength for Brass.

## 5. Increase diverse, qualified positions

### 5.1 Define qualification separately from strength

For IL, “qualified” continues to mean an admitted demonstration source with verified rules transitions, permitted observations, complete legal choices, valid selected actions, complete terminal outcomes, and valid partition/source receipts. It does not mean every demonstrated move is optimal.

For PPO, “qualified” means a valid completed fresh ordinary trajectory from the exact frozen collection policy, with reconstructible behavior probabilities and valid return targets. Nonforced learner decisions are actor-eligible; forced decisions remain critic-only. Do not import teacher search-coverage thresholds into ordinary PPO, where no search labels exist.

Never increase reported diversity by copying rows, replaying old PPO samples, treating repeated epochs as new positions, relabeling model versions, or weakening admission. Do not discard valid losses, insolvency cases, or uncommon actions because their outcomes look undesirable.

### 5.2 IL dataset expansion

Use the existing audited Features-4 joint dataset as the starting inventory. Do not use the old joint helper's default model-continuation command: it imports batch7's MLP weights. Reuse its manifest-union and semantic-game-deduplication logic through an architecture-neutral preparation entry point.

Initial expansion target: **at least 25% more distinct qualified nonforced training-position fingerprints than the starting manifest**, sourced from additional independent games. This is a preparation target to measure, not an assumed outcome or permission to lower quality requirements.

Preparation steps:

1. Inventory compatible existing manifests, their raw hashes, unique setup families, semantic games, nonforced rows, phases, action families, industry choices, and source profiles.
2. Merge additional compatible qualified games where available. Preserve source ownership, frozen train/validation/test partitions, and raw content hashes. Conflicting identities or cross-partition setup families are errors.
3. Measure the actual increase after game deduplication and position accounting. If it is below target, report the shortfall explicitly.
4. When further collection is required, provide a separate offline generation command using fresh setup seeds and balanced admitted profile/seat combinations. Start with a requested new-game quota of 25% of the baseline unique training games and assess the resulting position increase. Do not automatically launch an unbounded generation job to reach the target.
5. Use diagnostics to choose subsequent offline source/profile batches. Do not fabricate rare states, filter for winners, or claim duplicate oversampling increases coverage. Any new generator/source profile must pass the existing qualification process before inclusion.

Train on the full admitted union. Preserve the existing teacher/era/action weighting and threefold weight cap initially. Shuffle without replacement within each epoch; do not introduce a second rare-state oversampler in this revision. Report both natural and balanced validation metrics, with the existing validation-based checkpoint selection. Test data must remain outside fitting and model selection.

### 5.3 Ordinary self-play expansion

Set the new recipe's initial `games_per_iteration` to **480**, compared with the current 320. This is 50% more fresh completed tables per update cycle. Keep PPO at two epochs; increase data collection rather than the number of presentations of each row.

Preserve the current game-count curriculum exactly:

| Committed ordinary games | Latest-model tables | Historical tables | Direct legacy tables |
| --- | ---: | ---: | ---: |
| 0–19,999 | 15% | 25% | 60% |
| 20,000–99,999 | 45% | 25% | 30% |
| 100,000 onward | 60% | 25% | 15% |

The clock is committed ordinary games, not optimizer steps, planned jobs, or iteration count. A collection crossing a boundary must apply each game's correct quota. Preserve the normal seat rotation and existing neutral/industry context schedule.

At a full 480-table collection wholly inside each stage, the planned learner-trajectory counts are respectively 696, 1,128, and 1,344: four trajectories per latest-model table and one per historical/legacy table. These are trajectory counts, not unique-position estimates. Report actual completed counts and actor eligibility separately.

Historical opponents remain frozen neural policies, using the existing initial/IL, recent, and older-reservoir strata. Prefer distinct models within a selected stratum when alternatives exist; keep the initial anchor and report unavoidable repeats in small pools. Keep the stratum-selection probabilities and deterministic per-game RNG. Never fill a historical seat with an unrecorded live learner update.

Use new seeds from the existing training-partition seed mechanism. Freeze collection weights and job specifications before dispatch. Do not opportunistically update the learner while other games belonging to that collection are still running.

### 5.4 Distinctness and usage accounting

Track three different identities:

- Sample occurrence: game ID, observer, decision index, and collection behavior identity. Duplicate transport/replay occurrences are errors.
- Permitted-position fingerprint: a versioned canonical hash of the permitted observation and full semantic legal-action set, excluding incidental transport/game identifiers. Preserve physical-card multiplicity and all meaningful public history. Canonicalize seats consistently.
- Encoded-input fingerprint: exact Features-4 state/action data plus context and mapping. This measures distinct model inputs and may merge different permitted histories that the current representation intentionally omits.

Do not drop legitimate independent PPO occurrences merely because they share a position or input fingerprint. That would change the on-policy state distribution. Use fingerprints for diversity measurement and exact-copy detection, not a new outcome-dependent PPO sampler.

Keep cumulative fingerprint accounting in an owned disk-backed index with bounded insert/read batches, rather than an unbounded Python set or copied observations. Bind its format to the canonicalization contract. Merge an iteration's counts into committed history idempotently only with its successful checkpoint/state commit; replaying interrupted reporting must not double-count positions or advance seed progress. Inventory passes should stream prepared/raw rows and leave source artifacts unchanged.

For IL and PPO, report generated/read, admitted, actor-eligible, critic-eligible, distinct positions, distinct rows actually used, total presentations, unused eligible rows, and rejection counts/reasons. Early KL stopping can leave admitted PPO rows unused; expose that difference.

Use the existing phase boundaries: early Canal rounds 1–4, late Canal 5–8, early Rail 1–6, late Rail 7–8. Implement a lightweight phase-contract accessor without importing teacher search machinery solely for metrics. Report per-phase sample usage, action families, industry choices, forced fractions, and economic coverage such as shortfall and saleability, with unknown/inapplicable bins preserved.

## 6. IL integration and data compatibility

Pass an explicit model specification from `step5.py imitate` through `train_bc`, `_train_bc`, checkpoint loading, completion validation, and handoff. Remove the hard-coded medium-MLP constructor from the new architecture path while retaining old behavior for old configurations.

For the first transformer IL run, initialize transformer weights randomly. Do not pass `--from-imitation-run` pointing at a completed MLP, use partial state-dict loads, copy shape-compatible tensors opportunistically, or rewrite old provenance.

Keep the current IL optimizer/loss starting settings: AdamW, learning rate 1e-4, weight decay 1e-4, value weight 0.5, current win/rank/VP losses, nonforced actor filtering, and the existing early-stop semantics. Initial limits remain ten epochs, minimum three, patience two. These are continuity choices, not evidence they are optimal for the transformer.

Keep effective batch size 2,048, but start with microbatches of 16 and at most 8,192 padded candidate cells. Add attention-aware memory limits described in section 9. Gradient accumulation must reproduce effective-batch loss normalization, including variable nonforced counts and final partial batches.

Preserve the repaired disk-backed prepared store. Store existing float32 state/action arrays and compact indexes; gather/project tokens just in time. Do not precompute learned token embeddings into a persistent dataset cache. They become stale after every optimizer update and would expand storage substantially.

Eligible existing Features-4 demonstrations need no new gameplay or feature conversion solely because the backbone changes. Confirm this through the actual production loader. Reuse or rebuild derived prepared caches according to their identities; do not overwrite or re-sign existing cache receipts. Raw-data admission may tolerate the existing narrowly permitted architecture-source difference, while all rule/feature/encoder/target and teacher-source checks remain strict.

This compatibility claim also requires proving that the static token metadata comes from the same admitted board/rules profile, and that every token/reference gather consumes the unchanged flat tensors. Store the new static metadata and topology receipt with the transformer run. Verify tokenization of cached arrays against tokenization of freshly encoded observations for selected real IL rows, including both eras and a pending multi-part action. Do not demand equality between randomly initialized MLP and transformer predictions; compatibility concerns data meanings and admission, not network outputs.

Preserve the full training/validation/test split and shard recovery behavior. A worker failure must not silently omit a shard. New transformer-specific metadata belongs to the new run/model and token manifest, not a modified historical demonstration header.

## 7. Make self-play genuinely search-free

### 7.1 Explicit disabled mode

Add `training_mode: "direct_ppo"` to the new recipe. Resolve it before teacher budget migration, scheduling, resource allocation, or signature generation. Old recipes keep their existing behavior.

Canonical disabled values for this new configuration format:

```json
{
  "training_mode": "direct_ppo",
  "search": null,
  "search_ready_games": null,
  "search_every_games": null,
  "search_games": 0,
  "search_collections_per_iteration": 0,
  "teacher_work_budget": null,
  "distill_steps": 0,
  "distill_weight": 0.0,
  "distillation": null,
  "teacher_value": {"enabled": false},
  "rehearsal_manifest": null,
  "rehearsal_weight": 0.0,
  "rehearsal_until": null,
  "search_wall_fraction": null,
  "legacy_opponents": {"mode": "direct"},
  "legacy_boost": null
}
```

This is a proposed configuration contract, not valid input to the current 2.2 validator. Implement mode-specific validation; do not simply edit an old JSON file to these values.

The resolved execution settings must allocate zero teacher/search worker groups. Ordinary actor processes and direct-opponent evaluators still exist. Do not instantiate unused teacher pools, register teacher model snapshots, prepare rehearsal tensors, probe teacher drift, run distillation, or perform teacher auxiliary optimizer steps.

`next_search_games` is null in new run state. Scheduling and compact-history code must handle this explicitly. Branch around the current rehearsal decay calculation before arithmetic: `0 * expression` does not make division by a null/zero horizon safe.

`search_wall_fraction=None` in Python and `null` in JSON retains its existing meaning: no cumulative teacher-versus-ordinary time restriction. It does not enable teaching and must not be used as a substitute for the disabled mode. Normal game/inference cancellation and failure timeouts remain active.

Reject contradictory new-recipe settings, such as enabled teacher value updates, nonzero distillation steps, searched legacy mode, or a non-null teacher schedule. Do not silently accept them as ignored.

### 7.2 Direct opponents, including legacy

Latest-model tables use one frozen collection snapshot in all four seats. Historical tables use one learner and three frozen historical neural policies. Both choose from current legal-action probabilities without tree search, lookahead, or best-of-rollout selection.

The legacy adapter requires additional work. Existing `search_free=True` / `incumbent_only=True` still calls native `plan` or `plan_forest` and constructs multi-action incumbents. It must not be advertised as satisfying the new direct mode merely because `completed_simulations == 0`.

Add a separately identified direct heuristic policy that ranks the current legal choices using admitted public information, the existing action features, and retained profile preferences. An existing single-choice heuristic can be reused only after verifying its call graph. It must not create belief worlds, expand a search tree, simulate future opponent turns, enumerate multi-action continuations, or select among simulated turn plans. Score each choice and execute one; choose again when the rules return the next legal decision.

Normal rules execution, legal-choice enumeration, and the existing deterministic feature calculations for candidate consequences remain permitted. They are required inputs, not permission to add future-turn search. Do not change those feature semantics to satisfy the opponent implementation.

Give this direct policy a new immutable identity and provenance. Do not overwrite the frozen searched-v4 bank or pretend the new policy has its former search-based strength qualification. Its actions are opponent behavior, not newly qualified IL labels. If no compliant direct legacy adapter exists yet, the implementation stage is incomplete; do not silently call the old planner or change the curriculum to hide it.

Add fail-closed execution guards and tests that reject calls to teacher collection, native planning, neural MCTS, determinization, or distillation from the direct recipe. Inspect both attempted and completed work counters. Keep `direct_heuristic_decisions` separate from any legacy telemetry field historically named `search_elapsed_seconds`.

### 7.3 PPO correctness and unchanged recipe elements

Preserve the exact behavior mixture:

```text
p_behavior = (1 - exploration) * p_policy + exploration * q_contextual
```

Learner exploration remains 0.08 with temperature 1.0. The contextual action-family/industry exploration distribution remains unchanged. All four current-model self-play seats contribute eligible trajectories; only the learner contributes in historical/legacy tables. Frozen historical neural exploration remains at its current 0.05. Preserve each direct external policy's explicitly recorded selection rule; do not assign it the learner's exploration setting.

Record actual chosen-action probabilities, policy identity, source identity, context, model identity, token-layout identity, and behavior settings. Strictly reconstruct all collected learner behavior before updating. Cached PPO data from another architecture, token layout, model, source, or exploration mixture must be rejected or recollected through the existing mechanism.

Keep PPO learning rate 1e-5, weight decay 1e-4, two epochs, clip 0.2, target KL 0.01, entropy weight 0.005, value weight 0.5, gradient clip 5.0, advantage normalization, and actor nonforced filtering. Keep win/placement/VP reward coefficients 1.0/0.1/0.001 and insolvency event cost 0.005 capped at 0.075. Keep champion gates and champion-opponent rollout disabled for this latest-model recipe.

Ordinary utility fitting/calibration remains where required by the existing combined-return critic, with targets from current ordinary games only. There are no teacher value targets, teacher-policy labels, or supervised rehearsal updates after PPO. Record every optimizer-step category so an unexpected auxiliary update is visible.

PPO's repeated minibatch passes use the newly collected on-policy batch; the [original PPO formulation](https://arxiv.org/abs/1707.06347) does not make arbitrary old behavior samples interchangeable with that batch. A diversity archive may support reporting, but is not a source for fresh PPO updates.

### 7.4 Scheduled evaluation

Keep scheduled evaluation cadence in committed games, but generate an explicitly direct-policy evaluation configuration for this recipe. Neural references and any legacy references must also use their declared direct modes. Monitor/broader evaluation must not re-enable search through old defaults.

Use fixed published policy-selection settings across evaluations. Separate execution smoke results from seat-balanced held-out playing-strength evidence. In a symmetric four-player matchup, the reference win share is 25%, not 50%. Report opponents, seats, setup counts, ties, and setup-level uncertainty; do not make promotion automatic.

## 8. Handoff, continuation, population, and retention

The initial sequence is existing qualified data → new transformer IL → completed transformer IL checkpoint → new direct self-play run. IL-to-PPO transfer copies the exact base-model weights, binds the completed IL receipt, and creates a fresh PPO optimizer as the current workflow does. Add/initialize the utility extension through an architecture-aware version of the existing mechanism.

Starting this new architecture necessarily starts new self-play progress. Do not claim it preserves an old MLP run's optimizer or counters. It preserves the old run by leaving it untouched and keeps the new model's IL lineage explicit.

After the transformer run exists, same-run continuation must preserve weights, Adam moments, RNG, committed generation/game counts, curriculum position, seed cursor, context counter, population, evaluation anchor, and parent hashes. Persist any new sampler/diversity cursor. If new stochastic code uses Python or NumPy RNG, serialize its state as weights-only-safe primitives/tensors rather than arbitrary pickle objects.

Do not reapply fresh-IL initialization on resume. Do not reset collection schedules, import old MLP population entries, or rewrite signatures to accept pending incompatible work. Discard/recollect incomplete work only through the existing checked freshness protocol, leaving committed progress intact.

Retain the current population capacity: 16 recent plus 48 anchors/reservoir entries. Preserve the initial transformer IL anchor and active opponent/checkpoint references during cleanup. Historical diversity counts checkpoints retained on disk, not models resident on the GPU.

Existing source-review and resume protections remain strict. Add explicit reviewed transitions only for proven equivalent same-transformer revisions. Architecture changes require fresh runs, even when a user passes `--resume-reviewed-source`.

## 9. Performance and resource integration

The self-play model's calculated FP32 weight storage is about **112.4 MB** in decimal units. FP32 weights, gradients, and two Adam moments alone are approximately **449.6 MB** before activations, temporary buffers, master copies, and inference replicas. These estimates are not a measured peak-memory budget.

The transformer blocks reuse their weights across **187 tokens**. Their dense matrix operations alone require approximately **5.09 billion MACs per position**, calculated as `8 * (4*T*d² + 2*T²*d + 3*T*d*f)` for T=187, d=512, f=1,408. This excludes tokenization, heads, normalization/nonlinearities, and candidate attention; it is not measured latency. Do not estimate speed by scaling the 2M MLP's inference time by its parameter ratio.

Relative to the earlier unimplemented 165-token proposal, token-linear work grows about 13.3% and pairwise attention work about 28.4%, before the FFN/head changes. These are arithmetic comparisons between drafts, not benchmark results. The richer card/industry representation must justify its real inference cost. Removing teachers also does not guarantee a faster iteration after increasing model size and ordinary games.

Initial bounded settings for the new recipe:

| Resource | Starting setting / requirement |
| --- | --- |
| IL/PPO effective batch | 2,048 |
| Training microbatch | 16 positions |
| Training padded candidate cells | At most 8,192 |
| Inference batch | At most 8 positions initially |
| Inference padded candidate cells | At most 4,096 initially |
| Attention limits | Bound both `B * heads * T²` and `B * heads * A * T`, not only `B * A` |
| Resident frozen models | Byte-bounded LRU, initial maximum eight resident models per inference service; aggregate worker replicas must also fit the host/device budget |
| Retained population | Still up to 64 checkpoints on disk |
| Teacher/search worker groups | Zero for this recipe |
| Precision | FP32 correctness path; explicit BF16 pilot only on supported lab hardware |

The numeric batch limits are conservative initial settings to benchmark, not hardware-independent optimal values. An oversized candidate set must be chunked, never truncated. Preserve all legal choices and global probability normalization. If a single position cannot fit the configured memory contract, report a resource error rather than silently dropping it.

Update the current `max_models >= population + 3` validation assumption for this architecture. Preserve a full identity catalogue while lazily loading immutable opponent checkpoints under a resident-byte limit. Pin the learner and in-flight requested models, serialize load/eviction safely, and never serve a request with another model's weights. Report model loads, evictions, resident bytes, and batch stalls. Bound host RAM as well as GPU residency.

Keep existing encoder-cache reuse. Cache model-dependent encoded tokens/keys/values and projected reference banks only under the exact model and token-layout identity plus the permitted input/context and numerical execution contract. Invalidate them after updates. Do not reuse detached model embeddings across training steps, even if raw feature-cache entries remain valid. Static metadata buffers need no per-turn reconstruction; only the four dynamic relation channels depend on each position.

Do not eagerly materialize all layers' dense `B * heads * T * T` relation biases together. Reuse compact relation/bucket indexes and form each layer's needed bias. Dense additive bias may affect fused-attention eligibility: benchmark the actual path with these biases enabled, report the selected backend, and retain the reference path for parity/export. A fast no-bias microbenchmark cannot establish performance of the proposed model.

Refactor manual inference paths, including `learning/batching.py`, to call an architecture-neutral interface that can return pooled context plus state tokens and cached K/V. Preserve one state encoding per position across candidate chunks. Update state-only value detection explicitly; the current exact-type check recognizes only `FourSeatNet` and `UtilityNet`.

The laptop inspected during planning has a Core Ultra 5 135U and approximately 16GB RAM. It is an inference/small-debug target; substantial IL/self-play should use the lab GPU. No transformer latency has yet been measured on either machine.

## 10. Planned source changes

| Files/modules | Work |
| --- | --- |
| New `learning/entity_tokens.py`; `configs/learning/entity-tokens-1.0.0.json` | Generated feature/metadata ledger, identities, card semantics, topology/dynamic relations, action-reference maps, coverage checks |
| New `learning/transformer_model.py`; `learning/model_factory.py` | Strict transformer configs, SwiGLU encoder, state-conditioned candidate head, shared per-player outcomes, utility variant, capability interface |
| `learning/contracts.py`, `checkpoint.py`, `batching.py`, `utility_model.py`, `export.py` | Architecture-aware fingerprints/load/save/inference/export; preserve old contracts |
| `recovery/demonstrations.py`, `prepared_store.py`, `feature_engineering/data.py` | General model selection; reuse audited Features-4 data; preserve bounded preparation and admission |
| New `recovery/position_diversity.py` | Versioned identity and coverage/usage accounting with bounded storage |
| `recovery/config.py`, `league.py`, `ordinary.py`, `curriculum.py` | New direct recipe, 480 games, explicit disabled teacher path, unchanged game-count quotas |
| New `recovery/direct_legacy.py`; existing execution/legacy integration | Direct heuristic selection with new provenance and no planning calls |
| `recovery/updates.py`, `utility.py`, `inference.py`, `execution.py`, `shared_transport.py` | PPO parity, state-only values, memory-aware batching, model residency, worker dispatch |
| `recovery/teacher_collection.py`, `teacher_value.py`, scheduling callers | Keep old behavior; make new-recipe invocation impossible/guarded |
| `recovery/population.py`, `retention.py`, `resume.py`, `history.py`, `review_metrics.py` | Preserve anchors and restart state; retain compact diversity/no-search evidence |
| `step5/cli.py`, `league.py`, `evaluation.py`; config generators | Carry model specification end to end; explicit recipe/evaluation mode; updated descriptions |
| New `scripts/transformer-training.py` | Inspection, dataset inventory/union, config binding, export, bounded benchmark helpers |
| New transformer IL/self-play/evaluation presets | Explicit new paths; do not modify an existing signed run configuration |
| `scripts/export-training-reports.py` and compact evidence helpers | Include model/token manifest, data-diversity, actual usage, and zero-search evidence |
| New `tests/Transformer/` regressions and smoke harness | Focused checks listed below |
| New `docs/TRANSFORMER_IL_SELFPLAY_LAB.md` and implementation report | Verified commands, measured counts/latency, limitations, and continuation instructions |

Audit additional constructors and `isinstance`/exact-type assumptions in personality/export packaging. Direct base/utility export is required; personality adaptation is outside this first revision. Unsupported personality workflows must give an explicit incompatibility error rather than silently reinterpret the transformer as an MLP. Preserve the existing MLP personality path.

## 11. Implementation stages and acceptance criteria

### TR0 — Freeze the contract and inventory

Capture current source/config identities and relevant input hashes. Inventory the actual Features-4 IL manifest and available additional qualified sources without loading all dense features. Generate the proposed **187-token** mapping, static-metadata/topology/action-reference receipt, and parameter ledger. Record direct-mode semantics and the legacy-planning finding. Verify the source-derived card, industry, merchant, and three-region-link semantics before freezing the token manifest.

Exit: all 3,128 state inputs and 327 candidate inputs accounted for; no private-information expansion; source manifests identified; new output paths chosen; no existing artifact modified.

### TR1 — Implement tokenizer and transformer

Add the manifest, static/dynamic relation handling, architecture factory, SwiGLU model/utility variants, reference-conditioned chunkable candidate attention, shared per-player outcome head, and architecture-aware checkpoint dispatch. Use the current flat-feature encoder boundary. Keep a reference attention implementation and an exportable SiLU path; no custom kernel is required for correctness.

Exit: coverage/semantic tests pass, actual counts reconcile within 26–29M, all relevant modules receive finite gradients, incompatible checkpoints fail clearly, and candidate chunking preserves outputs/gradients within declared tolerance.

### TR2 — Integrate IL and diversity preparation

Generalize standalone IL model selection, preserve lazy prepared storage, and add read-only inventory plus immutable dataset-union output. Implement actual sample/position usage metrics. Provide optional offline expansion commands; do not launch a full dataset-generation job as part of implementation validation.

Exit: a tiny existing qualified Features-4 dataset loads through the production path, trains the transformer, validates, saves, resumes, and produces an admitted completed-IL receipt. Original source/cache/checkpoint hashes remain unchanged. Report actual diversity increase or an explicit shortfall; do not claim the 25% target from row duplication.

### TR3 — Implement direct self-play and direct opponents

Add mode-specific validation and generators, remove teacher/rehearsal execution from this mode, implement direct legacy selection, preserve four-seat PPO behavior, and add the 480-game production setting. Keep all no-search guards active across curriculum boundaries and scheduled evaluations.

Exit: instrumented tests show zero planner/search/teacher calls, correct learner attribution/exploration, unchanged PPO mixture reconstruction, and no hidden teacher auxiliary optimizer step. Direct legacy policy identity and qualification scope are explicit.

### TR4 — Integrate resources, continuation, retention, and reports

Add attention-aware batching and resident-byte limits; preserve encoder reuse; complete source/freshness checks; include diversity, actual usage, and disabled-teacher evidence in full and compact reports.

Exit: restart continuity and memory-admission tests pass; retained population is independent of resident model count; anchors and in-flight opponents survive cleanup; unused eligible samples remain visible.

### TR5 — Export, focused validation, and bounded measurement

Complete the tests and smoke below, produce ONNX export/parity evidence, and run a small isolated latency/throughput benchmark. Do not start long training, a large demonstration-generation job, or a strength tournament. Publish measured results and a lab runbook, then leave production execution to an explicit subsequent run.

Exit: each required check has a pass/fail/skipped result and evidence path; no unexecuted check is reported as passed; any deployment limitation is explicit.

## 12. Focused regression matrix

| Area | Required checks |
| --- | --- |
| Feature coverage | Exact pre-projection gather/unpack of all 3,128 components; 327 action inputs unchanged; no structural padding; all 738 static values traced to admitted metadata |
| Semantic cases | Exact zero/unknown/inapplicable; interval/one-sided bounds; pending/debt choices; empty slots; exhausted mats; final Rail; own mat/development alignment |
| Cards/industry | All 27 types; dual-industry and wild semantics; correct `/8` versus `/3`; repeat-card weights; public discards per seat; feature/context industry orders mapped by name |
| Identity/topology | Three-region incidence; nonempty link–merchant relation; era distances and unreachable/nonspatial buckets; ownership/acceptance changes; changed static profile rejected |
| Permutation consistency | Permuting token storage with IDs, relations, reference maps, CLS/player gathers, and metadata consistently preserves outputs; this is not permission to permute game semantics |
| Architecture | 187 tokens, eight independently initialized blocks, SwiGLU width 1,408, calculated counts reconciled, finite gradients, dropout zero, no causal masking |
| Attention parity | Reference versus optimized outputs/gradients with actual additive bias; zero-initialized relation coefficients; explicit SDPA dropout and correct mask polarity; finite logits/gradients |
| Action references | All target/merchant/industry/card maps; absent references produce zero; no all-zero argmax fallback; preprojected banks match direct computation including gradients |
| Heads/masks | Shared `[B,4,3]` player head with correct outcome-axis extraction and stable-seat remapping; legal probabilities sum to one; illegal rows have zero probability; values independent of candidates; utilities keep their contract |
| Chunking/cache | One state encode, K/V preparation, and reference-bank preparation per position; chunked/un-chunked logits and gradients agree; no per-chunk softmax; stale embeddings/biases rejected |
| IL admission | Existing qualified Features-4 data accepted; cached-versus-fresh token/reference parity; bad hashes/rules/features/teachers rejected; static-metadata source verified; no MLP weight import |
| IL storage | Bounded prepared records/IPC/handles; worker retry retains committed shards; invalid data is not skipped; no persistent learned-embedding cache |
| Diversity | Duplicate game/transport handling; independent repeated states remain legitimate; no split leakage; true generated/eligible/distinct/used/presentation counts |
| Direct mode | Teacher scheduling, MCTS, determinization, native `plan`/`plan_forest`, distillation, rehearsal, and teacher auxiliary updates never called |
| Mode validation | Contradictory settings rejected; null horizons cannot cause arithmetic failures; `search_wall_fraction=null` remains distinct from enabling search |
| Curriculum | 480-table quotas, boundary crossings at 20k/100k, normal rotations, no old boost/search-threshold activation |
| PPO | Four current-model learner seats versus one in mixed tables; 0.08 contextual mixture; strict pre-update behavior verification; forced actor filtering; cached incompatible behavior rejected |
| Utility | Ordinary-return calibration only; reward and insolvency values unchanged; value-only fast path produces identical values |
| Handoff/resume | Exact IL base-weight transfer; fresh PPO optimizer once; subsequent Adam/RNG/counters/lineage/population preserved; incompatible pending work rejected |
| Residency/retention | Eviction never changes served model identity; memory limit respected; initial anchor and active references retained; population can reach 64 |
| Legacy isolation | Old MLP load/config/feature paths still work in focused fixtures; old signed artifacts remain byte-identical |
| Reports | Per-phase coverage and actual usage retained in compact history; disabled teacher drift is null/not applicable, not a measured zero |
| Export | Dynamic B/A, static T=187, SiLU/gating, dynamic relation construction, all reference gathers, real legal masks, value-only path, candidate padding/chunking, CPU ONNX parity |

Use small deterministic fixtures and the smallest relevant existing regressions. A full historical test suite or repeated broad reruns is not required unless a failure reveals wider impact.

## 13. Bounded smoke and benchmark protocol

Use one new scratch directory under `artifacts/transformer/<timestamp>/`. Keep production checkpoints and datasets read-only. Use the production-sized transformer for at least one forward/backward, checkpoint, and export check; tiny fixture models alone cannot establish integration of the 28M target.

Smoke limits:

1. Admit an existing tiny qualified dataset with distinct train and validation setup families. Use at most 16 selected rows per partition for one IL optimizer update and validation. Selection is smoke-only and recorded.
2. Save and complete a diagnostic IL run; verify checkpoint load, RNG/optimizer restoration where applicable, and the completed-IL handoff.
3. Start a new diagnostic direct self-play run. Collect at most three complete games, one per table type, using a test-only explicit composition. Collect four learner trajectories from the latest-model table and one from each other table. Run one bounded PPO optimizer step and ordinary utility calibration as needed.
4. Run with teacher/search entry points replaced by fail-fast sentinels. Exercise a direct legacy game, not merely a neural-only substitute.
5. Verify a read-only resume and retention calculation from the committed checkpoint; compare weights, optimizer state, counters, lineage, and protected references. Do not collect another generation just to test loading.
6. Export the base/utility policy with the named outputs needed by the direct consumer, and compare PyTorch versus ONNX on a few real/synthetic shapes, including forced choices and padded candidates.

Set an explicit harness wall limit, initially 15 minutes, plus existing per-game decision/cancellation limits. If a platform cannot complete the bounded smoke, stop safely and report incomplete coverage. Do not weaken data checks or silently replace the target with a tiny model to claim success.

Benchmark separately on the available laptop CPU and, when available, a lab GPU. Use warmups and a small fixed number of measured calls; cap benchmark wall time at five minutes. Suggested shapes: B=1 with A=1/32/128/512, and B=8 with A=32/128. Add a small set of real authored positions and one larger legal-action set if available.

Report full model/feature/token identities, runtime/provider, hardware, threads, precision, candidate/token counts, warmup/sample counts, median/p95 latency, peak host/GPU memory, and model residency. Separate encoder cache misses (`new_encoder`), raw-feature warm hits (`warm_hit`), tensor-ready neural inference, and end-to-end selection. Measure candidate chunks and count actual state-encoder invocations.

Within that bounded benchmark, attribute time to token gathering/metadata, dynamic bias construction, the eight encoder blocks, state K/V and reference-bank preparation, candidate attention/scoring, and feature extraction. Compare reference and optimized execution of the **same** model with real biases, not a separate baseline architecture. Measure whether entity preprojection avoids repeated dense work as intended. ONNX export should lower SwiGLU to standard linear/multiply/sigmoid operations when needed; a backend-specific fused operator must not make direct deployment dependent on that backend.

Initial parity tolerances should follow existing export checks and be recorded per output, not silently widened until a test passes. Compare logits/probabilities and head values as well as selected actions; document near-tied argmax changes separately. Optimized attention/ONNX paths must satisfy the strict pre-PPO probability checks. Do not relax that check just to accommodate BF16 or another backend.

No latency, memory, training-stability, or strength result in this section is currently measured. Model-size arithmetic and planned benchmark shapes are not substitute evidence.

## 14. Reporting contract

Every new run's resolved configuration and compact report should identify the transformer architecture/token layout, exact parameter count, IL origin, data manifest hash, direct mode, exploration, reward, curriculum stage, committed game counts, and effective resource limits.

Record per phase and table type:

- Completed/failed games, learner seats, actor-eligible/forced rows, qualification rejections.
- Unique setup families, semantic games, permitted positions, encoded inputs, source/opponent strata, and historical-model repeat rate.
- Actor/critic distinct rows used, presentations, unused eligible rows, and early-KL-stop effects.
- Actual behavior entropy and action-family/industry frequencies, with forced decisions excluded from actor summaries and measurement scope stated.
- PPO KL/drift, clip fraction, losses, gradient norm, utility-calibration evidence, insolvency events/cost, and economic activity summaries.
- Collection, preparation, update, evaluation, export, and checkpoint time; games/eligible positions per second; cache and residency metrics.

For a small fixed diagnostic minibatch, also report per-layer/head attention-logit spread and attention entropy, residual activation norms, and FFN/attention gradient norms. Keep diagnostics off the ordinary inference hot path; do not log full attention tensors for every training row. These may reveal saturation or optimization failures, but neither high attention entropy nor an attractive attention map demonstrates playing strength. Report IL train/validation loss gaps by phase/action family to expose limited coverage or overfitting of the larger model.

Record an explicit no-search receipt: attempted/completed searches 0, teacher collections 0, teacher-qualified labels 0, distillation steps 0, rehearsal steps 0, teacher auxiliary steps 0, and no instantiated search worker groups. Teacher insufficient-coverage and teacher-policy-drift measurements are **not applicable/null**, with reason `disabled_by_direct_recipe`. Do not fabricate successful search coverage or teacher stability from their absence.

Extend compact history and the report export script so these fields survive cleanup. Export the token-layout/parameter receipt, resolved configuration, latest reports, compact history, IL completion/data-diversity receipts, and evaluation receipts when present. No need to copy full model weights into the review bundle.

## 15. Planned lab commands and rollout

The helper/config names below are implementation targets and do not exist yet. Do not run these commands against today's checkout expecting the new recipe to work. The implementation runbook must replace this section with commands verified against the actual CLI.

Run from the lab repository with its normal environment. `IL_MANIFEST` must point to the actual qualified Features-4 manifest from the user's completed data-generation work; do not guess a different historical dataset path.

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training
export IL_MANIFEST=/absolute/path/to/the/existing/qualified-features4/manifest.json

# Proposed helper: inventory first; no training or dataset mutation.
python scripts/transformer-training.py inventory \
  --manifest "$IL_MANIFEST" \
  --output artifacts/transformer/data-inventory-v1

# Proposed helper: optional immutable union with additional audited manifests.
# Omit this step if additional compatible sources are not yet available;
# report the diversity-target shortfall and use the original manifest honestly.
python scripts/transformer-training.py prepare-il-data \
  --manifests "$IL_MANIFEST" /absolute/path/to/additional-qualified/manifest.json \
  --output artifacts/il-replay/transformer-features4-diverse-v1

# Point this at the actual selected manifest, original or qualified expanded union.
export IL_MANIFEST=artifacts/il-replay/transformer-features4-diverse-v1/manifest.json

python scripts/transformer-training.py configure-il \
  --manifest "$IL_MANIFEST" --preset entity28m \
  --output artifacts/transformer/configs/il-entity28m-v1.json

# New transformer IL; no --from-imitation-run pointing at an MLP.
python scripts/step5.py --device cuda \
  --config artifacts/transformer/configs/il-entity28m-v1.json imitate \
  --manifest "$IL_MANIFEST" \
  --run artifacts/step5/il-transformer-entity28m-v1 \
  --epochs 10 --min-epochs 3 --patience 2 --learning-rate 0.0001 \
  --batch-size 2048 --microbatch-size 16 --max-candidate-cells 8192 \
  --workers 8 --precision fp32

# Bind the actual completed transformer IL checkpoint and its immutable lineage.
python scripts/transformer-training.py configure-selfplay \
  --imitation-run artifacts/step5/il-transformer-entity28m-v1 \
  --games-per-iteration 480 \
  --output artifacts/transformer/configs/selfplay-entity28m-direct-v1.json

# First production pilot; separate from the implementation smoke.
python scripts/step5.py --device cuda \
  --config artifacts/transformer/configs/selfplay-entity28m-direct-v1.json train \
  --run artifacts/step5/selfplay-transformer-entity28m-v1 \
  --imitation-run artifacts/step5/il-transformer-entity28m-v1 \
  --fresh --iterations 1

# Continue only after confirming the first iteration committed generation 1.
python scripts/step5.py --device cuda \
  --config artifacts/transformer/configs/selfplay-entity28m-direct-v1.json train \
  --run artifacts/step5/selfplay-transformer-entity28m-v1 \
  --imitation-run artifacts/step5/il-transformer-entity28m-v1 \
  --expected-generation 1 --iterations 20
```

`--expected-generation` is an optional guard; use the actual committed starting generation or omit it deliberately. `--iterations 20` means 20 additional iterations. Reuse the same signed configuration for continuation. Do not add `--fresh` on resume. Add `--resume-reviewed-source` only for an applicable exact source review; it never authorizes MLP-to-transformer conversion.

The separate implementation smoke/benchmark commands must use their own diagnostic paths and explicit bounds. The commands above describe future production operation and are not authorization to execute it while writing or implementing this plan.

## 16. Completion evidence and remaining scientific questions

Implementation is complete when the new model, eligible-data reuse, IL handoff, direct self-play, continuation, retention, metrics, export, focused tests, and bounded smoke work as specified, with a reconciled parameter count and measured inference results. A diagnostic-only implementation that cannot run the direct legacy table or load real qualified Features-4 examples is incomplete.

Data expansion is a separate measured deliverable: report whether the 25% distinct-position target was achieved, the source of the additional games, and every unresolved shortfall. Increasing to 480 ordinary games provides more collection opportunities; it does not guarantee 50% more unique observations or successful optimizer usage.

Training stability and playing strength remain open questions. The larger model may underfit available supervision, overfit repeated positions, slow collection, or learn different weaknesses after removing teacher corrections. Establish those effects through subsequent ordinary logs and held-out direct-play evaluations. Do not create automatic rollback thresholds or restore search/teaching without an explicit recipe change.
