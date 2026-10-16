# Main MLP model: grouped v11.1

V11.1 retains the widened v11 MLP and local neighborhoods, and injects the selected local vectors before the action GELU and LayerNorm. It retains Features-4, the factored action scorer, and shared player-conditioned outcome predictions. See the [v11.1 validation report](MLP_V11_1_REPORT.md) and [current lab commands](../docs/MLP_V11_LAB.md). The [v10 report](MLP_V10_REPORT.md) is historical.

## Versioned contract

| Contract | Current identity |
| --- | --- |
| Architecture | brass-four-seat-mlp-resnet/11.1.0 |
| Neural features / learning compatibility | permitted-features/4.0.0 / four-seat-learning/4.0.0 |
| Derived encoding | permitted-derived-context/3.0.0 |
| Feature manifest | configs/learning/features-4.0.0.json |
| Learned projection layout | grouped-state-layout/6.0.0 |
| Learned layout manifest | configs/learning/state-groups-6.0.0.json |
| Learned layout SHA-256 | 9dcb18c8bfc04ca9b020b43ddcfa20a27785d3baf08d19679c8751747a18d53d |
| Local entity layout | grouped-mlp-local-entities/2.0.0 |
| Local entity SHA-256 | a4e1ca6a7588f8839275ff527918a3080f324db6d300215e7328410fb430a4e9 |
| Static topology manifest | configs/learning/local-entity-topology-2.0.0.json |
| Static topology file SHA-256 | 7e4a3e88c1c7f55c580ce8b180d7277d5628d24d0195df18e7c300cd31f4ae5b |

Earlier identities are immutable. The original Features-4 field ordering and state-groups-5.0.0.json remain intact, including their historical widths. `state_layout()` continues to expose that existing feature layout; `model_layout()` supplies v11's learned projections and v6 identity. The separate Transformer implementation retains its token/feature identities. Equal tensor shapes alone do not admit a checkpoint.

## Representation and network

| Group | Inputs | Projected outputs |
| --- | ---: | ---: |
| turn_global | 31 | 32 |
| cards_history | 218 | 64 |
| player_economy | 109 | 48 |
| industry_development | 1,314 | 320 |
| board_resources | 1,247 | 352 |
| current_action | 197 | 48 |
| policy_context | 12 | 12 passthrough |
| Total | **3,128** | **876** |

An exact-only fact still emits [value / scale, known, applicable]. Exact zero, unknown and inapplicable remain distinct. Compact fields reject real intervals and one-sided bounds. Retained bounded fields keep all seven components. Candidates retain 184 categorical/card components, 36 compact facts × 3 and five bounded facts × 7: **327 inputs**. Legal rows, action IDs/order, card multiplicity and observer-relative seat mappings are unchanged. No hand-coverage features are introduced.

~~~text
state [B,3128]
  -> static gather and six disjoint splits
  -> six Linear(input,width) -> GELU projections
  -> concatenate 864 projected values + 12 context values
  -> Linear(876,512) -> GELU
  -> eight residual blocks at width 512 -> LayerNorm(512)

candidate [B,A,327]
  -> Linear(327,160)
  -> add selected target / merchant / industry local embeddings
  -> GELU -> LayerNorm(160)
  -> candidate slice of Linear(672,192)
  -> add shared state slice projection and bias, computed once per position
  -> GELU -> Linear(192,1) -> legal mask

existing state fields -> shared family Linear(input,64) -> GELU -> LayerNorm(64)
  -> 49 slots / 69 links / 9 merchants / 6 own industries / 1 market
  -> bias-free, zero-initialized role projections 64 -> 160
  -> prepend one zero row per bank; select by categorical index (argmax in graph)

existing player fields -> own/other player encoders (284/287 -> 64)
  -> add relative-player identity embedding
  -> shared Linear(512+64,128) -> GELU -> Linear(128,3), for each player
  -> win softmax / rank sigmoid / VP scaling -> stable-seat mapping

utility residual: Linear(512+4,32) -> GELU -> Linear(32,4)
residual(x) = x + Linear(512,512)(GELU(Linear(512,512)(LayerNorm(x))))
~~~

Residual branch final weights/biases also start at zero, as before. Global encoding, entity encoding, bank role projections and the scorer's state projection occur once per observation/model, including across candidate chunks. Zero-initialized role projections initially contribute zero; their gradients allow learning the local branch, with gradients reaching its encoders once those projections move away from zero. Shared outcome, utility and personality output semantics remain unchanged.

Each selector is admitted as binary one-hot or all-zero. A prepended zero selector entry makes argmax return bank row zero for an absent reference; real references select rows 1 onward. A flattened bank and batched offsets provide indexed gathers without a one-hot matrix multiply. Entity families already use one precomputed index tensor, one batched gather and one encoder call each; the three other players are encoded together and the outcome head processes all four seats together. The 327-wide action input is preserved, including for PPO and ONNX. This is the training form. The ONNX form retains constant action weights for runtime prepacking: `pre = Linear(actions) + selectors @ projected_banks`, then the same GELU and LayerNorm. Static slices from the action manifest select 128 target, 9 merchant and 6 industry columns. The concatenated `[B,143,160]` bank is prepared once per position, with no absent row or per-candidate gather; it is reusable across candidate chunks through `prepare_candidates` / `score_prepared`. A standalone ONNX call still encodes its inputs once per call; it does not cache state across independent session calls. Own and other seat encoders run in one grouped export MatMul with their distinct weights and normalization affine parameters preserved.

## Local neighborhoods and static admission

| Bank | Input width | Gather order |
| --- | ---: | --- |
| Slot | 72 | Own 18-component row first, other rows at its location in feature-schema order, trailing zero rows to four slots |
| Link | 148 | Four owner bits, then all endpoint-location slot rows in pinned endpoint/schema order, trailing zero rows to eight slots |
| Industry | 219 | Own 36-component mat, the three relative opponents' 36-component mats, then own 75 derived components |
| Merchant / market | 7 / 2 | Unchanged |
| Own / other player | 284 / 287 | Unchanged public economy, history, finance/order and industry inputs |

Location-to-slot and link-to-region tables come from the pinned native board definition, not guesses from ID spelling. The Canal and Rail Worcester/SouthWest/Kidderminster junctions have **three endpoint regions**, and all their slots are included. The board's maximum is four slots per location and seven slots across a link's endpoint regions, so the requested four/eight capacities do not truncate any entity. Padding is explicitly masked to zero; it cannot read a real feature at a sentinel index. Empty board slots remain represented, and existing action IDs retain location identity.

Admission validates complete slot/link ID sets, slot types, link endpoint sets and eras, native source/assembly pins, and the frozen topology against its board definition. Reference encoding, prepared encoding (including cache hits and validated=True) and replay admission enforce these assumptions. Unsupported topology fails before inference or training. Existing merchant, tile and other static-profile safeguards remain active. Checkpoint loading verifies generated gather buffers as well as layout identities; ONNX metadata includes group/entity identities.

## Parameter and arithmetic budgets

| Component / model | Parameters |
| --- | ---: |
| Six projections | 889,920 |
| Fusion | 449,024 |
| Eight residual blocks | 4,210,688 |
| Final state LayerNorm | 1,024 |
| State encoder | **5,550,656** |
| Base | **5,904,644** |
| Utility | **5,921,320** |
| Personality | **6,023,137** |

Base and utility ceilings are **6,500,000**; the personality ceiling is **6,600,000**. The personality extension's separate allowance is 110,000 parameters (actual 101,817), accommodating its unchanged head at the wider trunk.

Medium base dense work is B × (8,235,232 + A × 83,232) MACs. Utility/personality fixed work is 8,251,872 / 8,353,504 MACs, with the same candidate work. Indexed selectors remove 22,880 MACs per candidate from the original v11 implementation. The static-action-plus-selector ONNX form adds 22,880 MACs per candidate, for 106,112 total; it trades indexed dispatch for a single small matrix multiply. It builds 22,880 dynamic bank values per position rather than a 52,320-value effective action matrix. Counts include entity/player multiplicities. One MAC is two FLOPs; feature extraction, selector argmax/gathers, nonlinearities, normalization, transport and search are excluded. Lower dense arithmetic alone does not establish lower runtime.

The small diagnostic preset keeps a width-128 trunk, with v11 projections, eight blocks and the same action/policy/outcome heads. Its base/utility/personality counts are 1,499,780 / 1,504,168 / 1,539,553. Main training uses width 512.

## Dataset, training and continuation

The existing audited **Features-4 heuristic IL dataset is reusable as-is**, including manifests, raw shards, labels and partitions. Feature/encoder/derived/rules/target hashes remain unchanged. Existing heuristic admission permits the architecture-source difference; it does not admit incompatible neural replay or search targets. Source-bound derived caches are rebuilt through the normal guarded loader when required. No demonstration regeneration or native re-audit is necessary for this model-only revision.

Use **fresh v11.1 IL weights and a separate self-play directory**. V11 and all earlier checkpoints are rejected clearly, including with source-review flags. A completed v11.1 IL run transfers its weights exactly into fresh PPO, with new PPO Adam state. Same-target continuation preserves model/Adam/RNG, warmup progress, game/iteration counters, population, curriculum and lineage. Source and pending-collection freshness checks remain active.

V11.1 has a new architecture and source identity despite retaining identical parameter dimensions. The historical selector source-equivalence receipts do not match v11.1 and do not allow old weights or old neural trajectories into this model. Features-4 heuristic data remains compatible. No existing run, checkpoint or immutable source receipt is rewritten.

Provisional defaults are **IL peak LR 6e-5 / PPO peak LR 6e-6**, 0.6 times the preceding width-304 recipe. Linear warmup starts at 10% and reaches the peak on update 500 / 100 respectively; gradient norm clipping is **1.0**. The persisted counter advances on shared-network optimizer updates, including distillation when used; existing head-only utility calibration keeps its rate and does not advance warmup. No post-warmup decay is introduced. Effective rates and schedule progress are reported in full/compact histories.

The current teacher clocks remain **9/12/12/12 seconds**, with eight search workers per teacher turn and five simultaneous games (capacity for 40 private workers). Each clock remains shared across workers, actions and continuation choices. Generic opponent/evaluation workers stay at four. Exploration 0.08, temperature 1, entropy/value weights 0.005/0.5, 70/25/5 curriculum, four-seat collection, disabled champion gate, insolvency cost 0.005 capped at 0.075, teacher cadence 160 and eight distillation steps are preserved. search_wall_fraction=None/null still disables only the cumulative restriction. Label-quality, source, reward and calibration safeguards remain active.

This is an experimental change to where local information enters the action encoder. Correct execution and export parity do not establish improved playing strength or training stability.
