# Native Features-4 input contract

`brass-native-features/1.0.0` implements `permitted-features/4.0.0` for the
frozen L01 v11.2 model schema. It does not load weights or admit gameplay.
The generated `native/brass-ai-core/data/feature-layout.json` records the source
digest, exact ordered feature selections/scales and grouped/entity gathers.
`scripts/prepare-rust-feature-layout.py --reference <frozen-source> --check`
verifies it without importing the working Python encoder.

## Typed production boundary

`FeatureEncoder::encode` consumes an acting seat's typed `Observation`, the
complete ordered `Choice` slice and an explicit `FeatureContext`. It returns
borrowed reusable buffers: state `[3128]`, actions `[N,327]`, legal mask `[N]`,
stable-to-relative mapping `[4]`, stable-order remaining utility budget `[4]`
and relative-order critic descriptors `[4,34]`. The batch dimension and padding
are added by L09. Existing grouped/local-entity gathers remain inside the model;
they are not extra feature vectors or a replacement architecture.

The chosen manifest fixes these dimensions; other feature versions must be
admitted explicitly. Layout/schema/model routing mismatches must fail before
inference. In L08, model/reward/profile IDs are validated, carried through and
tested for lifetime isolation; they are labels, not proof that weights have
loaded. L09 must bind them and the maximum cost to the admitted manifests.

Unknown critic descriptors are the deployment default: four rows with only
role index 4 set. Explicit diagnostic/frozen-opponent descriptors require the
existing 34-value schema and five-way role one-hot; unknown rows have zero
lag/fingerprint fields. Input rows arrive in stable seat order and are gathered
to relative order. Remaining budgets arrive and remain in stable seat order;
the unchanged model performs its own relative gather. Budget values are finite
and bounded by the explicitly supplied maximum cost (g890: 0.075).

Style `[4]` and intent `[8]` are finite in `[-1,1]`, with the exact existing tail
order. No model value, reward or descriptor is synthesized from hidden cards.
Every newly expanded actor uses that actor's permitted view. Action, payment,
continuation, bonus and debt inputs are supported. Setup, chance, terminal and
nonacting views reject policy encoding; search handles those paths separately.
Input support does not imply search calibration/readiness or authorize debt
search in a wrapper that still routes it through policy-only handling.

The caller supplies legal choices from the qualified rules engine. The encoder
validates IDs, duplicates, vocabularies, referenced objects and tensor facts;
it does not re-enumerate legality or prove that an external slice is complete.
The direct simulator-handle entry point obtains the full legal slice itself.
All supplied candidates are encoded, up to an explicit bound of 8,192; a larger
set fails instead of truncating. No candidate score/policy cutoff is applied.

## Reuse, allocation and cancellation

Common public facts and connectivity are prepared once per encode. Fixed indexed
storage and typed fact arrays replace Python dictionary/key construction. Card
variants share identical 143-value derived action effects within the call;
their 184 base values still encode their exact card counts and identities.
`effect_hits` / `effect_misses` measure that reuse, not neural evaluation-cache
hits. Each new call clears effects and candidate metadata while retaining buffer
capacity. There is no persistent feature/result cache in L08. L10/L11's future
evaluation cache must include permitted observation/recall, ordered candidates,
model/reward/schema identities, personality, remaining budgets and descriptors.

A cooperative checkpoint is called before work, during candidate preparation
and encoding and before publishing completion. Any rejection or cancellation
makes `tensors()` unavailable. Callers cannot obtain a partially completed result;
Rust borrowing prevents encoding again while a previous borrowed result is used.
The final tensor capacity is bounded by the input contracts. Native allocation
and normal-allocator timing are measured separately from JSON transport and
observation/semantic-choice construction.

## Diagnostic protocol

The `features` capability adds `encode`, `from_state` and `page` commands.
Each of two independent one-thread sessions owns one encoder and one bounded
immutable diagnostic snapshot. `from_state` requires an owner-local playing
handle and exact revision; it projects a native permitted observation and legal
choices directly, without exporting a board for Python feature construction.

Each accepted preparation gets a monotonic owner-tagged encoding ID. Every page
includes that ID, observer/revision, model context, layout identities, total
candidate count, exact choice IDs and `[start,end)` candidate offsets. Pages
contain at most 64 rows, preserving the 1 MiB frame bound and the complete set.
New preparations invalidate old pages before validation; failed preparations
cannot revive an older result. Foreign/stale IDs and invalid page bounds reject.
An existing snapshot represents its captured revision even if a world later
changes; L09 must match request/root freshness before accepting any evaluation.

The Python diagnostic reader holds the transport lock across all pages and
checks shapes, masks, mapping, identities and full ordering. Corrupt replies
retire the worker. It never translates observations into neural features.

## Qualification and admission

Exact float32 equality is required for both full feature arrays and every
discrete mask/ID/order/seat field. Same-runtime PyTorch policy, win, rank, VP and
utility outputs use `atol=rtol=1e-6`; actual maximum differences are reported.
Unknown and explicit descriptors are compared, including actor independence.
The oracle is an immutable extracted L01 source tree with a separately recorded
candidate diagnostic-client overlay. Corpus/data/model/source and executable
hashes accompany the result. The unchanged g890 checkpoint is mechanical test
input only: missing utility calibration keeps live/strength/teacher admission
false. ONNX export/inference/batching remain L09, Gumbel search remains L10–L12.
