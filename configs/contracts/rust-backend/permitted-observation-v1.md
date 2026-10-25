# Native permitted observations and sampling — L06

This additive local diagnostic interface preserves the frozen `0.2.0` observation,
four-player standard rules, and `BrassSpielState` recall semantics. It does not
admit live Unity search or training. `target_rules_qualified` remains false until
the simulator adapter is qualified in L07.

## Information boundary

`state::observation::Observation` is a typed, recursively strict allowlist. It
contains one sorted own hand, public hand sizes/discards/known wildcards, public
board/player/turn/pending fields and public semantic history. It has no opponent
hands, actual deck order, setup burn identities, checkpoint, seed or live RNG.
The L02 permitted request's broader model/search/budget envelope remains a later
search interface; L06 accepts only its view and allowed recall components.

An empty merchant's `types` field may be omitted or be an empty array. Both
representations are valid and retain their distinct wire/information identity
through import, sampling, transitions, both eras and save/restore. Explicit
`types:null` is rejected, as required by the frozen observation schema. This
optional field is also preserved by the trusted semantic state DTO; existing
L04/L05 fixtures containing arrays keep exactly their original representation.

`NativeState::observation(seat)` rejects temporary chance states. An
`ObservationMemory` retains the last permitted public projection and each seat's
hand/recall until the entire draw or Rail shuffle completes. Its information
envelope is `{view, setup:false, chance:null|"draw"|"rail-shuffle"}`. A standalone
trusted import during chance has no past reveal to recover: its view is null and
recall empty. Saving/restoring an existing session preserves its real prior
reveal and recall, including during partial chance.

At a reveal, each seat appends `{revision, hand}` only when its sorted hand
changes. Known wildcards derive from public Scout/discard history in the current
era. Canal setup burns remain hidden; Rail aliases before `discard_starts` do not
become public current-era discards. Sampled opponents receive their sampled own
hands and fresh current-hand recall, never the original live opponent's recall.

`InformationState` has complete typed equality over the observation envelope and
that seat's recall. Diagnostic SHA-256 matches the frozen Python
`information_state_string`: a compact sorted JSON object containing the Python
sorted/spaced observation string and recall. This JSON/hash conversion occurs
at the diagnostic boundary; it is not required for native tree lookups.

## Detached sampling

`PreparedRoot::new(view, recall)` has no full-state overload. It validates the
view, cards, topology, phase, history and recall, reconstructs the board once,
and retains a shared immutable public projection. It subtracts own cards,
current-era public discards and opponents' publicly known wildcards from the
public composition. It shuffles the sorted remaining normal-card multiset,
fills opponents' unknown hand slots in seat order, samples Canal setup burns in
seat order, and uses the remainder as the deck. Wild decks remain explicit.

Recall is attached **after** sampling, exactly as in the reference. It changes
information-set identity but does not add historical or behavioral conditioning.
Absent tiles become removed tiles and historical Rail discard aliases are
omitted, matching C# `ImportPosition`, without reading their actual live zones.
Each world owns all mutable data; immutable definitions/public projection may be
shared. Per-world construction does not parse JSON, replay history, invoke the
reference, or export/import a full DTO.

Sampling draws are either:

- `{kind:"recorded", indices:[...]}`: exactly one descending Fisher–Yates index
  for each `i=n-1,...,1`, with support `0..=i`. Reject missing, extra and out-of-range
  draws. Duplicate card types keep their physical multiplicity.
- `{kind:"chacha12", seed:u64, stream:u64}`: pinned `rand 0.8.5` and
  `rand_chacha 0.3.1`, `ChaCha12Rng::seed_from_u64`, explicit stream, uniform
  `u32` range draws. The identity is
  `brass-permitted-sampler/1.0.0:rand_chacha-0.3.1:ChaCha12Rng:fisher-yates-u32`.

Native seeds do not reproduce Python seeds. Deterministic cross-language
fixtures replay the actual shuffle indices from the unchanged Python sampler.
Explicit simulated game chance continues through L05's multiplicity-weighted
semantic card support; sampler randomness and game chance are separate.

## Worker operations

All operations run on one of the existing **two independent one-thread owners**.
No extra inference, search or simulation threads are created.

| Operation | Request fields | Response |
| --- | --- | --- |
| `observation_snapshot` | `session`, `seat` | `observation`: one permitted `information` record and reference digest |
| `belief_prepare` | `session`, `view`, `recall` | `prepared`: sampler identity, unseen-card count and draw count |
| `belief_sample` | `session`, `random` | `sampled`: sampled state, legal choices and actual draws |

The last response is **trusted diagnostic output**, containing a sampled world's
hidden cards; it is not a policy request. The future production search consumes
the native prepared root directly. Preparation replaces only the owner's
prepared root. Sampling commits its fully validated successor/memory only after
legal generation and response preparation succeed. Rejected requests preserve
prepared/live/saved states, recall and the other owner. The single save slot
includes observation memory, while the independent prepared root survives
ordinary state imports and restores. Input identities/revisions and all nested
unknown fields are checked at the applicable existing typed boundaries.
