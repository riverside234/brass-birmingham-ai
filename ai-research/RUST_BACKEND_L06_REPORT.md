# L06 — Permitted observations, recall and sampled worlds

Date: **2026-10-25**. Status: **complete; final qualification passed**. The sealed
reference capture is under `artifacts/rust-backend/20261025-l06-capture/`;
the final qualification is under
`artifacts/rust-backend/20261025-l06-qualified-v3/`. See the
[qualification receipt](../artifacts/rust-backend/20261025-l06-qualified-v3/qualification.json)
and [final evidence verification](../artifacts/rust-backend/20261025-l06-qualified-v3/final-verification.json).

## Implementation

`native/brass-ai-core/src/state/observation.rs` produces the exact four-player
permitted observation: sorted own hand, public hand counts/discards/known
wildcards, public player/board/turn/pending fields and semantic history. Typed
Serde DTOs reject unknown fields recursively. Complete cards, deck order, setup
burn identities, checkpoints and live RNG are outside this interface.

Observation memory shares one immutable public projection and owns each seat's
hand and recall. It appends recall only when the sorted hand changes. Partial
draws and Rail shuffles retain the prior permitted reveal until chance completes.
Native information-set equality includes the complete view, chance kind and own
recall. The diagnostic SHA-256 agrees with the frozen Python string/digest
format; Serde handles JSON with the reference's separators.

`state/belief.rs` prepares a world from the detached permitted view. It parses
public history and reconstructs the board once, then reuses this typed template
and public projection. Each sample owns its mutable state and fills only the
unknown normal-card zones. It preserves own cards and publicly known wildcards,
samples opponents and Canal setup burns, and places the remainder in the deck.
Reference `ImportPosition` conventions for removed tiles, Rail discard aliases
and debt reconstruction are retained.

Recall is attached after sampling, exactly as in `BrassSpielGame.from_observation`.
It changes information-set identity without adding history-conditioned beliefs.
Sampling accepts recorded descending Fisher–Yates indices or an explicitly
identified ChaCha12 seed/stream. It directly reuses the already locked `rand
0.8.5`, `rand_chacha 0.3.1`, Serde and RustCrypto SHA-256 crates. No package version
or upstream source changes were needed. Python and Rust seeds are not claimed
to produce identical outcomes; cross-language comparisons use actual recorded
draws.

The worker adds `observation_snapshot`, `belief_prepare` and `belief_sample`,
with corresponding Python diagnostic methods. Prepared roots, live worlds and
save slots remain private to **two independent sessions, one thread each**.
Save/restore now includes observation memory. Preparation/sampling failures leave
the prior prepared root, live/saved state, recall and other owner intact. The
sample response is trusted diagnostic output; policy consumers receive only a
selected seat's information.

## Evidence and coverage

| Qualification check | Result |
| --- | --- |
| Rust formatting and strict Clippy | Passed |
| Native tests | 29 passed in each of debug and release |
| Python/process tests | 27 passed on each of debug, release and profiling |
| Reference parity | Zero unexplained differences in every build |
| Execution ownership | Two independent sessions, one compute thread each |
| Preservation and archives | Source, binary, dependency, reference and prior-evidence identities verified |

All three binaries identify native source SHA-256
`f0c052bb03c86aa38c79a6940cd9feb102892a57958d20ecb8e221d5854e4639`
and Cargo lock SHA-256
`c8525c36187f0b93038c7ee8741e2faafa5684a987c436c71e08cf1aea8720e7`.
The release executable SHA-256 is
`5eaa844e979a96aa524a6a2656999988ccfc57366b757ea85d4e9c73ea0f491f`.
The evidence directory contains the implementation and executable archives,
per-build parity reports, command logs and preservation receipts.

The capture executes the original Python sampler and frozen C# runtime. Its
deterministic fixtures contain the actual `_randbelow` indices and their bounds.
Existing L02/L04 roots and L05 transitions remain immutable, including the six
previously corrected authored variants and original rejection cases.

| Capture | Coverage |
| --- | --- |
| Accepted roots | 546; every seat's permitted observation |
| Decision roots eligible for sampling | 426; all five decision phases in both eras |
| Sampled worlds | 1,704: every eligible root × four observing seats |
| Sampled continuations | 40 era/phase/observer groups; 440 staged/chance steps |
| Original full-game corpus | 20 games; 8,344 transitions; 33,456 seat observations including initial states |
| Optional merchant fields | Eight additional sampled worlds; 456 steps including a complete 372-transition two-era game |

The process suite executes the frozen `BrassSpielState.refresh`,
`observation_string` and `information_state_string` methods as its recall and
identity oracle. It checks sampled private state/ordered cards, legal choices,
all four sampled observations, every continuation, full-game recall and partial
chance reveals. Hidden counterfactuals permute opponent cards, deck order, setup
burns and own hand order without changing the allowed view. Recorded sampled
worlds and subsequent rules/chance behavior must remain identical. Changing
allowed recall changes only information identity.

Adversarial tests cover nested hidden fields, unsupported seats/protocols,
inconsistent counts, non-wild "known wildcards", future recall, malformed draw
tapes, extra checkpoint/RNG payloads, two-owner isolation and saved memory.
Native tests additionally exhaust the three-card shuffle support, check duplicate
multiplicities, census unseen-card marginals over 4,096 fixed ChaCha streams,
replay recorded draws, and run a complete two-era chance/recall regression.
The earlier state/rules/reward and alternative-move regressions run alongside L06.

The wire-schema audit found a valid representation absent from the original
corpus: empty merchant slots may omit `types`. Native state and observations now
preserve omission separately from an empty array, including across Rail reset
and save/restore; explicit null is rejected by the schema. The frozen C# edge
capture under `20261025-l06-edges/` covers each observer with one/all empty
merchants omitted and follows one case through a complete game. This is an
additive supported representation; original L04/L05 fixture bytes are unchanged.

## Reference preservation and reproduction

Reference execution uses the hash-verified 462-file L01 archive plus the separately
pinned L02 harness. The current native Python client is an explicitly recorded
candidate overlay. Live training-source drift is recorded and never substituted
for the reference. The g890 temporary checkpoint, installed fallback assemblies,
original saves, upstream modules and prior qualification archives are verified
before and after the run.

Frozen sampler SHA-256:
`a9dd26f89e3c9926db08d44673ed8efa4a73940190a2d5e40d41b6a79baa1d88`.
Frozen OpenSpiel adapter SHA-256:
`7c0756a81061c9b01c49fc54a64c143ffc39e30557cd51b56247c1b52b1d84ac`.

```powershell
python scripts/qualify-rust-privacy.py --out artifacts/rust-backend/my-new-l06-run
```

This reuses the sealed capture. To regenerate it with the pinned exporter and
isolated reference, run `scripts/capture-rust-privacy.py --out <fresh-directory>`
and pass that directory through the qualifier's `--capture` option. Builds use
the existing offline toolchain. Completed evidence cannot be overwritten.

The preliminary qualification directory `20261025-l06-qualified/` records a
Windows executable-lock failure during debug linking while an earlier test
still held that executable open. That test completed successfully; subsequent
qualification sequences builds and tests. This was not a rules/privacy mismatch.
The `20261025-l06-qualified-v2/` run was stopped during full-game checks when the
schema audit identified the omitted-merchant case; v3 includes that fix and its
new reference regressions.

## Remaining gates

L07 supplies the usable reference simulator adapter and its profiling. This
milestone does not implement neural search, run the NN, install a Unity backend,
or migrate lab training. `target_rules_qualified` remains false. A standalone
trusted import at chance returns no invented past view/recall; an existing
session's save/restore retains its actual prior reveal.

The later matched NN/search benchmark remains **4 seconds for a complete turn**,
with two independent one-thread sessions. Local settings remain **4/8/10/12s**.
No search speedup or L07 allocation/performance gate is claimed here. The g890
checkpoint remains a temporary test model; its live/strength calibration gate
and the local-before-lab promotion order are unchanged.

See the [interface contract](../configs/contracts/rust-backend/permitted-observation-v1.md)
and [migration plan](RUST_AI_BACKEND_MIGRATION_PLAN.md).
