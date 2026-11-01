# Local native transport, version 1

L13 packages `brass-ai-local.exe` and the qualified g890 ONNX export for
Windows x64. `brass-local-native/1.0.0` is distinct from the research protocol.
The backend identity is `brass-gumbel-is-mcts/alpha-g890-v1`; its neutral profile
is `alpha-test-g890-neutral/1.0.0`. This is a temporary local test model.

## Admission and readiness

Unity verifies the package file hashes before starting the child process. The
worker verifies the policy hash supplied on its command line, model manifest,
export, checkpoint and reward identities. The explicit acceptance has version
`brass-local-utility-acceptance/1.0.0`, scope `part1-local-alpha-test`, decision
`user-2026-10-25`, `final_model: false` and an explicit null calibration field.
Missing fields, another identity, lab scope or a fabricated receipt are rejected.
Historical mechanical-only manifests are preserved byte for byte.

Acceptance is separate from measured readiness. Each of the two persistent
CPU owners loads its own ORT session with one compute thread, warms the full
engine/features/NN/search path, then publishes build/source/model identities,
provider and successful neural position counts. Startup is outside the timed
turn. The L12 singleton policy remains cap 1 / pending 1 / immediate flush.
The conservative frozen L12 startup estimate is retained; L14 owns retuning.

## Framing and ownership

Messages are UTF-8 JSON with a four-byte little-endian length, bounded to 1 MiB.
A request is `{protocol, id, request}` with strictly increasing positive IDs.
Operations are `health`, `search`, `cancel` and `shutdown`; readiness uses ID 0.
Search contains a full ticket, `allowance_ms`, `remaining_ms` and `input_json`.
The ticket binds game instance, saved game ID, generation, revision, actor,
profile, exact UTF-8 input digest and native source fingerprint. Only the
permitted view, actor's recall, an accepted industry feature context and bounded reward
ledger are admitted. Full/private state and scripted/shuffle inputs are absent.

Local release 1.4.46 admits exactly seven g890 profiles:
`alpha-test-g890-{neutral,cotton,manufacturer,pottery,brewery,iron,coal}/1.0.0`.
Readiness advertises the ordered `industry_focuses` list. The ticket profile must
equal the input profile, and style/intent must exactly match the selected tag's
`step5.policy.context_for` encoding (not merely contain finite values). Neutral
uses twelve zeros; Cotton/Manufacturer/Pottery use style `[1,0,0,0]`, and
Brewery/Iron/Coal use `[0,0,1,0]`. Intent is the corresponding one-hot channel in
that six-industry order, followed by two zeros. This conditions the existing
search request's NN evaluations; reward and search_v1 are unchanged.

One existing Unity scheduler job owns one Pair. Each search has two independent
single-threaded sessions. The reader/control thread only handles bounded I/O;
it performs no inference or search. The original ordinary-turn allowance is
4,000, 8,000, 10,000 or 12,000 ms. Action two and retries inherit its remaining
time. Same-object retries retain a weakly associated clock; the versioned save
journal preserves reduced allowance and permitted recall after action one and
between liquidation choices. Forced/debt policy paths keep L11/L12 semantics.

Cancellation acknowledges the request without claiming retirement. The adapter
drains both that acknowledgement and the terminal search response before the
scheduler lease completes. EOF/shutdown retire both owners. A failed child is
terminated and waited for before releasing resources. The response is compact:
complete semantic action paths, provenance, identity, work and timing counters.
The adapter rejects mismatched or late results. Main-thread generation/revision
checks and isolated complete-action validation precede each live commit.

## Unity and saves

`Alpha Test` is offered for standard four-player games. It chooses NN for every
AI seat; V4 chooses V4 for every AI seat. Opposite-family seat choices are omitted.
Human seats remain supported. Difficulty is independent per seat: different
4/8/10/12-second choices can coexist within either family. A family change
preserves the other seats' difficulty choices. Unsupported configurations omit
Alpha Test and cannot start an existing unsupported NN draft.

As of local release 1.4.47, the NN / Alpha Test page offers one NN choice with
neutral context. Former industry-tag menu preferences normalize to neutral while
retaining the other setup settings. Apply persists the choice and Cancel discards
the draft. Signed assignments from existing games retain their bound profile;
the backend's seven-profile admission remains available for save compatibility.
Industry tags cannot be selected for new games. Older signed neutral saves
without `nn_focus` remain readable without rewriting.

Native saves use `brass-local-native-save/1.0.0`, family `nn`, exact model/export
identities and signed assignments/records. Existing V4 save mappings remain
unchanged. Mixed/incompatible saves fail without rewriting their payload.
NN debt prompts use the qualified public-state contract's color order for both
human and NN debtors; each sale uses the existing persisted liquidation event.
V4 prompt ordering and valuation are preserved.

Worker failure, model mismatch or a runtime disable flag stops the NN table;
no V4 seat is substituted. `BrassNative/disabled` removes Alpha Test from setup
and prevents further NN commits. Rollback restores the exact previous helper
assemblies and installation receipt, preserving NN saves for later resumption
with the matching Alpha Test package. L13 shadow/integration tests do not
establish L14 performance/strength or L15 full-game release acceptance.

## Reproduction

Run `scripts/qualify-rust-local.py --out artifacts/rust-backend/<new-run>` with
the existing pinned offline Windows build environment. The script builds the
runtime and C# candidate, runs process/replay/Unity/preservation checks, and seals
the evidence and native ZIP only after success. `--audit` recomputes every
recorded artifact hash. `scripts/install-rust-alpha.ps1` accepts this sealed
L13 receipt; `-Action Rollback` restores its backup and `-Action Recover`
handles an interrupted installation. These are local Alpha Test deployment
commands, separate from production release qualification.
