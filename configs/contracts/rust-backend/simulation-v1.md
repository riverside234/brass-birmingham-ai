# Native diagnostic simulator — L07

`select_backend(backend="rust-simulator", root=..., executable=...)` returns a
`NativeSpielGame`. The default selector still returns the existing reference
game. No environment variable selects Rust automatically. This is Variant B:
the existing Python search consumes the Rust simulator's OpenSpiel-compatible
state API. Native search, features and NN inference follow in later milestones.

The supported configuration is four-player standard Brass, target rules
`steam-brass-patch-1.1.0`. Setup explicitly chooses order, nine merchants and all
64 normal cards with reference semantic IDs/probabilities. Rust constructs the
initial board and deal. All subsequent staged rules, chance, observation memory
and terminal returns execute in Rust. Returns are the reference win-share
utilities; the existing Python reward/cost ledger remains responsible for a
model's separately identified training reward.

## Ownership and failure

The existing process still has two independent owner threads, one per session.
One diagnostic game claims each session; callers can share a `NativeWorker`
between games with `session=0` and `session=1`. Transport remains synchronous;
this adapter is not the native concurrent search scheduler.

Each owner has at most 256 live simulator handles. Handles monotonically advance,
encode their owner and never alias a retired state. Clone/restore share immutable
native worlds; stepping creates an independent successor. No journal replay,
state DTO import, or JSON board transfer is needed to switch between live clones.
The Python wrapper copies its cached public views when cloning/restoring.
It releases handles on state close/collection, and releases remaining handles
when closing the game. Explicit close is available for deterministic cleanup.
Process retirement destroys all remaining worlds.

Invalid moves/revisions, invalid checkpoints, unsupported configurations and
capacity failures preserve existing states. No partially built successor is
published. An interrupted Python search closes its trajectory generators;
transport timeouts retire the worker and cannot publish a late successor to
another game. Native search cancellation remains an L12 responsibility.

## Control surface

The existing framed protocol adds `simulation {session, command}`. Commands are
strict tagged objects with `kind`:

| Command | Inputs | Result |
| --- | --- | --- |
| `new` | none | Initial explicit-chance state and handle |
| `import` | trusted `state` DTO | New handle; no fabricated prior reveal at chance |
| `prepare` | permitted `view`, allowed `recall` | Prepared detached root; shuffle draw count |
| `sample` | L06 `random` descriptor | New sampled world and handle |
| `snapshot` | `handle` | Context, legal/chance choices, four permitted views/recalls, terminal result |
| `step` | `handle`, `revision` (null during setup), semantic action ID | Transactional successor and snapshot |
| `clone` | `handle` | Independent handle to the same immutable world |
| `restore` | destination `handle`, `source` handle | Replace destination with source snapshot |
| `release` | `handle` | Retire the handle |
| `export` | `handle` | Trusted portable checkpoint |
| `deserialize` | trusted `checkpoint` | Validated new handle |

Snapshot context omits the C# process's transient `game_id`; the native handle
and revision identify diagnostic operations. Semantic action hashes, ordering,
observation strings and information-set digests match the reference adapter.
Setup and partial chance keep hidden picks out of permitted observations.

## Sampling and checkpoints

`from_observation(view, seed, recall)` prepares the typed root once for consecutive
identical permitted inputs. The adapter emits exactly the descending shuffle
indices consumed by Python's `Random(seed)`, preserving the reference search's
RNG stream without reconstructing worlds in Python. Rust deals all card zones.
Native ChaCha sampling remains separately identified by the L06 contract.

Portable serialization is `brass-native-spiel/1.0.0` with action history and a
`brass-native-simulation/1.0.0` checkpoint. The latter contains either validated
setup picks, or a complete trusted state plus all four observation memories.
It preserves the prior reveal during incomplete draws/shuffles. Import validates
state invariants, observer ordering, public agreement, recall, current views or
prior-reveal revision. These checkpoints contain hidden state and are trusted
diagnostic artifacts; they must never enter production policy requests.

All four views are materialized only for this Python diagnostic boundary.
Criterion measures native rules/sampling/copying separately; adapter profiles
include IPC, JSON, Python copies and four-view materialization. Those timings
are not an NN search-speed claim or a Unity deployment qualification.
