# Native CPU evaluation contract — L09

`brass-native-evaluation/1.0.0` builds on the [L08 feature contract](features-v1.md).
This is mechanical CPU inference admission. It does not authorize live search,
strength comparisons or teacher generation with the uncalibrated g890 model.

## Model and runtime admission

The caller pins the SHA-256 of `brass-native-model/1.0.0` before loading.
The manifest pins the checkpoint/model ID, ONNX bytes, export metadata, reward,
feature/group/entity layouts, CPU runtime bytes, binding, opset and execution
settings. Bundle paths must stay inside the bundle. Reads have explicit size
bounds. ONNX is loaded from the verified bytes, avoiding a second model-file read.
Deployment must supply a trusted manifest pin; accepting an arbitrary path/hash
from an untrusted party is not a model-signing or authorization scheme.

The qualified contract is the existing v11.2 base/utility model, float32, opset 17,
`ort = 2.0.0-rc.10` / ONNX Runtime 1.22.1, CPUExecutionProvider, all graph
optimizations with exact GELU. The graph's actual input/output names, types and
dynamic/static dimensions are checked. This does not admit the distinct
personality-network goal/confidence heads or heterogeneous opponent models.
Style and intent are already part of L08's 12 feature inputs; do not apply a second
personality transformation to these logits.

Each owner holds one persistent frozen model, constructed/warmed/dropped on that
owner's thread. It is neither Send nor Sync. Two owners mean two model/session
handles. Loading another identity requires a new worker. Intra-op and inter-op
threads are both one, sequential execution, spinning disabled, no global pool,
async ORT forward, inference dispatcher or cross-owner batch. The runtime DLL
identity is fixed for the process. Loading/warm-up are outside timed evaluation.
The worker still starts and operates rules without loading a DLL or any model.

## Typed evaluator and tensors

`Evaluator` accepts a bounded slice of `Input` borrows and a caller-owned
`Control` deadline/cancellation flag. `CpuEvaluator` binds L08 tensors directly
for singleton evaluation. Mixed batches copy into reusable contiguous buffers,
zero-pad actions, and use false masks past each complete candidate count. Legal
choices are never truncated. Every accepted legal logit must be finite; padding
logits must be negative infinity. The maximum is four positions, 8,192 candidates
per position and 32,768 padded candidate cells, with a configurable smaller cell
limit. Memory pressure/backpressure rejects or flushes work instead of dropping
actions.

All six input arrays are explicit, including unknown critic descriptors by
default. Check each row's stable-to-relative map, stable remaining budget,
relative descriptors, model/export/reward/profile/generation/request identity,
candidate IDs/order/mask, shapes and finite values. The feature encoder validates
observation semantics; callers of the typed interface must supply its matching
revision and complete legal choice slice. The diagnostic native-state path
obtains both directly from the simulator and rejects stale/foreign handles.

The five output tensors are legal policy logits and stable-seat win share,
placement, VP and remaining utility. The exported graph already performs value
seat remapping: do not rotate its outputs again. Preserve logits as f32; compute
stable log-softmax/probabilities in f64. No probability floor or two-player sign
flip is permitted. `visited_priors` normalizes the visited logits directly, even
when their full-set probabilities underflow to zero. Search temperatures, Gumbel
draws and completed-Q improvements belong to L10's separately identified search.

`ReferenceEvaluator<T: ReferenceTransport>` implements the same interface and
rejects reordered, partial, nonfinite or stale service responses before publishing
them. The transport must synchronously retire its use of borrowed tensors before
returning, including on cancellation/error. This supplies the reference-service
boundary; GPU networking/collection remains Part 2. Python's diagnostic reference
adapter calls the existing frozen `Graph` evaluator only in parity harnesses.

## Cooperative batching and ownership

The default is immediate singleton: cap 1, pending limit 1, flush 0 microseconds.
The sweep supports caps 1/2/4, pending limits up to four, and flush 0/500/1000/2000
microseconds. `Queue::take_ready` flushes when full, old, near expiry, constrained
by padded-cell limits, or unable to produce more work. It never sleeps waiting
for a future request from the same thread. Only one model, generation and schema
may enter an evaluator batch. Future search owns the pending trajectory/tensor
slots; L12 adds tree reservations and transactional backups.

Input lifetimes cover the entire synchronous runtime call. ORT output handles
remain alive through shape/mask checks and copying. Replies own their data;
subsequent packing/encoding cannot overwrite them. The evaluator retires after a
runtime/output failure. A panic retires its owner; a process failure retires the
transport. Cancellation and expiry discard every result in the diagnostic job;
they do not assert that ORT stopped in the middle of a kernel. The owner remains
busy until the call returns and its ticket is drained. No runtime cancellation
thread or premature buffer reuse exists.

## Diagnostic protocol

`inference/load` verifies and warms the pinned bundle. `inference/start` starts a
bounded asynchronous job on one owner and returns an owner-tagged monotonic
ticket. Each job has a newer per-owner generation, batch ID, remaining budget and
up to four permitted observations or native-state handles. This job-level
generation is a diagnostic lifecycle epoch, not the future search tree's epoch.
Neither a new ticket nor a retry may refresh a live turn's allowance; the future
coordinator must pass its remaining allowance.

`poll`/`cancel` use the exact active ticket. The control thread may process the
other owner while evaluation runs. Old, foreign and duplicate tickets are
rejected. Other operations on a busy owner are rejected. Cancellation responds
`cancelling`; only draining the final reply releases the slot. Polling rechecks
expiry/cancellation even if a completed reply was waiting in the channel. Python
also rejects delivery past its own deadline, including transport time. L11-L13
still owe whole-turn clock mapping, complete plans and Unity admission.

The diagnostic wait helper also has a drain watchdog: five seconds after the
caller deadline (or the shorter configured transport timeout), a nonretiring
runtime kills and retires the worker. Individual pipe exchanges retain their
transport timeout. This is failure containment, not a five-second extension of
accepted search work or a qualified Unity response-time guarantee.

The one-MiB frame bound remains. Oversized diagnostic replies return an explicit
error; they do not truncate candidates. Use smaller diagnostic batches. Native
typed search does not serialize observations, features or NN results per leaf.

## Qualification and measurements

Exact L08 input equality remains required. Cross-runtime output tolerance is the
existing frozen exporter's `atol=2e-4, rtol=2e-4`, across PyTorch, Python ORT and
Rust ORT, including singleton/mixed batches and critic-only changes. This differs
from L08's same-PyTorch-runtime comparison; it is not a relaxation of input parity.

Record dispatch, summed position queue age, encoding, packing, binding, synchronous
inference, output copying and postprocessing separately. Queue age overlaps other
work and is not additive CPU time. Record copied tensor bytes, valid/padded cells,
actual batch sizes, shared-process work intervals, two model handles, OS thread
counts/CPU and memory. Output-copy counters include the temporary raw head copy
and owned row copies; metadata/string allocation is measured separately.

The allocation probe reports aggregate Rust allocations across two owners,
including validation and owned replies. It cannot see the DLL's C allocator;
RSS growth is a separate coarse observation. Never use instrumented timings as
performance results. The batch sweep is an encoder/NN microbenchmark, with
diagnostic orchestration in its wall time. Final trajectory batching, 4-second
whole-turn comparisons and strength evaluation remain L10-L15.
