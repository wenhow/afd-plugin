# C5 native geometry / manager boundary — 2026-11-09

This is the implementation contract for the new CPU geometry module. It builds
on [the pre-prepare source review](pre-prepare-dispatch-contract-review-20261109.md)
and [the phase protocol proposal](manager-control-contract-proposal-20261109.md).
Production C4 code and already staged Graph component files were not changed by
this work. Native model Graph/Engram/checkpoint admission remains closed.

## Implemented API and concrete consumers

`afd_plugin/v1/worker/npu/hccl_v41_graph.py` now exports:

```python
@dataclass(frozen=True, slots=True)
class HcclV41StageGeometry:
    request_start: int
    request_stop: int
    token_start: int
    token_stop: int

@dataclass(frozen=True, slots=True)
class HcclV41GraphGeometry:
    descriptor: BatchExecutionDescriptor
    stages: tuple[HcclV41StageGeometry, ...]

@dataclass(frozen=True, slots=True)
class HcclV41GraphDispatch:
    candidate: HcclV41GraphGeometry
    eager: HcclV41GraphGeometry

def preview_hccl_v41_geometry(
    batch_state: BatchReqState, descriptor: BatchExecutionDescriptor
) -> HcclV41GraphGeometry: ...

def capture_hccl_v41_geometry(
    descriptor: BatchExecutionDescriptor,
) -> HcclV41GraphGeometry: ...

def dispatch_hccl_v41_graph(
    manager: CudaGraphManager | None,
    batch_state: BatchReqState,
    *,
    parallel_config: ParallelConfig,
    allow_ubatching: bool,
    need_eager: bool = False,
    max_query_len: int | None = None,
    num_active_loras: int = 0,
) -> HcclV41GraphDispatch: ...
```

The planned consumers are the A execute scope's early dispatch/controller and a
real native-manager NPU component probe. They must consume these return values;
the helper does not itself install a new factory or change a runtime selector.

The module imports native types and the native pure slicing helper, but no
transport, model, execution scope or connector. The protocol owner keeps its
`HcclGraphStage` / `HcclGraphKey` / candidate / report / decision types free of
native runtime imports. The A consumer explicitly projects the four bounds and
descriptor fields into those wire types. There is no reverse dependency from
the distributed protocol to this worker module.

Geometry excludes dynamic token validity, request/token IDs, history values,
transaction IDs, graph objects and cache state. Those values do not become
static simply because a Python dataclass is frozen. A real cache entry separately
owns its graph, buffers, metadata views/events and storage generation.

## Selection and fallback semantics

Dispatch requires native DP1 rank zero. It computes U from the actual raw token
count using native `check_ubatch_thresholds` and `get_num_ubatches`, and requests
the real manager's `dispatch(..., num_ubatches=U)`. U2 cannot be produced by
renaming a returned U1 cache hit. Profile/forced eager, prefill, non-query-one
input and active LoRA return a fresh NONE descriptor before cache lookup.

Only FULL uniform query-one, no-LoRA descriptors are accepted as graph geometry.
Varlen, PIECEWISE and other unsupported native candidates fall back early. A
local NONE result remains a local cache miss; the phase resolver must still
independently preview and check a peer's proposed FULL key. It must not mistake
that local miss for a fatal native-input disagreement.

`eager` always describes actual requests and tokens, with the native
threshold-selected U. On all-peer miss, return that descriptor to native execute
before preparation. Do not replace only a FULL descriptor's mode while retaining
unproven graph padding. Preview rejects that shortcut explicitly.

For FULL, the helper uses the actual native ordered CPU scheduled vector and
the exact native common U2 slicing function, followed by the two GPU request
clamps. For eager, it uses actual dimensions. Uniform q1 capture creates the
same all-ones CPU scheduled vector implied by native `InputBatch.make_dummy`,
without constructing a device InputBatch or inventing a BatchReqState.

The real-four/padded-eight/U2 boundary remains visible: runtime trailing request
bounds are `3:8`, whereas full dummy capture uses `4:8`. The graph/cache consumer
must compare that complete layout. The helper deliberately does not report a
native descriptor as a ready model graph simply because manager dispatch found
its token bucket.

## Exact pinned manager lifecycle

Sources remain vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` and vLLM-Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`, read through locked Git objects.
The following signatures are future patch boundaries, not patched by this new
module:

| Boundary | Exact native signature |
| --- | --- |
| CPU batch gather, actual selected Ascend owner | `gather_batch_req_state(self, scheduler_output: SchedulerOutput, dummy_run: bool)`; no return annotation |
| Native imported dispatch | `dispatch_cg_and_sync_dp(cudagraph_manager: CudaGraphManager \| None, num_reqs: int, num_tokens: int, uniform_token_count: int \| None, dp_size: int, dp_rank: int, max_query_len: int \| None = None, need_eager: bool = False, num_active_loras: int = 0, parallel_config: ParallelConfig \| None = None, allow_ubatching: bool = False, uniform_decode: bool = False, dp_sync: DPSyncState \| None = None) -> tuple[BatchExecutionDescriptor, DPSyncState \| None]` |
| Base native capture | `CudaGraphManager.capture(self, create_forward_fn: CreateForwardFn, progress_bar_desc: str = "Capturing CUDA graphs") -> None` |
| Capture factory protocol | `CreateForwardFn.__call__(self, desc: BatchExecutionDescriptor, warmup: bool) -> Callable[[CUDAGraphMode], None]` |
| GPU model replay | `ModelCudaGraphManager.run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor \| tuple[torch.Tensor, list[torch.Tensor]] \| IntermediateTensors` |
| Actual Ascend model replay | `ModelAclGraphManager.run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor \| tuple[torch.Tensor, list[torch.Tensor]]` |

`CudaGraphManager.capture:436–485` calls a fresh factory first with
`warmup=True`, executes the returned callback, then calls a fresh factory with
`warmup=False` before entering the device graph context. **Both FULL callbacks
receive `CUDAGraphMode.NONE`**. Host phase must come from the factory's warmup
argument and controller state, not the callback mode.

The graph is inserted in `manager.graphs[desc]` only after the graph context
exits, at line 485; `_graphs_captured=True` is set after the complete capture loop
at line 488. The owner must not advertise an in-progress graph as ready.

The smallest correct graph-control ordering is:

1. At entry to the next factory, synchronize and COMPLETE the previous phase
   before writing any buffers for the next one.
2. Derive capture geometry from `desc`, reserve/agree the current phase, then
   delegate native fresh input/metadata preparation.
3. Validate prepared layout and submit actual counts, then return a callback
   whose graph body contains only tensor/event operations.
4. After the outer native capture returns successfully, synchronize and COMPLETE
   its last pending phase outside capture. Only then expose a ready cache entry.

A duplicate capture key must be globally rejected or explicitly recaptured in a
new agreed generation before its peer enters warmup. A local early return leaves
the remote HCCL launch unmatched. Preparation/capture failure is terminal in this
first protocol; it does not send a normal COMPLETE or silently reuse the session.

The controlled native-manager component probe can call the base capture API
directly with its real factory and prove that ordering. Production
`ModelAclGraphManager.capture` delegates through `ModelCudaGraphManager.capture`,
whose factory is a local closure, not a replaceable instance attribute. A future
production patch must either intercept the base capture call for the one target
manager in a scoped exact-signature hook, or explicitly copy the native model
capture factory. A scoped delegation exception preserves native graph pools,
offloader synchronization, descriptor order and fresh metadata preparation;
it must restore the binding and join the existing sole signature table.

U2 needs additional model-owner work: the native model factory calls
`prepare(..., FULL, for_capture=True)` and `begin_capturable_run`, which starts
native full-model handoff threads. The current HCCL V4.1 runner rejects FULL and
uses a synchronous layer-major eager body. Geometry alone cannot make that body
capturable. The future Graph runner must retain native per-stage preparation
while providing a synchronous tensor-only capture body, and preserve one parent
Engram producer plus native Ascend retirement. The existing C4 host-cursor
send/receive functions cannot execute inside capture.

## Verification and next concrete tasks

The new mirrored CPU file contains 46 parameter cases. It calls the actual
native slice function and `CudaGraphManager.dispatch`; its preloaded CPU manager
candidates represent software cache fixtures, not constructed device graphs.
The shared builder constructs the real native BatchReqState NamedTuple in an
explicit order. No test asserts private output state or helper call counts.

Local Ruff, formatting and AST checks passed. The local environment lacks numpy
and torch; collect-only returned exit 5 without collected tests, so this report
does not claim a local behavioral pass. Actual-dependency CPU execution and the
native-manager NPU probe are still pending for these exact source bytes.

Next implementation tasks are concrete consumers: wire this selector into the
existing gather/dispatch execution scope; map native geometry into the strict
phase/key protocol; run the base native manager factory with actual tensor
exchange and native U2 capture stream; then implement and independently review
the model Graph body/Engram/metadata manager hooks. Keep the native manager
component, real ModelAclGraphManager integration, GE/Dynamo and full-checkpoint
acceptance as separate claims.

## Actual-dependency CPU result, 2026-11-10

The A3 record [control-001](a3-control-cpu-20261109/control-001/results/summary.json)
completed five targeted modules with **317 passed, zero failures/errors/skips,
exit 0**. Its JUnit XML contains all 46 geometry cases. I independently compared
the geometry source, test and shared BatchReqState fake against `identity.json`;
all three working-tree hashes match the recorded manifest:

| File | SHA256 |
| --- | --- |
| Geometry source | `bb8a73e4d1ea46dc012af7980cc4a5edcad1f915e7b9c62d6942b47c09da9777` |
| Mirrored test | `6d06a19e5ffeddd082164a64713273155c94940d327d72663b8ad0acd06e545f` |
| Shared native batch-state fixture | `76516b74b001ffc5cfbea8621c325136540230d67275dffaf3f98b7ae000e347` |

The snapshot manifest is
`e40db49ad7416e556f6507c114bbbd59a6a9c1a92cc77aca92fef42d7611fb2c`.
No device probe was started by that run. This replaces the pending CPU status
for these exact three files; it does not validate native NPU capture/replay or
the pending model lifecycle consumer.
