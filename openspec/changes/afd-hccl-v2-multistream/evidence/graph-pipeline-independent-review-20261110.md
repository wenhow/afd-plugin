# C6 graph compute pipeline independent review

Static read-only review on 2026-11-10. This reviews one independent scheduling
component and its host seams. It does not accept the later native model driver,
capture/replay integration, device overlap, performance or production admission.

## Reviewed snapshot

| File | SHA256 |
| --- | --- |
| `afd_plugin/connectors/npu/hccl_graph_pipeline.py` | `12eab892912906240d116ac064e8bc1b449b7a8a278433547143f6e1041f7bfd` |
| `tests/unit/fakes/hccl_graph_pipeline.py` | `cd0b8707a497465523ca2ca1185b019c3156ec54f1bedde6a87390fa59cb307f` |
| `tests/unit/connectors/npu/test_hccl_graph_pipeline.py` | `44250a7e148076822d053e26549b4a4d09b82b119c0b32020ce3e8bf555c978a` |

Source comparisons use donor `4ddd69a391dfb5b3e362c7601399e538a31de702`:
`afd_plugin/connectors/npu/p2p_hccl.py:726-807,1241-1317`,
`afd_plugin/model_executor/models/deepseek_v4.py:1067-1244`, and
`afd_plugin/v1/worker/npu/ffn_model_runner.py:1061-1140`. The donor's graph
compute helpers wait matching receive events on a side stream, record every
crossing tensor's consumer, and keep graph HCCL on the parent stream. This
component is a new composition API; it does not copy a V4 continuation tuple
into V4.1 or replace an upstream function. V4.1 HC/pre_mix/Engram/auxiliary
semantics remain responsibilities of its future actual model consumers.

## Earlier findings and closure

Root identified two issues in the withdrawn initial candidate. Non-hybrid
final compute needed a real fence for the complete parent communication
frontier, and Attention needed a canonical complete-layer host schedule.
The reviewed snapshot has distinct typed tail `ready` and `done` events;
non-hybrid final compute records ready on the parent after all wire steps and
waits it on compute. Attention requires all stages' current-layer compute
before any exchange, and all preceding-layer exchanges before next-layer
compute. Final compute requires the complete last wire frontier. These close
the two static scheduling gaps without changing the C4/C5 transport cursor.

No further hard blocker was found in this independent component's state/event
contract. Actual NPU graph event behavior remains a separate validation.

## State, stream and bank reuse

`allocate` requires an indexed NPU bank outside capture and allocates streams
and events there. Manual CPU construction is explicitly a host dependency
trace. Step/tail events cover precisely positive-row stages; event objects
cannot alias. Static zero-row stages perform no dependency waits or wire
operations. A whole-empty parent remains the existing eager protocol boundary,
not a fabricated positive Graph bank.

`run` binds a distinct same-device actual parent stream once. The object is
single-use for one warmup/capture factory; recorded replay never re-enters its
Python state machine. Communication scopes require that parent, enforce
layer-major/stage-minor wire order and record completion after their body.
Compute scopes select the side stream and reject a body that leaves it current
on exit. Nested runs and reuse after recording/failure reject.

Attention's first compute waits a parent readiness event. Later hybrid compute
waits its own preceding-stage receive, allowing that dependency to be narrower
than the remaining queued parent work. Non-hybrid compute records/waits the
current parent frontier. Parent exchange waits the actual compute producer
before BF16/original-FP32/raw-ID copies and HCCL. Its receive event therefore
fences the next same-stage continuation before the next exchange can overwrite
the reusable output buffer. Final continuation has a separate tail event, and
the parent waits every positive stage's tail before consuming returned outputs.

FFN receive waits the same stage's preceding send before reusing its bank slot.
Compute waits the current receive, send waits compute, and join waits all final
stage sends. The present wire schedule remains receive/compute/send ordered;
side-stream allocation alone does not prove overlapping FFN receive/compute or
any throughput gain.

All caller-supplied crossing inputs/outputs must be included in the typed tensor
tuples. This covers original FP32 and raw IDs as well as BF16/residual/HC inputs.
Each tuple is checked for the bank device and recorded on its actual consumer
stream. Python retention holds the latest tuple per stage plus joined outputs;
replaced tuples already had their previous consumer stream recorded. Failure
keeps stream, event, bank and current frontier references for supervision and
does not claim to cancel queued device work. The graph owner must retain this
component and all native model/bank/context dependencies until graph reset.

## Integration boundaries and cost

The scheduling policy is a capture constant. The current physical graph key
does not encode hybrid/stream policy: actual owners must fix the policy for a
session or invalidate availability through generation. A new generation makes
old entries miss while retaining their resources; the existing cache still
rejects duplicate-key recapture, so this is not an in-session recapture API.

Native helper stream caches also need review when real model consumers land.
Locked vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`,
`vllm/utils/torch_utils.py:792-848`, caches `current_stream()` in thread-local
state and updates it through patched CUDA `set_stream`. Merely entering an NPU
stream context does not establish that a cached vLLM stream follows it. The
selected locked V4.1 Attention/Engram/MoE code inspected here uses
`torch.npu.current_stream()` at its relevant event boundaries; Ascend's separate
cached stream helper is used by PyHccl weight transfer, outside this scope.
No claim is made that every future native kernel/TP/helper consumer has been
qualified. A real consumer using a cached helper needs a scoped synchronization
and restoration of that existing owner rather than a new stream registry.

For L layers and U positive stages, this allocates one compute stream,
4*L*U step events and 2*U tail events; host bookkeeping is O(L*U). The component
does not allocate payload banks, create process groups or add control frames.
It adds explicit event edges and allocator consumer records; tensor tuple
retention stays bounded by the current per-stage frontiers. Device event cost,
graph pool reuse and achieved overlap have not been measured by this review.
Removal depends on native stage-aware compute/event hooks preserving the same
parent HCCL, continuation and final-join ownership.

## Validation separation

The mirrored host tests use a stream/event call recorder and actual CPU bank
arithmetic. They observe protocol order, unchanged payload values, zero-stage
skips, temporary lifetime and failure retention. They do not execute NPU event
dependencies or simulate graph replay. Installed NPU API source review and
actual-dependency CPU/device results must be recorded separately with their
own hashes; none is implied by this static review.

## Installed API and directed CPU follow-up

The separate stdlib-only A3 source collection is
`a3-pipeline-cpu-20261110/torch-npu-stream-api-20261110.json`. It records
torch_npu `2.10.0.post4` without importing torch/torch_npu or executing a device
operation, and retains the complete three existing Python source files. This
review recomputed SHA256 from the archived source strings and matched each
recorded hash:

| Installed Python file | SHA256 | Relevant boundary |
| --- | --- | --- |
| `torch_npu/npu/streams.py` | `241741b356db5006797dbec119007afb33b73b0e1753bfb06ab27c989c5bfac4` | Stream construction lines 24-26, Event construction 134-136, record 143-151, wait 153-161 |
| `torch_npu/npu/utils.py` | `608b5984180706f4179c7d6427242ec9163b840b5d4fb56459f92d359c3a84b0` | StreamContext 176-229, stream wrapper 232-242, current_stream 260-272 |
| `torch_npu/npu/__init__.py` | `355df777433fd8c85670db75d8aacf7fa076332092470d59b6ba26202ed7fd2f` | Public NPU namespace |

The actual constructors are `Stream.__new__(cls, device=None, priority=0,
**kwargs)` under a device guard and `Event.__new__(cls, enable_timing=False,
blocking=False, interprocess=False)`. Ordinary Event sets `graph_external=False`;
the separate ExternalEvent sets it True. `record(self, stream=None)` and
`wait(self, stream=None)` accept the component's explicit stream argument and
delegate to their native base implementations. The stream context saves the
actual previous stream, selects the target through NPU set_stream and restores
the previous stream(s) on exit. `current_stream(device=None)` queries the native
stream for the selected device. These Python signatures fit the component;
ordinary graph-internal event dependencies still require actual capture/replay
proof from the native implementation.

The attempted `utils/_tensor_methods.py` path did not exist, so the collection
does not establish the underlying tensor.record_stream/allocator implementation.
The pinned donor and locked V4.1 `model.py:1523-1525` do use
`tensor.record_stream(stream)` at their own cross-stream boundary. Its runtime
allocator/capture behavior is still an explicit device validation item.

Directed actual-dependency CPU run `a3-pipeline-cpu-20261110/component-001`
reports 31 passed, zero errors/failures/skips, return code zero and
`source_files_match=true`. This review independently read its XML, summary and
manifest and matched the three reviewed component/fake/test hashes above. No
device probe was started. This accepts the reported host dependency/lifetime
seams for those bytes, and does not upgrade the result to NPU graph execution,
real native model scheduling, overlap or performance acceptance.
