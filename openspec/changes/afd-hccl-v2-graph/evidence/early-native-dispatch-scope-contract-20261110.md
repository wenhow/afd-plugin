# C5 early native dispatch scope, 2026-11-10

This is the implementation contract for the reviewer-authored
`afd_plugin/compat/patches/npu/hccl_graph_dispatch.py`; it does not self-certify
independent architecture acceptance. The separate state-audit reviewer read the
final scope and tests and reported no blocking finding for the explicit DP1
owner boundary. Actual-dependency regression is recorded separately below when
available. No full-model Graph enablement is claimed.

## Exact native seam and ownership

The selected Ascend method is
`NPUModelRunner.gather_batch_req_state(self, scheduler_output: SchedulerOutput,
dummy_run: bool)` with no return annotation, at
`7e37e29a7803d6897a7113088e2b771f83205bc0:vllm_ascend/worker/v2/model_runner.py:615`.
The temporary bound replacement has that exact signature. It calls the selected
native method once, retains its real `BatchReqState | None` and uniform count,
and returns the original tuple object without changing order, values or storage.

The imported `gpu.model_runner.dispatch_cg_and_sync_dp` comes from
`ced6857afa0ea7b2e3f0846a62e1394e90f15607:vllm/v1/worker/gpu/dp_utils.py:214`:

```python
def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    need_eager: bool = False,
    num_active_loras: int = 0,
    parallel_config: ParallelConfig | None = None,
    allow_ubatching: bool = False,
    uniform_decode: bool = False,
    dp_sync: DPSyncState | None = None,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]: ...
```

The temporary imported replacement preserves all thirteen parameters, defaults
and the return annotation. A local AST comparison against both locked Git
objects independently matched argument lists and return annotations. The two
runtime signature checks were added to the existing sole parametrized table;
the existing A5 quantization rows are preserved.

The native execute calls gather at line 1660 and dispatch at 1685, before
`prepare_inputs` at 1714, dummy preparation, metadata and ModelState preparation.
Native request-state updates earlier in execute still happen normally; the
scope is not a rollback mechanism for those updates. Native execute, its input
and attention builders, ModelState/Engram preparation and its sole final
`finish_execution` remain their existing owners.

## Consumed callback contract

```python
with use_hccl_graph_dispatch(owner, select):
    # The existing selected native execute invokes gather and imported dispatch.
    ...

select(
    actual_state: BatchReqState | None,
    arguments: HcclGraphDispatchArguments,
    native_descriptor: BatchExecutionDescriptor,
) -> BatchExecutionDescriptor
```

`arguments` records all thirteen dispatch inputs. Its `need_eager` includes
profile/other native forcing, missing real state and an already-eager reused DP
agreement. The selected imported native dispatcher runs once first. The
controller receives that actual native result and returns the final descriptor;
the scope always returns the native dispatch's original DPSync object unchanged.
A forced eager invocation cannot be promoted to Graph by the controller.

Missing real state is not fabricated into `BatchReqState`. Dummy/profile and
zero-token inputs still call the selector with `None`, an effective eager flag
and the real native eager descriptor, allowing an early all-peer eager
reservation. The controller owns the authoritative native U-specific lookup,
exact geometry and wire decision. In particular, a native U1 cache hit is not
renamed to U2 by this scope.

The root-owned controller is the callback consumer. Binding this scope into the
normal production execute owner remains a separate integration step; this file
alone does not claim a complete startup, successful model execute or capture.

## Isolation, exceptions and cost

The scope allows one authoritative gather and one dispatch on the owning host
thread, checks DP1/rank zero, the current owner's manager identity (including
`None` during initialization/profile), the provided parallel-config identity,
uniform count and real batch dimensions. Dispatch before gather, repeated
batches, another manager/config, cross-thread use and nested scopes are rejected.
The native call site always supplies its owner's parallel config, so profile
manager `None` does not remove that ownership check.

The imported temporary bound method identifies nesting without a new global
registry. Concurrent execute scopes are excluded: this is not a process-wide
concurrency lock. On all exits, the original module alias and selected gather
binding are restored, including an existing instance override. Neither an
exception nor the scope itself destroys transport groups, retires Engram state
or announces a recoverable network abort.

The scope retains one CPU batch-state reference until exit, performs constant
host validation and introduces no device allocation or tensor copies. Native
dispatch followed by the controller's U-specific dispatch is an intentional
additional host lookup. Wire agreement and native preparation remain owned by
the controller and native execute respectively. Delegation is necessary here
to preserve the selected gather and imported backend dispatch binding; copying
their native bodies would bypass those selected behaviors and duplicate DP
synchronization/reuse logic. Remove these wrappers when upstream provides an
equivalent pre-input-preparation callback with real batch state.

## Validation boundary and frozen sources

Local Ruff and formatting passed. The local dependency-light pytest run reports
12 skips because native runtime dependencies are absent; it is not behavioral
acceptance. The mirrored scope module contains 23 behavior cases covering native
DP1 dispatch, actual gather provenance, dummy/zero/profile forcing, DPSync
preservation, errors and restoration, invalid ownership, nesting and thread
boundaries. The geometry test fixture now shares the native CPU graph-manager
builder with the controller and scope tests; this tests real dispatch methods
with preloaded CPU candidates, without constructing device graphs.

| File | SHA256 |
| --- | --- |
| `compat/patches/npu/hccl_graph_dispatch.py` | `4b8800f54dd6cde11a7c1bbf21dc12b8ac24dd70272bbbd4f0f7654693255e0c` |
| mirrored scope test | `f2a6d5319cc886c7bfc4bc21faf12bc88ea10b8f5c5ae196245fe791990581ae` |
| shared selected-owner fake | `ffd2aaed97c81b70cff8cc7a3d1c6d990bc16e08ed5f3bc22d28afd6bc042ac2` |
| sole signature table | `0195ae3913f79afd6ae1fa3be99a1115fd519cecb305c0aa0da15b02e24b6022` |
| geometry test using shared manager | `80b46d6f7a38ec9cc9c0a5ec94251ff6ebdb645666f97a99c0d3ef574e641ce2` |

CPU seam results cannot establish HCCL device ordering, NPU Graph replay,
ModelAclGraphManager integration, V4.1 Engram preparation or checkpoint output.

## Actual-dependency CPU regression

I inspected [consumer-005](a3-control-cpu-20261109/consumer-005/results/summary.json)
and its JUnit XML: **129 passed, zero failures/errors/skips**, exit 0, with no
device execution. Counts are controller 36, geometry 46, new scope 23, the sole
upstream signature table 10 and cache 14. The scope, mirrored scope test, owner
fake, signature table and shared-manager geometry test all match the final
hashes above byte for byte in `identity.json`; the shared native manager builder
is `b26ce0e57efe5b407699be61b5257f933e4ecdc548ca11a6bbbd07d06c34599c`.
This closes the source signature and CPU behavior requirements for the scoped
entry. Native execute/model/Graph device acceptance remains a separate boundary.
