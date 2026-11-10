# Stage diagnostics across serial, eager tickets and Graph

Status: source contract and proposed next software increment only. No frozen
runtime, helper, session, driver or FFN file was changed for this audit. No SSH,
torch import or device experiment was performed. The
[source record](stage-diagnostics-family-source-20261110.json) pins the current
HEAD, fourteen source SHA256s and thirty-six function bodies/line ranges.
Existing CPU and NPU results keep their independent acceptance boundaries.

## Current gap

`HcclStageDiagnosticRun.__post_init__` rejects eager execution when its actual
compute stream is not the parent. `_fields()` consequently has only `serial`
and `graph-streamed` modes and always reports actual send/receive on the parent.
This correctly describes the existing Graph parent-transport override; simply
removing that eager rejection would mislabel ticket execution as Graph.

The actual eager owner posts torch.distributed operations on its
`send_stream` / `receive_stream`; its loans select the passed consumer stream
and install that stream in vLLM TLS. The new A/F consumers use those actual
objects. Neither eager session nor owner construction currently requires all
stream objects to be distinct: a valid indexed-device stream can alias the
parent or another role. A configured/logical side stream is therefore not
proof of a distinct actual stream, much less concurrent device execution.

The serial FFN parent and Graph FFN body already accept diagnostics. The A
serial stage loop and the A/F eager ticket consumers do not yet receive
progress. `HcclV41EagerSession` is U2-specific, so it cannot be the sole settings
owner for serial U1 diagnostics.

## Minimal actual execution schema

Retain `selection.diagnostic_record()` unchanged for configured, effective,
logical and selected physical fields. Add an explicit immutable actual
execution description to `bind_execution`, separate from that selection:

```python
HcclDiagnosticExecution(
    mode="eager-streamed",          # serial / graph-streamed / eager-streamed
    compute_stream="compute",       # parent / compute
    send_stream="send",             # parent / send
    recv_stream="recv",             # parent / recv
)
```

The name is an API proposal, not an implemented class. This small value has no
torch dependency, allocations, environment access, synchronization or logger
state. Existing `actual_compute_stream` callers can map to a legacy description
until their containing increment is updated together; existing Graph output
and default behavior must remain stable.

The active executor constructs this description from the actual selected
owner/streams, never from `selection.logical_*` or `selection.physical_*`.
For eager tickets, classify each role as `parent` when its stream compares equal
to `owner.parent_stream`. A non-parent stream keeps its compute/send/recv role.
If physical identity is exposed, record the installed stream's native handle
as an integer and its device index, rather than Python wrapper `id()`.
Equal native handles must remain visible as aliases; do not force a new
distinct-stream restriction merely to simplify diagnostic output.

`actual.mode=eager-streamed` identifies the Work/ticket executor. It does not
claim that its streams are distinct or that operations overlap. When streams
alias, emit their real role labels/handles honestly. Keep
`actual.compute_pipeline_enabled` as a side-compute fact, not as proof that a
Graph pipeline object or an overlap exists. Do not infer the executor family
from that boolean.

| Actual route | Mode | Compute/send/recv | Compute-sync diagnostic |
| --- | --- | --- | --- |
| A serial U1/U2 | `serial` | parent / parent / parent | Off |
| F serial live target U1 | `serial` | parent / parent / parent | Only when explicitly requested |
| F serial U2, profile, warmup or dummy | `serial` | parent / parent / parent | Off |
| A/F eager target U2 tickets | `eager-streamed` | Actual consumer / owner send / owner receive, including aliases | Off |
| Graph body with side compute | `graph-streamed` | compute / parent / parent | Off |
| Graph body with compute disabled or U1 | Existing `serial` record with Graph phase | parent / parent / parent | Off |

Graph remains subject to its actual parent HCCL override regardless of logical
Attention-three or FFN-receive/cross-layer settings. A fallback that actually
runs the serial eager executor reports serial actual streams even if a logical
setting would have preferred a pipeline.

## Settings, rank configuration and ownership

The existing A/F owner binds one immutable `HcclOverlapSettings` instance and
optional diagnostic sink during explicit setup. Parsing the supplied
environment mapping belongs to that setup, not to the diagnostic helper,
decoder, ticket method or capture body. A serial U1 owner needs this independent
settings/sink binding; do not allocate a U2 stream session to turn on U1 logs.
When an eager session or Graph owner also exists, reject contradictory settings
before any payload work instead of silently rereading environment variables.

After an accepted parent and actual route are known, each participating A and F
rank creates its own selection/progress value. Validate exact transport, role,
role rank, world rank, stage count and parent identity before binding it. With
either diagnostic enabled, emit one configuration record per rank/parent before
model work, even for an all-empty parent. Include transaction ID and actual
mode/streams alongside configured/effective settings and profile/warmup flags.
No leader-only logging, extra gather, new collective or global logger registry
is needed. STOP is not an extra parent and does not fabricate stage records.

Configuration may describe settings that are ineffective in a particular
phase; this difference must be visible. In particular, U1 and profile/dummy
work do not acquire the U2 ticket executor just because the session flag is on.

## Concrete propagation points

1. `HcclV41Execution.install_context` accepts the authoritative parent before
   native model invocation. Carry the immutable settings/sink to the execution
   scope, then pass one matching diagnostics value through
   `execute_hccl_v41_stages` to `run_deepseek_v41_stages`. The same path serves
   U1 and serial U2; no second Engram producer or parent preparation is added.
   A serial loop records Attention/HC preparation, input send, output receive,
   remote continuation and final collapse as host enqueue boundaries.
2. `HcclV41UBatchRunner.run_staged` chooses the eager session before launching
   the model. Construct the actual descriptor after that choice. For a ticket
   parent pass progress through the existing stage driver and its delayed
   `_run_pipelined_stages`; for a serial fallback pass the serial descriptor.
   Keep preparation, native child contexts, Engram events and HC continuation
   storage ownership unchanged.
3. The FFN runner's `execute_hccl_v41_step` passes the same setup settings to
   either `execute_hccl_v41_parent` or `execute_hccl_v41_eager_parent` after its
   current route decision. Add an optional diagnostics argument to the latter;
   default `None` preserves the frozen consumer. Validate rank/parent identity
   before receive or failure handling, preserving the stale-owner protection.
4. Pass the bound eager progress to the owner once for that parent. Instrument
   actual post/loan/reuse/finish boundaries there, since A's `post_attention`
   contains both its three input sends and its output receive. Logging only
   around the outer API would hide which posting or dependency is blocked.
   Model consumers own compute/HC events; the owner owns communication and
   loan events, avoiding duplicate events or new waits.

These are explicit opt-in plumbing points. They do not enable a production
model guard, select a new backend or change the wire protocol.

## Eager milestones and completion vocabulary

Keep existing Graph event names unchanged. For eager tickets add explicit
`send_post_begin/returned`, `recv_post_begin/returned`,
`loan_wait_begin/returned`, `compute_begin/returned`,
`stage_reuse_wait_begin/returned` and `parent_finish_begin/returned` as needed.
First/last-layer filtering remains the default stage-volume policy; use the
issuing ticket's original layer/stage for delayed consumption or stage reuse,
rather than the next layer's current loop coordinate.

Posting records describe host submission; Work.wait records describe installed
dependencies or host return. Neither is a device-completion timestamp. A
`compute_returned` record means the native call returned/enqueued work. Only
the existing explicit live U1 synchronizer may produce a compute-sync-complete
event. Do not add `event.query`, sleeps, Work waits, device synchronization,
event allocations or allocator releases solely for logging.

Diagnostic values and emitted dictionaries contain scalar metadata, not tensor,
Work, event, stream or model references. The real owner remains responsible for
lifetime, ready/done events, storage loans and exception retention. Preserve the
current sink-error propagation behavior; a diagnostic exception in live work
must retain that exact owner's resources, not fail a foreign/finished parent.

## Keep Graph replay and U1 synchronization contracts

Graph configuration remains bound before native capture when its factory builds
the retained body. Python progress executes only while that body is run for
warmup/capture; native graph replay does not reenter the helper or model Python
loop. Do not reconstruct a diagnostics object or add Python stage callbacks,
NPU events, allocations or waits to each replay. Cached actual descriptors and
settings stay immutable with the resources that were captured.

`compute_sync_active` must require all of: F role, phase eager, actual serial
parent route, target parent, exactly one stage, explicit setting on, and neither
profile nor warmup/dummy. An A rank, U2 parent, eager ticket owner or any Graph
phase always reports this diagnostic false and never calls the synchronizer.
Bind its device explicitly at the worker as the existing FFN seam does.

## Focused validation for a later implementation

Use helper/output and native-call seams, not assertions about source text or
private attributes. Parametrize A/F, TP1/TP2 ranks, U1/U2, serial/ticket/Graph,
empty and padding-only stages, enabled/disabled logs and aliased streams.
Assert configuration records for every participating rank; actual fields must
follow selected stream objects despite conflicting logical settings. Compare
communication order, FP32/IDs and native outputs with diagnostics disabled.

Add stale/foreign diagnostic identity rejection before payload/producer work,
borrowed-context and TLS restoration on sink/model failure, exactly one normal
parent completion, and the U1-only synchronizer exclusion matrix. Existing
Graph tests must show unchanged configuration and no extra Python stage events
when replaying a retained graph. These CPU results qualify the software seams;
any later device log probe remains separate from full-model or overlap/perf
acceptance. This design itself starts no new hardware job.
