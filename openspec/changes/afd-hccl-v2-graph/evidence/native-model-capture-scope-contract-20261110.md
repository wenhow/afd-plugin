# Native capture factory scope and phase binding

Date: 2026-11-10. Author's contract record for the new explicit compatibility
scope and persistent runner binding. Production Graph admission remains closed.

## Exact locked seam

vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` defines
`CudaGraphManager.capture(self, create_forward_fn: CreateForwardFn,
progress_bar_desc: str = "Capturing CUDA graphs") -> None` at
`gpu/cudagraph_utils.py:398-488`. Native ModelCudaGraphManager.capture defines
its model factory locally at 644-745 and passes it to the base at 747. Its
prepare_dummy_inputs at 664 precedes both the U1 Attention helper and U2
runner preparation.

The new `use_hccl_graph_capture(manager, *, before_factory, after_capture, fail)`
scope wraps that base seam, preserving its exact signature. The selected native
capture executes once. Other managers delegate unchanged. Before each native
factory call, `before_factory(desc, warmup)` may retire the previous phase,
reserve the next phase and arm its execution owner. It then invokes the native
factory once. `after_capture()` runs only after native capture returns and has
stored the final graph. No callback is inserted into the graph tensor body.

The scope rejects nesting, repeated selected capture and selected calls from
another thread, and restores the original class binding even on failure.
Failures mark the caller's phase owner terminal; they do not fabricate a
coordinated cancel. Concurrent capture is outside this controlled scope.

Delegation is deliberate: the native function owns graph pools, streams,
memory profiling, graph recording, persistent outputs and abort cleanup.
Duplicating it would split those owners. Remove this patch when native before-
factory/after-capture callbacks are available. The existing sole signature
table owns its alignment row. AST arguments/defaults/return annotations were
also compared directly to the locked Git object.

## Persistent runner owner

`HcclV41GraphSession(transport)` can be passed as the runner's graph owner during
native KV construction, before any capture reservation exists. Its transport
property is read-only. A coordinator calls `arm(current_execution)` after
accepting one reservation and before native preparation, and `clear()` after
the accepted plan and reservation are retired. It delegates require_full,
prepared and forward_parent to that owner. An armed owner cannot be replaced;
another session/transport, inactive transport or binding-to-binding recursion
is rejected. Failures retain the owner and dependencies for external supervision.

This solves the construction order without creating mutable global state or
opening a user flag. It does not create the coordinator, select production
Graph mode, capture a model, send COMPLETE or own native Engram retirement.

## Cost and validation boundary

The patch adds one host callback per native factory invocation and a final host
flush. The runner binding adds one local delegation per preparation/model
callback. Neither introduces a device operation or per-replay tensor copy.
No performance qualification is claimed.

Ruff and exact locked-signature checks passed. Local collection on the host
without runtime dependencies skipped its gated modules. The new CPU tests
check notifications before preparation, final-flush ordering, selected versus
foreign managers, failures/restoration, nesting, owning thread and repeated
capture; an explicit CPU boundary replaces native graph execution. Session
tests check successive phases, public output/context behavior, transport
identity, invalid retirement and terminal failure. These tests must not be
reported as NPU capture acceptance. Actual-dependency results are appended
separately when the new snapshot completes.

## Next U1/runtime integration boundary

The native U1 helper `prepare_inputs_to_capture:765-838` constructs the actual
AscendInputBatch internally, calls selected EngramModelState.prepare_attn with
NONE and for_capture=True, and returns metadata plus layer slot mappings.
A selected-instance prepare_attn scope can observe that real batch before
native model execution without copying the factory. Native metadata run_build
already treats for_capture=True as FULL task ownership. Retain FULL on the
ModelState after that selected call, build one FULL child context from the
returned metadata and native slot mapping, and submit counts outside capture.
U2's existing selected prepare_attn override owns its children and must not be
wrapped a second time.

A native U1 capture forward creates a NONE parent context with no batch
descriptor. Its sidecar installer therefore identifies the actual retained
metadata, rather than using batch_descriptor as the discriminator. It installs
only the parent tensor body: no counts, wire agreement or wrapper recursion.
The original ModelWithContext must execute once. Runtime U1 FULL uses the same
actual preparation hook; runtime U2 additionally needs an explicit handoff for
ModelAclGraphManager's parent metadata read. Native run_fullgraph and final
Engram/metadata retirement remain authoritative.
