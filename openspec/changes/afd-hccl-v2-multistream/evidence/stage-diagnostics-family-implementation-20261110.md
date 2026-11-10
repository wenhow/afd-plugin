# Diagnostic execution families: implementation and CPU evidence

This increment implements the earlier
[source contract](stage-diagnostics-family-contract-20261110.md) without
changing native HC/Engram/MoE semantics or enabling diagnostics by default.
The independent A3 actual-dependency
[018b CPU run](a3-pipeline-cpu-20261110/diagnostics-family-018b/results/summary.json)
passed 512 selected cases. Its source manifest is authoritative; this document
does not qualify subsequent bytes, device progress or model accuracy.

`HcclDiagnosticExecution` is an immutable actual execution description. Its
`mode` is `serial`, `graph-streamed` or `eager-streamed`; compute uses `parent`
or `compute`, send uses `parent` or `send`, and receive uses `parent` or `recv`.
The optional alias tuple reports actual compute/send, compute/receive and
send/receive equality in that order. Owner bindings derive these fields from
the concrete streams. They do not infer actual execution from logical overlap
settings. Aliases are legal and remain visible; contradictory parent labels or
an impossible equality relation are rejected.

`HcclStageDiagnostics.bind_execution` accepts `actual_execution` and the
accepted `transaction_id`. The existing `actual_compute_stream` keyword remains
available for captured Graph callers, with its previous record schema when no
new transaction or alias data is provided. Supplying both descriptions is an
error. The resulting run verifies role, world/role rank, stage count, complete
layer indices and transaction via `validate_parent`. `validate_execution`
compares actual roles and any explicitly supplied aliases.

The rank's immutable `HcclV41DiagnosticBinding` is bound to its exact transport,
settings and explicit sink/synchronizer during setup. After acceptance it
constructs a configuration record on each rank. No helper reads environment
variables or keeps mutable global state. The optional consumer keyword is
uniformly `progress=None`. The live eager ticket owner accepts
`bind_progress(progress, compute_stream=...)` once before any payload posting;
reusing the same object is idempotent. A different parent, rank, actual executor
or actual consumer stream is rejected before tensor work. Attaching a new run
after an FFN receive is also rejected even before its output has been posted.

The owner records send/receive post entry and return, loan Work-wait entry and
return, stage reuse and parent finish. These are host enqueue observations;
they neither introduce device waits nor establish device completion. Reuse and
loan records use the original ticket's layer/stage coordinates. Live sink
failure retains the current owner and its storage. The returned finish record
is emitted after COMPLETE: a failing sink there cannot mark a retired owner or
a newly accepted parent failed. Outer execution checks preserve this rule.

The A serial model records native embedding preparation, Attention/HC layer
compute, serial send/receive, HC continuation and final parent collapse. The
eager model records the same native compute/continuation boundaries while the
owner reports actual asynchronous posting and loans. F eager compute records
surround exactly its native MoE call inside the existing received-input loan
and native context. Exceptions restore those scopes and retain the exact live
owner. Independent FP32 routing values and raw IDs keep their original path;
these diagnostic changes add no reduction, padding mask or producer call.

Explicit FFN compute synchronization remains limited to live serial eager U1
FFN execution. A-side, eager tickets/U2, Graph, profile, dummy and warmup paths
do not enable it. Captured Graph replay runs no new Python logging. With
`progress=None`, the new consumer paths take only the optional branch; they
allocate no diagnostic objects and perform no diagnostic I/O, events or fences.

Validation remains separated: pure local tests passed 56 helper cases while
three runtime modules skipped for missing local dependencies. Python 3.11
grammar accepted 321 `afd_plugin`/`tests/unit` Python files; owned files passed Ruff. A
whole-tree Ruff pass found only 26 line-length errors in unrelated Graph work
in progress, recorded before upload and left to its owner. 018b used the
unchanged installed torch/vLLM stack, matched all ten upstream source pins and
started no NPU process. Native model weights, operators, HCCL runtime and
performance still require their separate acceptance evidence.

An independent source review found one outer-executor boundary after the owner
itself had already been hardened: an unconditional exception handler could call
`fail()` after a successful COMPLETE and mask the sink's original exception.
The executor now fails only its exact live owner. The independent
[020 CPU follow-up](a3-pipeline-cpu-20261110/diagnostic-boundaries-020/results/summary.json)
passed 247 cases in six affected modules, including an actual ticket COMPLETE
with a failing returned sink, U1/U2 configuration sink failures before native
producers, and both Graph/diagnostic binding orders. Its ten upstream source
pins match and it starts no device probe. 018b and 020 retain their own source
manifests and original results; their counts are not combined.

Separate actual A3
[TP1 and TP2 ticket-owner diagnostic sessions](a3-eager-pipeline-20261110/README.md)
subsequently passed with six configurations and 146 scalar records per rank,
exact transaction/rank/stream alias data, an empty-parent configuration and
inactive U2 compute synchronization. The original payload/STOP/native-TP checks
remained successful and all eight boards were idle before and after each run.
These sessions validate the real owner/binding host observations on NPU with
controlled arithmetic; they do not execute the native A/F model consumer or
qualify device overlap, sink-fault recovery or performance.

Graph-protocol eager fallback now uses the same explicit rank binding. The
pre-KV bridge accepts optional immutable settings and sink; native target,
dummy, profile and empty parents emit configuration only after their real
accepted plan. The actual execution is serial on the parent stream, even when
the logical U2 settings request side compute. Dummy/profile geometry remains
the native U1 geometry. UBatch receives the binding only for the execution
scope; its prior binding, native preparation and forward context are restored
on normal return and sink failure. Coordinator eager fallback, empty parents
and capture-eligibility preflight use that same binding. A bridge that already
requested settings requires the coordinator to retain them; an old bridge
with no requested settings permits later diagnostics. FULL replay does not
enter this Python diagnostic path.

The independent
[021 CPU attempt](a3-pipeline-cpu-20261110/graph-fallback-diagnostics-021/results/summary.json)
passed all 49 pre-KV cases, 81 coordinator cases and 74 UBatch cases. Its full
four-module result remains **212 passed, 2 failed, 26 setup errors**, zero
skips: the Execution mirror used two retired profiler fixture names following
the concurrent H14 API rename. Those fixture failures are retained for the
H14 owner's independent complete-module closure. The
[source review](stage-diagnostics-graph-fallback-review-20261110.md) records
settings ownership, accepted-parent ordering, output/auxiliary and producer
preservation, sink-failure restoration and the unchanged captured replay
boundary. All ten installed upstream pins match, sources match the immutable
snapshot, and this attempt starts no NPU process. No earlier CPU or device
result is relabeled to accept these new bytes.
