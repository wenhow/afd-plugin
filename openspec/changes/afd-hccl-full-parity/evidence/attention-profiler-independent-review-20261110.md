# Attention profiler execution and shutdown ownership review

The reviewed source bytes and exact locked signatures are recorded in
[the source review artifact](attention-profiler-independent-review-20261110.json).
This review found no blocking issue in the frozen increment. It is source
evidence; actual-dependency CPU and device results remain separate.

The affected upstream owners are `NPUWorker.profile`, `NPUWorker.shutdown`
and `NPUModelRunner.execute_model` at Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`. The three signatures, including
return annotations, match exactly. The worker overrides document their reason,
behavior, delegation exception and removal condition, and mark their AFD
branches. Delegating native profiling and cleanup preserves native KV,
weight-transfer and profiler ownership without duplicating those bodies.

Attention creates a deferred session with its native role rank. The narrow
execute context surrounds the existing native/AFD branches and applies requests
on their execution thread. Successful calls advance its execute-step counter;
legal `None` and KV-only returns also count. This counter does not establish
that a model parent ran, and it differs from the F daemon's parent counter.
Execution exceptions remain the original exceptions, including when profiler
export fails. A failed backend remains retained for process supervision.
Constructor/load operations no longer belong to this serving trace.

The original native worker releases KV and weight-transfer resources before
calling runner shutdown. The new role-env branch therefore first invokes the
runner's idempotent preparation: verify the profiler execution thread, finish
peer STOP, close its retained Graph cache, then finish/export the profiler.
Only successful preparation permits delegated native KV/model/group release.
Foreign-thread, STOP, cache or export failures escape before that release.
The runner's later native shutdown sees successful preparation and does not
send a second STOP. Disabled role-env profiling preserves the native worker
order; partially constructed workers still enter native shutdown.

No model arithmetic, router precision, Engram history, KV layout or TP reduction
changes in this increment. Enabled profiling adds short request/status locks
and profiler calls at the existing host boundaries; no lock spans device work
or export. Disabled profiling retains a single optional context and its legacy
step behavior. Trace size, export cost and useful device timing need actual
profiling evidence. The role-env and native profiler configurations are
explicitly exclusive. Collective RPC and status support do not register a new
HTTP route.

Remove these worker shims when the native worker exposes a pre-release callback
and a runner-owned profiler control/status interface. Until then, the plugin
owns this lifecycle seam without modifying either pinned upstream checkout.
