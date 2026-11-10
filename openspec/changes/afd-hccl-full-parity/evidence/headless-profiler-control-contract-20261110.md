# Headless F local profiler control: source and CPU contract

This increment supplies the external control entry missing from a headless F
EngineCore. It does not construct another engine or change the HCCL protocol.
`AFD_NPU_FFN_PROFILER_CONTROL_PATH` explicitly enables one local Unix socket.
Absent that variable, the existing core creates no listener or management
thread. Configuration validation requires an NPU HCCL F role with the F
role-env profiler enabled and no conflicting native profiler configuration.

The existing EngineCore management thread binds before `start_worker_loop`
and polls the listener after its native input-queue processing. It accepts
only newline-delimited status/start/stop commands and forwards the fixed
`afd_profiler_status` or `profile` RPC to the core's actual executor. The
execution-owner thread applies desired state at the existing safe boundary.
A successful start/stop reply acknowledges the request; it does not assert
that a blocked receive has returned, the owner has applied the request, or
that an NPU trace contains useful events.

Client IO is nonblocking, with four connections, a 4096-byte request limit,
a 65536-byte response limit and a five-second incomplete-IO deadline. Each
poll may issue at most one synchronous executor RPC, whose separate timeout
is thirty seconds. These limits do not establish management latency under
model load. Partial reads/writes retain their state for later polls. Invalid
JSON, unsupported operations, RPC failures and invalid or oversized status
responses return bounded errors. No arbitrary executor method is exposed.

The socket has mode 0600. Its private parent directory must already exist.
Binding refuses any existing path, including a dangling symlink. Cleanup
unlinks only the socket inode bound by this listener; a replacement file is
preserved. Core exit closes local IO before its existing worker STOP policy.
The implementation adds no profiler backend, device calls, thread, mutable
global registry or upstream signature change. The integration changes the
plugin-owned F busy-loop helper, whose reason/removal comments are explicit.

The native contracts were read at vLLM commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`: EngineCore input-queue processing
and Executor collective RPC signature/timeout. The
[source identity](headless-profiler-source-identity-20261110.json) records
these and fifteen inherited source pins. No upstream checkout was modified.

The immutable A3 CPU attempt
[headless-profiler-027](a3-profiler-cpu-20261110/headless-profiler-027/results/summary.json)
passed 91 cases with zero failures/errors/skips: local control 31, EngineCore
29 and profiler lifecycle 31. All seventeen installed source pins matched.
Real Unix socket tests cover strict framing, fragmented writes, bounded
connections, IO expiry, RPC errors, nonserializable status, fixed commands,
owner-pending state, existing-endpoint protection and inode cleanup. Core
cases send commands during the public management-queue loop and observe
normal STOP and service-error shutdown. CPU executor/profiler seams remain
explicit fakes; no NPU probe or model started, package changed or upstream
file changed in that attempt.

Local Windows verification separately passed 65 cases and skipped 26 because
the host lacks AF_UNIX. Skips are not Unix socket acceptance. The A3 result
does not qualify full-model management reachability, real device traces,
pressure, A5 or performance. Those remain in the consolidated
[A5 operating guide](../../../../docs/npu/DSV41_A5_PROFILER_VALIDATION_ZH.md).

Exact source hashes, raw artifact hashes, JUnit counts and explicit delivery
scope are recorded in the root review and delivery manifest alongside this
file. Older attempts in the CPU index remain immutable and are not summed
into this result.
