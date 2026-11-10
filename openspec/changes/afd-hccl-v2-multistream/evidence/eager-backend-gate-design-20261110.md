# Existing eager endpoint capability gate

Status: implementation prepared and locally checked; **no NPU gate session has
run**. This is the prerequisite experiment for the eager ticket proposal in
`eager-u2-inflight-contract-20261110.md`, not implementation or acceptance of
C6 eager/U2 scheduling. No production code, packages or upstream files change.

## Endpoint and source boundary

`HcclStagedTransport.initialize()` calls `init_afd_process_group(backend="hccl")`
for its public `data_groups`. Its current eager input/output methods use
`torch.distributed.send/recv` on those groups. The locked donor likewise uses
`dist.send/recv` in its eager `_send_tensor`/`_recv_tensor` (1730–1778 at
`4ddd69a391dfb5b3e362c7601399e538a31de702`). A side stream in that donor establishes
intent; it does not prove the installed backend returns to the host with an
unmatched peer operation pending.

The locked Ascend `PyHcclCommunicator` at
`7e37e29a7803d6897a7113088e2b771f83205bc0` exposes `all_reduce` and `broadcast`,
without P2P methods. This gate does not substitute that communicator, Graph HCOM
ops, another backend, another default world or newly invented data groups.
It calls `dist.isend/irecv` on the real transport's **`data_groups[0]`**.
`eager-endpoint-source-audit-20261110.json` retains the exact original Git-blob
hashes, method names, line bounds and relevant source excerpts for this chain.

The read-only collector `collect_eager_endpoint_source.py` obtained installed
declarations without importing torch or torch-npu. Its raw result is
`eager-endpoint-source-20261110.json`, with original-file SHA256 values:

| Installed source | SHA256 | Contract observed in source |
| --- | --- | --- |
| `torch/distributed/distributed_c10d.py` | `31155344233ae77360a630bbeacf1b43cfa18f1923cf2103cd5b3142db39ec5c` | `isend` 2449–2491 returns `group.send`; `irecv` 2494–2536 returns `group.recv`; synchronous send/recv subsequently invoke `Work.wait()`. |
| `torch/_C/_distributed_c10d.pyi` | `b3f981c9c6979517d3f98cadfeca1339438003c90360ff3ba72f6ed63fe97101` | Work declares `wait(timeout: timedelta) -> bool`, `block_current_stream`, `synchronize`. A declaration is not the HCCL implementation or progress evidence. |
| `torch_npu/distributed/distributed_c10d.py` | `6afceb0425ee4229e1bdab21dd1c3965f7e4a599c8f495a42cba70fad997fec7` | Installed NPU batch helper is recorded; this probe uses individual installed P2P calls. |
| `torch_npu/distributed/__init__.py` | `19b6221f90249b150390b64724454d69dff995ff4c98dc3f4dcd552d8d489f89` | Registration file pinned alongside the P2P source. |

Native default TP is world-size one per role; the separate AFD group has F rank
zero and A rank one. Calls use installed `group_dst/group_src`, avoiding default
world rank translation. The runtime report records the real group backend/rank,
default native rank, function signature/file SHA and returned Work type/doc.

## Minimal execution and CPU proof

`tests/npu/hccl_eager_backend_probe.py` requires explicit
`--operation isend|irecv`. Attention is the selected poster; FFN is its matching
peer. Device indices default to FFN=0, Attention=2. Only model configuration is
read through the existing `hccl_staged_probe.build_config`; no weights, native
model body, MoE, prefill/decode, Graph, accepted parent or transport cursor is
created or modified.

Both sides initialize the normal independent native TP group and the existing
transport. A matched operation for each payload type warms the **same data
group and both explicit side streams** before the gate. Warmup checks exact
payload and consumer values. Native device synchronization separates warmup
and newly allocated payloads from the measured host call.

Each payload then gets this handshake, using worker stdin/stdout pipes and the
CPU supervisor, independently of HCCL/Gloo data readiness:

1. Both roles announce `gate-ready`. The matcher waits for `match:<kind>` and
   cannot enter its data op before receiving that command.
2. Supervisor grants `post:<kind>` only to Attention. `post-enter` precedes the
   actual installed `isend` or `irecv` on a real NPU posting stream.
3. On `post-return`, supervisor records whether matcher permission is still
   withheld, then grants `wait:<kind>` to Attention. `Work.wait()` executes in
   an actual consumer stream under the production NPU/TLS stream scope.
4. If wait returns while the matcher remains held, Attention enqueues tensor
   arithmetic followed by an Event on that consumer stream. The first query is
   diagnostic only. Supervisor keeps MATCH withheld for a complete observation
   window, then grants a separate `query:<kind>` CPU command. Attention reports
   this late query before supervisor grants MATCH. A fleeting pending state
   immediately after a kernel enqueue is not dependency evidence.
5. If post or wait does not return within the five-second observation bound,
   supervisor grants MATCH to permit real work to drain. That gate remains
   unproven/failed even if subsequent cleanup succeeds. No abort is inferred.
6. On the sending side the consumer **reuses the actual send allocation with
   an in-place addition**, ordered after Work.wait on its consumer stream. The
   receiver must still obtain the original reference; sender storage must
   contain the new value. On the receiving side the consumer reads the received
   tensor into a separate output. Both results are checked exactly against CPU
   references. Buffers/Work/events remain retained until actual consumption
   completes. This creates a storage-lifetime observable in both directions.

Payloads are BF16 `[8,16]`, **independent original FP32** `[8,16]` with values
BF16 cannot represent, and int32 IDs `[8]`. FP32 is not BF16 converted back.
The consumer arithmetic after Work.wait detects an early read of sentinel
receive storage or an early overwrite of send storage. A post return alone
says nothing about data completion; an unrecorded or unrelated event is never
used as receive readiness.

This tests a Work.wait-based cross-stream handoff. It does not yet establish
`Work.block_current_stream()` behavior, progress with multiple in-flight stages,
large-payload backpressure, TP2 posting, allocator reuse under pressure, native
continuations or performance. A five-second bound classifies this observation;
it is not proof that a slower call can never return asynchronously.

## Classification, retention and shutdown

`result.json` keeps per-payload observations separately:

- `post_returned_while_matcher_held`: actual host posting returned before the
  peer received permission to enter its matching op.
- `wait_returned_while_matcher_held`: actual Work.wait returned under the same
  withheld-permission condition.
- `late_query_while_matcher_held`: the separate query ran after a complete CPU
  observation window with matching submission still forbidden.
- `late_consumer_event_pending`: observed pending state at that late query.
  `observed_inflight_at_late_query` combines held permission and that state.
- Exact receiver payload, send-source reuse, consumer result and completed
  event are reported by each real worker after matching/drain.

Capability pass requires host post/wait progress and the late held query for
all three payloads, exact receiver/send-reuse/consumer results and normal close.
For **irecv**, that late event must remain pending before the peer is permitted
to send. For **isend**, an already completed late event can be legitimate small
message buffering; pending remains an observation, not a mandatory condition.
The source reuse and exact original receiver payload provide the relevant
lifetime check without misclassifying lawful early send completion as backend
failure. A cleanly drained capability miss returns code 2;
runtime/setup/cleanup failure returns code 1; capability pass returns code 0.
These statuses must not be called a regression of the already accepted C4
synchronous transport. No accepted parent is fabricated for this raw primitive
gate. Once all real work drains, Attention sends the normal idle STOP, FFN
receives/acknowledges it, plugin groups close, and each native TP group executes
its own all-reduce successfully before native cleanup.

The inner supervisor has a 240-second total bound. It records descendants of
the launched workers and their `/proc` start identities; only these identities
can receive a final SIGKILL if genuine matching/drain cannot finish. The outer
harness retains the established 280-second ownership-checked supervision. No
foreign process, name-based kill, production CANCEL or HCCL abort is introduced.
First raw results are immutable, including a capability miss or failed drain.

## Harness and proposed commands

`run_a3_eager_backend.py` reuses the existing environment preparation and
ownership helpers, preserving CANN/ATB paths and deployed FastAPI 0.123.10.
The immutable archive contains the complete plugin source, probe/helpers,
`source-expected.json`, installed API raw JSON and a manifest with no duplicate
members. It retains the previous 31 installed-source checks and adds the two
previously absent Work/registration sources, for **33** source checks. Broader
source checks preserve environment identity; they do not enlarge execution
scope to those modules. The stdlib version preflight explicitly proves no
torch import before the eight-board idle check. Both sessions sample eight
boards before/after and record own PIDs, all raw worker logs and commands.

Local preparation completed without SSH/device execution:

```powershell
.venv/Scripts/python.exe openspec/changes/afd-hccl-v2-multistream/evidence/run_a3_eager_backend.py --operation isend --label archive-precheck-001 --prepare-only
```

`a3-eager-backend-20261110/archive-precheck-001/identity.json` records 136 unique
archive members and 135 manifest entries with all archived bytes matching.
Its source manifest SHA is
`965b6c070d3429fe58376082a0d8aa496f629ccf0edd7f7449d7d88b605cdfe5`.
Local Ruff, Python 3.11 AST and both CLI help checks pass. The preparation
snapshot is not device evidence and will not be overwritten.

Independent source review rejected that first candidate's instantaneous-event
classification and lack of send-source reuse. The revised probe adds the late
query/real reuse above. The original preparation/static evidence remains as
history; a new final preparation identity will bind the reviewed revision.

The independent reviewer rechecked corrected probe `86ddf599` and harness
`4a3df039`: no remaining source finding blocks the bounded component gate.
The final local preparation is
`a3-eager-backend-20261110/archive-precheck-003/identity.json`, with manifest
`f1c58339e600b90a58b8ff3b7b3ad0dbbda80cf0fd98736630cf90b386bdba62`:
136 unique members, 135 manifest entries, 33 installed-source pins, no SSH or
device execution. `static-check-002.json` contains the corrected probe's
successful Ruff check/format, Python 3.11 AST and CLI checks. Prechecks 001/002
and static 001 remain unmodified history, not acceptance of their old bytes.

After parent/independent source review and explicit device authorization,
proposed serial commands are:

```powershell
.venv/Scripts/python.exe openspec/changes/afd-hccl-v2-multistream/evidence/run_a3_eager_backend.py --operation isend --label isend-001
.venv/Scripts/python.exe openspec/changes/afd-hccl-v2-multistream/evidence/run_a3_eager_backend.py --operation irecv --label irecv-001
```

No gate device command has been executed at this document's creation.

## Candidate file identity

| File | SHA256 |
| --- | --- |
| `tests/npu/hccl_eager_backend_probe.py` | `86ddf59910776a8b50b3dc76b5e56484f4d207e9eff9e8a2d375f271fdf91b34` |
| `run_a3_eager_backend.py` | `4a3df039d861b3d6106bdf6fed8e716277972966661ace5b29703bbab9de420a` |
| `collect_eager_endpoint_source.py` | `7621e53c0b205410acd14a3e42f2d01f2c95b4e291f9c21231833f4ac399e673` |
| Installed API raw JSON | `d4d8a6937dc1f41971c969183924e22aaa5f954be3682bf0650d1011dab41573` |
| Existing production `hccl_staged_transport.py` | `9ae40a90dab184fc4f1d17f934520318c2bd50eea75b5d082ed284e9a4e79128` |
| Existing production `afd_process_group.py` | `eab623c56c995f4cd4f00ba89c14e8448d83dbf32b08565eda7fe0973e97c1c5` |
