# Eager endpoint gate preparation

Both initial authorized NPU gate sessions passed. The installed-source
collection and preparation remain distinct from these actual device results.
This verifies the warmed existing endpoint primitive, not eager/U2 scheduling.

| Device evidence | Result | Manifest SHA256 |
| --- | --- | --- |
| `isend-001/results/summary.json` | PASS; 43.99 seconds, eight idle boards before/after | `46c69cf22e0d81da16fa3aa98ee11b60bbc38c25111561e707eef4ced00ed113` |
| `irecv-001/results/summary.json` | PASS; 44.47 seconds, eight idle boards before/after | `56293f939f2fa27479cd4cd8f4281a308d0c5f987f4a065e09d0b10cd7aaa8f8` |

Both sessions matched all 33 installed-source pins and executed probe
`86ddf599` / harness `4a3df039` at HEAD `8a6eeb2a`. For BF16 `[8,16]`, original
FP32 `[8,16]` and int32 IDs `[8]`, actual post and Work.wait returned before
the peer received matching permission. The consumer event remained pending at
the separate query after a five-second held window. The real receiver obtained
the original exact payload, while the sender's actual allocation was safely
modified after its Work dependency; consumers and final events were exact.
Both workers acknowledged STOP, closed plugin groups and retained native TP
all-reduce `[1.0]` before/after close. No timeout, kill or failed drain occurred.

Full gate/event/worker records are in each `results/probe/result.json`;
`validation-summary.json` indexes raw file hashes and exact session identities.
The complete archive includes contemporaneous Graph work-in-progress source
only as working-tree identity; no Graph body participates in this endpoint gate.

The scope is paired TP1 after matched warmup on the same group. It does not
prove first-use host progress, multiple in-flight stages, TP2, larger-message
pressure, allocator stress, native model execution, HCCL abort or performance.

The reviewed design is [eager-backend-gate-design-20261110.md](../eager-backend-gate-design-20261110.md).
It uses the existing transport's actual HCCL data group with individual
torchdist isend/irecv. Independent CPU permission withholds the matching peer,
then separately observes host post, Work.wait and a late consumer event. A
send-source reuse and exact receiver payload check protect the storage seam.

Preparation history is preserved:

| Raw evidence | Meaning |
| --- | --- |
| `archive-precheck-001/identity.json`, `static-check-001.json` | Initial candidate; source review found instantaneous-event and send-reuse gaps. Never run on NPU. |
| `archive-precheck-002/identity.json` | Intermediate corrected candidate, before final diagnostic phase label. Never run on NPU. |
| `archive-precheck-003/identity.json`, `static-check-002.json` | Reviewed final probe `86ddf599` / harness `4a3df039`. 136 unique archive members, 135 manifest entries, 33 source pins; Ruff/format/3.11 AST/CLI checks pass. No SSH/device launch. |

The final preparation manifest is
`f1c58339e600b90a58b8ff3b7b3ad0dbbda80cf0fd98736630cf90b386bdba62`.
The independent source reviewer found no remaining blocker to a bounded
component experiment. The parent subsequently authorized the two serial device
sessions above. First raw results remain immutable. A future clean capability
miss must stay distinct from runtime/drain failure and the accepted C4
synchronous path. `preflight-001` preserves npu-smi's missing driver library
failure; `preflight-002` uses the established CANN/driver/ATB paths and confirms
33 matching sources, no torch import and eight idle boards before execution.
