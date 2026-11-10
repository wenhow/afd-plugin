# Independent review: actual eager ticket/loan component probe

Read-only review by the profiler/C7 implementation agent, separately from the
eager owner/probe author. No SSH, device job, or production admission change was
performed for this review.

Reviewed exact SHA-256:

| Path | SHA-256 |
| --- | --- |
| `tests/npu/hccl_eager_pipeline_probe.py` | `d45ab50581e863d94372435cd0ed3265ebb38533c640e376aa60dbb4d5b747d7` |
| `afd_plugin/connectors/npu/hccl_eager_pipeline.py` | `b758b9e8cb398cebdf91c8bd36087b060e2d10e7cbfc6636ff0ba385e0cb6e31` |
| `afd_plugin/connectors/npu/hccl_staged_transport.py` | `ae4c38de38cb90d5e511fe0c7d0dbd6fda9fc318292f2d10784d33b3c8d422f1` |

No blocking issue was found for a separately authorized, bounded paired TP1
component experiment. This is not device acceptance and does not authorize the
experiment by itself.

The probe uses the actual transport warmup, accepted local-count plans, public
owner posts and loans, finish, STOP, and close. It does not use a mock Work or
the earlier primitive gate's raw payload operations. Six consecutive U2 parents
cover changing padded extents, positive padded rows with zero valid tokens, one
empty stage, and an entirely empty parent. Positive stages run two layers.
Attention posts its positive stages before the output loans; FFN retains its
existing stage receive/consume/send ordering. Borrowed output is cloned while
its actual compute-stream loan is open before feeding the next layer. Separate
FP32 deltas, raw IDs, and full padded outputs are checked against a CPU
recurrence. Readbacks run on the actual consumer stream, so these correctness
checks deliberately serialize portions of the workload and prove no overlap or
performance benefit.

Owner methods retain payloads/Work/events and attach allocator `record_stream`
uses. The reuse frontier waits for the prior consumer event and all prior Works
on the parent stream, then places an event dependency on send/receive streams
before reusing storage. Host Work return is not labeled device completion.
Failure retains the owner and its storage for supervision. The final native TP
all-reduce occurs after normal plugin STOP and group close, checking group
isolation for this workload rather than native MoE equivalence.

The supervisor observes child PIDs and Linux start ticks, enforces a finite
deadline, and reaps only its owned live processes through the previously
reviewed supervision helpers. Workers do not pretend to abort broken HCCL
communications. A failure/timeout is a failed bounded component experiment,
not proof of in-process recovery or a regression of the default serial path.

Before execution, the outer immutable harness must preserve these exact bytes,
the imported supervision/helper bytes, installed source pins/package versions,
device allocation, and before/after NPU inventory. Warmup here uses matched
blocking send/recv; first use of the actual asynchronous owner remains part of
the new experiment. Earlier warmed primitive-gate results do not replace it.
TP2 requires a distinct run and explicit four-device allocation after TP1 has
drained. No V4.1 model, HC, Engram, quantized MoE, scheduler, actual speculative
sampling, independent communication stream performance, or full-model outcome
is covered by this probe.

## Separately executed TP1 evidence

After the parent authorized the architect's run, the reviewer independently read
`a3-eager-pipeline-20261110/owner-tp1-001/identity.json`, `results/summary.json`,
and `results/probe/result.json`. The immutable manifest digest is
`ccf53188e398340b70e4975fc2b856e1ddce8f0a06056438f12b2199d591ea54`;
the owner, transport, and probe entries exactly match the three hashes reviewed
above. All 33 installed-source expectations matched, two rank exits were zero,
the run timed out neither internally nor externally, no owned processes were
killed, and all eight devices were idle before and after.

Each rank completed six target U2 parents and 16 positive layer/stage checks,
including the padding-only and empty cases. BF16/FP32/ID/full-output checks
passed; both ranks recorded normal STOP and a native TP collective result of
`[1.0]` before and after plugin group close. This qualifies this actual warmed
paired TP1 ticket/loan recurrence component. It does not extend to TP2, the
native delayed HC/Engram/model consumer, failure recovery, allocator pressure,
or overlap/performance. The earlier review remains source-only; this paragraph
is the distinct actual component result.

## Separately executed TP2 evidence

The reviewer independently read the separately authorized
`a3-eager-pipeline-20261110/owner-tp2-001/identity.json`,
`results/summary.json` and `results/probe/result.json`. Its immutable manifest
digest is `4c58361ff41090bc58e5d87e1f69f124753be97c703a5c36102f540157ce27c7`.
Owner, transport and probe match the hashes reviewed above; the outer harness is
`5f549fc23396ba07b6481c2e3f1c2490e7ee39d2e7c8f109384e0dbbdf6f1ea8`.
All 33 installed-source pins matched. The four rank exits were zero, no timeout
or owned-process kill occurred, and all eight devices were idle before and after.

Each A/F rank completed six target U2 parents and 16 positive layer/stage checks,
with every recorded BF16, independent FP32, raw-ID and full padded-output check
true. The empty and padding-only parents retired normally. All ranks recorded
matched stage-group warmup, normal STOP, and native TP collective results of
`[2.0]` both before execution and after plugin group close. This extends the
actual ticket/loan component evidence to paired TP2. It does not qualify native
MoE, delayed HC/Engram/model execution, failure recovery, allocator pressure,
timeline overlap or performance; the oracle readbacks intentionally serialize
the controlled arithmetic workload.
