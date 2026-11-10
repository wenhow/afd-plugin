# Independent Graph bulk-v1 source review, 2026-11-10

This is an independent read-only review of the seven production paths and
their existing mirrored tests/shared fakes in
[`graph-bulk-v1-local-validation-20261110.json`](graph-bulk-v1-local-validation-20261110.json).
All 20 current file hashes matched that frozen manifest when checked. Its base
HEAD is `c279c4f6f95445b9e764d2487829af39b3a757a3`. The reviewer changed no
runtime/test files and launched no device task.

No blocking source finding was identified in this controlled component scope.
The conclusions below concern host construction, static tensor/wire contracts
and resource ownership. They do not demonstrate actual multi-group HCCL Graph
enqueue/replay or overlap.

## Negotiation and controlled eligibility

`HcclPlanCapabilities.graph_exchange_order` defaults to `stage_serial`, and the
default field is omitted from encoded HELLO bytes. Decode accepts only the two
defined names, and negotiation requires exact equality; a peer cannot silently
downgrade one half of a bulk session. `bulk_v1` requires graph execution v1,
counts v1, max stages two and paired native TP1/TP2. Existing generic fanout
topologies do not enter it.

The bank projects a U1 key to `stage_serial` even in a bulk-capable session.
Existing serial exchanges remain send-three/receive-one per positive stage.
The bulk U2 A/F model bodies require an actual compute-pipeline owner before
producer or payload work; there is no serial-body use of the new order.

The A coordinator and F service freeze the negotiated capabilities and reject
later capability drift. They check bulk compute eligibility at whole capture
sequence preflight, including when the first captured descriptor is U1. If
either side lacks U2 compute support, it reports unsupported before any native
factory callback; the common control protocol then retires an honest empty
eager preflight. It does not enter a partial bulk warmup and later fall back.
Normal eager fallback continues through the existing eager executor and cursor;
it does not invoke the Graph bulk bank.

The graph key does not add an exchange-order field. That is sufficient only
because caches, managers and bank bundles remain bound to this one transport
session and the newly frozen capability equality. Cross-session reuse is not
authorized by identical geometry/digests.

## Matched wire and event frontiers

For each positive stage `S`, input packets retain the exact order BF16 hidden,
original independent FP32 hidden, raw int32 IDs. A bulk layer constructs:

```text
A parent: send inputs S0; send inputs S1; receive output S0; receive output S1
F parent: recv inputs S0; recv inputs S1; send output S0; send output S1
F compute: wait recv S0 -> native MoE S0; wait recv S1 -> native MoE S1
```

F enqueues S0 side computation between its S0 and S1 receives, while deferring
the parent output send until all positive stage receives/computes have been
enqueued. A sends every positive stage's packet before receiving its first
output. Thus the old `A sends S0 -> waits output S0` / `F waits input S1 ->
sends output S0` cycle is not introduced. Packets still use their negotiated
per-stage data groups and actual communication stays on the parent stream.

The pipeline rejects an early A receive, out-of-order A send, serial exchange
in a bulk bank, and F output send before all same-layer receives/computes. Its
wire and compute frontiers remain layer-major; zero-row stages are omitted on
both sides. Neither the tensor body nor these frontier methods sends host
JSON, advances the eager cursor, performs `.item()` or emits COMPLETE.

A hybrid mode keeps each next-layer stage compute dependent on that stage's
prior output receive; non-hybrid uses the existing parent all-communication
fence. Continuation completion precedes that stage's next-layer preparation
and bank reuse. F retains the existing preceding same-stage send dependency,
the selected non-cross-layer join and its final send join. The new static host
order does not remove the event dependencies or change native TP collectives.

## Tensor and failure lifetime

Bulk A defers wire until both native preparations are enqueued. It now retains
the produced BF16/FP32 tensors together with the complete native HC/pre-mix
continuation and stage tensors, then records parent-stream consumption before
the bank copy/send. The returned output remains a borrowed bank view and is
consumed by continuation before that stage's next output receive overwrites it.

Bulk F retains every output until both receives/side computes have been
enqueued, then copies/masks/sends through the same native-output contract. The
bank still owns one fixed set of payload buffers per stage, reused across
layers; this change adds no layers-times-bank allocation. `record_stream`
describes actual consumers, and the pipeline holds one current frontier per
stage. On failure its sticky state retains those references for supervision.
The added F weak-reference test checks that S0 output survives an S1 compute
exception before any output send. This is a CPU lifetime assertion, not a
device allocator or network-abort demonstration.

Native complete MoE, its TP reduction, independent router FP32, raw-ID masking,
parent Engram production/events, auxiliary timing and HC continuation stay with
their previous owners. No output is rebuilt from BF16 to manufacture FP32.

## Reviewed production hashes

| Path | SHA-256 |
| --- | --- |
| `afd_plugin/distributed/hccl_plan.py` | `68fbfa4b5ce38c8dfdd6ad3cebe8dc797f71a7b634d68b876375eaa56bdaec7d` |
| `afd_plugin/connectors/npu/hccl_graph_stage.py` | `4504702c1c8c36083c280e71d2628941ae50ae08373b49af0d049f096c750498` |
| `afd_plugin/connectors/npu/hccl_graph_pipeline.py` | `af043a551fdb40fbdb665bd78112b9100a6c6b84ba8c831fee8939d858149405` |
| `afd_plugin/model_executor/models/npu/deepseek_v41_graph.py` | `09065402433f51cef2378ad984b29f2d0cec2b48f4961fb7bbca4315ee8f9347` |
| `afd_plugin/v1/worker/npu/hccl_v41_ffn_graph.py` | `c28ecf032bd1614feb3d776b263dfbcab16bf801f9bc8f7addb822de946c8dc3` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph_coordinator.py` | `043408c12148438586cb6cd5b48b5a28cf9dfccb97752c9ca537ee335fbcaf82` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph_service.py` | `7329b52b39fb6a4b9ba1d6efc2f66b6fc938fa2d99e42b0bacd01e1a6b6e2ec2` |

## Validation and cost boundary

The reviewer read the mirrored tests for packet matching, early/misordered
frontier rejection, zero-row stages, layer joins, numerical CPU outputs,
original FP32/IDs, native continuation/Engram/aux behavior, capability drift,
whole-session preflight support and output retention on failure. Their fakes
observe public payloads and stream/event host traces; those traces do not prove
device execution. The implementation manifest reports local Ruff, Python 3.11
AST and pure codec checks; the reviewer did not rerun those tests.

Actual-dependency CPU and a new bounded NPU multi-stage-group capture/replay
check were pending when this source review was written. Prior `stage_serial`
NPU evidence cannot qualify the new order. Full V4.1 model/MoE/Engram Graph
acceptance, delayed-device/error behavior, timing and overlap performance also
remain separate. No production feature guard or automatic factory choice is
opened here.

This changes fixed host wire construction and temporarily retains at most one
layer's current outputs/continuations per positive stage. It adds one optional
HELLO field and extra bounded frontier checks; it does not add new communicator
groups or per-layer payload banks. Graph replay executes the recorded tensor
DAG without Python diagnostics. Remove this plugin scheduling adapter when
native Graph stage hooks provide the same negotiated matched order and
resource/event ownership.

## Follow-up: actual CPU and bounded bulk artifacts

The reviewer subsequently read the raw XML, identities and source checks in
`a3-graph-bulk-20261110/graph-bulk-019a` and `graph-bulk-019b`. The first snapshot
ran seven modules: **552 passed, one failed**, no skips/errors. Its failing
Graph-service capacity case changed session capacity while constructing an
unrelated invalid model/session binding; the binding guard correctly rejected
that earlier boundary. The subsequent snapshot changed the shared fixture,
kept all seven production hashes above unchanged, and ran the Graph-service
module: **64 passed**, no skips/errors. This is a directed closure, not a
repeated all-green 553-test run. Neither CPU artifact starts an NPU probe.

The reviewer also inspected the complete bulk delta in
`tests/npu/hccl_graph_manager_probe.py`
(`2df95803f6283ce8b38ddd8bb4540bff2c5529cc6f54ae697c210319e78e0b5e`)
and `afd-hccl-v2-graph/evidence/a3-native-manager-20261109/run_a3_graph_manager.py`
(`7f2ca8487d7ae2bdc0271deb7af0ad4859a16a5e011ac90eb7893fb7a8743006`).
The default remains `stage_serial`. Explicit `bulk_v1` is passed to both roles,
negotiated in HELLO, and rejected without a stage bank and enabled compute
pipeline. A enqueues every positive stage input before receiving outputs; F
receives every input and enqueues its compute before sending retained outputs.
The original recurrence, independent FP32/IDs oracle, dynamic valid counts,
buffer-address checks, STOP, owned-process supervision and 31 installed source
pins remain present. No additional static blocker was identified.
`bulk-prepare-001` is only an archive check (139 members, 138 manifest entries).

After authorization by the parent, the implementer ran `bulk-tp1-001`. The
reviewer read its raw identities and summaries: manifest
`7c5007bcc046f0266f27eff70f65da476a42d7e6e5aca11e454644a5fad965a2`,
source files matched, 31 installed sources checked, both process exits zero,
`status=passed`, eight boards idle before and after, no timeout or owned-PID
termination. This qualifies that bounded native-manager/NPUGraph/HCCL
arithmetic component with `bulk_v1`; it does not qualify native V4.1 model,
MoE, Engram, GE, arbitrary topology, timing, or overlap performance. Actual
parent-stream identity is checked by the probe; no hardware timeline or
numeric stream-handle proof is claimed. No additional device run was launched
by this reviewer.
# Stage-owned output follow-up

Root additionally reviewed the generic native-output lifetime boundary. The
locked checkpoint's ordinary shared-expert path produces a fresh result, but
the plugin's native-call interface does not guarantee ownership for every
quantization backend. Retaining a tensor reference alone cannot prevent a
subsequent native call from rewriting borrowed workspace.

The follow-up introduces `HcclGraphStageBank.prepare_output`, preserving the
original exact shape/dtype/device/contiguity checks. Only deferred bulk compute
copies native output into its existing stage bank before recording compute
completion. Later parent masking/send consumes that bank; the serial path
retains its original one-copy behavior. Same-stage prior-send, receive-done
and compute-done dependencies protect reuse without another buffer, reduction
or control packet. No blocking finding remains in this narrow source change.

The independent `graph-owned-output-024` CPU run passed both complete affected
mirrors: bank 41 and F Graph 73, 114 total, with no failures, errors or skips.
It includes a native callable that returns the same writable workspace across
successive stages and verifies final public output, plus invalid native output
rejection. Historical bulk device results do not qualify this new copy location;
a new bounded capture/replay attempt is recorded separately.
