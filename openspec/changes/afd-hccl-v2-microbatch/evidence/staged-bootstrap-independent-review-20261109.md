# Staged bootstrap and count protocol independent review, 2026-11-09

The reviewer independently read the root agent's connector, shared policy,
validation route, FFN bootstrap and count-protocol implementation. The reviewer
did not author or edit those runtime files. The separate Attention execution
increment was authored by this reviewer and has its own contract record and
independent peer review; it is not presented here as independently reviewed.

Two concrete ownership/error findings were reported and fixed before this
review was accepted:

1. A reply may already be consumed and its transport cursor advanced when
   copying into the caller's output buffer fails. `recv_ffn_output` now marks
   the endpoint terminal and rethrows that post-consumption failure. Layout
   rejection still occurs before receiving and permits a corrected buffer.
2. A public binder could previously accept a different, equally capable
   transport from the one held by its staged adapter. Service and close
   ownership could then split across two sessions. The FFN binder now requires
   identity with `connector.transport` for the new adapter. The corresponding
   Attention fix and negative test belong to the author contract record.

Both repairs were read in the final source. No unresolved static blocker was
found in this limited synchronous eager bootstrap. This is not full native
model, checkpoint, Graph, overlap or production-admission acceptance.

## Normal route and compatibility

`HcclStagedAFDConnector` implements `AFDConnectorBase` directly and composes a
real `HcclStagedTransport`. It does not inherit from or impersonate the old v2
connector. There is one accepted parent/cursor and one transport lifecycle.

`create_npu_runtime_connector` selects it for V4.1/HCCL/speculation-off only when
the configuration enables DBO/ubatching. Other configurations use the existing
factory. Shared validation uses the same predicate. This additional U2
condition preserves default C2 U1 and the C3 DSpark pair: F's existing role view
clears speculation, so selecting solely on `speculative_config is None` would
have sent C3 F to v3 while A retained v2. The corrected condition keeps both
default-U1 roles on v2. `get_npu_ffn_config` rejects DSpark+U2 before clearing
speculation, preventing that role view from concealing an unsupported
combination. A and F must configure the same U2 service; there is no new enable
flag or silent protocol fallback. The existing model guard remains closed.

The selected normal U2 session supports native per-forward U1 fallback. This
does not mean default U1 configuration automatically selects v3. Explicit
staged adapter/binder components may still be constructed for logical U1.

Common policy retains BF16/text/eager/compilation-NONE, no SP, DP1/PP1/CP1,
non-EP/EPLB/elastic/fault-tolerance, no LoRA/PD and speculation-off restrictions.
Binding compares the actual native rank/device and full model capability:
hidden width, vocabulary, complete target layer tuple, A/F/native-TP topology,
counts extension, idle state and scheduler-bounded token capacity. Counts
protocol 1 requires one native TP world per role: `A == F == TP`. Transport-only
aggregation/fanout capabilities do not widen this model admission policy.

## Base tensor API

Each transfer is checked against an accepted transaction, full layer membership,
stage index and the local native/aggregate row count. A transfer state records
the parent transaction, layer and stage. Output requires the original context
identity; send/receive pending state prevents overlapping exchanges through the
base API. Correct layer-major sequence and raw tensor/device/dtype/ID checks
remain enforced by the real transport's cursor and tensor methods.

BF16 activations, separate FP32 router values and raw IDs are passed directly.
They are not reconstructed or cast into one payload. The Attention output
buffer's shape, dtype, device and contiguity are checked before consuming the
reply. Copy failure after consumption is terminal. The adapter allocates no
native TP group, does not add a TP reduction, and does not own COMPLETE or STOP
independently of the runner/service. Explicit `close` delegates group release
to the same transport; the worker/runner must obey its successful-STOP and
native-cleanup contract before calling it.

The CPU queue tests replace only the transport boundary. They check adapter
payloads and rejection behavior, not real device transfer or backend scheduling.
Real transport cursor behavior is covered by its own protocol/device evidence;
the queue is not accepted as a substitute for that cursor or NPU behavior.

## Counts, dispatch and resource ownership

Protocol 1 uses `HcclLocalForward(mode, valid_tokens, padded_tokens)` and a
solicited count round. Attention0 sends REQUEST to every other Attention rank;
each peer receives REQUEST or idle STOP before sending REPORT. This avoids an
unsolicited blocking report racing STOP. Attention0 drains every requested
complete reply, including semantically invalid decoded frames, before
broadcasting ERROR. Framing/backend failures escape to terminal supervision;
the implementation does not promise to drain an invalid-length frame's body.

Attention0 constructs the canonical stage plan. All Attention ranks validate
their own saved report's ID, mode and native valid/padded columns before ACK.
The complete plan digest binds all peers' ACCEPT. `HcclStageLayout` retains the
native TP requirement that replicas agree on each stage's counts. Both one-
and two-stage parents may follow each other within the session. COMPLETE clears
the saved local report, transaction and payload state; completed IDs cannot be
reused.

Capability 0 retains its old v3 HELLO representation by omitting the extension;
capability 1 is strict and does not silently downgrade. This compatibility is
for the prior staged v3 component, not a wire conversion to the separate v2
protocol. Counts-enabled Attention cannot bypass the report round with public
manual begin/receive calls. All-A `stop_attention()` sends no synthetic counts:
Attention0 publishes STOP, other A ranks acknowledge it, and global STOP_ACK
completes the idle shutdown boundary.

The FFN normal constructor uses the same connector helper. After the real
device has been resolved, the worker binds its connector's transport before
initialization. The existing worker service owns receive -> one complete FFN
parent -> COMPLETE; False means acknowledged STOP only. Native MoE and native
TP reduction remain in the established FFN executor, with no request scheduler
or Engram state moved onto F. The service preserves its original exception
across poll/stop and cannot restart a terminal session. Normal close follows
peer STOP and thread join. A failed or still-blocked receiver retains resources
for process supervision rather than being destroyed underneath device work.

## Snapshot, cost and remaining validation

Read-only source comparison uses vLLM `ced6857a` and Ascend `7e37e29a`, never an
approximate installed source. The reviewed root-owned snapshot is:

| File | SHA256 |
| --- | --- |
| `connectors/npu/hccl_staged.py` | `11acab13b114d891f14c62d9936ed0efd34ff15733f22cc8509d6210f9adc9df` |
| `connectors/npu/hccl_config.py` | `9272b89a5fc93bbb90248e1fe9c840b37a181fe1016c088d390f9815f0049e15` |
| `validation.py` | `a79edfaf134f8fe0842ed7d040212e071108a8cc8b0878c912f6932d37791096` |
| `v1/worker/npu/ffn_model_runner.py` | `008bd85229a2b00b4423fb698d2804a42098c5334e4b87a0d755068f4b91b44a` |
| `v1/worker/npu/ffn_worker.py` | `b3c8c9bb5e2a048355ca509a5ea688e484fb64230e88329498fcfbc6686b49c5` |
| `compat/npu/runtime_config.py` | `193779218efd9c973e64bb7ebb69ed23b880f5223d3d6a07aff35f81bc4cb540` |
| `connectors/npu/hccl_staged_transport.py` | `76878b3103a31338cb88033f4315b99ca33371183cc1ea4942c7b2c6d4d6d344` |
| `distributed/hccl_plan.py` | `b3da12187d1355b05630f9167b1da198bbb9b8a18b25e7fc7c13e675983128b2` |

All paths in this table are relative to `afd_plugin/`. AST parsing with Python
3.11 grammar passed for these sources. The root agent and A3 validation agent
own final runtime test logs and manifest matching; follow-up results must be
read and appended rather than copied from an older source snapshot.

This increment adds constructor-time validation and a bounded control count
round per parent. The adapter adds context checks and one final output-buffer
copy for callers of its base API; the staged model driver uses the transport
directly. Existing synchronous payload/COMPLETE costs remain. No new permanent
registry, weight copy or communication worker is introduced. Upstream backend
connector selection and prepared-parent callbacks could replace these seams.

Actual native V4.1 metadata/Engram execution, complete model weights, checkpoint
inference/equivalence, Graph/U2, overlap, DSpark/PD and mid-device-failure recovery
remain separately pending. Automatic Python wiring under a closed guard does
not establish those results.

## Installed-dependency follow-up

The reviewer read `a3-forward-counts-20261109/integration-cpu-008/identity.json`
and `results/cpu-integration.xml`. The eight reviewed runtime files match that
immutable manifest exactly. These relevant modules record zero failures,
errors or skips: adapter 15 tests, shared HCCL policy 108, plan codec 116,
staged transport 50 and FFN worker 19. These are Python/CPU contract results.
They include the final post-reply copy failure and transport identity repairs.

The complete integration run is not represented as all green: it has 2238
passed, 4 failed, 2 skipped and 4 deselected. Four older constructor test
fixtures still patch a removed module-level `AFDConnectorFactory` alias after
the normal runtime helper migration. Their tests-only correction and targeted
follow-up are tracked by the root agent. No passing subset is being promoted
to device or full-model acceptance.

The final fixtures-only repair was subsequently verified in
`a3-forward-counts-20261109/integration-cpu-009/results/summary.json` and
`cpu-integration.xml`: **2242 passed, 2 skipped, 4 deselected, return code 0**,
with zero failures/errors. Every reviewed runtime hash in the table matches
that final immutable manifest. The selected CPU regression is now green; the
two skips and four deselected device-marked cases do not supply missing hardware
coverage. The earlier 008 failure remains historical evidence, not an unresolved
runtime finding.
