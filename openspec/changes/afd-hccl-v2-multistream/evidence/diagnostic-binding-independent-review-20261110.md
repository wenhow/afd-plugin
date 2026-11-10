# Independent eager diagnostic binding review, 2026-11-10

Reviewer: the independent C7/configuration agent, not the author of these
diagnostic bindings. Scope is the root-owned eager A/F integration and the
diagnostic helper/owner seams it consumes. Runtime files were read only during
this review. The base is `24c8d51df5d43d57dc8b4d3de0833fe592b42aa5`;
concurrent Graph work is outside this conclusion.

## Findings and source closure

1. **Completed-owner diagnostic exception was masked.** The initial
   `_execute_hccl_v41_parent` outer catch unconditionally called
   `eager_pipeline.fail()`. A new `parent_finish_returned` sink can raise after
   COMPLETE has retired that owner. Calling `fail()` then raised the stale-owner
   error and hid the original sink exception. The component's own `finish`
   intentionally emitted this final record outside its live-owner failure
   catch, but the A integration undid that protection.

   The fixed A catch retains `accepted_plan`, fails only the identical active
   owner/plan that is neither finished nor already failed, and re-raises the
   original exception. The UBatch and F service catches also compare their
   retained accepted plan and owner before failure. This preserves refs for live
   failure without attempting to fail a retired or foreign owner. An already
   failed endpoint can still be terminal; none of this is an HCCL network abort.
   The initial WIP bytes were not separately archived by this reviewer; the
   fixed source hashes are listed below, rather than inventing a pre-fix hash.

2. **Graph binding lacked reverse mutual exclusion.** F diagnostic binding
   rejected an existing Graph service, but the reverse Graph-service binder
   did not reject existing eager-session/diagnostic bindings. The explicit
   legacy connector injection path can use different transports, so the
   staged connector's exact-transport check alone did not close this state.
   Initial reviewed F source was
   `1ebc104fd6ec76b95528e7d5fde6cf5b57a83d9f98a956df23523352667988bf`.

   F Graph binding now rejects either existing binding before assignment. A
   Graph-eager binding likewise rejects existing eager diagnostics. Existing
   eager/diagnostic binders retain their Graph-capability and exact idle
   transport checks. Public regression cases and follow-up CPU execution are
   owned by the implementer; no new broad feature permission was introduced.

No further blocking issue was identified in the reviewed fixed source scope.

## Owner and timing checks

`HcclV41DiagnosticBinding` is immutable and tied to one transport. Binding
requires an initialized idle non-Graph session. An eager stream session and
diagnostic binding compare the same immutable settings in either bind order.

For U1, `HcclV41Execution.install_context` first accepts the real parent, then
binds/emits configuration before native `nn.Module.__call__` and its Engram
producer hook. Contexts without a batch descriptor remain native KV maintenance
and publish no diagnostic parent here. That maintenance conclusion is source
analysis, not a PD request test.

For U2, the selected UBatch runner opens the actual optional ticket owner before
binding progress. `execute_hccl_v41_stages` checks transaction, rank, role,
layer/stage geometry and actual executor streams before `prepare_engram`. The
ordinary A-side producer is still consumed once; diagnostics do not replace
native request/history or retirement.

F binds after `receive_forward`, before receiving a layer or invoking MoE. The
serial consumer validates exact accepted progress and keeps raw IDs/independent
FP32 inputs and native TP reduction unchanged. Compute synchronization requires
the explicit setting, actual serial parent execution, live target eager U1 and
F role. Profile, warmup, dummy, Graph and eager U2 do not gain that fence.

Actual eager stream labels and alias relationships come from the owner's real
stream objects, independently of logical flags. Host post/loan/reuse/return
records state enqueue boundaries; they do not claim device completion. The
default `None` binding preserves the old route. Disabled diagnostic settings
produce no sink or synchronization work. Config-sink errors before compute
terminalize the accepted session while retaining live owner resources for
external supervision.

## Fixed source identity

| Path | Reviewed SHA-256 |
| --- | --- |
| `afd_plugin/v1/worker/npu/hccl_v41_diagnostics.py` | `749994fc17b163286b17c9b6cfde57744a9fa1b5a2cfd7c531229dc23faf074a` |
| `afd_plugin/v1/worker/npu/hccl_v41_execution.py` | `a7629882fce03f4be9f8ec91281e2107adc661a0abb61d9013c15b711a31f433` |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `112491c774c1baedbfa3fb78e1be7b2363bea337b033b2be06666220ec61c252` |
| `afd_plugin/v1/worker/npu/attention_model_runner_v2.py` | `de6457cbaf51cadafc3a890bcc105dc79e0a405dc6146ac80097ed77ef059d7b` |
| `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `ec1321df7d09b1db054098ceb49676705c0b5f7736c14d4aa551928e43ddfe53` |
| `afd_plugin/v1/worker/npu/hccl_v41_ffn_runner.py` | `d324da6aecfe01ec2febe012aa74bc4b1f2345039629dd1cbc90622ea33fcb3b` |
| `afd_plugin/v1/worker/npu/ffn_worker.py` | `ecbeffe2d28c3bd04287cef44198b0ed0ddd205b61056017d6be3a81291897ba` |
| `afd_plugin/connectors/npu/hccl_diagnostics.py` | `45f34d9ddbe821c2d142ad34063da48d8d009ae8e4fa4149bf95e8fd453f6549` |
| `afd_plugin/connectors/npu/hccl_eager_pipeline.py` | `19db743a77cfdf7b2db27fb90077a7291f826797576dfd8ff4989084c7beebd2` |

## Validation boundary and cost

The reviewer read the actual
[`diagnostics-family-018b` XML and manifest](a3-pipeline-cpu-20261110/diagnostics-family-018b/identity.json):
512 cases, zero failures/errors/skips, exit 0, ten matching baseline native
source checks and no device probe. Manifest:
`3254497cb19e97cdb842af769aa239ca23ed83fe1a6d5723815480237334dced`.
Its snapshot matches the helper, fixed `hccl_v41_execution`, F serial consumer
and worker hashes above. The UBatch, A runner and F runner have later ownership
or mutual-exclusion fixes, so 018b is not acceptance of those final bytes. New
finish-sink and reverse-binding regressions were not included in that historical
512-case result.

The reviewer subsequently checked the separately frozen
[`diagnostic-boundaries-020`](a3-pipeline-cpu-20261110/diagnostic-boundaries-020/identity.json)
summary, XML and manifest:
`0d1d08edb730ad18d7baaa6055f79c28300f5f08d436aeb3444326e39a72e1c9`.
Its six targets passed **247 cases, zero failures/errors/skips**, exit 0,
and all seven root-owned production hashes in the table still matched that
snapshot. This follow-up includes the two native U1/U2 configuration-sink
failures before producer, the actual ticket COMPLETE-returned sink failure
preserving its original exception, and both F reverse-Graph-binding cases
for existing eager/diagnostic bindings. The two reported findings are therefore
closed for these CPU public seams. This does not replace 018b with a fictional
new 512-case run, nor qualify device diagnostics or NPU failure behavior.

This adds bounded per-parent configuration/coordinate checks and conditional
host records. Enabled sinks add Python/logging cost; the explicit live serial
U1 F compute-sync setting intentionally adds a device fence. No new shared
mutable registry, wire kind, device polling or blocked-HCCL cancellation is
added. Remove the adapters when native diagnostics expose equivalent accepted
parent identity and actual stream/enqueue boundaries. This is not model/NPU
timeline, overlap or performance acceptance.

## Bounded ticket-owner diagnostic follow-up

The reviewer later read the raw identities, outer summaries and all role
reports in `a3-eager-pipeline-20261110/diagnostics-tp1-002` and
`diagnostics-tp2-002`. Both freeze manifest
`b7d1964c67db1c6e5692ab363a1d90eb1480485ba290d4c4b2012e844384d20c`,
probe `c8b4b0b2becb3fe11d958666b8e204e8757df09f418a32850bf76b12fd787fc6`
and harness `e808603c73d5bd5b353db108d9dab63f902fd1dcc860469944efe60c46e1700e`.
The complete diagnostic-only probe/harness delta was reviewed before these
parent-authorized runs: it appends scalar records through the actual binding
and ticket owner without adding a wait, event, query or synchronize.

Each role report has six accepted-parent configuration records, 146 total
records and 16 checked positive steps. Rank/transaction and actual stream
alias checks, inactive U2 compute-sync checks and the empty-parent check all
pass. Native TP results are `[1]` and `[2]` before and after plugin closure;
all two/four rank exits are zero. Both outer summaries match source bytes and
33 installed source pins, show eight idle boards before/after, and record no
timeout or owned-PID termination.

These are real ticket-owner/HCCL component records with controlled arithmetic.
They do not execute the complete native model/MoE or prove the native consumer
timeline, compute/communication overlap or performance. They do not supersede
the separately scoped CPU binding/exception tests or accept later profiler
changes. The reviewer launched no additional device attempt.
