# Independent review: explicit eager session binding and parent stream scope

Read-only review by the C7/profiler agent. The session, A/F consumer, and runner
bindings were implemented by the parent agent. This review did not launch a
device experiment or change those production files.

## Reviewed snapshot

| Path | SHA-256 |
| --- | --- |
| `afd_plugin/v1/worker/npu/hccl_v41_eager.py` | `8591e7a5860843bb8c4537212a71112114afc7a9179801087b377c67a2ab2695` |
| `afd_plugin/v1/worker/npu/hccl_v41_execution.py` | `8027fed83fab1db933c6609a2f1a18cffb055ed3bca23a9cf2d7143a404fb529` |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `fb456f3568b99dfc89f62a6389cd0b0183a2976d423a718c6013fb059739e932` |
| `afd_plugin/v1/worker/npu/attention_model_runner_v2.py` | `139c58d443fb96cf63d205e9c2dfbd1b3ae19f2c3d956e79b24d19946cf359a0` |
| `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `d805d19f96cfd6c5ea87b9c78abe3a8939a0b357ba341e94fafc5c0f352a8993` |
| `afd_plugin/v1/worker/npu/ffn_worker.py` | `6f36918afc2dbd0d0afedc7a908be676f111d9398a4858a34556a3beaf0d2e7e` |

## Findings sent to the implementation owner

1. **Medium: F runner's public eager binder did not require its connector's
   exact transport.** `AFDNPUFFNWorker.bind_hccl_v41_eager_session` checks the
   worker transport, but the independently public runner binder only validated
   role/config/device. With an `HcclStagedAFDConnector`, a same-capability foreign
   transport could be bound directly to the runner, then passed to execution,
   while native forward metadata still names the original connector. Require
   the same connector/transport identity check already used by the graph service
   binder, before accepting the session. An injected legacy connector remains a
   separate supported explicit construction case.

2. **Medium: A's public `run_staged` checked a foreign transport too late.**
   With a bound eager session, `run_staged` opened that session's parent before
   checking its `transport` parameter. The downstream model validates the
   pipeline identity, but the parent Engram producer can already have been
   consumed by that point. Require `session.transport is transport` before
   opening the pipeline. Keep the existing no-session standalone interface
   unchanged. A public negative should demonstrate no parent open and no
   producer consumption.

3. **Medium: A's eager binder accepted a previously initialized but now dead
   transport.** `session.initialized` records completed warmup, not current
   transport liveness. The UBatch binder checked only plan/reservation state.
   Reject uninitialized, terminal, or closed transport at this boundary, so the
   API actually fulfills its idle/live precondition before storing the binding.

These are entry/ownership guards; this review did not find an already observed
normal-path numerical failure. They remain open in this snapshot until the
implementation owner supplies a follow-up and corresponding public negatives.

## Other inspected boundaries

The session constructor accepts only eager U2, graph protocol version zero,
paired TP1/TP2, the model's native rank/device/config, and immutable overlap
settings. Initialization requires an idle live transport and performs matched
warmup only for the enabled eager family. U1/profile/warmup/dummy select the
serial executor before posting tensor work. The active U2 family owns one
pipeline and cannot switch the schedule mid-parent. Both roles must be
explicitly constructed with the matching family: this component does not add
HELLO agreement of rank-local settings. Closed production model admission is
unchanged.

For A, the new outer execution scope verifies the actual parent stream and
uses `use_npu_current_stream` around the parent Engram producer, child
construction, staged execution, merge, and COMPLETE. Side compute scopes use
the same compatibility helper. Its exact prior vLLM TLS value or absent state
is restored in `finally`, and the native NPU stream context is restored too.
This addresses the parent/side cached-stream ownership boundary without a new
global registry. The synchronous path keeps its prior scope. The native
Engram producer remains one parent-owned consume, and native Ascend keeps the
producer retirement; the session adds no history submit or second retirement.

F's eager consumer checks the exact active pipeline/plan, native topology and
rank, executes the complete native MoE under existing per-stage forward
contexts, passes original FP32 and raw IDs, and adds no TP reduction or padding
output mask. Its parent and side scopes use the same TLS helper. Tensor loans
remain open through actual native consumption, and explicit continuation refs
keep borrowed storage alive until reuse/finish. Failure retains active owner
resources for process supervision rather than issuing a network abort.

This is static/source ownership evidence. CPU tests can cover the public
guards, scheduling, and TLS restoration but cannot demonstrate native MoE,
Engram event execution, NPU stream concurrency, or performance. The separately
reviewed actual ticket/loan probe has its own device acceptance scope.

## Narrow source follow-up

The implementation owner closed all three findings, and the reviewer re-read
the changed functions and public negative tests. Final production hashes for
this follow-up are:

| Path | SHA-256 |
| --- | --- |
| `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `69ff9109e51dd363491d2c208a08b7dd48ffa893eec4ecb45e9093313ebb3260` |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `c76eedb4b43a42a29edde890a397f24aab3763c0327c7691680fab8471844a1d` |

The F runner now rejects a staged connector's foreign transport in the binder's
precondition, before storing the session. Its public negative is in the
existing `test_hccl_v41_ffn_runner.py` mirror. A `run_staged` now compares the
bound session transport to its explicit parameter **before** `open_parent` or
native Engram preparation; the negative asserts the producer was not consumed
and the peer has no messages. A's binder rejects all uninitialized, closed, and
terminal transport cases before storing the binding; the mirror parametrizes
those three states.

The inspected fixes address the original findings without narrowing the
no-session standalone path or introducing a fallback after posting. There is
no remaining static blocker in the reviewed entry/stream scope. Actual
dependency execution of these added negatives is separately scheduled as 015;
this source review does not pre-accept that result or qualify a new device
model consumer.

## Actual-dependency guard follow-up

The reviewer read `a3-pipeline-cpu-20261110/binding-guards-015/identity.json`,
`results/summary.json`, and the JUnit cases. Manifest digest
`2d016616630638c6955eb4b6fc70063a2fb5304570cea1ccd4647c42016497aa`
fixes the two production hashes above and signature table `acd00460…`.
All 134 selected cases passed with zero failures/errors/skips; source bytes
matched, and no NPU probe started. The XML explicitly contains the F foreign
connector-wire negative, A foreign-wire-before-producer negative, all three
dead-session binding states, and the new exact native F `profile` signature
row. This closes the three public guard findings at the actual-dependency CPU
boundary. NPU model/stream/timeline/performance limits above remain unchanged.
