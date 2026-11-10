# DSpark target shape: phase-one software contract

The synchronous V4.1 FFN worker now retains the target's validated query width
before isolating its private configuration. This fixes MC2 capacity sizing
after F removes the speculative configuration. It does not enable DSpark
Graph serving, change HCCL capability bytes or construct a drafter on F.

`DeepseekV41TargetShape` is a frozen, slotted value containing only
`target_query_len`. The resolver first applies the existing same-checkpoint
V4.1 DSpark validation, then derives N+1 from the actual speculative-step
count. Speculation off produces one; other model families produce None.
Boolean, noninteger and nonpositive widths/steps are rejected. A target width
above the configured native token capacity is rejected before F isolation.
HF draft block size and draft query width are not used as substitutes.

The explicit ownership chain is:

```text
original validated target config
  → AFDNPUFFNWorker.__init__: immutable target_shape
  → get_npu_ffn_config: private spec=None, Engram=None
  → init_device / create_npu_ffn_model_runner(target_shape=...)
  → synchronous AFDNPUFFNModelRunner(target_shape=...)
  → load_model / native set_mc2_tokens_capacity(..., target_query_len)
  → native loader and MoE constructor
```

The plugin-only keyword defaults to None. The synchronous runner checks
model identity and capacity, and checks equality when it still receives an
original speculative config. An isolated config cannot reconstruct N, so the
worker supplies the immutable dependency explicitly. The AsyncCAM factory
selection and its existing admission rules are unchanged; non-V4.1 runners
retain their previous sizing path. Session/capability equality belongs to
phase two and is not implied by this local value.

Ascend at `7e37e29a7803d6897a7113088e2b771f83205bc0` derives the target
width in `worker/v2/model_runner.py:257-262`. Its
`ascend_forward_context.py:243-278` chooses scheduler capacity for prefill
MC2/MegaMoE, capture capacity when present, or maximum requests times target
query width otherwise, then applies native TP rounding and kernel limits.
That implementation remains upstream-owned and unchanged. This increment
introduces no per-token work or new NPU buffers.

The shared CPU fixture runs the actual plugin worker constructor,
`init_device`, factory, synchronous runner constructor and `load_model`.
Device/admission/connector/weight boundaries are substituted explicitly.
The actual pinned native capacity helper is called; the CPU loader allocates
a routing workspace using its public result. Eighteen cases cover q1/q3/q6,
TP1/TP2 and decode/capture/scheduler capacity branches. Assertions inspect
the resulting workspace, private config and unchanged original target, so a
lost q or incorrect native capacity changes the result.

Actual-dependency evidence is in
[the CPU run index](a3-dspark-shape-cpu-20261110/README.md).
The first snapshot's four core modules passed 218 cases. Its optional Async
regression exposed ten stale constructor/profiler fixture failures, retained
in `component-001`. The separate Async-only `component-002` retained two
further missing-profiler-default failures. Tests-only fixture corrections
closed the complete 55-case Async module in `component-003`; neither failed
attempt has been relabeled as passed. All three snapshots matched eleven
installed source pins, including the actual native capacity helper.

These are CPU configuration, construction and capacity contracts. No device
probe, checkpoint loading, MC2 kernel, HCCL session, Graph capture or DSpark
acceptance/rejection was run. Production V4.1/Graph guards remain closed.
The separate phase-two work must negotiate target q exactly, propagate it
through graph keys and native request/token geometry, and retain target-only
versus A-local draft scope. Native U2 may split inside a q-wide request;
child token counts must not acquire an invented divisibility constraint.

Independent source review:
[target-shape review](dspark-target-shape-independent-review-20261110.md).
The frozen phase-one identities and raw artifact hashes are recorded in
[validation-summary.json](a3-dspark-shape-cpu-20261110/validation-summary.json).
