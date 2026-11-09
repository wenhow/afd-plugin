# Explicit native model U2 capture callback contract

Date: 2026-11-10. Scope: the controlled HCCL V4.1 UBatchRunner preparation
and synchronous callback component. This document describes the author's own
implementation; it is not an independent review of that runner.

## Locked source and entry

The source pair remains vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` and vLLM-Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`. Both upstream repositories were
read only.

- Native `gpu/cudagraph_utils.py:673-695` prepares FULL inputs and invokes
  `ubatch_runner.begin_capturable_run(..., for_capture=True)` directly. The
  callback must therefore be adapted independently of the ordinary `run` seam.
- Native `gpu/ubatch_utils.py:329-387` owns the actual request/token slicing,
  child query/sequence buffers, Attention preparation and slot mapping.
- The native callback at `gpu/ubatch_utils.py:432-524` starts complete-model
  child threads. The HCCL component returns a synchronous parent callback
  instead, preserving the caller's native ModelWithContext and factory output
  storage. It does not install a parked-thread completion callback.
- Native Ascend `patch/worker/patch_v2/patch_input_batch.py` explicitly aliases
  both `gpu.cudagraph_utils.InputBatch` and `gpu.model_runner.InputBatch` to
  AscendInputBatch. The actual consumer must load that native patch; this
  component adds no substitute alias patch.

An AST comparison against the locked Git object confirmed exact arguments,
defaults and return annotations for `prepare`, `_make_forward_contexts`,
`begin_capturable_run` and `run`. The existing sole signature-alignment table
also owns the new `hccl-graph-begin` row.

## Ownership and behavior

An optional explicitly injected graph owner implements `require_full`,
`prepared`, `forward_parent` and `fail`, and exposes its read-only transport.
The owner and runner must reference the same transport instance. Construction
requires FULL_DECODE_ONLY, speculation off, graph/counts protocol v1, U2 and
paired native TP roles with DP1/PP1/CP1/non-SP native rank coordinates. Existing
eager construction keeps its previous admission; production Graph admission
remains closed.

FULL preparation validates the reservation before native input mutation. It
retains FULL on EngramModelState until native parent input preparation, so that
the native FULL producer path is selected. Both child ForwardContexts and their
Ascend additional context use FULL. A scoped preparation-mode field is restored
after preparation, so a later eager parent uses NONE. Failure restores the prior
parent producer mode and marks the graph owner terminal.

When TargetDeviceMetadata exists, one native `build` context collects all
independent child builders. The returned child resources receive the same
native executor. This preserves the source's single pending task frontier;
calling native `run_build` independently for each child would replace or reject
that pending frontier. The supplied native ModelWithContext still performs
`begin_forward` and `finish` once around the parent. This CPU interface proof
does not establish the A5 metadata kernels or their captured stream behavior.

The callback accepts only the currently prepared FULL state, requires capture
mode, and executes once on its creating host thread. It calls the owner's
`forward_parent`, which must invoke the supplied native ModelWithContext once
in a parent context and route only the inner model through the staged tensor
body. Warmup and capture are identified by the accepted owner phase, not the
native factory callback's NONE argument. Failures are terminal and leave
communication resources to external supervision. Native Ascend execute_model
retains final Engram/ModelState retirement.

## Validation and limits

Ruff and whitespace checks passed for the runner, mirrored tests, three shared
fake modules and sole signature table. Dependency-free local collection safely
skipped unavailable runtime imports. Actual-dependency CPU results are recorded
separately when the immutable A3 snapshot returns.

The added tests use actual native child slicing/ForwardContexts and the actual
TargetDeviceMetadata state machine. Its stream executor is an explicitly CPU
task boundary; the model is CPU arithmetic under the actual ModelWithContext.
These tests cover mode restoration, both metadata frontiers, one parent model
call, errors/context restoration, one-shot/thread ownership, and graph-mode,
transport-identity and capability rejection. They do not emulate NPU operators.

No full-model capture, SFA/DSAV41 device update, Engram graph producer, A5 packed
metadata, GE/Dynamo graph, checkpoint inference, production startup or
performance acceptance is claimed. The native model manager/factory consumer
and retained cache lifetime must be assembled and validated separately.

Initial frozen SHA256 values before actual-dependency CPU:

| Path | SHA256 |
| --- | --- |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `2e0940e9c3a07c43d0b4a942a8dca19c6f217dbff019a7832a53f12ae65663c5` |
| `tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py` | `0df085f7143f3ed4a0165eb0a95580c192ac3e03d23be135fd5317715a9542dd` |
| `tests/unit/fakes/input_batch.py` | `b9b09ea2f2743ad8f4b7e909223e3c340a19fd65d0c0aa696425bbc5c388e623` |
| `tests/unit/fakes/device_metadata.py` | `4c67aa93612ad1583cd72b35c61d5ac3dd4a1be4643adf2344faf0060e72f617` |
| `tests/unit/fakes/hccl_v41_ubatch.py` | `05da495574a673bf77177160c2314c79cc791109e9f8d281d6db40ea2c64494e` |
| `tests/unit/compat/patches/test_signatures.py` | `d022d80eea1cb57730d26b4c4b3e6c6f0f89286b43d66a2dbbc69300e65bfa99` |
