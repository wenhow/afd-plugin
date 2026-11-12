---
title: Standalone NPU FFN entry point
kind: module
status: draft
owners:
  - "@hsliuustc0106"
  - "@jiangkuaixue123"
primary_code_paths:
  - "afd_plugin/entrypoints/npu_ffn.py"
related_code_paths:
  - "afd_plugin/compat/patches/engine_core.py"
  - "afd_plugin/v1/worker/npu/ffn_worker.py"
depends_on:
  - "ffn_runtime.md"
  - "compatibility_and_patches.md"
validation_paths:
  - "tests/unit/entrypoints/test_npu_ffn.py"
  - "tests/unit/fakes/engine_client.py"
upstream_refs:
  - "vLLM 0.30.0 ced6857afa0ea7b2e3f0846a62e1394e90f15607"
  - "vllm.v1.engine.core_client.SyncMPClient"
  - "vllm.v1.engine.utils.CoreEngineProcManager"
verified_platform_refs: []
---

The standalone entry point supplies the local frontend handshake needed by an
FFN EngineCore. Native `vllm serve --headless` connects to another frontend's
handshake socket; an independent DP1 FFN process has no such frontend. An HTTP
generation service is unnecessary because FFN workers receive connector work.

`AFDFFNClient` inherits the complete pinned `SyncMPClient` constructor, including
its native local input/output sockets, `launch_core_engines` handshake, readiness
barrier and output receiver. It overrides only `start_engine_core_monitor(self)`
with the identical upstream signature. Native request clients classify every
engine exit as unexpected, whereas acknowledged peer STOP normally ends FFN.
The entry point instead calls the same process manager's liveness monitor on its
main thread. This adds no model buffers or device synchronization.

Before process creation, the entry point requires FFN/HCCL, NPU ModelRunner V2,
one local DP1 `mp` engine and DSpark/PD off. The existing model qualification guard
runs on the worker's existing detached FFN configuration view; the original
Engram configuration remains available to normal worker initialization. This
entry point does not expand the qualified model or execution modes.

`monitor_engine_liveness()` waits for an engine exit and calls manager shutdown.
The DP1 entry point then requires exactly one finished engine, preserves its
nonzero exit status, maps terminating signals to shell statuses, and finally
calls native client shutdown. The zero-exit marker is emitted only after those
steps complete. It proves process exit codes and client cleanup, while peer STOP,
worker/resource cleanup and device idle state require their separate evidence.
External interruption produces a nonzero launcher status and follows native
client cleanup; it is not a substitute for Attention-owned STOP.

CPU tests cover configuration rejection, original configuration preservation,
exit-status propagation, missing/extra engine outcomes and client resource
release. They do not simulate model loading or establish NPU/HCCL acceptance.
Remove this adapter once the native frontend supports connector-service exit
policy and standalone bootstrap.
