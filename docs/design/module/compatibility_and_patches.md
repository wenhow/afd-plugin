---
title: Compatibility and patches
kind: module
status: draft
owners:
  - "@hsliuustc0106"
  - "@jiangkuaixue123"
primary_code_paths:
  - "afd_plugin/compat/__init__.py"
  - "afd_plugin/compat/vllm.py"
  - "afd_plugin/compat/npu/__init__.py"
  - "afd_plugin/compat/npu/feature_validation.py"
  - "afd_plugin/compat/npu/forward_context.py"
  - "afd_plugin/compat/npu/runtime.py"
  - "afd_plugin/compat/npu/runtime_config.py"
  - "afd_plugin/compat/npu/stream.py"
  - "afd_plugin/compat/npu/profiler_api.py"
  - "afd_plugin/compat/npu/profiler_control.py"
  - "afd_plugin/compat/npu/kv_layout.py"
  - "afd_plugin/compat/npu/kv_layout_native.py"
  - "afd_plugin/compat/npu/kv_layout_connector.py"
  - "afd_plugin/compat/patches/**/*.py"
related_code_paths:
  - "afd_plugin/__init__.py"
  - "afd_plugin/v1/worker/**"
  - "AGENTS.md"
depends_on:
  - "plugin_boundary.md"
validation_paths:
  - "tests/unit/compat/**"
  - "tests/unit/package/test_package.py"
  - "tests/unit/v1/worker/test_runtime_classpaths.py"
  - "tests/unit/v1/worker/test_model_runner_v2.py"
  - "tests/unit/v1/worker/test_npu_runtime.py"
  - "tests/npu/pd_kv_layout_probe.py"
  - "tools/diagnostics/run_a3_pd_layout.py"
upstream_refs:
  - "vLLM vllm.v1.engine.core.EngineCore and EngineCoreProc"
  - "vLLM vllm.v1.engine.core_client.DPAsyncMPClient"
  - "vLLM vllm.forward_context.set_forward_context"
  - "vLLM vllm.engine.arg_utils.EngineArgs and vllm.config.VllmConfig"
  - "vLLM vllm.v1.worker.gpu.model_runner.GPUModelRunner"
  - "vLLM vllm.plugins.endpoint_plugins.interface.EndpointPlugin"
  - "vLLM-Ascend platform and fused-MoE symbols referenced by NPU patches"
verified_platform_refs:
  - "CPU/runtime compatibility tests in tests/unit/compat"
  - "Ascend patch evidence in NPU unit and E2E paths"
related_issues:
  - "#86"
  - "#129"
last_reviewed: 2026-10-13
---

# Compatibility and patches

The NPU current-stream scope keeps actual NPU stream selection and vLLM's
existing thread-local current-stream cache aligned for side compute. It restores
the prior value or absent field on normal and exceptional exit and creates no
new registry. Its public-query CPU seams live in
[`test_stream.py`](../../../tests/unit/compat/npu/test_stream.py); NPU event,
allocator and graph behavior require separate device validation.

The opt-in NPU profiler HTTP adapter in
[`profiler_api.py`](../../../afd_plugin/compat/npu/profiler_api.py) implements
the pinned native `EndpointPlugin` protocol. Its independent
`afd_npu_profiler` endpoint entry requires explicit `VLLM_PLUGINS`
allowlisting alongside the existing `afd` general entry. The native loader
attaches its `/afd/profiler` routes after core routers; initialization binds
the actual app's EngineClient and NPU AFD configuration and rejects disabled
role-env or conflicting native profiler configuration before HTTP startup.
GET forwards the fixed status RPC and preserves each worker snapshot; POST
strictly validates desired state and acknowledges the fixed profile RPC.
Neither operation starts/stops a profiler backend on the HTTP thread. Requests
use a bounded timeout, with pending/applied state and NPU trace acceptance
remaining execution-owned. Each app keeps an independent binding; no new
global registry, native patch, general RPC route or HCCL packet is added.
[`test_profiler_api.py`](../../../tests/unit/compat/npu/test_profiler_api.py)
owns actual ASGI request/validation and native loader seams. These CPU tests
do not establish a deployed server or hardware timeline.

The opt-in headless F adapter in
[`profiler_control.py`](../../../afd_plugin/compat/npu/profiler_control.py)
binds an explicitly configured local Unix socket before starting the existing
HCCL F worker loop. The EngineCore management thread polls bounded client IO
and forwards only fixed status/start/stop operations to its existing executor.
The executor RPC has a separate 30-second timeout; the five-second client IO
deadline does not bound RPC execution or establish management latency. Worker
execution owners still apply requests and export traces. Default-disabled
construction creates no socket or thread. Shutdown removes only this listener's
socket inode before following the existing worker STOP policy. Its real Unix
socket and EngineCore-loop CPU seams live in
[`test_profiler_control.py`](../../../tests/unit/compat/npu/test_profiler_control.py)
and [`test_engine_core.py`](../../../tests/unit/compat/patches/test_engine_core.py).
They do not establish full-model runtime reachability or device traces.

The independent V4.1 KV layout diagnostic in
[`kv_layout_native.py`](../../../afd_plugin/compat/npu/kv_layout_native.py)
collects actual native `KVCacheConfig`/specs and materialized tensor metadata.
It preserves shared-page component order, byte strides, dtype, page offsets,
physical-page alias topology and group transfer flags without exporting addresses or
reading tensor contents. The stdlib
[`kv_layout.py`](../../../afd_plugin/compat/npu/kv_layout.py) compares immutable
snapshots, validates every request's ordered source/destination block IDs and
provides a read-only CLI. Different local capacities are legal; each snapshot
and request map must still fit its own storage and block pool. Its two mirrored
tests fall under `tests/unit/compat/**`; native-type CPU tests gate runtime
dependencies before importing the collector. This tool does not attach to
native registration, exchange remote attestation, change PD guards or prove a
transfer. The [operator guide](../../npu/DSV41_KV_LAYOUT_VALIDATION_ZH.md) states
the controlled snapshot and live-object collection boundaries.

The explicitly selected
[`kv_layout_connector.py`](../../../afd_plugin/compat/npu/kv_layout_connector.py)
extends the pinned Mooncake connector through its external-module factory.
Its worker collects typed materialized V4.1 layout before native memory
registration or thread startup. The receiver requires the layout extension,
binds it to the ordered native address/stride arrays and validates each complete
request after native window-tail projection before its batched read. The
registration and peer resources have one fixed lifetime; partial initialization
cannot be retried in the same native engine. Scheduler and sending-thread
behavior remain native. Optional consumer retirement closes whole-batch
admission, waits through native DONE acknowledgements and joins the input
thread. This is consistency between cooperating
peers, not remote authentication. The mirrored connector CPU seams replace
network, transfer-engine and thread-start boundaries; they do not establish
NPU registration, PD transfer or prompt-history/Engram correctness. The V4.1
production guard remains closed. Remove these copied seams when upstream
exposes worker/thread factories and a typed metadata validation hook.

The separate [PD NPU probe](../../../tests/npu/pd_kv_layout_probe.py) uses the
real external constructor, native threads and transfer engine with controlled
typed NPU shared pages. It checks whole-slab bytes, window-tail projection,
incompatible layout and whole-page conflict rejection, native DONE and process
exit. Its [prepare/run harness](../../../tools/diagnostics/run_a3_pd_layout.py)
freezes installed sources, transfer binaries, model configuration and the owned
PID supervision dependencies before an explicitly selected device session.
Preparation imports no device package and establishes no transfer result. The
[component guide](../../npu/DSV41_PD_COMPONENT_VALIDATION_ZH.md) separates manual
slabs from native model allocation, and request/executor drain from the native
worker/GlobalTE wrapper's lack of a shutdown API. The underlying transfer engine
exposes memory unregistration; the probe does not invoke it or establish engine
shutdown. The opt-in `--retire-receiver` path uses two narrowly copied native
methods (private sentinel in `run`, observable ACK errors in DONE) and short
admission/finalization wrappers. CPU tests run real queues, receiver threads
and a CPU executor with gated socket acknowledgements; they do not establish
safe native-extension process teardown. One deadline bounds admission/drain
and receiver join; the final stdlib executor shutdown has no timeout and is
bounded only by the probe's process supervisor. Failure retains the receiver,
engine, cache storage and single-use registration state. No sender loop,
registration reset or engine-unregistration behavior is added. Remove these
retirement seams when the pinned native worker owns cooperative retirement.

The NPU V4.1 loader hook in
[`model_loader.py`](../../../afd_plugin/compat/patches/npu/model_loader.py)
copies the pinned `BaseModelLoader.create_model` seam and supplies the FFN role
class to native `initialize_model`. It preserves native ModelConfig/registry
identity and the native load/post-load/eval lifecycle. Its API regressions live
in [`test_model_loader.py`](../../../tests/unit/compat/patches/npu/test_model_loader.py);
signature alignment remains in the shared patch signature table. Full V4.1
runtime entry is still guarded.

The NPU microbatch construction scope reuses `use_npu_ubatch_sm_control` for
both the existing runner factory and paired Graph components. The pinned
upstream SM manager asserts CUDA/ROCm support at construction, so the NPU
scope supplies the existing no-op policy and restores the selected binding
on normal and exceptional exit. Native stream/buffer allocation remains
upstream; this scope adds no device cancellation or model Graph admission.

The explicit HCCL replay scope preserves the pinned
`ModelAclGraphManager.run_fullgraph` signature and delegates its native Engram,
graph update and `finish_replay` lifecycle once. A selected-instance callback
temporarily hands it the accepted child metadata; the original metadata is
restored before HCCL parent completion. This interface is limited to the pinned
V4.1 SFA/DSA backend update contract. It does not combine children into generic
UpdatableGraph metadata. Its exact signature belongs to the shared patch table.

The explicit `afd_dsv41_mxfp8` general-plugin entry point installs the
DS MXFP8 Linear and MXFP4 MoE odd scale-pair packing corrections only when
`AFD_DSV41_MXFP8_SCALE_PATCH=1`. It does not register AFD models or workers.
The copied post-load function at pinned Ascend `7e37e29a` pads the unused byte
of a final scale pair, preserving logical K, valid scale bytes and weights.
For K=288 the packed scale is `(5, N, 2)`, consistent with the base linear
implementation. CPU tensor tests live in
`tests/unit/compat/patches/npu/test_mxfp8.py`; signature alignment uses the
existing shared table. `tests/npu/dsv41_mxfp8_linear_probe.py` runs real A5 MX
quantization/matmul, including the last valid K group. This component probe and
the full native checkpoint remain separate acceptance steps. Remove this
compatibility patch when the pinned DS linear implementation handles odd pairs.

The same opt-in entry point copies the pinned DS MXFP4 MoE post-load function
and its expert-list scale method in `compat/patches/npu/mxfp4.py`. Both use
`mxfp_scales.py` to retain every E8M0 byte and pad an unused final pair slot.
For TP8 W2 with K=288, `(384, 5120, 9)` becomes `(384, 5, 5120, 2)`;
native weights, NZ conversion and EPLB list construction are preserved.
`test_mxfp_scales.py` checks real CPU tensor bytes; `test_mxfp4.py` checks
post-load tensor outputs with a fake NZ boundary and does not qualify devices.
The shared signature table owns alignment for both copied MoE methods.
`tests/npu/dsv41_mxfp4_moe_probe.py` exercises real DS post-load and W13/W2
GMM, including the last valid K group. Routing, SwiGLU, EPLB device behavior
and full-checkpoint execution require separate acceptance. Remove the MoE
patch when the pinned DS MoE methods support odd pairs. See the
[patch review](../../../openspec/changes/afd-v41-hccl-text-eager/evidence/mxfp4-scale-patch-review.md)
for source identity, loading costs and validation limits.

The A5 probe rejected W2 K=288 because its WeightNz kernel requires
`K == 64 * perTokenScale.shape[1]`. The opt-in DS patch therefore zero-pads
physical W2 K to 320 before NZ conversion, including the list path, and copies
the inherited native `apply_gmm2` to pad quantized activation K at the same
boundary. Valid FP4/FP8/E8M0 values and logical intermediate size are retained;
only unused input scale bytes are made finite. The probe calls this production
GMM2 hook, rather than applying an independent probe-only workaround.
CPU tests cover exact bytes, empty rows, strided scales and dequantized dot
products; they do not qualify NPU padding, GMM, EPLB or full-model execution.
W2 E=384/N=5120 adds 30 MiB per layer plus activation padding/scale copying at
runtime. Removal also requires upstream support for the paired weight/input
K alignment or target-kernel acceptance of the original K.

The subsequent A5 full-model profile reached the native fused SwiGLU MX
quantizer, which requires combined gate/up width divisible by 128. TP8 emits
576 values (two halves of 288), so the opt-in DS patch also copies inherited
`apply_gmm1_act_quant` and independently zero-pads each half to 320 just before
that quantizer. The 640-wide input produces the existing physical W2 width of
320, retaining valid gate/up pairs, clamps, group indices and native SITU
branches. Aligned inputs bypass the allocation. This adds a BF16 padding copy
per unaligned invocation, with no extra weight allocation or communication.
CPU helper tests verify pair/value preservation and clamped SwiGLU arithmetic;
`tests/npu/dsv41_mxfp4_activation_probe.py` separately exercises the real W13,
fused activation/quantization and W2 chain against a numerical reference.
Its 12 cases cover three widths, two clamp settings and count/cumulative group
lists, excluding routing, EPLB, MegaMoE, TP and full-model inference. The native
A5 launcher writes the probe result to files and starts the full model only
after the probe succeeds. Remove this hook when upstream handles the fused
alignment together with W2, following device and complete-model acceptance.

The connector-driven FFN forward scope also binds native current VllmConfig,
matching Ascend's configuration lifetime while retaining the real V2 platform
hook. Normal and exceptional exits restore the caller's configuration; this
scope owns no device buffers or communication state. The public config API
regression is in the existing compatibility runtime tests.

## Purpose and boundary

This document owns supported vLLM compatibility, compatibility adapters,
monkey-patch boundaries, upstream deltas, validation, and patch removal or
upstream plans. Backend mechanisms that do not alter upstream behavior belong
to [execution platforms](execution_platforms.md).

## Ownership and dependency direction

Compatibility code consumes the common plugin boundary and adapts pinned
upstream behavior for AFD roles. Role modules may rely on those adaptations;
patch modules must not become a general home for AFD-owned functionality.

## Supported upstream boundary

The package extra pins vLLM `0.30.0`, and
[`compat/vllm.py`](../../../afd_plugin/compat/vllm.py) enforces the same target.
Direct strict calls raise for a missing or different vLLM; plugin registration
calls the check with `strict=False`, so it warns and continues. This warning
policy does not make another vLLM release supported.

The NPU implementation is based on vLLM-Ascend commit
[`8d4409d62`](https://github.com/vllm-project/vllm-ascend/commit/8d4409d6256d8a6729140ddcc0d1889e3f96cdd6).
The repository does not declare a vLLM-Ascend package dependency in
`pyproject.toml`, so this source commit and the recorded NPU validation are the
compatibility evidence rather than a released package or container tag.

## Implementation evidence

| Area | Source | Focused validation |
| --- | --- | --- |
| Version policy | [`compat/vllm.py`](../../../afd_plugin/compat/vllm.py) | [`test_package.py`](../../../tests/unit/package/test_package.py) |
| Core vLLM patches | [`compat/patches/`](../../../afd_plugin/compat/patches) | [`tests/unit/compat/patches/`](../../../tests/unit/compat/patches) |
| CUDA benchmark routing | [`model_executor/routing_simulator.py`](../../../afd_plugin/model_executor/routing_simulator.py) | [`test_routing_simulator.py`](../../../tests/unit/model_executor/test_routing_simulator.py) |
| NPU adapters | [`compat/npu/`](../../../afd_plugin/compat/npu) | [`test_runtime.py`](../../../tests/unit/compat/test_runtime.py), [`test_ascend_ops.py`](../../../tests/unit/compat/test_ascend_ops.py) |
| NPU patch paths | [`compat/patches/npu/`](../../../afd_plugin/compat/patches/npu) | [`test_npu_runtime.py`](../../../tests/unit/v1/worker/test_npu_runtime.py) |
| CUDA native V2 DBO preparation | [`gpu_mrv2_ubatching.py`](../../../afd_plugin/compat/patches/gpu_mrv2_ubatching.py) | [`test_gpu_mrv2_ubatching.py`](../../../tests/unit/compat/patches/test_gpu_mrv2_ubatching.py) |

## Patch application lifecycle

The vLLM plugin entry point applies compatibility in this order:

1. perform the non-strict vLLM version check;
2. import `async_dp_engine`, `async_dp_forward_context`, `config_validation`,
   and `engine_core` in one best-effort block;
3. register the plugin-owned CUDA benchmark routing strategy as a required
   step;
4. register the plugin-owned DBO yield operator in a best-effort block;
5. register model mappings, which is required for registration to complete.

After Ascend platform initialization, AFD config normalization installs the
Ascend config factory and platform-config wrapper. The factory keeps the AFD
namespace out of native constructor arguments without changing the shared
source mapping. Synchronous Attention and FFN DP child entry points install
that factory before Ascend validates their config; async Attention keeps its
AFD engine entry point. NPU worker startup verifies those config patches and
installs the MLA graph resolver. CUDA-only plugin registration does not import
vLLM-Ascend.

To force-balance FFN-side experts, configure vLLM-Ascend's native
`additional_config.enable_force_eplb`. The native policy cycles through all
local routed experts. The Attention-side Async CAM `AFD_FORCE_BALANCED_TOPK_IDS`
switch is a separate AFD routing path.

Python module import provides process-level one-time execution in the ordinary
path. Some patches also preserve originals or set explicit sentinels, but this
is not consistent across the inventory. There is no transaction or rollback:
because core patch imports share one `try`, a failure can leave earlier patches
installed and later ones skipped. Registration logs these failures at debug
level and continues to model registration.

## Monkey-patch inventory

Every row describes current pinned behavior. `Guard` is the patch's own guard,
not the package dependency policy.

| Patch and upstream symbols | AFD delta and non-AFD path | Application, guard, and idempotence | Validation | Removal or upstream plan |
| --- | --- | --- | --- | --- |
| [`async_dp_engine.py`](../../../afd_plugin/compat/patches/async_dp_engine.py): `EngineCoreProc.run_engine_core`, `vllm.v1.engine.utils.launch_core_engines` and its imported client alias, `DPAsyncMPClient.add_request_async` | Async-DP Attention selects regular `EngineCoreProc`; the copied launch gates the DP coordinator's `enable_wave_coordination` to `is_moe and not is_afd_async_dp` so AFD async Attention keeps independent request-count publication while coordinator stats remain; `add_request_async` skips the DP-wave `FIRST_REQ` wakeup for AFD async-DP; other configs run the copied upstream branches. | Imported by `register_afd`; the installer keeps the target/dev guard. Its plugin-owned process target reapplies the engine-core loop in spawn children, and the Ascend late re-apply is scoped to AFD async Attention so vLLM-Ascend's own `run_engine_core` wrap is overridden only there. | [`test_async_dp_engine.py`](../../../tests/unit/compat/patches/test_async_dp_engine.py) covers coordinator-patch removal, async-DP Attention process selection, non-AFD MoE DP preservation, wave-coordination gating, target-version isolation, `FIRST_REQ` skip/preserve, backend-override rebinding, and reload idempotence. | Remove when vLLM exposes role-selectable async-DP engine scheduling, wave policy, and wakeup hooks. |
| [`async_dp_forward_context.py`](../../../afd_plugin/compat/patches/async_dp_forward_context.py): `vllm.forward_context.set_forward_context` plus already-imported worker aliases | Skips native `DPMetadata` construction/coordination only for AFD async-DP; otherwise uses the copied upstream flow. | Imported by `register_afd`; same target/dev/unknown guard. Rebinds known already-imported aliases so callers do not retain the old function. | [`test_async_dp_forward_context.py`](../../../tests/unit/compat/patches/test_async_dp_forward_context.py) covers async skip and non-async coordination. | Remove when vLLM supports a per-engine-role opt-out from native MoE DP metadata coordination. |
| [`config_validation.py`](../../../afd_plugin/compat/patches/config_validation.py): `EngineArgs.create_engine_config`, `VllmConfig.__post_init__` | For AFD-owned ubatching with a non-DeepEP backend, temporarily presents `deepep_low_latency` during upstream validation and restores the configured backend. After upstream platform normalization, maps an initial `worker_cls="auto"` to the role-specific CUDA or standard Ascend AFD worker. Native hybrid/multimodal DBO rejection remains intact, including Qwen3.5/3.6 GDN models. | Imported by `register_afd`; accepts the target version, development versions, or missing version metadata. Saves originals on upstream modules under AFD-specific attributes before installing wrappers. Explicit worker paths and non-AFD configs are not remapped. | [`test_config_validation.py`](../../../tests/unit/compat/patches/test_config_validation.py) covers backend relaxation, four role/platform mappings, explicit and non-AFD preservation, repeated validation, unsupported platforms, and dev versions. | Remove the backend branch when upstream validation distinguishes plugin-owned ubatching; remove worker mapping when vLLM offers plugin-owned role-aware worker selection. |
| [`engine_core.py`](../../../afd_plugin/compat/patches/engine_core.py): `EngineCore.__init__`, `_initialize_kv_caches`, `shutdown`; `EngineCoreProc.run_busy_loop`; `DPEngineCoreProc.run_busy_loop` | AFD FFN becomes a connector daemon: construct executor, skip scheduler/KV setup, return an empty KV-shaped result on late paths, start/monitor/stop the FFN worker loop, and use FFN-safe shutdown. Non-FFN branches copy pinned upstream behavior. | Imported by `register_afd`; Installs under the target/dev compatibility check (also permitting missing version metadata); no saved-original sentinel. Direct class assignment remains tied to the pinned upstream implementation. | [`test_engine_core.py`](../../../tests/unit/compat/patches/test_engine_core.py) covers FFN initialization, non-FFN behavior, and daemon start/stop; role runtime tests cover error propagation. | Remove when vLLM offers a headless connector-daemon engine lifecycle or an executor mode that does not require scheduler/KV ownership. |
| [`npu/ascend_platform.py`](../../../afd_plugin/compat/patches/npu/ascend_platform.py): `NPUPlatform.check_and_update_config` | Snapshots AFD DBO state, runs upstream normalization, and restores configured `enable_dbo`, `ubatch_size`, and `all2all_backend` in `finally`; non-AFD behavior is unchanged. | Installed through the config facade during AFD NPU normalization and verified again by the worker runtime facade; no version guard. Saves the original on the class and uses a class sentinel. The runtime facade caches success only after the wrapper is installed, so an early missing vLLM-Ascend import remains retryable. | [`test_runtime.py`](../../../tests/unit/compat/test_runtime.py) and [`test_npu_runtime.py`](../../../tests/unit/v1/worker/test_npu_runtime.py). | Remove when vLLM-Ascend recognizes plugin-owned DBO workers or no longer clears these fields. |
| [`npu/ascend_config.py`](../../../afd_plugin/compat/patches/npu/ascend_config.py): `vllm_ascend.ascend_config.init_ascend_config`, loaded factory aliases, and the AFD DP child entry point | Excludes the AFD namespace from native constructor arguments; preserves config identity, validation, singleton publication, and cache refresh. DP children install the factory before native Ascend initialization. | Installed by the NPU config facade; rebinds already-loaded aliases. The DP entry point is selected only for AFD synchronous Attention or FFN. | [`test_ascend_config_lifecycle.py`](../../../tests/unit/compat/npu/test_ascend_config_lifecycle.py), [`test_ascend_engine_core_binding.py`](../../../tests/unit/compat/npu/test_ascend_engine_core_binding.py). | Remove when Ascend accepts plugin namespaces and loads general plugins before config validation in every DP child. |
| [`npu/mla_graph.py`](../../../afd_plugin/compat/patches/npu/mla_graph.py): `vllm_ascend.attention.mla_v1.get_graph_params` | Resolves an AFD ubatch-owned `GraphParams` registry from the active forward context and otherwise delegates to the saved upstream process-global resolver. | Called through `apply_afd_ascend_patches_if_needed` during NPU worker startup; no version guard. Saves the original on the upstream module and uses a module sentinel. | [`test_runtime.py`](../../../tests/unit/compat/test_runtime.py) covers AFD-context resolution, upstream fallback, and idempotence; [`test_npu_mla_graph.py`](../../../tests/unit/v1/worker/test_npu_mla_graph.py) covers the owning graph lifecycle. | Remove when vLLM-Ascend accepts a forward-context-local MLA graph registry or exposes an equivalent resolver hook. |
| [`npu/mrv2_ubatching.py`](../../../afd_plugin/compat/patches/npu/mrv2_ubatching.py): native ubatch factory, dispatch, child DP metadata, input slicing, context and stream aliases | Builds `NPUUBatchRunner`, enables Async CAM DBO at DP1, omits Async child DP coordination, applies native truncation to Ascend CPU sequence lengths, preserves model-specific PP slices/nested auxiliary outputs, and provides NPU stream/failure handling during eager and capture. Native MRV2 owns execution and graph lifecycle. | Scoped to AFD initialization, preparation or execution; every replaced alias is restored in `finally`. The scope temporarily changes module symbols, so it is not an independent concurrent-runner API. | [`test_npu_model_runner_v2.py`](../../../tests/unit/v1/worker/test_npu_model_runner_v2.py), [`test_async_cam_dbo.py`](../../../tests/unit/model_executor/test_async_cam_dbo.py), [`test_mrv2_dbo.py`](../../../tests/npu/test_mrv2_dbo.py). | Remove when native vLLM/Ascend supports NPU child builders and streams, optional child DP metadata, DP1 DBO and peer cancellation. |

## Non-patch compatibility adapters

These modules adapt upstream behavior without replacing a global symbol:

| Adapter | Current purpose |
| --- | --- |
| [`model_executor/routing_simulator.py`](../../../afd_plugin/model_executor/routing_simulator.py) | Registers the opt-in `afd_balanced` strategy through vLLM's public `RoutingSimulator` interface. Select it with `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=afd_balanced`; `AFD_BENCHMARK_FORCE_LB_TOPN_PER_RANK` optionally limits each rank's expert pool. It provides deterministic source-rank phases and normalized weights for controlled CUDA performance tests without replacing router internals. EPLB remains outside the validated scope. |
| [`compat/npu/runtime_config.py`](../../../afd_plugin/compat/npu/runtime_config.py) | Preserves model SP while applying the non-SP backend fallback; normalizes Async CAM native ubatching to Attention DBO, checks thresholds, and derives model/PP tensor layout and active ubatch count. |
| [`compat/npu/feature_validation.py`](../../../afd_plugin/compat/npu/feature_validation.py) | Parses connector-owned typed extra information through the factory and fails before execution for unsupported NPU connector, quantization, graph, DBO, gate, or async MoE combinations. |
| [`compat/npu/forward_context.py`](../../../afd_plugin/compat/npu/forward_context.py) | Enters the pinned Ascend forward context for connector-driven FFN compute and installs AFD metadata in `additional_kwargs`. V1 uses `set_ascend_forward_context`; a V2-paired FFN uses native `set_forward_context` plus the Ascend MRV2 profile override so platform state follows the V2 proxy layout. |
| [`compat/npu/ops.py`](../../../afd_plugin/compat/npu/ops.py) | Discovers plugin-owned CAMP2P and routed-only async CAM operators with explicit missing-runtime errors; operator build/runtime ownership is documented in [execution platforms](execution_platforms.md). |

Several scoped upstream adaptations live with their semantic owners rather
than this patch directory. Model dummy runs and V2 execution temporarily wrap
`create_forward_context`; CUDA worker startup temporarily substitutes the
native V2 runner symbol; V2 graph capture temporarily wraps native input
preparation; and full-graph replay installs an instance-local manager hook.
Every scope restores symbols and runner state in `finally`. Connector process
groups also call private PyTorch/vLLM helpers as described in
[connector contracts](connector_contracts.md). All of these seams must be
included in an upstream upgrade audit even though they are not process-global
import-time patches.

The NPU V2 runner additionally scopes `_use_camp2p_dp_padding()` around native
dispatch for many-to-one CAMP2P: eager and graph-fallback token counts are
padded consistently within FFN receiver groups before input preparation.
It delegates graph selection and DP agreement to the pinned native function.
Replace this seam when upstream exposes a pre-input padding callback.

Async-DP queue statistics are a separate audit boundary. Current AFD selects
ordinary `EngineCoreProc`, retains the native coordinator/stat channels, and
disables wave coordination. Existing DPLB tests exercise native routing with
supplied counts;
they do not prove end-to-end publication from a running async Attention
scheduler. Audit the v0.30 publisher path before adding another queue-statistics
patch.

## Patch review and refresh procedure

For every new patch or upstream upgrade, reviewers must:

1. identify the exact upstream file, symbol, version/tag, and signature;
2. confirm that the patch targets the owning upstream layer rather than hiding
   AFD-owned behavior in a convenient global hook;
3. copy the pinned upstream function when feasible and mark only AFD deltas
   with `# ### PATCH START` / `# ### PATCH END`;
4. document any original-function delegation exception immediately above the
   function;
5. test both the selected AFD branch and the non-AFD/upstream branch, including
   initialization, failure, shutdown, and idempotence where relevant;
6. assess hot-path, graph/compile, memory, and distributed-lifecycle impact;
7. record an upstream contribution or concrete removal condition.

When the pinned upstream version changes, copied bodies must be refreshed from
that version before reapplying marked AFD deltas. Passing unit tests against a
different signature is not sufficient compatibility evidence.

## Failure visibility and resource ownership

- Patch modules own symbol replacement only; the patched engine or connector
  continues to own its runtime resources.
- Patch import errors are currently best effort at plugin startup and may be
  visible only in debug logs. Runtime paths must therefore fail explicitly if
  a required adaptation is absent rather than silently serving incorrect
  results.
- Copied non-AFD branches are intended to preserve pinned upstream behavior;
  they still require regression tests because global replacement affects every
  process that loads the plugin.
- A patch that saves an original callable must never overwrite that saved
  original with its own wrapper on reload.
- Patch removal requires deleting registration/application code and focused
  tests only after the replacement upstream behavior is part of the supported
  pinned stack.

## Candidate invariants

The following RFC candidates are non-normative while this document is draft:

- `PATCH-INV-001`: new or modified patches follow `AGENTS.md`, including
  upstream context, marked AFD deltas, validation, and a removal or upstream
  plan.
- `PATCH-INV-002`: a global patch preserves the pinned non-AFD path and has a
  focused regression test for it.
- `PATCH-INV-003`: unsupported upstream versions are not treated as supported
  merely because best-effort registration continued.
- `PATCH-INV-004`: patch initialization is idempotent or explicitly guarded,
  and partial application is observable before the affected runtime executes.

## Upstream relationship and validation requirements

Patch signatures and copied code must match their pinned upstream symbols.
Every modification requires focused tests for AFD and non-AFD behavior plus
the architectural review required by `AGENTS.md`. Version guard changes also
require package compatibility tests. Ascend patch refreshes require an exact
vLLM-Ascend source reference and the NPU unit/E2E evidence for the affected
path.

## Limitations and open issues

The inventory covers current production patch files, but the document remains
**draft** because:

- `engine_core` has a target/dev version check but no saved-original
  idempotence sentinel; scoped native DBO adapters rely on pinned private APIs;
- best-effort grouped imports can produce a partially applied patch set;
- the exact vLLM-Ascend source commit is documented but not pinned as package metadata;
- owners have not approved the candidate invariants as normative contracts.

Runtime refactor decisions in
[#86](https://github.com/JiusiServe/afd-plugin/issues/86) may remove or relocate
patches. The documentation RFC remains tracked by
[#129](https://github.com/JiusiServe/afd-plugin/issues/129).
