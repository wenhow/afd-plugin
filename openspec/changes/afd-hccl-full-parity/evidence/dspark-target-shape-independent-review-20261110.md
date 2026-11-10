# Independent review: DSpark target shape, phase one

No source blocker was found in the seven files below. The reviewer read the
three runtime changes, the shared CPU worker/loader boundary and all new
mirror cases. This increment preserves the original target width through F
role isolation and consumes it in the native MC2 capacity helper. It does
not admit DSpark Graph serving or change a protocol/capture key. The reviewer
changed no runtime, test, upstream tree or device state.

| File | Reviewed SHA256 |
| --- | --- |
| `afd_plugin/compat/npu/runtime_config.py` | `79f7041a41aef55129f094a1a0b7954d50b80b0285edbb25f0eb103f25569d40` |
| `afd_plugin/v1/worker/npu/ffn_worker.py` | `685d2afad186a5236f791a972e6a79b5c22b13f109b991f8fb74f49d6864cc73` |
| `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `dce2aa725607f9e2d2453ab6e557e309f2a98b049642cfcd1c89f7d6d54a197d` |
| `tests/unit/fakes/ffn_model_runner.py` | `92884402ae4dc06e8583af2c7964f97bbc5690e8108bfbfaefa3d8176c28910f` |
| `tests/unit/compat/npu/test_runtime_config.py` | `59bbd7512bb9cc94c86de5b3d784d4d950dca97e926ea604244af8690703edfd` |
| `tests/unit/v1/worker/npu/test_ffn_worker.py` | `1b7442c2b26f0904052bcb4b236f66a1ec6f581354bc30fab0b8a0cdd3c84beb` |
| `tests/unit/v1/worker/npu/test_ffn_model_runner.py` | `abb127408b91f09cda03e778036b16cda39d919293dad4931de6236e5b95b7b9` |

## Pinned source and concrete consumer

The reviewer read Git objects at the locked revisions, without switching or
modifying either upstream checkout:

| Lock and source | Exact contract |
| --- | --- |
| vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, `vllm/v1/worker/gpu/model_runner.py:276` | The target's `num_speculative_steps` comes from `vllm_config.num_speculative_tokens`; the generic runner derives its decode width after creating ModelState at line 439. File SHA256 `174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce`. |
| Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`, `vllm_ascend/worker/v2/model_runner.py:257-262` | The selected NPU runner explicitly uses `self.num_speculative_steps + 1`, then passes that width to MC2 capacity and mask setup. File SHA256 `a3471a9f181b45742b8f985c86137cdaafc2871d2ef6f96aa6a45236a242969b`. |
| Ascend same lock, `vllm_ascend/ascend_forward_context.py:243-278` | `set_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len)` selects scheduler capacity for prefill MC2/MegaMoE, capture capacity when configured, otherwise `max_num_reqs * uniform_decode_query_len`; it applies native TP ceiling and kernel limits, then exposes the result through `get_mc2_tokens_capacity()`. File SHA256 `848112214c9f56371b4bf43a59d39fadbed0e0e864581e300a3a89778d4083cb`. |

`resolve_deepseek_v41_target_shape` first identifies the exact V4.1 family
and runs the existing same-checkpoint DSpark validator. Its new frozen,
slotted value contains only the positive integer target query width, equal
to one without speculation or N+1 for validated speculative steps. It rejects
bool/noninteger/nonpositive widths and a width above native token capacity.
It does not retain a draft config, draft model, HF draft block size, request
state, Engram buffer or mutable configuration child.

`AFDNPUFFNWorker.__init__` resolves that value from the original config
before `get_npu_ffn_config` removes F's Engram, speculator and private native
KV transfer bindings, and before delegating to the native worker constructor.
The existing signature and marked isolation boundary remain intact. Later,
`init_device` passes this same value through the plugin-owned factory's new
documented keyword-only argument to the synchronous FFN runner.

The runner rejects a target shape for a foreign model or an excessive width;
when the original speculative config is still present it also checks that
the retained value agrees with it. An already isolated config cannot recover
N, so the retained immutable value is the explicit dependency. `load_model`
uses its width in the actual native capacity helper before native model
construction. It does not copy that helper, initialize a draft runner or
restore the speculator on F. For non-V4.1 models the resolver returns None
and the pre-existing speculative sizing path remains in place. The AsyncCAM
factory branch is unchanged; this increment's propagated consumer is the
synchronous runner.

## Meaningful CPU boundary and pending evidence

The shared fixture runs the actual plugin worker constructor, factory,
synchronous runner constructor and `load_model`. It substitutes native
device initialization, feature admission, connector/device allocation and
checkpoint loader boundaries. The loader allocates a CPU routing workspace
using **the output of the actual locked native capacity getter**. Tests
assert that final workspace extent for target N values, native TP1/TP2 and
decode/capture/scheduler capacity branches. They do not assert the arguments
or call count of an internal capacity helper. Source speculator and Engram
configuration remain unchanged; the config consumed by the F model has
neither binding. Frozen/value-error and foreign-model boundaries are tested
separately.

These fixtures do not execute an NPU MC2 kernel, construct a native checkpoint
model or prove distributed capacity agreement. They reset the native
capacity singleton through monkeypatch for deterministic CPU isolation and
leave the capacity implementation itself installed. The owning agent is
running a separately frozen actual-dependency A3 CPU snapshot; no result is
claimed here before its raw identity and XML are available.

### Actual-dependency CPU follow-up

The reviewer subsequently parsed each immutable identity, summary and JUnit
XML under `a3-dspark-shape-cpu-20261110`, preserving the failed attempts:

| Attempt and manifest | Actual result |
| --- | --- |
| `component-001`, `c8285ef70c6ec5bbae7c3b0fe5819e64620f2db00842077074f24955a324ea7e` | 263 passed, ten failed, no skips/errors, 4.669 s. The four core modules passed in full: config 97, NPU F worker 47, NPU F runner 19 and existing runtime 55, totaling 218. All ten failures were in the additional old Async startup module, whose other 45 cases passed. |
| `component-002`, `831e7d8fc42e052ac8d2e599e532b9e9d42f9300b4160c2fd99957dac83c3d8e` | Only the affected Async startup module: 53 passed, two failed, no skips/errors, 3.700 s. |
| `component-003`, `ff2f5763ac7d3644555e0a73f08df3f4ce4b1aba56df4225c5718bd6cb70170f` | The complete affected Async startup module: 55 passed, zero skips/errors/failures, 3.675 s. |

The additional failures were stale CPU construction/profiler fixtures; their
test-only fixes supplied the actual unbound worker/session lifecycle fields
instead of adding obsolete production aliases or weakening a binder. The
three production hashes above did not change. All three attempts matched
eleven installed source pins and recorded no device probe. This is a
directed test-fixture closure, not a single fabricated 273-test green rerun.
The eighteen worker cases execute the native capacity-output/workspace seam
described above; their success does not qualify NPU MC2, DSpark sampling or
Graph serving. This review is frozen after the follow-up for scoped delivery.

## Phase-two contract list; no admission change in this review

The next increment must preserve target q as one session invariant and use
it in every geometry/cache/owner consumer. Merely deleting q1 checks would
leave inconsistent request and token dimensions. Exact remaining locations:

| Location | Required contract before DSpark Graph can be selected |
| --- | --- |
| `distributed/hccl_plan.py:HcclPlanCapabilities` and its HELLO codec; `connectors/npu/hccl_config.py` | Agree strict positive target q between A's original validated config and F's retained `target_shape`; preserve default q1 bytes or explicitly version the new optional field. Reject mismatched peers before QUERY/capture. Leave existing graph/spec and native-model qualification guards closed until the consumers exist. |
| `distributed/hccl_graph.py:HcclGraphKey.__post_init__` | Replace only the fixed q1 eligibility with an agreed uniform q contract: `num_tokens == num_reqs * q`, uniform/max-query fields and exact request/token stage geometry. Keep digest, request-row metadata shapes, ordered capture-sequence agreement and strict no-LoRA scope. |
| `v1/worker/npu/hccl_v41_graph.py:_uniform_decode`, `_validate_full_descriptor`, `preview_hccl_v41_geometry`, `capture_hccl_v41_geometry`, `validate_hccl_v41_capture_coverage`, `dispatch_hccl_v41_graph` | Native target q replaces `DECODE_QUERY_LEN=1`, `num_reqs==num_tokens`, `np.ones` dummy counts and q1 canonicalization. Real ordered scheduled counts and native `maybe_create_ubatch_slices` plus request clamping remain authoritative. Capture request count is T/q, not T; U2 may split at a token coordinate inside a q-wide request, so exact native stage geometry must still be compared. |
| `v1/worker/npu/hccl_v41_graph_control.py:key_for`, `reserve_capture` | Normalize a native `max_query_len=None` to the agreed target q; remove the separate hardcoded 1 from duplicate native-descriptor comparison. Local manager hits, authoritative peer preview, storage generation and final prepared geometry must all retain that q. |
| `v1/worker/npu/hccl_v41_graph_coordinator.py:__init__`, `_capture_scope`, `capture_native`, runtime/cache resource validation | Admit only validated same-checkpoint target DSpark at the explicit seam. Keep the real target manager's q-sized descriptors, ordered auxiliary hidden outputs and target metadata storage identity. Crucially, pinned native `GPUModelRunner.capture_model:1043-1045` captures the local speculator after target capture: no target reservation/provider may accidentally send F control for that A-local drafter. |
| `v1/worker/npu/hccl_v41_graph_service.py` and `ffn_model_runner.py` binding | Keep F `speculative_config`/Engram absent; compare wire q with `runner.target_shape`, token capacity and retained cache resources rather than reconstructing an F drafter. A cleared F config is not evidence that the target originally used q1. |
| `model_executor/models/npu/deepseek_v41_graph.py` and `v1/worker/npu/hccl_v41_graph_execution.py`; runtime/capture tests | Prove q-wide native Engram history/query buffers, token-indexed masks, same shared-KV/HC state, and ordered target auxiliary outputs under warmup/capture/replay/early eager fallback. The existing tensor payload rows remain token rows; no ID/FP32 recreation or extra native TP reduction is authorized. |

Pinned native `GPUModelRunner.sample_tokens:1997` owns acceptance/rejection
and calls local `speculator.propose` at line 2143. Target phase retirement
must occur before that continuation, while draft capture/proposal remains
entirely A-local and emits no F FORWARD. Ordinary MTP, DSpark draft block
size and target N+1 are separate contracts. Adaptive verification with U2,
PD+DSpark, Graph PD and unsupported topologies remain rejected. The general
V4.1 feature guard in `compat/npu/feature_validation.py:35-40` is unchanged.

The phase-one cost is one small immutable object per V4.1 F worker and a
constructor dependency; no per-token operation or new device buffer is
introduced. Remove the value handoff if upstream supplies a target-only
scheduling descriptor after role isolation. Graph q propagation requires
the separate source/CPU/protocol/device gates above; this review does not
declare it implemented or accepted.
