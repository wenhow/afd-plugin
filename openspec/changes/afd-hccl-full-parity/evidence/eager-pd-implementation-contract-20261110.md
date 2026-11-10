# C7 eager PD implementation contract, 2026-11-10

Status: the initial source audit and proposal below preceded implementation.
The final section records the later, narrowly authorized plugin/configuration
increment and its separate validation scope. No upstream trees, package
installations, device jobs or task checkboxes were changed for either step. The preceding
[DSpark / PD audit](dspark-pd-source-contract-20261110.md) describes the broader
owner chain; this follow-up checks whether its native SFA candidate is actually
admissible for V4.1 before changing an AFD guard.

## Sources and current comparison

All upstream files were read from immutable Git objects:

| Label | Repository | Commit |
| --- | --- | --- |
| V | `D:/openSource/vllm` | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| A | `D:/openSource/vllm-ascend` | `7e37e29a7803d6897a7113088e2b771f83205bc0` |
| D | `D:/openSource/afd-plugin/afd-plugin-phase1-delivery` | `4ddd69a391dfb5b3e362c7601399e538a31de702` |

Current plugin comparisons are working-tree snapshots, not an assertion that
these files equal HEAD or a previously tested manifest:

| Path | SHA-256 |
| --- | --- |
| `afd_plugin/compat/npu/runtime_config.py` | `304e31b3275378571b161282907aeb4a09dc422ee13ee03f90dbd335f5e99805` |
| `afd_plugin/connectors/npu/hccl_config.py` | `ceb1f99377f84eff6e223cb14e218c7862e99af3d1114e6013ac71b769683bc7` |
| `afd_plugin/v1/worker/npu/hccl_v41_execution.py` | `0e91c47b3d52bdc55b1580808b468d18074b349ea6c86c66904214243bb3ffab` |
| `afd_plugin/v1/worker/npu/attention_model_runner_v2.py` | `139c58d443fb96cf63d205e9c2dfbd1b3ae19f2c3d956e79b24d19946cf359a0` |
| `afd_plugin/compat/npu/feature_validation.py` | `f36266695f24c04f977fb8df5bccc4f3c9b4525e0ff5be311aa4e305bb96e4ff` |

Concurrent C6 development can change other files. No result below promotes
those working-tree bytes to device acceptance.

## Native lifecycle already retained

V `vllm/v1/worker/gpu/model_runner.py:1631` owns `execute_model(self,
scheduler_output, intermediate_tensors=None, dummy_run=False,
skip_attn_for_dummy_run=False, is_profile=False, context_len=0,
valid_dummy_state_slots=False) -> ModelRunnerOutput | IntermediateTensors | None`.
Its zero-scheduled-token path at 1641-1654 updates request/block state then calls
`kv_connector.no_forward`; it does not run a model. Normal eager execution calls
`pre_forward` inside the real parent context, invokes the native selected UBatch
runner or raw model, then calls `finish_forward` (approximately 1820-1908).

V `vllm/v1/worker/gpu/kv_connector.py:55-142` owns the actual active connector:

- `pre_forward(self, scheduler_output, **kwargs) -> None` binds native metadata
  and starts synchronous loads before computation.
- `finish_forward(self) -> None` delegates native connector completion.
- `post_forward(self, finished_req_ids, wait_for_save=True) -> KVConnectorOutput
  | None` starts deferred asynchronous loads, collects native success/failure
  results, and clears metadata.
- `no_forward(self, scheduler_output) -> ModelRunnerOutput` performs the same
  maintenance without a model context or A/F plan.

The current plugin `HcclV41Execution.install_context` ignores contexts without
a `batch_descriptor`. Its existing U1 runner metadata hook independently ignores
the native PD maintenance context with neither descriptor nor attention
metadata. Neither path should fabricate an empty F target merely because native
KV maintenance created a ForwardContext.

A `vllm_ascend/worker/v2/model_runner.py:486-553` retains the native outer
`execute_model` signature. It handles preemptions, delegates the native target,
calls `model_state.finish_execution(failed=...)` once in its `finally`, then
performs `_maybe_send_draft_kv` at 555-581 where selected. The AFD execute scope must remain
inside this delegation and must not copy request/cache/history retirement.
Source `gather_batch_req_state` at 615-638 also owns PD decode-recompute request
classification (615-636); eager U2 must use its returned actual request order and slices.

F is an activation/MoE owner. The concrete missing isolation is that
`get_npu_ffn_config` currently removes `engram_config` and `speculative_config`,
but retains `kv_transfer_config`. `AFDNPUFFNWorker.__init__` calls this helper
before `NPUWorker.__init__`, so this is the correct pre-construction seam to
remove F's PD connector ownership. Native A `worker/worker.py:1181-1202` calls
`ensure_kv_transfer_initialized` before runner KV initialization. Merely preventing
an F forward does not establish that a leaked native connector was never built.

## History handoff is a native request contract

V scheduler `vllm/v1/core/sched/scheduler.py:1324-1334` merges resumed requests
into the V2 new-request list and passes `req._all_token_ids` to
`NewRequestData.from_request`. V `sched/output.py:53-76` preserves this as
`prefill_token_ids`; runner `model_runner.py:1120-1158` places it together with
`num_computed_tokens` in native request state. A Engram
`models/deepseek_v41/engram/model_state.py:99-154` gathers lookback from the same
`req_states.all_token_ids.gpu` and computed lengths. Its `finish_execution` at
65-72 owns lookup retirement.

No plugin Engram tensor needs to be invented in PD metadata. The real proxy/API
ingress must nevertheless supply the full transferred/computed token prefix,
including any generated or resumed prefix. Source examination of the internal
V2 scheduler-to-runner path cannot prove an external ingress supplied those IDs.
The actual request/history check remains a device and service qualification item.

## Confirmed incompatibility: SFA is not a V4.1 shortcut

A `sfa_pd_rd2h/connector.py:48-110` provides `SfaRemoteD2HConnector` and requires
the producer's sparse offload disabled and the consumer's sparse offload enabled
(83-90). A `ascend_config.py:1756-1763`,
`SparseKVOffloadConfig._validate_preconditions(self, vllm_config) -> None`,
immediately rejects a text configuration containing `compress_ratios`.
V4.1 requires this field: A `models/deepseek_v41/model.py:522-573`,
`build_layer_plan(config) -> DeepseekV41Topology`, reads it unconditionally.
The independent packed-main-cache restriction at `ascend_config.py:1041-1047`
is an additional boundary, not the reason to bypass the compression guard.

Consequently this exact pinned combination cannot be admitted by changing only
an AFD PD predicate. The native SFA DSpark functions remain useful owner/source
contracts, but their existence is not evidence of a runnable V4.1 path. Deleting
checkpoint fields, silently disabling D offload, enabling debug
`keep_device_kv_cache`, or reopening an upstream guard would not repair it.

## Candidate for the first eager increment

The native registered `MooncakeHybridConnector` is the relevant first candidate
for compression/shared pages. Its A source is
`distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`:

- `_reconstruct_shared_pages(kv_cache_config, kv_caches) -> list[_SharedPage]`
  at 114-175 reconstructs shared/strided physical pages from real cache views.
- `MooncakeConnectorWorker.register_kv_caches(self, kv_caches)` at 1886-2037
  handles compression layouts at 1931-1966, verifies shared page base addresses,
  registers configured physical regions, and publishes native metadata.
- `_get_prefill_decode_size(self, vllm_config)` at 1864-1884 consumes real
  `prefill`/`decode` TP/DP/PP values from native connector extra config.
- `KVCacheRecvingThread._transfer_kv_cache_all_groups(self, req_meta)` at
  764-884 still aligns logical group IDs and zips address arrays at 838-870.
  Reconstructing each local physical layout does not validate a heterogeneous
  remote layout. Remote metadata does not carry logical layer/component names.

The minimal config-only software proposal is therefore **controlled native
Mooncake eager PD with speculation off first**, alongside F isolation. Retain
the closed global V4.1 feature guard and graph-PD rejection. Limit the native KV
connector to exactly `MooncakeHybridConnector`, no custom module override,
`MultiConnector`, SFA or `kv_both`. Require separate producer/consumer roles,
matching configured P/D TP, DP=PP=CP=1, and the existing native-TP paired A/F
constraints; disable local prefix caching initially. Preserve native KV load
failure handling, cache dtype/layout choices and external request identity.
These config checks are eligibility, not proof that a remote checkpoint/layout
actually matches; that requires actual layout collection and P/D qualification.

Proposed immediate-consumer API in `compat/npu/runtime_config.py`:

```python
def validate_deepseek_v41_pd(vllm_config: VllmConfig) -> None:
    """Validate the controlled HCCL eager native-PD eligibility boundary."""
```

Call it from HCCL runtime validation before rendezvous and from
`get_npu_ffn_config` **before** stripping the original target fields. The latter
then shallow-copies VllmConfig and clears F's `kv_transfer_config` together with
the existing A-only state; preserve original target config, ModelConfig identity,
native parallel scheduling and non-HCCL behavior. Direct F runtime config with a
remaining transfer connector must still fail. Both U1 and staged U2 use the same
PD rule; connector selection must not depend on the already stripped field.

Do not introduce an unused wire schema or replace the native KV connector.
No upstream patch, new management thread or F KV handshake is required for this
first eligibility/isolation increment. The actual A runner is already the
consumer of native request, load, target forward and retirement APIs.

## DSpark requires a distinct prompt-draft proof

The existing same-checkpoint DSpark validator proves model metadata, equal
target/draft TP, ordered aux boundaries and non-synthetic rejection; it does not
prove that D has the P-computed prompt draft KV.

Do not assert that native P never writes draft KV: V
`gpu/model_runner.py:2143-2184` calls native `speculator.propose` before
`kv_connector.post_forward`. V `gpu/spec_decode/dflash/speculator.py:300-427`
pre-inserts current target context KV in the draft cache before generating query
tokens. However that function only consumes this step's target rows. It does
not reconstruct a previously transferred prompt from token IDs. The native SFA
special path explicitly projects all prompt chunks, checks allocation generation
and contiguous offsets, and joins target/draft readiness; generic Mooncake has
no such model-specific named transfer or readiness implementation.

A `models/deepseek_v41/cache_config.py:147-228` also puts `.mtp.` draft SWA
resources in physical slot tuples and a separate logical draft group. Enabling
drafts on only D can therefore change the ordered physical-page and logical-group
contract. A generic address zip must not silently substitute for that proof.

Before admitting DSpark+PD, establish all of these with an actual consumer:

1. Both loaded P/D draft models expose matching logical draft cache layers,
   dtype/block geometry and sufficient blocks; P generated the whole prompt
   draft context, not just the final decode query.
2. Published transfer readiness happens after those device writes are complete,
   and target, index, compressor/SWA and draft pages match at every physical
   placement. Named matching or an explicit identical-layout contract must be
   validated before a potentially truncating zip.
3. D admits a request only after both target and prompt draft KV are ready for
   the same external request/allocation generation on all TP ranks. Failure or
   stale/retry handling cannot mark it ready early.
4. The AFD target transaction retires before unchanged native sampling and local
   drafting; no draft FORWARD is sent to F, and only native post-update writes
   accepted history/computed lengths.

The first native-PD increment should reject DSpark+PD explicitly until this
contract is implemented/validated, rather than weakening the existing DSpark
validator or claiming the SFA code solves V4.1. A plugin-owned native-connector
subclass/adapter with a real layout/readiness consumer is a possible subsequent
increment; the exact patch/override must be reviewed against this pinned class,
not resurrected from the donor v0.23 monkeypatch.

## Focused verification and cost

Expected CPU/API seams for the first increment: A original KV config survives;
F receives `kv_transfer_config=None` before native construction; unsupported
roles/connectors/module overrides/asymmetric topology/Graph/speculation fail
before isolation; same valid A/F U1/U2 target capabilities are retained; native
KV-only steps publish no A/F parent and leave F waiting for its next real target;
mixed eager target slices preserve native aux, raw IDs and independent FP32.
Use the existing shared configuration and execution fakes; do not rebuild the
native scheduler/model graph or simulate PD device acceptance in unit tests.

Then qualify actual P/D cache layouts, prefill/decode request/history, retries,
failure-policy handling, cancellation and following requests, with the same
checkpoint/tokenizer and native comparison. Graph PD remains pending throughout.
No tests were run for this source-only proposal.

The proposed validator adds bounded startup configuration checks and one existing
private VllmConfig copy. No per-token wire, readback, additional KV copy, remote
F cache, new mutable global or duplicate native history update is proposed.
Remove the narrow eligibility restriction after native/plugin PD layout and
model qualification covers a wider connector/topology; keep F role isolation.

## Source file hashes

| Pinned file | SHA-256 |
| --- | --- |
| V `vllm/v1/worker/gpu/model_runner.py` | `174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce` |
| V `vllm/v1/worker/gpu/kv_connector.py` | `b89a4333f1e552d86e1815674b59e9f1335f0f49c0a7b46082439076f62c6a81` |
| V `vllm/v1/core/sched/output.py` | `3a66c8efaefe10c572f6f33cf1f8aa4610758c945130cad62eb9f7331005677c` |
| V `vllm/v1/core/sched/scheduler.py` | `c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884` |
| V `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | `f9c100f42f9326e48eb84e89281897e859ba761943f54d593523555ccd98dc63` |
| V `vllm/config/kv_transfer.py` | `506fae7a9d7f32cb3646156cf9249a856f6317dff6c4c85db70efd34418c429b` |
| A `vllm_ascend/ascend_config.py` | `1a6b5f644dc412c35c112c47cfbf9649fec00af7455a1e1e6dc27be5cee2a676` |
| A `vllm_ascend/worker/v2/model_runner.py` | `a3471a9f181b45742b8f985c86137cdaafc2871d2ef6f96aa6a45236a242969b` |
| A `vllm_ascend/worker/worker.py` | `51cfb3da18046af8c72ccefe940de5f24d9502bfdae75b941e1798c0a165cda4` |
| A `vllm_ascend/models/deepseek_v41/model.py` | `989aec6db1e344dca76159dcbef0d5876874facedaa36b47e993146d5b8a2b9f` |
| A `vllm_ascend/models/deepseek_v41/cache_config.py` | `467dddca82e2528ebbec14c78682486a39bffb4f096675d8b1d2d140f1a7014b` |
| A `vllm_ascend/models/deepseek_v41/engram/model_state.py` | `677cb13bb86ad80a154e0ff9a43620aa3344e7e93170a05e27aba0cc1f10aa8b` |
| A `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/connector.py` | `ed7722a252ff647ca39c011c9f923889d80402ceab1c781f0e372152a7e7a946` |
| A `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_context.py` | `296bd051f967d34a21f310cd64561f71819af96d423ff0edceaaeed68e65f93a` |
| A `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py` | `a6a9c3a870b7d577fb2a39435067fe378646b6ab39bc669140641beba92bdf84` |

## Plugin implementation follow-up

Base HEAD at this increment is `24c8d51df5d43d57dc8b4d3de0833fe592b42aa5`.
The following working-tree hashes freeze the implementation; they are distinct
from the proposal's earlier comparison table:

| Path | SHA-256 |
| --- | --- |
| `afd_plugin/compat/npu/runtime_config.py` | `9c6083e6b309d8c60eecc90fc4051648fb43737301294c0bb2ce17b702875e45` |
| `afd_plugin/connectors/npu/hccl_config.py` | `8f947e0f35bd79c9cf520707bdb177c321132896dbfcf7c6025b480c086a9b7d` |
| `tests/unit/compat/npu/test_runtime_config.py` | `df4890c36d49007f30f53f35c92d98c65cf221742ef1fa0c0afe5e9842d64218` |
| `tests/unit/connectors/npu/test_hccl_config.py` | `df99b91b36c2a5d8fde9e4ec9302685060356b13e2785d38879ebaa3e01b98d4` |
| `tests/unit/fakes/vllm_config.py` | `81c51dfb65d7db7a8e57243893509b484763722f9c0e3f1403e5e04c1a599dee` |

`validate_deepseek_v41_pd` now checks the actual native `KVTransferConfig`
fields. The pinned `vllm.config` publicly exports that class. The validator reads
the same `kv_connector_extra_config["prefill"/"decode"]` TP/DP/PP entries that
the pinned Mooncake worker consumes at 1864-1884; no plugin-only layout claim is
substituted for those native fields. Equal configured layout is local
eligibility only. Actual P/D model identity, physical register regions, ingress
history and device readiness remain unqualified.

Only the exact native MooncakeHybrid connector, separate producer/consumer,
eager/no-Graph, no-speculation, no-prefix-cache, no-sparse-offload and
PP/DP/PCP/DCP=1 combination is eligible here. The existing HCCL topology,
activation, compilation and capacity checks still apply. SFA, custom/multi
connectors, `kv_both`, asymmetric configured TP, DSpark+PD and Graph PD are
rejected. The global V4.1 model feature guard is unchanged.

For HCCL F construction, the original target is validated before creating the
private shallow config. F then receives `kv_transfer_config=None`, together with
the existing Engram/drafter isolation. Native ModelConfig and ParallelConfig
objects retain identity, and A's original config is unchanged. Other NPU
connectors retain their existing KV-config behavior. The native A scheduler,
KV connector, shared-cache registration, Engram/history and request retirement
are not patched.

The public seam tests cover source preservation, private F ownership, valid
A/F capability parity with actual native `KVTransferConfig`, and rejection
before isolation for unsupported fields/layout/speculation. The native-type
cases run only when torch/vLLM are available; their local dependency skips are
not evidence that those behaviors executed.

Local execution used `.venv/Scripts/python.exe` (Python 3.14.3) and
`-m pytest tests/unit/compat/npu/test_runtime_config.py
tests/unit/connectors/npu/test_hccl_config.py -q -ra`. Exit code was zero;
stdlib cases passed and torch-dependent cases skipped because torch is absent.
The original double-quiet terminal result did not print a numeric pass total,
so none is reconstructed here. Separate collection reported 215 collected
cases; collection does not demonstrate execution. Ruff check, Ruff format check,
Python 3.11 AST parsing and `git diff --check` passed for the five frozen files.
This paragraph is a transcription of local tool results, not a raw stdout file.

Actual-dependency CPU validation subsequently completed in the separate
immutable [`a3-pd-cpu-20261110/pd-eligibility-017`](a3-pd-cpu-20261110/README.md)
snapshot: 215 passed, zero failures/errors/skips, exit 0. Only the two config
mirrors ran; concurrent diagnostic/Graph work was excluded from the claim by
not collecting its executor/selector modules. The manifest is
`f19716bfa9931cbce30a25e3a2fc316c56cbc8f65bc6f1898ad55a9a97d41ff8`.
All five frozen plugin/test/fake hashes match it. The ten baseline native source
checks passed, and a separate stdlib-only collector verified that the actual
installed `KVTransferConfig` and Mooncake connector source hashes match the
locked Git files. This source collector did not import torch or torch-npu.

No NPU/PD/full-model acceptance, KV-only maintenance behavior, network transfer
correctness or performance is claimed by this implementation record. No donor
v0.23 PD patch was restored. The initial sandbox SSH-config permission failure
preceded any upload; the evidence index records that local failure separately
from the successful A3 test execution.

Cost is bounded startup validation plus the existing private config copy.
There is no new per-token wire, readback, KV copy or mutable registry. Keep F's
role isolation permanently; widen or remove these eligibility restrictions only
with separate native P/D layout/history and runtime qualification.
