# V2 staged HCCL native entry contract review — 2026-11-09

This increment adds a controlled U2 construction and `run` entry. It does not
enable the production V4.1 HCCL model path or complete C4 model acceptance.
The existing explicit `run_staged` API remains available.

## Source and patch contracts

Read-only source comparison used vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` and Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`. No upstream checkout was modified.

The patched factory remains the alias actually called by native
`vllm/v1/worker/gpu/model_runner.py` during KV initialization. Its inner
`maybe_build_ubatch_runner` retains the native signature, including return type:

```python
def maybe_build_ubatch_runner(
    vllm_config: VllmConfig,
    device: torch.device,
    model_state: ModelState,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_num_reqs: int,
) -> ubatch_utils.UBatchRunner | None:
```

The AFD context manager adds an explicit keyword:

```python
use_npu_ubatch_factory(owner, *, hccl_transport=None)
```

Only a non-`None` transport selects `HcclV41UBatchRunner`. Omission retains the
existing NPU factory admission, including the HCCL DP1 result `None`. Explicit
injection permits DP1 construction but rejects disabled ubatching,
U1/U3, non-eager or Graph mode, non-HCCL connector, and non-Attention endpoints.
It also rejects PP/PCP/DCP greater than one, SP, and a non-native
`EngramModelState` before native allocation. Native construction is restricted
to `data_parallel_size == 1` and `data_parallel_rank == 0`. Negotiated transport
TP must equal configured TP; the Attention role size must equal that TP world;
`parallel.rank` must equal `transport.role_rank`. At the locked
`vllm/v1/worker/worker_base.py:86`, native WorkerBase writes the actual worker
rank into `parallel_config.rank`. Under DP1/PP1/CP1, this is the TP coordinate,
so the strict role-rank check binds the single native TP world without inventing
a DP mapping. This matches the existing F helper's restriction. DP>1 native
construction remains pending; standalone prepare/run_staged DP child-view tests
keep their separate interface-only scope. An unsupported explicit combination
raises; it cannot silently fall through to `None`.

`HcclV41UBatchRunner.__init__` retains the `NPUUBatchRunner` positional arguments
and the existing `kernel_block_sizes` and `timeout_s` keyword arguments. Its
additional required `transport` keyword is documented immediately above the
method. The caller owns initialization, accepted plans, failure supervision and
close; construction only validates the transport's Attention role.

The new override exactly matches native
`vllm/v1/worker/gpu/ubatch_utils.py::UBatchRunner.run`:

```python
def run(
    self,
    model: Any,
    model_inputs: dict[str, Any],
    ubatch_state: UBatchState,
) -> Any:
```

Native heterogeneous annotations are preserved only at this entry boundary.
`run` requires an initialized, open, nonterminal transport and delegates to the
existing typed stage execution seam. The accepted Attention plan is checked
before consuming the parent Engram producer. Device or model failure remains
terminal and restores the parent forward context through the existing scopes.
No native thread/handoff loop, parent history retirement, sampling or KV lifecycle
is copied into this override. `prepare` retains its existing native signature.

Reason/functionality/removal comments and PATCH markers identify the new factory,
constructor and `run` differences. The factory imports stay lazy because the
runner imports the factory module through its NPU base class.

## Ownership and construction

The existing scoped replacement of `create_sm_control_context` with
`nullcontext` is preserved. Native SMControl otherwise asserts CUDA/ROCm, so
constructing this runner outside the NPU factory scope is not an accepted entry.
Both native aliases restore on normal exit and exception. This scope is intended
for the existing single-threaded KV initialization boundary; no new mutable
global registry or user configuration is introduced.

The constructor still runs native `UBatchRunner` and `NPUUBatchRunner` buffer,
stream and metadata-builder construction. In particular, the locked native
constructor creates two CUDA-named streams; actual Ascend alias behavior needs
the separate NPU constructor probe. CPU constructor substitution validates only
factory selection and method execution, and is not device-construction evidence.

The stage execution ownership previously reviewed in
`v2-stage-integration-review-20261109.md` is unchanged: native parent preparation
owns Engram lookback/history; `run_staged` consumes its producer once in the parent
context, slices lookup/mask rows, executes layer-major child stages and merges
parent outputs. Ascend `execute_model` retains its single `finish_execution`.

## Production integration gaps

At the pinned native `GPUModelRunner.execute_model` lines 1787 and 1935, only a
descriptor with `num_ubatches > 1` invokes the runner's `prepare`/`run`. U1,
threshold fallback and some profile/dummy paths call the model directly around
line 1948. This increment does not route those calls and does not force U2.

No production caller currently supplies the new factory keyword. Existing HCCL
runtime admission and the full-model feature guard remain in force. Production
integration still needs an authoritative gather of every A rank's native child
valid/padded counts, coordinator plan acceptance before `run`, a v3 U1 path, and
F worker service dispatch to the existing v3 F helper. Uniform native DP metadata
must not substitute for the actual peer counts. The old v2 metadata/control path
must not also publish a second parent when v3 is connected.

The separate transport component's topology evidence continues to apply to
communications. It does not prove native model group construction, Engram
producer/events, nonzero-offset sparse-index operators, HC/MoE kernels, native TP
model outputs or checkpoint inference. Graph, stream overlap, DSpark/PD and full
checkpoint U1/U2 remain outside this increment.

## Cost and removal

Selection adds configuration checks and one instance dependency at construction.
It introduces no new per-step control collective, tensor copy or stream. Native
U2 allocates two streams and existing per-stage buffers/builders even though this
synchronous driver does not use native handoff threads. The existing synchronous
per-layer/stage HCCL and parent COMPLETE costs remain unchanged; no performance
claim is made.

Remove the factory override when upstream provides backend runner injection and
NPU-safe construction. Replace the `run` override when upstream exposes a
synchronous prepared-stage callback with native parent ModelState ownership.

## Snapshot and validation

| File | SHA256 |
| --- | --- |
| `afd_plugin/compat/patches/npu/mrv2_ubatching.py` | `1e53b24475946f0cc83d6a05b5a83af5d4746287bd9ba04f357021df27905ecb` |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `7d9ec420eb7d15192819e47587e37116e846dde7967998ea615c284b95ea3435` |
| `tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py` | `ba7ac5e1af8e325a7d404f8a1ebae635e31a27f479d7ac1d5f3edc075e414d9d` |

Local static commands completed with exit 0:

```text
.venv/Scripts/python.exe -m ruff check <the three files above>
.venv/Scripts/python.exe -m ruff format --check <the three files above>
Python ast.parse(..., feature_version=(3, 11)) for the same three files
```

Local targeted pytest collected `0 items / 1 skipped` on Windows Python 3.14.3
without torch. Its observed process exit was 1. This is dependency-gated CPU
collection, not a passing runtime result. The suite now describes 43 cases:
native slicing/context behavior, parent producer/output ownership through both
entries, terminal failure, inactive/unaccepted transport, actual scoped factory
selection/restoration, unchanged omitted-injection admission and explicit
unsupported-combination rejection, including five DP/TP/role coordinate failures.
The factory CPU seam substitutes native device
construction only; its slicing and forward-context methods remain real.

A3 actual-dependency CPU results and the separate real constructor probe must be
recorded against their immutable snapshot before claiming those validations.
The limited entry API has no unresolved source ownership blocker within the
controlled eager U2 scope; independent acceptance review and NPU evidence remain
separate from this author contract review.

Independent review found the first candidate lacked a native TP/role binding.
That candidate (`a42d8cf5...`, 38 test cases) is superseded, not accepted as the
final construction contract. The final snapshot above adds the DP1/TP/role/rank
checks and five rejection cases. The reviewer confirmed the scope and compared
the native method/factory signatures independently; any earlier device logs must
retain their original snapshot identity. Final A3 results must use these bytes.

## Final A3 CPU and constructor evidence

The final immutable evidence was read directly from:

- `a3-stage-state-20261109/native-entry-cpu-008/identity.json`,
  `results/summary.json`, `results/cpu-seams.log` and `results/cpu-seams.xml`;
- `a3-v2-entry-20261109/constructor-001/identity.json`, `results/summary.json`,
  `results/probe/result.json`, `results/probe.log`, `results/source-check.json`
  and `results/npu-after.log`.

Both manifests match the final factory and runner SHA256 above. CPU008 also
matches the final test SHA256. The installed-dependency CPU suite completed with
**112 passed, 14 warnings, exit 0**. XML records 112 tests and zero failures,
errors or skips; 43 cases belong to `test_hccl_v41_ubatch_runner`. The other 69
are the existing stage-state and model-driver seams. Arithmetic model/producer
fakes and the CPU device-construction seam keep their stated scope; this result
does not execute native model kernels.

The separate constructor probe's SHA256 is
`49d6a407788f95d7d700a6683d3b8205e786ed8cf82a54690c16f5ce3d9c97e1`,
also matching its manifest and the current file. It ran on A3 chip0 with Python
3.12.9, torch 2.10.0, torch-npu 2.10.0.post4, vLLM 0.30.0+empty and Ascend
0.1.dev1+g7e37e29a7. It exited 0 without timeout or a forced kill; all eight
boards were idle before and after. The ten recorded upstream source checks are
true. FastAPI remained the user-approved deployed version 0.123.10.

This probe calls the actual `EngramModelState` constructor and actual
Hccl/NPU/native UBatchRunner constructors through the factory. It uses weightless
native model shells with no Engram and empty Attention groups. It verifies two
distinct native NPU streams and distinct per-stage, zero-initialized int32 query
buffers of shape `(3,)` and sequence buffers of shape `(2,)`; it also verifies
restoration of the factory and SM-control aliases. An actual uninitialized
`HcclStagedTransport` is rejected through the new native `run` entry and remains
caller-owned and uninitialized until explicit close.

The pinned state constructor dependency chain is ModelState -> DefaultModelState
-> AscendModelState -> EngramModelState. Its minimal configuration supplies model
length/dtype/embedding-size access, scheduler request/token capacities, disabled
prompt embeddings and no encoder cache, initialized Ascend configuration with
sparse offload disabled, and a model with `token_lookback_depth == 0`. The last
condition avoids registering a producer pre-hook or allocating lookback storage.
No state or runner constructor is bypassed with `__new__`.

The native `torch_cuda_wrapper` from
`vllm_ascend/worker/v2/utils.py` maps CUDA-named APIs to NPU APIs; its pinned
`finally` does not restore those mappings. The probe uses an isolated process.
The verified restoration claim concerns the scoped factory and SM-control
aliases only.

Construction is now demonstrated for this DP1/TP1, no-Engram, empty-group scope.
Attention metadata builders, real Engram history/producer/event consumption,
HCCL initialization or accepted plans, TP2 constructor execution, native sparse
index/HC/MoE operators, successful model `run`, and checkpoint inference were not
executed by this probe. Those remain distinct pending validations; production
injection, U1 routing, plan gathering and F service integration remain unchanged.
