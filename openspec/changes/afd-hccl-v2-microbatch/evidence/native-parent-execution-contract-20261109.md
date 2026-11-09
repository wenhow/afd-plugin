# Native V2 parent execution contract, 2026-11-09

This is the author contract record for the Attention entry increment. The
independent counts/bootstrap review is recorded separately. It describes
software wiring under the existing closed V4.1 model guard; it does not qualify
checkpoint inference, native model operators or overlap.

## Pinned upstream and entry

The upstream objects were inspected with `git show`, without changing either
checkout: vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` and Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`. The working tree was based on
`6d43f0390eb1bb846c4109d82f56b661bdb21aab`; concurrent A5 changes were retained.

The added wrapper has the exact native
`AscendDeepseekV41ForCausalLM.forward` signature from `vl_model.py:261`, including
the `torch.Tensor` return annotation. The execute wrapper retains the exact
Ascend `NPUModelRunner.execute_model` signature from `model_runner.py:486`, with
no return annotation. Native Ascend delegates the large request/KV execution
body and calls `model_state.finish_execution(failed=...)` in its own `finally`.
The added code neither copies that lifecycle nor calls its retirement method.
Returning `None` from native execution remains a valid native result.

The scoped `prepare` wrapper matches vLLM `UBatchRunner.prepare` at
`gpu/ubatch_utils.py:329`: `input_batch`, `block_tables`, `slot_mappings`,
`cg_mode=NONE`, `for_capture=False`, returning `UBatchState`. Its difference is
marked, its delegation exception is documented, and its signature/restoration
check was added to the existing sole parametrized signature table. Existing A5
rows were preserved.

Normal constructor selection uses the backend-local runtime connector helper.
For a V4.1/HCCL configuration with U2 enabled, the selected staged adapter is
automatically bound before `load_model`; model loading then initializes that
same transport, skipping the old v2 rendezvous. The negotiated capacity is
validated again after HELLO and defensively in the U2 constructor. Default U1
and C3 DSpark retain their v2 selection. An explicitly supplied staged transport
still supports U1. Configuring U2 permits native threshold/profile fallback to
U1 within the same v3 session; it does not force every forward to two stages.

The native `ParallelConfig.use_ubatching` property is `enable_dbo or
ubatch_size > 1`, and `num_ubatches` is `2 if enable_dbo else ubatch_size`
(`parallel.py:596-601`). Those are derived properties, not fields to overwrite.
The existing Ascend platform compatibility scope snapshots/restores the real
DBO/ubatch fields for valid AFD configurations before worker serialization and
revalidation. No new enable flag or configuration mirror was introduced.

## One accepted parent

`HcclV41Execution` scopes the existing native call. It establishes eager Engram
mode before parent `prepare_inputs`, but leaves history and lookup production
with the native ModelState. Native U2 `prepare` supplies the real slices; for
each bounded slice the report is `valid=max(0,min(stop,num_tokens)-start)` and
`padded=stop-start`. Dummy reports have zero valid tokens even when native
dummy storage contains rows. U1 obtains padding from the native parent batch
descriptor. No tensor-mask host read or global count estimate is used.

The parent context provider submits one `HcclLocalForward`. Attention0 assigns
the parent ID; other Attention ranks receive it through the solicited counts
round. Native request/KV maintenance contexts without a batch descriptor do
not submit a parent. Child contexts do not create another count round.

U1 executes the outer model through real `nn.Module.__call__`, allowing the
native Engram pre-hook to merge the parent's produced kwargs once. The typed
context sidecar then calls the shared stage executor with `engram_prepared=True`.
This also works before KV initialization, when no UBatchRunner exists. U2
instead enters native `ubatch_runner.run`, calls the same stage executor, and
consumes `prepare_engram()` once in the parent context because native U2 bypasses
the outer model pre-hook. Both routes slice the already-produced lookup/mask,
keep child DP metadata absent in the admitted DP1/non-SP mode, invoke the
existing layer-major driver, reconstruct native parent coordinates, and finish
the accepted transport parent. Native Ascend then retires the parent producer.

The scoped prepare alias, metadata provider and DP1 threshold dispatch alias
restore on normal return and exceptions. Compute/producer/communication failures
mark the endpoint terminal and retain failed communication resources for
process supervision. There is no unsolicited mid-payload abort packet. Normal
shutdown performs all-A STOP, native cleanup, then closes the bound adapter;
native cleanup failure retains the transport. A staged adapter must be bound to
its own transport, preventing a second session from acquiring separate close
ownership. Explicit old-adapter injection retains its separate transport owner.

## Cost and removal

The increment adds bounded host bookkeeping, one context sidecar per parent and
the counts exchange before the existing FORWARD/ACCEPT round. Native U2
preparation, request/KV buffers and TP groups remain in their original owners.
The layer-major driver and COMPLETE device synchronization retain their
previous costs. Execution is synchronous; no stream overlap benefit is claimed.

Replace the scoped prepare/context/DP1 hooks when upstream exposes native
prepared-state and model-call callbacks. Replace the wrapper sidecar when that
callback can route U1 and U2 without patching the model wrapper. Keep upstream
Engram retirement ownership when doing so. Graph, DSpark/PD, DP>1, unequal model
role worlds and failed-device recovery require separate contracts and evidence.

## Validation

Local Ruff and Python AST checks pass. Local pytest dependency gates collect
without torch/vLLM; this is not runtime validation.

The actual-dependency CPU snapshot
`a3-forward-counts-20261109/attention-cpu-005` was read directly: `summary.json`
records return code 0, `cpu-attention-entry.log` reports **112 passed, 14 warnings**,
and XML records 19 new execution cases, 44 UBatchRunner cases, 41 model cases and
8 signature rows. This snapshot covers the initial normal automatic binding,
actual native input slicing/context construction, outer model pre-hook,
profile-before-KV, dummy reporting, U1/U2 alternation, negotiated-stage rejection,
and shutdown cleanup ordering. The upstream execute body is a single substituted
collaborator in these CPU execution tests; native scheduler, real Engram history,
native metadata builders, device kernels and inference are not simulated as
accepted behavior. The earlier real device constructor probe retains its
original, narrower evidence and source snapshot.

CPU005 precedes the final same-adapter transport identity rejection. Its A runner
SHA256 is `039f7d773c92ed80d3bd519ad23eb102f98507f0f1de4372a2ef6ee77199c576`
and its execution-test SHA256 is
`a6cd66eacc68dde6f48ed5a0b0a85468c1330d583d8cfd1f238fd2baec884a93`.
Final identities and that follow-up's result must be appended after verification;
the earlier 112-case result is not silently transferred to changed bytes.

### Final source and installed-dependency CPU follow-up

The immutable `a3-forward-counts-20261109/integration-cpu-008/identity.json` and
`results/cpu-integration.xml` were read directly. All eight files below match
that manifest byte-for-byte. The four Attention/signature modules record
**113 passed** (20 execution, 44 UBatchRunner, 41 model, 8 signature), with zero
failures, errors or skips in those modules. This includes the final same-adapter
identity rejection and the exact copied native wrapper body. The wrapper's
ordinary path now calls `self.language_model(...)` directly as upstream does;
its two tests observe the output and Engram kwargs through real model pre-hooks.
The prepare wrapper's comments explicitly explain why it delegates the selected
Hccl/NPU override instead of bypassing that instance with a base-body copy.

| File | SHA256 |
| --- | --- |
| `afd_plugin/model_executor/models/npu/deepseek_v41.py` | `5e1690b161674180faca8407eba806c8ad642418655b1e7eb4d43e8bc27f6007` |
| `afd_plugin/v1/worker/npu/hccl_v41_execution.py` | `bcd29066f59e7c053b183b77a6c249b92d22ec2f634f56ef87ac8a6548a85af4` |
| `afd_plugin/v1/worker/npu/attention_model_runner_v2.py` | `1fdfc6ef13b5fdefa20a98c762e0a1dc9ec23068d2b5683245ed4d1286d1079c` |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `2317b9e3a911d79533761f1bc81bc994dc27638eceebaaf069ee00c90e3c0824` |
| `tests/unit/v1/worker/npu/test_hccl_v41_execution.py` | `14a2b90c69da8108f66da837dffae4fcb8d363e2bc6631d8f924e3df32bc5d0f` |
| `tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py` | `187c2ce4536ad046bbbeba0d7b4370aad893ccf89dccb568c91a357d609b4d0a` |
| `tests/unit/model_executor/models/npu/test_deepseek_v41.py` | `b88b327e0b51ad4820808c62fd792736670cfbd5ff17a796923dba4da917cf31` |
| `tests/unit/compat/patches/test_signatures.py` | `932d8a40e329a4aa7d75a55fff162acdf4225e837c44dbcfd07ba4430bffdf8d` |

The surrounding integration run was **not** all green: its total is 2238 passed,
4 failed, 2 skipped and 4 deselected. The four failures are older constructor
fixtures patching the removed `AFDConnectorFactory` module import after the
runtime helper migration. They must be corrected and validated separately;
they are not concealed by the passing Attention subset above. This follow-up
still runs CPU Python seams with installed dependencies and establishes no
native device/model execution result.

### Final integration closure

The root agent corrected those four obsolete constructor fixtures without
restoring an unused production import. The reviewer read
`a3-forward-counts-20261109/integration-cpu-009/results/summary.json` and
`cpu-integration.xml`: return code 0, **2242 passed, 2 skipped, 4 deselected**,
zero failures/errors. XML contains 2244 cases including the two skips. All eight
Attention/source-test hashes above match that final immutable manifest; the
runtime bytes did not change during the fixtures-only repair. The final CPU
regression is therefore green within its declared selected scope. Device/model
acceptance boundaries remain as stated above.
