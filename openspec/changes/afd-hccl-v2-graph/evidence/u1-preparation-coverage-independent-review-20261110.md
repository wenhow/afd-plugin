# U1 preparation and capture coverage independent review

## Scope and source identity

Read-only architectural review on 2026-11-10; no upstream edits or device
experiments. The source identities below are file SHA256 values, not a claim
that an NPU run used these bytes.

| Plugin source | SHA256 |
| --- | --- |
| `afd_plugin/compat/patches/npu/hccl_v41_u1_graph.py` | `cedc8b23f80b0ec1978b696a3670a0375012856a8fc05321ecb41d5dd3e00646` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph.py` | `38994c5ba42a7325409bece462b2ea385d05bd5ffeb3f908d7feba26dee848da` |
| `afd_plugin/connectors/npu/hccl_graph_cache.py` | `82a8c52ce2f6429ded6768b1c53949486d3281606a650b05e11836e95a10c285` |

Pinned upstreams inspected with `git show`, without changing their checkouts:

- Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`:
  `vllm_ascend/models/deepseek_v41/engram/model_state.py:77-99`,
  `vllm_ascend/worker/v2/model_states/default.py:168-318`,
  `vllm_ascend/worker/device_metadata.py:188-252`, and
  `vllm_ascend/worker/v2/aclgraph_utils.py:256-337`.
- vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`:
  `vllm/v1/worker/gpu/cudagraph_utils.py:398-490,644-767,823`,
  `vllm/v1/worker/gpu/attn_utils.py:build_slot_mappings_by_layer`, and
  native InputBatch/ubatch geometry constructors.

## U1 preparation patch contract

The inner `prepare_attn` at plugin line 75 exactly preserves the locked
`EngramModelState.prepare_attn` signature, including its return annotation:

```python
def prepare_attn(self, input_batch: AscendInputBatch,
                 cudagraph_mode: CUDAGraphMode,
                 block_tables: tuple[torch.Tensor, ...],
                 slot_mappings: torch.Tensor,
                 attn_groups: list[list[AttentionGroup]],
                 kv_cache_config: KVCacheConfig,
                 for_capture: bool = False,
                 ubatch_idx: int = 0) -> dict[str, Any]
```

No blocking finding in this scoped preparation observer. Native metadata
construction receives unchanged arguments, including native U1 capture's
`NONE, for_capture=True`. Its `DeviceMetadata.run_build` uses
`full_graph_mode OR for_cudagraph_capture`, so producer tasks stay deferred to
the native ModelWithContext begin/finish lifecycle. The observer retains FULL
for the later parent Engram input preparation; it does not submit history or
lookup work itself. Its slot dictionary uses the passed native persistent
slot-prefix storage. The extra context/state owns Python references and views,
not replacement KV/slot device allocations.

The selected-instance binding and mode restore on the covered error paths;
ordinary calls preserve native behavior, while native U2 temporarily replaces
and subsequently restores this outer binding. The patch documents the concrete
delegation exception, owner, reason, functionality and removal condition, with
short difference markers. Removing it requires a native callback exposing the
actual prepared U1 batch before model execution.

Limits: this observer alone does not attach the later native U1 model callback
to an HCCL owner. The separate capture coordinator must own that attachment and
completion. Empty actual requests cannot be passed to native FULL preparation:
the native metadata builder reduces `num_scheduled_tokens.max()`. Whole-empty
parents therefore retain the explicit eager path. Capture dummy requests are
positive q1 rows with valid count zero, which is a different supported case.
The geometry helper rejects a fabricated empty scheduled batch before native
preparation. None of these static conclusions qualifies native Engram, model
capture/replay, offset views or checkpoint inference on an NPU.

## Capture coverage and cache emptiness

No blocking finding in `validate_hccl_v41_capture_coverage` (line 204) or
`HcclGraphCache.is_empty` (line 54). Coverage preserves native descriptor order
and exact U1/U2 request/token geometry; it neither rewrites capture sizes nor
re-pads scalar totals on FFN. Both protocol versions must be one and the topology
must be paired native TP. Session `max_tokens` bounds one parent per rank,
without multiplying its allowance by Attention size. Equivalent query-length
None/one descriptors are rejected before duplicate key capture.

For hidden width H, padded parent T and U stages, the retained bank budget per
rank is `(8H + 9) * T + 4U` bytes. The payload traffic per layer per rank is
`(8H + 4) * T`; layer reuse avoids multiplying bank storage by model depth.
Native model/shared-attention buffers, graph pool and output storage are
additional costs. The mirrored test compares the reported budget with tensors
owned by an actual CPU stage bank; this verifies accounting, not device memory
behavior. Preflight/phase agreements and COMPLETE synchronization remain graph
external costs.

`is_empty` includes pending and ready captures, including references retained
after failure. Successful close synchronizes and resets graphs before clearing
them. It reports dependency emptiness, not cache usability; lifecycle callers
must continue checking terminal/closed transport/cache state.

Validation separation: this review inspected source and mirrored test behavior.
Actual-dependency CPU regression is reported separately by its immutable run
manifest. No hardware task is completed by this review and no production Graph
guard is opened.

Follow-up evidence inspected: the `service-010/results/cpu.xml` artifact under
`a3-control-cpu-20261109` contains 14 U1 preparation cases with zero failures,
errors or skips. That run as a whole failed in a separate F-service scripted
peer, which omitted the real protocol's global ACCEPT after FORWARD. Its raw
failure is retained. The peer was corrected without changing production wire
behavior; `service-fix-011/results/cpu.xml` then contains 29 F-service cases
with zero failures, errors or skips. These are actual-dependency CPU results,
not NPU capture or inference acceptance. Coverage/cache changes await the
separate final broad-regression manifest for their current bytes.

Later targeted validation: `capture-seams-013` passed its other ten modules
(450 cases), while two F-service test assertions incorrectly referenced a
nonexistent `HcclParentPlan.all_empty` property. Those assertions now inspect
the actual stage padded-token counts. The unchanged raw failure remains.
`service-close-014/results/cpu.xml` has 42 F-service passes and zero failures,
errors or skips, with `source_files_match=true`. Its source SHA256 is
`cda23c7e17c192a4cd59c55e99dafd8e7168ae063a2999c7327285b0fda0a2c5`;
test SHA256 is
`0f445b60d5c1f8248f4ba540d50c1500bbe69f51200a1eaae9a91dc11789f16f`.
The latter run includes acknowledged-STOP/cache-close ownership and public
cache-close-before-STOP rejection. Counts from separate snapshots are not
presented as one aggregate run or as device evidence.
