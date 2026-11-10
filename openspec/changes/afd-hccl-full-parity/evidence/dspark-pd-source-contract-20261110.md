# C7 DSpark / PD source contract (2026-11-10)

Status: independent read-only source audit and implementation proposal. No runtime,
tests, task checkboxes, upstream checkouts, package installations or devices were
changed by this audit. Source availability is not DSpark inference or PD acceptance.

## Scope and fixed sources

Read the C3 `afd-v41-hccl-dspark` proposal/design/spec/tasks and its existing
`parallel-development-20261109.md` / source review; also read C7
`afd-hccl-full-parity` design/tasks and H12/H13 in the capability matrix. C3's
target-only, native aux, real rejection and independent fallback requirements
remain applicable. C7 additionally owns the five PD combinations and the ordinary
MTP branch mapping. Existing task completion is not expanded here.

All upstream reads used `git show` / `git grep` against these objects, without
switching any checkout:

| Prefix | Repository | Commit |
| --- | --- | --- |
| V | `D:/openSource/vllm` | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| A | `D:/openSource/vllm-ascend` | `7e37e29a7803d6897a7113088e2b771f83205bc0` |
| D | `D:/openSource/afd-plugin/afd-plugin-phase1-delivery` | `4ddd69a391dfb5b3e362c7601399e538a31de702` |

Current plugin comparisons use worktree bytes, not an assertion that all files
equal HEAD. Their exact hashes are recorded below; other development continues
concurrently. Source line references below are to the pinned full files.

## Native proposal, verification and state owners

1. **Model identity and construction.** V01 `SpeculativeConfig` explicitly
   distinguishes `method="dspark"`; its V4.1 classic-MTP rejection is at
   1347-1355. An `mtp.*` checkpoint namespace does not make this ordinary MTP.
   V04 `DSparkSpeculator.load_draft_model(self, target_model,
   target_attn_layer_names) -> nn.Module` at 84-114 uses the DSpark loader.
   The loader in `gpu/spec_decode/dspark/utils.py:57-106` retains the target
   ModelConfig in VllmConfig and supplies the independent draft ModelConfig to
   `get_model`. A02 `DeepseekV41DSparkModel.__init__` at 146-222 constructs all
   draft decoder layers, main projection, Markov head and confidence head on A.
   F must not intercept this model identity or load those complete draft layers.

2. **Target execution and aux.** V02 `GPUModelRunner.execute_model` at 1631
   updates native request/block-table state and returns early through the KV
   connector for a no-forward step (1641-1654). Its actual `BatchReqState`
   comes from native gather (1660), not scheduler-dictionary iteration. Target
   `prepare_inputs` reads `scheduled_spec_decode_tokens` (V02:1292-1323;
   A05:697-721). A01 `DeepseekV41Model.forward` captures
   `hidden_states.mean(dim=1)` at a configured one-based layer entry, before
   Engram injection (1601-1607), then returns ordered aux alongside final hidden
   output (1649-1651). HC collapse/final normalization are not aux sources.
   The current plugin staged driver already preserves this timing and merges
   each aux layer across parent token slices; that existing operation is not
   proof of an enabled DSpark/U2 runtime.

3. **Actual verification.** V02 `sample(self, hidden_states, input_batch,
   grammar_output) -> tuple[SamplerOutput, Tensor, Tensor]` at 1512-1588 uses
   native ordinary sampling if there are no draft tokens; otherwise it calls
   `self.rejection_sampler(logits, input_batch, self.speculator.draft_logits)`.
   `sample_tokens` at 1997 consumes and clears `execute_model_state`, obtains
   actual `num_sampled` / `num_rejected` at 2048, calls native postprocessing at
   2115, then calls native `speculator.propose` at 2143. Acceptance is not an
   AFD F service phase and must not be forced or reconstructed from block size.
   `num_sampled` includes the native sampler's recovery/bonus conventions; it
   is not a universally valid standalone accepted-draft count.

4. **Accepted/rejected history.** V02 `postprocess_sampled` at 1590-1619 calls
   V03 `post_update`. Its kernel at 552-609 appends only actual sampled IDs to
   `all_token_ids`, advances `total_len` by `num_sampled`, updates the last
   sampled token, and advances `num_computed_tokens` by
   `query_len - num_rejected`. Masked request indices are ignored. Do not add
   a plugin copy of those updates or a generic cache-clearing rollback.
   Physical rejected KV can remain in owned storage until native subsequent
   writes/length masks supersede it; this audit does not establish full
   compressor/ring-state rejection correctness on NPU.

5. **Next draft context and queries.** A06 `AscendDSparkSpeculator.propose`
   (405-471; complete native signature includes `input_batch`, attention/slot
   metadata, last/aux hidden states, `num_sampled`, `num_rejected`, last sampled
   and next prefill tokens, temperature, seeds, DP sync and dummy/profile flags)
   delegates to the DSpark/DFlash native implementation. V05 `propose` at
   300-346 concatenates ordered aux and uses `model.combine_hidden_states`.
   At 410-427 it eagerly pre-inserts valid context KV outside the draft graph.
   Its input kernel at 538-586 excludes the rejected suffix from context,
   writes `PAD_SLOT_ID` for that suffix/null blocks, and at 589 onward starts
   query positions after the last valid context position. The draft has its
   own query graph manager (V05:434-480; A06:138-146). V04:253-271 runs one
   parallel draft backbone followed by sequential Markov sampling; draft layer
   count, query width, proposed-token count and remote target MoE count differ.

6. **Engram owner.** A03 `EngramModelState.prepare_engram_inputs(self,
   input_batch, req_states) -> dict[str, Any]` at 99-154 gathers the lookback
   window from the same `req_states.all_token_ids.gpu` and accepted native
   `num_computed_tokens.gpu`. Dummy paths use no real history. FULL publishes
   persistent producer inputs; eager publishes one pending parent producer.
   `finish_execution(*, failed: bool) -> None` at 65-72 retires lookup bindings
   and delegates to native ModelState. A05 `execute_model` calls it exactly
   once in its native `finally` (527-530), before returning to sampling.
   AFD stage execution must not create per-child history updates, duplicate
   native finish, or leave a target provider installed around draft proposal.

The resulting owner chain is:

```text
native execute(target request/KV + Engram parent)
  -> target-only A/F MoE work and ordered target aux
  -> native finish_execution
  -> native sample/reject
  -> native post_update(accepted IDs + computed lengths)
  -> A-local native DSpark context KV + draft proposal
  -> req_states.draft_tokens
  -> subsequent scheduled target verification
```

No sampler/proposer rewrite is justified by this source chain. Native A06
lmhead/DCP/PCP behavior must remain native; this review does not admit those
additional parallel topologies. A source comment in the pinned Ascend rejection
helper (`worker/v2/spec_decode/rejection_sampler_utils.py:235`) also says block
verification is accepted as an argument but not implemented on NPU. Do not
claim an AFD patch supplies that separate native algorithm.

## Donor mapping: DSpark is not remote ordinary MTP

| Donor branch | Exact source seam | V4.1 destination |
| --- | --- | --- |
| DSpark complete draft on Attention | D02:1998-2014 recognizes `AscendDSparkProposer` and skips attaching the remote draft connector; D01:355-363 excludes DSpark from `requires_mtp` | Preserve full native V4.1 draft on A; only target plans consume F |
| Ordinary MTP eager N steps | D03:1221-1233 `_mtp_ffn_forward_sequence` loops each speculative step after target | Retain for applicable existing model families; not a V4.1 DSpark phase |
| Ordinary MTP merged stage 0 / header | D01:1448-1485 sends shape/DP header; 1554-1589 enforces merged stage 0 and speculative-step bounds | A separate compatibility feature for ordinary MTP models; not encoded by DSpark block/stage count |
| Target graph followed by remote MTP | D03:624-643 replays/executes target, receives MTP phase-ready, then services draft | V4.1 target graph should retire, then A executes its own native draft; F waits for the next target reservation |
| Full remote ordinary-MTP draft graph | D02:2018-2041 wraps a native ACLGraph draft runnable with phase announcement; D01:1652 onward receives static graph headers | Keep the ordinary-MTP model regression separately; V4.1's own draft graph manager does not use AFD phase-ready |

The capability matrix's ordinary-MTP N=1/2/3 coverage must remain pending until
the applicable model path is ported/tested. It cannot be checked off by any
V4.1 DSpark test, nor deleted because V4.1 rejects classic MTP.

## PD: upstream already covers some donor purposes

**Async allocation:** D04's v0.23 allocation shim suppresses speculative
lookahead while remote KV is loading. V06:1131-1137 already computes zero
effective lookahead whenever `load_kv_async` and lookahead is nonzero, and
passes it to native allocation at 1164-1166. Re-enabling D04 by widening its
version gate would duplicate this behavior. Its hybrid `failure_policy=fail`
branch is a separate question: V06:3181 onward still delegates invalid-block
handling to native helpers. This audit has not proved equivalence of every
heterogeneous failure marker; preserve fail/recompute tests before deciding
whether a minimal new plugin patch is needed.

**Native SFA DSpark draft-KV path:** A07 selects `SfaRemoteD2HConnector`, also
inside MultiConnector (25-47), not every Mooncake connector. P must load the
native draft to generate prompt KV locally; P-side prefill currently requires
eager, CP1 and local prefix caching disabled (50-75). This is compatible with
AFD F having no draft: P and A are request/KV owners; F remains an activation
compute role.

The concrete call chain is A05 `initialize_kv_cache` (352-403) ->
`configure_dspark_kv_transfer` -> identify loader-owned draft cache names ->
keep persistent draft pages separate from layerwise target scratch -> native
KV allocation -> bind D's `DSparkContextReceiver`. A05's native execute then
calls `_maybe_send_draft_kv` at 555-581 after successful target execution,
before `sample_tokens` clears `execute_model_state`.

A07 `send_dspark_prefill_kv` at 222-304 concatenates real target aux, selects
only prompt rows, validates contiguous chunk offsets and allocation generation,
calls A06 `initialize_local_context(chunk, aux_hidden_states,
draft_block_ids_by_group) -> None`, and sends completed draft pages. A06's
`get_draft_context_group_layout() -> tuple[tuple[int, ...], tuple[int, ...],
dict[int, int]]` at 149-166 uses actual loaded layer/group/block mappings.
A08 `build_draft_kv_metadata` (104-144) uses those names; its
`build_draft_kv_read_batches(remote, local, source_blocks, dest_blocks,
prompt_tokens)` (147-199) matches by layer name and validates page geometry,
dtype and logical block coverage, copying valid prompt rows only.

A07 receiver state is keyed by external request ID and allocation generation.
`begin_direct_transfer` (363-394) handles retry/stale/backpressure;
`finish_direct_transfer` (396-408) only marks prompt initialization after real
transfer success; `ready_requests` (417-428) additionally requires target KV
done, no failure and no in-flight transfer. The connector must join readiness
across all D TP ranks before `retire_ready_request` (430-451). Resident draft
KV remains runner-owned after ingress bookkeeping retirement.

**Target shared-page layout is still an independent check:** A04
`get_layer_tuples` and `group_cache_specs` build long-KV/index/folded-index,
compressor state, SWA and optional draft views over physical slots, including
zero layer stride. A09:1931 onward reconstructs these shared pages for
registration. That improves the native physical-page description but does
not itself prove heterogeneous P/D matching: `_transfer_kv_cache_all_groups`
still indexes logical groups and zips local/remote address arrays
(838-860). A07/A08's named **draft** transfer does not establish named
**target** transfer equivalence. Compare actual P and D placements before
choosing native SFA or a new narrow target-layout adapter. D05's old
layer-addressed patch is useful evidence of the required validation, not a
drop-in solution for V4.1's `.mtp.` names and overlapping/shared views.

**Engram history handoff:** V02:1120-1158 initializes D request state from
`scheduled_new_reqs.prefill_token_ids` and `num_computed_tokens`; A03 then
gathers Engram history from those native buffers. Therefore absence of an
Engram tensor in PD metadata alone is not a proven architectural blocker.
The actual ingress must retain the complete token IDs needed for lookback,
including resumed/generated prefixes, alongside the transferred computed
length and allocation generation. This must be checked on the real P->A
request path; a successful KV transfer/healthcheck cannot substitute for it.

## Smallest useful next software increment

Implement **controlled eager target DSpark x U2 admission and ownership**, using
the existing real target stage engine and native sampler/proposer. Keep the
production V4.1 model guard closed. Do not start with an unused proposal codec,
remote draft service, graph-key relaxation, or a copied rejection sampler.

Proposed module seam (names are a design proposal, not implemented APIs):

```python
@dataclass(frozen=True, slots=True)
class HcclV41DSparkTargetContract:
    aux_layer_ids: tuple[int, ...]  # native one-based ordered boundaries
    block_size: int
    draft_stages: int
    num_speculative_tokens: int
    target_tp_size: int
    draft_tp_size: int

def validate_hccl_v41_dspark_target(
    vllm_config: VllmConfig,
) -> HcclV41DSparkTargetContract: ...
```

This seam must have an immediate consumer in the existing explicit A target
execution/binder and F pre-construction config isolation, rather than exist
alone. Reuse `validate_deepseek_v41_speculation` for model identity/checkpoint
metadata. Explicit contract admission may allow actual U2 **target** work;
it must still reject non-DSpark methods, synthetic rejection, mismatched TP,
wrong draft/target identity, SP/CP/PP/DP outside current paired scope and PD.
F receives a private spec/Engram-free configuration before native construction,
with negotiated target capabilities unchanged. Do not copy U1 config to trick
selection and do not infer target rows from speculative-token count.

Keep the current `HcclV41Execution.scope` around exactly one native
`execute_model` and collect real native `prepare` slices/counts, including
variable-length verification requests. Existing stage aux merge returns the
native parent order to `ExecuteModelState`. On leaving the scope, the target
transaction must be retired and provider/model bindings restored. Invoke
unchanged native `sample_tokens` later, so rejection/postprocess/draft execution
has no live F transaction. No additional phase-ready packet is needed for a
local draft. The native `finally` retains sole ModelState retirement ownership.

First CPU/API tests should observe actual public seams: native DSpark class
selected instead of ordinary MTP; same-checkpoint F isolation before init;
two nonuniform target stages preserving original FP32/raw IDs and ordered aux;
exactly target FORWARD/COMPLETE with no new frames during native draft contexts;
native errors/empty target/provider restoration; and closing speculation
restoring current eager behavior. CPU tests must not fabricate accepted NPU
state or assert private helper calls. A separate device test must run the
unchanged sampler with real acceptance/rejection and compare request history,
computed lengths, target/draft cache behavior, subsequent requests and output
to native on A5. Source/software work need not wait for that device gate.

## Concrete remaining interfaces and controlled boundaries

| Gap | Current source boundary | Required next contract |
| --- | --- | --- |
| DSpark/U2 role admission | `runtime_config.get_npu_ffn_config:89-96` rejects DSpark+ubatching before F isolation; staged HCCL rejects speculation at `hccl_config:258-263` | Explicit validated target-only admission consumed by A/F setup; full draft remains A-native; production model guard still closed |
| Target graph verification queries | `HcclGraphKey.__post_init__` requires q=1 and T=R; native geometry accepts FULL q1 only; coordinator requires spec=None | Query-aware native descriptor/metadata/stage preview and speculative schema in session/key; agree all peers before prepare; same early/final decision binding and graph generation ownership. Never relabel q>1 descriptor as q1 |
| Adaptive/nonuniform target queries | A05:723 onward can compact/reallocate verification rows; current pre-prepare FULL preview excludes this | Initially honest eager fallback; exact metadata/valid geometry proof before any later graph admission. Runtime counts alone cannot certify request geometry |
| Independent draft graph | A06:138-146 and V05:434-480 own the native query graph manager | Keep local draft graph/config separate from target cache and F service; first controlled target graph may use enforced-eager native draft |
| PD runtime admission | `hccl_config:264-265` rejects KV transfer; coordinator's strict owner excludes it | Validate chosen actual KV connector and owner callbacks; preserve native pre/post-forward/no-forward and `_maybe_send_draft_kv`; F owns no KV/history |
| PD heterogeneous target layout | A04 physical alias plan plus A09 group/array transfer; A08 named draft transfer only | Collect both native layouts, match logical target source/cache component + dtype/stride/block/alias geometry before reads; reject mismatches before any partial transfer |
| PD readiness / history | Native receiver generation + target/draft ready joins; native request all-token history | Verify complete ingress token history/computed length; native all-TP ready before target admission, retirement/cancel/failure generation rules, resumed/prefix requests |
| Old ordinary MTP capability | D01/D02/D03 remote phase/header/N-loop paths | Independent existing-model implementation/regression; V4.1 DSpark cannot qualify those branches |

These are software gaps and acceptance boundaries, not evidence that an A5
model request has failed. Resource acceptance remains pending: full A-local
draft weights, workspace, persistent target/draft KV, graph pools, cold-load
peak and topology. Existing C3 encoded-weight budget is not an HBM measurement.
No new per-token host readback or sampler synchronization is necessary for the
eager ownership increment; actual request counts remain CPU native inputs,
with accepted/rejected tensors owned by native sampling. Extra Graph and PD
storage/copy costs must be budgeted when those separate owners are admitted.

## Reproducibility identities

SHA256 of exact `git show` bytes read in this audit:

| ID | Path | SHA256 |
| --- | --- | --- |
| V01 | `vllm/config/speculative.py` | `9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19` |
| V02 | `vllm/v1/worker/gpu/model_runner.py` | `174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce` |
| V03 | `vllm/v1/worker/gpu/input_batch.py` | `d5dd956eb319bd69dd9e762047833d81ad40b21fce619fcb2133759e73620694` |
| V04 | `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | `dcf4abdedbd2bddd0d76be198ccf5b4e467c46f645038c35fcab821491a35b03` |
| V05 | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | `f9c100f42f9326e48eb84e89281897e859ba761943f54d593523555ccd98dc63` |
| V06 | `vllm/v1/core/sched/scheduler.py` | `c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884` |
| V07 | `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` | `86af2bf30604a184312e8f076097b91ecfec704f409f9d68d634b03d6b318070` |
| A01 | `vllm_ascend/models/deepseek_v41/model.py` | `989aec6db1e344dca76159dcbef0d5876874facedaa36b47e993146d5b8a2b9f` |
| A02 | `vllm_ascend/models/deepseek_v41/dspark.py` | `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601` |
| A03 | `vllm_ascend/models/deepseek_v41/engram/model_state.py` | `677cb13bb86ad80a154e0ff9a43620aa3344e7e93170a05e27aba0cc1f10aa8b` |
| A04 | `vllm_ascend/models/deepseek_v41/cache_config.py` | `467dddca82e2528ebbec14c78682486a39bffb4f096675d8b1d2d140f1a7014b` |
| A05 | `vllm_ascend/worker/v2/model_runner.py` | `a3471a9f181b45742b8f985c86137cdaafc2871d2ef6f96aa6a45236a242969b` |
| A06 | `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` | `51faf4e6b4d633863d1f7c74db84986fb7c15453e7ce0305d3d965abbd5d14c1` |
| A07 | `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_context.py` | `296bd051f967d34a21f310cd64561f71819af96d423ff0edceaaeed68e65f93a` |
| A08 | `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_kv.py` | `a67222bc667ddde0679d1f3c90ddd767ac4e65fa0850c95982c796ac45c397d1` |
| A09 | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py` | `a6a9c3a870b7d577fb2a39435067fe378646b6ab39bc669140641beba92bdf84` |
| A10 | `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py` | `83cfdc9672f558820c3fcf18bdf68c1a89279b14b20de5104b636ed207458bfd` |
| D01 | `afd_plugin/connectors/npu/p2p_hccl.py` | `594ac53c8c8008ba05da4d12db1c38511e94fdcb8e13340322a0587e940df0ef` |
| D02 | `afd_plugin/v1/worker/npu/attention_model_runner.py` | `ba7713356f7ecfbefde289004ea911ba68150cc46d34c1bb0555a995907af89e` |
| D03 | `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `f95532370a9a5157b16817c5879d78af7405c321d5739d975cd6e2785fe54ccb` |
| D04 | `afd_plugin/compat/patches/npu/dspark_pd.py` | `b2ba263bd73235245077eda7d11e3a206e2523fc75154cb7c7e31df22c2a21ac` |
| D05 | `afd_plugin/compat/patches/npu/dspark_pd_layout.py` | `d8b2f9c6217ae8678710e27d861ad300218dfb623a077535ea4abc5ba625774a` |

Current plugin worktree SHA256 at audit read:

| Path | SHA256 |
| --- | --- |
| `afd_plugin/compat/npu/runtime_config.py` | `193779218efd9c973e64bb7ebb69ed23b880f5223d3d6a07aff35f81bc4cb540` |
| `afd_plugin/connectors/npu/hccl_config.py` | `8390a5a49633b60c452206f804963bd97ca7c88572dc0fabb0a52095dc7fba99` |
| `afd_plugin/compat/npu/feature_validation.py` | `f36266695f24c04f977fb8df5bccc4f3c9b4525e0ff5be311aa4e305bb96e4ff` |
| `afd_plugin/distributed/hccl_graph.py` | `ed9f425e0fd5a486a7d66adce77a17e57c7dd5ea2103b78a3d53740188804742` |
| `afd_plugin/model_executor/models/npu/deepseek_v41_staged.py` | `62d47c8686f4257047a5290803f88231bdcc83c1ef526a314df60d7a53b12322` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph.py` | `7e7f56c640b01d7e101b39747096ee520f46e08370b7deac7a22426ac595015a` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph_coordinator.py` | `496d4bec05048b3e33de1ecf24a1d88a584696a29305109ca166df549afee008` |

`compat/npu/feature_validation.py:35-40` still rejects V4.1 NPU AFD globally
pending full-checkpoint/eager/state lifecycle acceptance. This audit does not
authorize opening that guard. No new tests were run because this increment is
source analysis and a proposed implementation boundary only.
