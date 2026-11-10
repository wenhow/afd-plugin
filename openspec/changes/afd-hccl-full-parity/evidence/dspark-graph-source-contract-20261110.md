# C7.1.3 DSpark target Graph: source contract and next software wave

Status: independent source review, 2026-11-10. No production source, tests,
packages, upstream worktrees, task checkboxes or devices were changed. This is
not DSpark/U2/Graph execution or acceptance. The source manifest and exact
excerpts are in [the companion JSON](dspark-graph-source-contract-20261110.json).
It records 13 locked upstream files, 18 current plugin files and the actual HEAD;
the plugin comparisons describe those bytes, not the whole concurrent worktree.

The smallest useful implementation is **fixed-width DSpark target Graph/U1/U2
with the native draft kept eager**. The locked Ascend implementation already
supports that distinction. A uniform target batch uses its negotiated target
query width; mixed or shortened batches select an all-peer eager fallback before
preparation. Adaptive/dynamic verification stays outside this first wave.

## Exact sources and existing owners

The manifest uses `V` for vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`, `A` for vLLM-Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`, `D` for donor
`4ddd69a391dfb5b3e362c7601399e538a31de702`, and `P` for current plugin
worktree bytes. All upstream reads use locked Git objects. Full path, Git blob,
SHA256 and numbered excerpts are recorded per ID.

| Contract | Exact source |
| --- | --- |
| Target query width | V01 `gpu/model_runner.py:439-442` computes speculative steps plus new sampled tokens; A01 `worker/v2/model_runner.py:255-261` fixes the Ascend width to `N+1` and sizes MC2 state with it. |
| Draft query width | V05 `gpu/spec_decode/dspark/speculator.py:46-55` uses `N` for anchor-as-first-prediction, otherwise `N+1`. This is not the target query width. |
| Uniform Graph descriptors | V02 `gpu/cudagraph_utils.py:312-340` rounds the configured token bucket up to a multiple of query width, sets requests to rounded tokens divided by that width, records `uniform_token_count`, deduplicates and creates eligible U2 twins. |
| Adaptive descriptors | V02 `:265-307` has separate dynamic schedules and varlen descriptors. These are a larger contract than fixed-width uniform decode. |
| Native U2 geometry | V02 `:221-244`, V03 `worker/ubatch_utils.py:166-216`, V04 `gpu/ubatch_utils.py:49-75` retain the native token midpoint and request overlap/clamping. |
| Separate draft-eager switch | A02 `worker/v2/spec_decode/dspark/speculator.py:138-146` changes only the draft manager mode to `NONE` when `speculative_config.enforce_eager=True`. V01 `:716-742` creates the target manager separately, then initializes the speculator manager. No new patch is needed for this policy. |
| Capture order | V01 `gpu/model_runner.py:1024-1045` captures the target manager, then invokes `speculator.capture()`. V06 `dflash/speculator.py:148-164` and V07 `dflash/cudagraph.py:77-121` prepare and capture the draft with its own manager/factory. |
| Draft context and dispatch | V06 `:410-443` precomputes context KV eagerly, then dispatches the draft manager with its own `num_query_per_req`. It does not use the target runner's gathered-batch dispatcher. |

The existing [DSpark/PD source audit](dspark-pd-source-contract-20261110.md)
continues to own the detailed proposal, rejection, accepted history and Engram
call chain. This increment should keep native `sample_tokens`, rejection sampler,
`post_update`, draft context KV and proposal code. P14/P15 already merge target
auxiliary outputs in parent token order, at native layer-entry timing. A q-aware
Graph change must preserve that behavior rather than adding F-side draft steps.
The donor explicitly excludes DSpark from remote ordinary-MTP routing
(D01 `p2p_hccl.py:354-369`); its per-speculative-step draft headers are not the
V4.1 DSpark protocol.

## Concrete adaptation points

| Current boundary | Required narrow change |
| --- | --- |
| P01 `distributed/hccl_graph.py:99-104`: q=1 and `tokens==requests` | Validate positive fixed q, `uniform_token_count==max_query_len==q`, and `tokens==requests*q`; retain full stage-layout identity and existing limits. A codec that can parse q>1 must still reject keys that disagree with the session's negotiated q. |
| P02/P09 capabilities and binding | Add strict immutable `target_query_len`, default 1 and omitted when 1, with exact HELLO equality and no silent downgrade. Build it from the original target config on both roles; after A model load, verify native `decode_query_len` matches before session use. Existing q=1 bytes must remain unchanged. |
| P10 `runtime_config.py:155-188`, P11 `ffn_model_runner.py:134-146` | Preserve immutable target shape before F's private config clears speculation. Pass that shape explicitly into capability/binding and F MC2 capacity initialization. Keep `speculative_config=None` and `engram_config=None` on F; do not rebuild a drafter there. |
| P03 `hccl_v41_graph.py:89-106,203-222,255,304-330` | Parameterize uniform detection, descriptor validation, capture scheduled counts (`full(R,q)`, not ones), normalization and native dispatch with session q. A FULL parent has `T=R*q`. Eager geometry still uses actual request/token counts. |
| P04 `hccl_v41_graph_control.py:94,256-258` | Normalize absent native `max_query_len` to the descriptor's validated q, and use the same canonicalization for retained-native-descriptor detection. Replacing every value with 1 would collapse the new contract incorrectly. |
| P05/P08/P09 Graph speculation rejection | Replace broad A-side spec rejection only for validated same-checkpoint DSpark, fixed q, paired TP/DP1/PP1/CP1/non-SP, native draft eager and existing model/device ownership. Keep other methods, adaptive/dynamic schedules, PD and LoRA rejected in this wave. |
| P12 F Graph service | Keep the no-drafter/no-Engram F invariant. Validate immutable target q against negotiated capabilities and Graph keys; do not relax the F `speculative_config is None` check. |
| Native producer eligibility | Keep P05 `:340-348` and P13 `:107-110`: unsupported native Engram producer must agree eager before preparation. Supporting q>1 does not establish Engram Graph eligibility. |

MC2 undersizing is conditional: A05 `ascend_forward_context.py:254-259` may use
the scheduler capacity or maximum capture size, so the current q=1 F derivation
is not proof that every Graph configuration allocates too little. Nevertheless,
that derivation loses the target shape and is unsafe as a contract. Preserve the
shape explicitly and test all three native sizing branches; do not restore F's
mutable speculative configuration to recover one integer.

U2 stages are token slices, not complete groups of q tokens. For example, a
uniform parent `q=3,R=3,T=9` splits at token 4: token slices `[0,4)` and `[4,9)`,
request slices `[0,2)` and `[1,3)`. A request legitimately appears in both stages.
Do not add `child_rows % q == 0`, infer child requests as `rows/q`, or replace the
native split with request-block partitioning. Store and verify the exact native
request/token bounds in the key, as the current q=1 path already does. Padding-only
children still require the existing native request clamping and dummy transport
rules. Query width is neither the stage count nor the number of remote MoE calls.

## Prepared-scope review: no confirmed draft contamination

The concern is real as a future regression boundary: P05 `:630-633` keeps its
outer `_capture_scope()` active around the entire native target-then-draft call.
However, reading the actual pinned call chain does **not** establish that the
current hooks wrongly prepare or publish draft work:

1. P06 `hccl_graph_capture.py:43-44` delegates a different manager directly to
   the original capture implementation. Only the exact target manager receives
   `before_factory` and `after_capture` callbacks.
2. The target manager's `after_capture` calls P05 `_finish_phase`; `:506-510`
   completes the target transaction, clears the session and clears current
   execution/context before the native caller reaches `speculator.capture()`.
3. P07 patches only the selected target `EngramModelState.prepare_attn` instance.
   The draft factory builds its own metadata directly via
   `_prepare_dflash_inputs_to_capture` (V07 `:24-70,99-109`); it does not call
   target ModelState preparation or target `HcclV41UBatchRunner.prepare`.
4. Although P16's temporary provider observes every newly created forward
   context, P05 `install_context:443-448` requires an active prepared target and
   exact attention-metadata identity. Draft capture has neither. A04 wraps the
   draft model with native `is_draft_model=True` separately.
5. Native runtime proposal normally runs from `sample_tokens`, after the target
   `execute_model` wrapper has exited. Native `_dummy_run` likewise calls
   `self.execute_model(...)` at V01 `:835-843`, then invokes the dummy/profile
   draft proposal after that call returns (`:850-911`). The draft also uses its
   separate module's dispatcher (V06 `:434-443`), not the target module symbol
   patched by P18. Preserve both boundaries. Native adaptive post-capture
   profiling can re-enter `_dummy_run` inside the outer capture call
   (V01 `:1046-1052`); that broader lifecycle remains rejected in the first wave.

Therefore this is a required **behavior regression**, not a confirmed current
bug to justify a broad compatibility patch. Test an actual native target-manager
capture followed by native draft factory/forward: no additional F transaction,
no target producer call, no target session re-arm, no target metadata/storage
replacement, and no retained target provider after return or draft failure.
Use two distinct manager and metadata identities. Do not prove isolation only
by asserting an internal callback's call count. The existing
`test_completed_dspark_u2_target_leaves_native_draft_forward_attention_local`
already covers an eager target followed by native draft `_run_model`; retain it,
then add the capture-specific boundary. If future work wraps the global
metadata builder or draft dispatcher, first narrow that new scope by the exact
target owner instead of reusing an active target reservation by query width.

The first wave should require `speculative_config.enforce_eager=True` and leave
native `speculator.capture()` in place. Its NONE manager has no draft graphs to
capture. Enabling native full-draft Graph later needs a separate review of draft
state/trace and replay; it need not add F work or alter the target Graph key.

## Minimal implementation and verification sequence

1. **Shape and protocol.** Introduce immutable target q before role isolation;
   extend strict capabilities/codec and key validation with default-q=1 byte
   preservation. CPU cases: positive q=1/q=3, bool/zero/negative/out-of-capacity
   rejection, mixed-peer HELLO rejection before data groups, mismatched Graph
   key q rejection before capture/inputs, and F native shape initialization with
   no draft construction. Do not open Graph admission in this first patch alone.
2. **Target geometry and scoped admission.** Make P03/P04 consume that q; allow
   only the native fixed DSpark target plus eager draft in A coordinator/UBatch
   validation. Preserve native bucket rounding/order, exact key/generation,
   producer eligibility and all-peer fallback. CPU cases should use native
   descriptor generation and `create_ubatch_slices`, including q=3 odd midpoint,
   padded requests, U1/U2 twins, shortened `[1,3]` queries, mixed prefill/decode,
   global F cache miss and all-zero parent. Include the target/draft scope
   isolation regression above and existing aux timing/order/output tests.
3. **Separate component and model evidence.** A weightless controlled HCCL q>1
   probe can qualify payload shape, independent FP32/IDs, U1/U2 Graph reuse,
   mixed-query eager fallback and STOP/native-TP preservation. It cannot qualify
   DSpark proposals or rejection. The manual A5 native-versus-AFD run must then
   record actual proposed/accepted/rejected tokens, aux outputs, next-step
   Engram history/KV behavior and Graph hit/fallback with the same checkpoint,
   sampler and request sequence. Keep the production acceptance guard closed
   until that evidence exists.

No epoch protocol, new remote draft phase, adaptive capture matrix, rewritten
sampler or extra hardware run is needed to start these software changes. Existing
C3 eager DSpark and C5/C6 q=1 evidence remains tied to its original bytes and scope.
