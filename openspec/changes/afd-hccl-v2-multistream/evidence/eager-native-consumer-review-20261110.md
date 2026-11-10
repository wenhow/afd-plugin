# Independent eager native-consumer source review

Reviewer: transport owner; independent of the Attention and FFN consumer authors.
This is a source review. Actual CPU and device results remain separately owned
raw evidence. No native model, MoE, or overlap acceptance follows from this text.

## Attention consumer

Reviewed the root-authored `deepseek_v41_staged.py` eager delta at initial SHA256
`0ed64e929443523ca41c1f92614550e7c04d73d4df90824923742414a082e203`.
The existing decoder continuation preserves native HC/pre_mix: the next layer
first borrows and consumes the preceding layer's output, then observes auxiliary
state, waits the original Engram events, and enters native Attention/norm. BF16
and original FP32 contiguous copies run in the actual compute scope. Both stage
posts precede the first output consumption. All four continuation tensors,
hidden/pre_mix and accumulated auxiliary tensors are explicitly retained by
posts and loans. The final compute event joins the parent before MTP copying,
HC collapse, norm and parent output merging. Empty stages have no ticket;
shared-Attention bindings and forward context restore in their existing scopes.
No blocker was found in those source boundaries.

The reviewer raised the same parent TLS risk fixed earlier for Graph: selecting
side compute alone restores a possibly foreign vLLM cached parent stream. The
root subsequently reports wrapping the complete eager parent in
`use_npu_current_stream(actual_parent)` with public-query tests. Those later
bytes and test results are owned by the root's immutable CPU evidence, rather
than silently relabelled as the initial source hash reviewed here.

## FFN consumer

Final reviewed source:

| File | SHA256 |
| --- | --- |
| `afd_plugin/v1/worker/npu/hccl_v41_ffn_eager.py` | `5e7183dea3436868aa1e22cb04056ffabb0ca714532b7090b623f0d56d2624ac` |
| `tests/unit/fakes/hccl_v41_ffn_eager.py` | `7da2d26ae0027a3d8b97ae6d4eb9ff2c1207af7c0ac3ccd6086dbb4e64ab20e6` |
| `tests/unit/v1/worker/npu/test_hccl_v41_ffn_eager.py` | `1e4a001160ecee25c284f5fa859fb080b5428cbccec8dd1f8862c75998bebd59` |

The FFN consumer checks exact live owner/plan identity before its failure scope,
preserves actual parent stream/TLS around the full operation, and validates
paired TP1/TP2, native rank, DP1/PP1/CP1/non-SP and speculation exclusion. Each
stage posts its receive, enters the actual compute loan and native Ascend
forward context, calls the full native FFN model once, closes the input loan,
and posts the output from the actual producer stream. It adds neither a TP
reduction nor output padding masking. Existing child metadata remains
stage-projected, matching the previously reviewed serial FFN contract.
Whole-empty parents require no model construction and finish honestly.

Two review findings were closed before final freeze: TP2 rank1's scripted global
ACCEPT now supplies coordinator `world_rank`; stale/foreign/finished owner
rejections occur outside the failure scope, and the exception handler fails only
the exact still-live same-plan owner. This preserves the original exception and
prevents an old owner from terminating another parent. Final reviewed source
has no remaining blocker in this explicit component scope.

The probe agent reports actual-dependency `eager-ffn-013` passed all 43 FFN
consumer cases and the 50 transport-owner cases. Its wider run contained two
unrelated old Graph worker fixture errors, separately closed by
`worker-followup-014` with 114 passed. Native FFN device/MoE and delayed model
scheduling remain unaccepted; the separate controlled-arithmetic TP1 owner run
does not exercise this native FFN consumer.

## Final direct Attention execution boundary

Independently read final `hccl_v41_execution.py` at
`0e91c47b3d52bdc55b1580808b468d18074b349ea6c86c66904214243bb3ffab`
and the final staged driver at
`8017c4095045ee735caef73caf7c80c6d0f825139f8d14871120cdf29d6845de`.
The direct public execution API verifies exact transport, accepted plan and
current owner identity, plus failed/finished state, before the producer or its
terminal exception handler. Rejecting an old or foreign owner therefore does
not consume Engram input, call the foreign owner's fail method, or terminal a
different wire. The actual parent stream check also precedes production.

The common `use_npu_current_stream(actual_parent)` scope covers parent Engram
production, all side-compute entries, final parent collapse/output, and owner
finish. Inner compute scopes restore that selected parent, and its outer
finally restores the caller's prior TLS even on failure. Pipeline None retains
the synchronous null scope. No remaining blocker was found in this direct API
and full-parent TLS boundary. Root reports the dedicated 016 execution subset
passed 28 actual-dependency CPU cases; its raw snapshot is distinct from this
source review and from the actual arithmetic owner TP1/TP2 probes.

## Final binding ownership check

The FFN binder at `ffn_model_runner.py:bind_hccl_v41_eager_session` delegates to
`hccl_config.validate_staged_hccl_binding` after checking the initialized session
and exact connector transport. That shared validator rejects closed/terminal
transports, an active plan, a Graph reservation when capability v1 is negotiated,
and mismatched model, rank, capacity or indexed device. A live Graph reservation
is unavailable through the public capability-v0 eager API. The initial reviewer
concern that the FFN binder lacked live/idle checks was disproved by tracing
this actual delegated validator; no duplicate inline checks are required.
The A runner also checks active/idle/exact session state before binding. No
remaining blocker was found in these explicit session binding boundaries.
