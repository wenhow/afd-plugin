# DSpark target-query Graph consumers: independent review

Read-only review of the eight production files listed in
`dspark-graph-consumer-candidate-20261110.json`, compared with the pinned native
contracts and the existing protocol/admission reviews. No source blocker found
for this explicit target-query increment. General model admission remains
closed; this is not DSpark inference or device acceptance.

| Production file | SHA256 |
| --- | --- |
| `afd_plugin/distributed/hccl_graph.py` | `3d410da36df91d875a2c595c6f473e8d15b82b26ee250dea23126b138bce9302` |
| `afd_plugin/connectors/npu/hccl_graph_stage.py` | `6e9d4f938632fadf14a0ea6d984deab2c5a0bd6c849aeede685a4a0212ff439c` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph.py` | `cb19a7b5caab26fc757f4d8ad85ab9b17a64513bf5353e63efd33677541dd20e` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph_control.py` | `c1cac3839e2aea64811fc832e8462eb626c36dc4ac832382a0d2f53966de3ee1` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph_coordinator.py` | `1bec79e6cc1c9ec03ad18e2b26cb33537dfbd09d013711fff874974e02c3583f` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph_service.py` | `5cd4e932b15584166ac55900751d4d7cd2eea00b9a8d8a65938e9c2139daf8d4` |
| `afd_plugin/v1/worker/npu/ffn_worker.py` | `6b55fd75e9f82cfcebda08b7f149735a871aded253db25b118e5eb2748381121` |
| `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `44e03cf92e93680f797f75734720cd70d6cd6959d51a3e28a15aed273173eb1d` |

All eight hashes were independently matched to the candidate and current
files. Native references remain vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`
and Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`.

## Query propagation and preparation boundary

`HcclGraphKey.__post_init__` now requires positive strict-integer q, matching
uniform/max-query widths and `T=R*q`, without changing stage geometry or LoRA
exclusion. `HcclGraphStageBank.__init__:66-73` additionally binds both key widths
to the negotiated session q before using tensor storage. A child need not end
on a q boundary: native T9/R3/U2 produces token extents 4/5, preserving native
request clamping instead of independently rounding child counts.

`hccl_v41_graph.py:124-189,213-237,296-364` preserves the pinned
`maybe_create_ubatch_slices` plus native request-bound clamping. Capture derives
the uniform vector from q; runtime uses the actual ordered gathered vector.
Dispatch passes q to the actual U-specific native manager lookup. Prefill,
profile, LoRA and native misses use fresh actual eager geometry before input
preparation. `None` max-query in a FULL native descriptor is canonicalized to
the session q for keys/duplicate checks, retaining the actual native descriptor
for manager lookup. Same T/U with a different request layout remains a miss
or rejection before tensor execution; it is not silently renamed a cache hit.

`HcclV41GraphControl.reserve_batch/reserve_capture` retain the early solicited
agreement, authoritative A-peer request/token preview, cache generation and
native manager graph identity. `verify_prepared:395-425` still requires the
same returned handle and accepted transport decision, exact actual geometry,
current generation and metadata key. No late fallback, host JSON, `.item()` or
COMPLETE is introduced into a tensor body.

## Native ownership and isolated FFN role

Coordinator construction `:149-155` compares the validated original target
shape, native runner decode q and negotiated q. `_require_storage:265-311`
revalidates shape and q with all existing native model/ModelState/input/KV/
manager identities before capture/runtime work. It retains the native
ModelWithContext, Engram parent producer, ordered auxiliary output and native
continuation owners. The new predicate requires eager draft and adaptive
verification off; Graph PD remains excluded by the accompanying admission
contract. It does not turn a draft forward into an F transaction.

The FFN worker/runner binders pass the already retained immutable target shape
to their public session validation. FFN service `:168-240` checks its isolated
spec/Engram/KV-free configuration and target shape against q each step before
receiving a reservation. `_resolve:297-365` checks key q before readiness;
bank allocation/count updates remain after accepted actual counts, outside
capture. Cache resources keep their exact model/bank/context/endpoint identities.
FFN target q is scheduling capacity, not a retained drafter or draft Engram.

## CPU evidence and its limits

Independently read `a3-dspark-graph-cpu-20261110/geometry-001` identity, source
checks, summary and XML. All 21 installed-source checks are true and the
summary reports source files matching, no device probe, and exit 1. The nine
mirrors record **551 passed and 4 failed**, without errors or skips. Eight
mirrors pass completely (455 cases); coordinator records 96 pass/4 fail.
The four failures are the q2/q3 by U1/U2 replay cases reaching the unchanged
native DP1-rank-zero guard: the shared fake incorrectly supplied decode q in
the positional DP-size field. The test-only correction passes actual parallel
configuration DP size/rank explicitly. Production bytes above are unaffected.

The newly added four coordinator cases pass in this snapshot with mirror
`7e26e6dc06fe3226780dd06c63251fafb6728ff4e7cbe91a55677fc0b4a260fc`
and fake `c317e797c9d247984c70b596fe17a9f416c900216ee6b111190b70352382fc18`.
They execute real pinned `GPUModelRunner.capture_model:998-1070` target-to-draft
ordering, real DFlash NONE-manager capture, and an explicit real
`DFlashSpeculator._run_model:242-265` after target publication while the target
scope still exists. The latter is an additional CPU isolation check: native
NONE-manager capture has no descriptor and does not itself call a forward.
Device memory/workspace and target factory/capture are explicit CPU boundaries.

For U1/U2 and draft success/failure, public transcripts remain unchanged after
target publication; draft metadata differs from target metadata and has no
target sidecar or AFD metadata. Target producer outputs and target cached
output identity/storage/value survive, retained graph handles remain usable,
and fresh forward contexts are clean. Draft failure makes the owning session
terminal while retaining target cache resources. This does not test native
DSpark proposal, accepted/rejected history, operators, HCCL or device capture.

Costs are additional q consistency comparisons and existing CPU geometry work;
wire payload and per-stage tensor ownership do not change. Remove the plugin q
projection/control once native A/F exposes equivalent early uniform-target
agreement and exact layout ownership, preserving old q1 wire compatibility.

## Narrow follow-up: geometry-002

Independently read `geometry-002/identity.json`, `results/source-check.json`,
`results/summary.json` and `results/cpu.xml`. All 21 installed checks are true;
the snapshot reports source matching, no device probe and exit zero. The
complete coordinator mirror records **100 passed**, zero failures/errors/skips,
in 2.876 seconds. Mirror SHA remains
`7e26e6dc06fe3226780dd06c63251fafb6728ff4e7cbe91a55677fc0b4a260fc`;
the corrected shared fake is
`e0c5d683638567895e926f730926b89b05395d660a5e2c9e5e30f623841ce048`.
Both hashes match the actual snapshot and current files. The correction is
limited to supplying native DP size/rank in `dispatch_runtime`, without changing
the four isolation cases or production code. This closes the four fixture
failures for the complete coordinator module; it does not turn historical
geometry-001 into a green nine-module rerun.
