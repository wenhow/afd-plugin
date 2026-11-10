# Uniform target-query Graph/HCCL component candidate

The later bounded actual run is recorded separately in
[target-query-cross-tp1-001](target-query-cross-tp1-001-README.md). The static
candidate and preparation records below retain their original scope.

This is a prepared artifact, not a device result. The two artifact files are
frozen at the hashes in [static validation](probe-static-validation-20261110.json).
No SSH, NPU work, checkpoint loading, draft model, package installation, or Git
delivery action was performed for this candidate.

The existing probe and harness now accept `--target-query-len 1|3` (default 1).
The real native `CudaGraphManager` receives this value as `decode_query_len`.
Both roles propose the same `HcclPlanCapabilities.target_query_len`; the existing
strict HELLO and keyed graph-reservation checks enforce exact agreement.
`InputBatch.make_dummy` receives actual requests `R=T/q`, tokens `T`, and
`max_query_len=q`. Native `create_ubatch_slices` supplies the prepared layout.
No child-row divisibility rule is added.

For q=1, capture sizes remain `(4,8)` and each rank retains the original 26
transactions: four warmups, four captures, twelve replays and six eager
fallbacks. The capability encoder still omits the default q=1 field; the
separate [protocol CPU result](../a3-dspark-graph-cpu-20261110/protocol-README.md)
checks its exact old HELLO bytes. This candidate does not repeat that CPU run or
claim a new q=1 device result.

For q=3, native token capture sizes are `(9,12)`, representing three and four
requests. The q=3/T9/U2 prepared layout must be exactly:

| Stage | Requests | Tokens | Padded rows |
| --- | --- | --- | --- |
| 0 | `[0,2)` | `[0,4)` | 4 |
| 1 | `[1,3)` | `[4,9)` | 5 |

All four captured buckets retain their full/1/T-1 controlled-valid replay
series. A separate final transaction uses actual R2/T6, whose native padded
layout matches the same T9/U2 key; its valid counts are exactly `(4,2)` and its
fixed bank capacities remain `(4,5)`. Thus q=3 expects **27 transactions per
rank**: four warmups, four captures, thirteen replays and six eager fallbacks.
The extra transaction checks native request-layout equivalence without replacing
the prior numeric coverage. These device-valid scalars are controlled component
inputs, not DSpark acceptance or rejection decisions.

The three A layout-miss transactions instead use actual R1/T3. Before device
input preparation, the public preview must show child 1's request start clamped
from cached 1 to actual 0; the key differs and all peers choose eager. The three
F generation-miss transactions retain their original schedule immediately after
the twelve baseline replays. There is no injected private cache state or forced
graph dispatch result.

`payload`, `verify`, `observe_bank`, `native_check`, and `complete_active` have
identical ASTs to HEAD `e18be5ca611bb9597b05f34b51d4e0883e4f1e0b`.
The BF16/original-FP32/raw-ID arithmetic, all-row oracle, padding comparisons,
fixed-address checks, per-layer recurrence, STOP and native TP checks are not
relaxed. The default stage-serial path remains available. The prepared q=3 case
selects the existing three-layer `cross_layer_v1` static schedule and all five
Graph axes; it does not establish physical side-stream HCCL or measured overlap.

## Static and preparation checks

- Python 3.11 grammar, Ruff check/format, and scoped `git diff --check` pass for
  both artifact files.
- Probe CLI default q=1 and explicit q=3/three-layer/cross arguments parse.
  Both artifacts reject q values `0`, `2`, `-1`, and `True` before execution.
- [Default prepare-only identity](prepare-q1-001/identity.json): 140 manifest
  entries, 141 archive members, 28 expected source pins. No streams are requested
  in this default preparation.
- [q=3/cross prepare-only identity](prepare-q3-cross-001/identity.json): 141
  manifest entries, 142 archive members, 31 expected source pins. Manifest SHA256
  is `280493e27b3dee708a40ad0cf24954d7a9d6646a38cab0c7c1e380374e2ab2b2`.
  Archive members were deduplicated and their bytes checked during preparation.

Preparation collects the locked local Git objects and previously recorded
installed stream API hashes. It does **not** verify the current remote packages,
eight-board idle state, graph capture, tensor transport, or device behavior.
Those checks remain in the existing bounded actual-run harness.

## Pending bounded command

Run only after root's separate device authorization and the geometry CPU review
have closed. The harness checks all eight boards idle and uses only FFN NPU 0
and Attention NPU 2. It retains the 31 installed-source checks, inner/outer
timeouts, owned-PID supervision, native TP before/after, normal STOP and final
eight-board idle check. A failed first run must retain its fresh raw directory.

```powershell
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/run_a3_graph_manager.py --label target-query-cross-tp1-001 --target-query-len 3 --layer-count 3 --stage-bank --exchange-order cross_layer_v1 --graph-compute 1 --graph-hybrid 1 --attention-three 1 --ffn-recv 1 --ffn-cross 1 --evidence-dir openspec/changes/afd-hccl-full-parity/evidence/a3-target-query-graph-20261110
```

This candidate covers controlled arithmetic over native base-manager and HCCL
components, with independent native TP1 groups. Full-checkpoint target/draft
execution, accepted-token counts, native sampling, MoE/HC/Engram, KV/history,
TP2 device, A5 acceptance, GE compilation and performance remain unverified.
