# Five fixed Graph axes: A3 component evidence

All four initial independent authorized device sessions passed. No retry,
timeout, inner/outer cleanup signal or package/upstream change occurred.
Every session checked all 31 installed-source pins and eight idle boards
before and immediately after its devices 0/2 run. Native TP was `[1.0]` before
and after plugin groups closed on both ranks.

Tuple order is compute, hybrid, Attention-three, F-recv, F-cross. All sessions
fixed eager overlap and diagnostics to zero and used the unchanged Graph wire.

| Initial raw session | Tuple | Elapsed | Result |
| --- | --- | --- | --- |
| `axes11111-001` | `1,1,1,1,1` | 30.20 s | PASS; active U2 pipeline, U1 pipeline None |
| `axes00000-001` | `0,0,0,0,0` | 30.66 s | PASS; all eight warmup/formal selections pipeline None, still Graph/U2 |
| `axes10000-001` | `1,0,0,0,0` | 30.18 s | PASS; canonical parent frontier and F per-layer joins |
| `axes10010-001` | `1,0,0,1,0` | 30.44 s | PASS; F-recv selected, F per-layer joins retained |

Each rank completed 26 parents: four warmups, four captures, twelve replays and
six intentional eager fallbacks. Four fixed keys cover U1/U2 and bucket 4/8.
Each Graph key replays controlled full/1/T-1 validity while physical native
dummy geometry stays fixed, including a zero-valid U2 stage and partial native
padding. BF16, original FP32, IDs, every layer output and Attention final
continuation match the same CPU recurrence. Dirty padded IDs are sanitized in
the graph bank body; F output masks and bank addresses/byte budget are checked.

The actual base manager/NPUBatchRunner constructors, InputBatch slicing,
GraphControl reservations, stage bank, pipeline streams/events and HCCL Graph
operations execute. Logical send/recv choices remain under the mandatory
**physical parent-stream override**. This is controlled component evidence,
not native V4.1 Attention/HC/indexer/Engram/MoE/ModelState/ModelAclGraphManager,
physical independent communication streams, exhaustive axis combinations,
eager tickets, allocator-pressure, timeline or performance acceptance.

The accepted source review and software plan are in
[c6-graph-axis-probe-design-20261110.md](../c6-graph-axis-probe-design-20261110.md).
`static-check-001.json` records Ruff, formatting, Python 3.11 AST, CLI help and
eight parse-only checks. `archive-precheck-001/identity.json` is preparation
only, preserving its own source identity before device snapshots.

All device sessions execute probe SHA
`1d7f7b1ef445dd852026d66919bc75c20e62b429e95c66e5ac47e2e286213365`,
harness SHA
`4a3e0b65a33562c9e63735f5dc0baa335522cbe468a58e53cde89b4f6773ac56`,
production pipeline SHA
`2105627f04b4693c44d5fc12c4ffac318cf540f6709fbb86228b60576540c066`
and overlap-selection SHA
`04391ecd16bbf4210bb978f6c7be7a16ab53a68f74dc9ad5e1ae1d452865a34c`.
Actual HEAD is `8a6eeb2a81506621f8571784179a46e3f4680d95` for all snapshots;
working-tree manifests, rather than HEAD alone, identify executed bytes.

First three session source manifests are
`242ce38a741bf5b3f08aebff06818ca0a1f6c1c43b173fef00c9622ec97f2899`;
`axes10010-001` has
`11d12b23552a0db535ec4182e5aff72296da48596488d68761533ae6b6692534`.
The complete plugin snapshot records concurrent source changes as identity;
specifically the fourth snapshot adds `hccl_eager_pipeline.py` and records
transport SHA `d99eaa6c42502e1695aaa4646ec9ed3782a459eb203eb73dc8ef9f8089e3bf48`
instead of the earlier `9ae40a90` transport. The probe does not open an eager
owner, and its Graph pipeline/overlap/probe/harness bytes are identical. The
transport was an in-progress eager candidate at snapshot time; its Graph-path
delta is an exclusivity guard when an eager owner is active, while this probe
never opens one. This is preserved as a separate actual candidate result,
without claiming it was the previously frozen transport or merging manifests.
Only the scope above is executed. Per-session `identity.json` retains every
source hash and command. `results/summary.json`, `results/probe/rank-*.json`
and original logs retain exact first outputs. `validation-summary.json` indexes
all raw file hashes, settings/projections, phase counts, validity and identities.
