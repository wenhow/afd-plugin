# C6 compute pipeline: controlled NPU graph component

Both first device attempts passed on A3 devices **0 and 2**, with one Attention
and one FFN process, each retaining its independent native TP1 group. The same
source manifest was used in two separate sessions:

| Session | Hybrid policy | Parent transactions per rank | Outer elapsed | Board idle before/after |
| --- | --- | --- | --- | --- |
| [hybrid1-001](hybrid1-001/results/summary.json) | per-stage receive ready | 26 | 30.70 s | 8 / 8 |
| [hybrid0-001](hybrid0-001/results/summary.json) | parent ready fence | 26 | 30.29 s | 8 / 8 |

Both workers exited 0 in both sessions. All 31 installed upstream/API source
checks matched. There was no timeout, supervisor cleanup or killed process.
Elapsed values include setup and supervision; they are not a performance
comparison. The final eight-board idle checks passed immediately in both runs.

This is real NPU stream/event and NPUGraph capture/replay around the production
`HcclGraphComputePipeline`, tensor bank and paired HCCL operations, with controlled
arithmetic. The existing real base CudaGraphManager, NPUUBatchRunner constructor,
InputBatch slicing and early GraphControl reservations participate. It does not
load checkpoint weights or execute native Attention/HC/indexer/Engram/MoE,
ModelState/ModelAclGraphManager or the production native-model A/F service path.
The snapshot contains model runtime files; their presence does not establish
that this component probe executed them.

Each rank completed four warmups, four formal captures, twelve replays and six
early eager fallbacks. Formal graph bundles retain their own pipeline, bank and
observation/producer storage. Warmup and capture use distinct phase owners and
zero valid rows. Attention records each layer's side producer arithmetic before
parent HCCL, makes the next layer depend on its returned output, then performs a
final side continuation and parent join. FFN records parent receive, side
arithmetic, masked parent send and final join. Per-layer observations preserve
earlier results when the bank is reused across layers. BF16 inputs/output,
independent FP32 values, sanitized IDs and final Attention continuation matched
the CPU oracle exactly; retained bank addresses/byte budgets were checked.

Every captured key kept its physical native dummy geometry and graph key fixed
while its three replay valid counts changed:

| Physical rows | Replay valid counts |
| --- | --- |
| U1 `(4,)` | `(4,)`, `(1,)`, `(3,)` |
| U1 `(8,)` | `(8,)`, `(1,)`, `(7,)` |
| U2 `(2,2)` | `(2,2)`, `(1,0)`, `(2,1)` |
| U2 `(4,4)` | `(4,4)`, `(1,0)`, `(4,3)` |

The accepted plan drives both peers' device-valid scalar refresh and oracle
masking. This includes a zero-valid child with positive static rows and partial
padding across stages. It does not claim that a smaller native request layout
hits the same graph; the existing request-shape miss still selects eager. The
FFN generation miss also selects eager before tensor payload, with no replay of
the stale graph. Native TP allreduce returned `[1.0]` before and after closing
the plugin groups on both ranks.

The immutable manifest is
`ae1b03e598476991dc259b321c0b487cb3a0e936b10e44efce4b5b27a1ae08d7`,
and both actual HEAD records are
`e8557317dd66efd8971566c758456ad2308c6e00`. Main executed source pins are:

- probe: `d2e780a99f080236a85fd3cbba1844d1b7077fac07c0dca62c52754433cccb87`;
- immutable harness: `746be9cebc107cf5b57957fe8af407a502b1a929b7b3654d7c2bd13c5e81d963`;
- production pipeline: `6473cc7ad538ddeaf45a5a9233eef7d5bae79b1cb1e6c95b883953ef05db73cd`;
- shared NPU/TLS stream helper: `6a881fa579e9c775f3b7142c7c456936e3d05c2ea37f9a87f3bcc386e7656771`.

`source-expected.json` and the installed torch-npu API raw JSON are included in
the per-file manifest. The independently collected installed stream/API source
pins were checked before importing the runtime or starting the probe. The
existing CANN/ATB environment and FastAPI 0.123.10 were preserved; no package or
upstream source was changed. Package versions and effective paths are retained
in each session's raw results.

Run from the checkout root, using fresh labels for future attempts:

```powershell
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/run_a3_graph_manager.py --label hybrid1-001 --evidence-dir openspec/changes/afd-hccl-v2-multistream/evidence/a3-pipeline-device-20261110 --stage-bank --compute-pipeline --pipeline-hybrid 1
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/run_a3_graph_manager.py --label hybrid0-001 --evidence-dir openspec/changes/afd-hccl-v2-multistream/evidence/a3-pipeline-device-20261110 --stage-bank --compute-pipeline --pipeline-hybrid 0
```

Raw evidence for each session includes `identity.json`, `delivery.json`,
`remote-run.log`, `results/summary.json`, `results/source-check.json`,
`results/upstream-identity.json`, `results/effective-env.json`,
`results/versions.json`, `results/owned-processes.json`, `results/npu-before.log`,
`results/npu-after.log`, `results/probe.log`, and each rank's JSON/log under
`results/probe/`. The [machine-readable summary](validation-summary.json)
checks both workers' phase totals, exact per-key valid schedules, unchanged
source manifests, TP results and successful immediate idle checks.

The separate [preflight](preflight-001/summary.json) used only stdlib source/env
reads, `npu-smi` and `/proc`: no torch import, no workload, eight idle boards and
no matching probe/test process. The [archive self-check](archive-precheck-002/archive-precheck.json)
verified 134 unique tar members and all 133 manifest hashes before any network
call. Its earlier local runpy relative-path setup error is preserved in
[archive-precheck-001](archive-precheck-001/setup-error.json); it did not execute
the runtime or device and required no harness source change.

No timeline was captured, so this does not establish actual compute/communication
overlap or throughput gain. Long allocator pressure, long-running request/bucket
switching, the six overlap settings' runtime integration and full-model
equivalence remain separate acceptance work. Old C5 device evidence is unchanged.
