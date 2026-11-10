# A3 DSpark target-shape CPU evidence, 2026-11-10

This directory records immutable CPU snapshots in the deployed A3 venv.
No device probe, checkpoint, package installation or upstream modification
was performed. Installed FastAPI remains 0.123.10. Eleven installed source
pins matched in each run, including the actual Ascend MC2 capacity helper.

| Snapshot | XML result | Exact scope |
| --- | --- | --- |
| [component-001](component-001/results/summary.json) | 263 passed, 10 failed, 0 errors/skips | Config 97, FFN worker 47, FFN runner 19 and legacy runtime 55 all passed. Optional Async startup module passed 45 and failed 10 on old fixture fields/factory names. |
| [component-002](component-002/results/summary.json) | 53 passed, 2 failed, 0 errors/skips | Async startup module only; two remaining old `__new__` runner fixtures lacked `prof`. |
| [component-003](component-003/results/summary.json) | 55 passed, 0 failures/errors/skips | Complete Async startup module after tests-only fixture repair. |

The attempts are separate results; this index does not turn them into one
combined successful run. The three production source hashes stayed fixed
through all attempts. The immutable identity manifests contain the actual
HEAD and complete uploaded working-tree hashes, including unrelated files;
the claims above apply only to the selected targets and their documented seams.

The worker's eighteen new capacity cases execute the plugin
worker→factory→synchronous runner→loader lifecycle with CPU-only boundary
fakes. They consume the actual native MC2 helper's result as model workspace
size for q1/q3/q6, TP1/TP2 and three native sizing branches. This verifies
configuration/isolation/capacity behavior, not NPU MC2 or model inference.

Reproduce the initial scope with:

```powershell
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-full-parity/evidence/run_a3_dspark_shape_cpu.py --label <fresh-label>
```

Async-only closure uses `--targets tests/unit/test_async_cam_startup_wiring.py`.
The harness creates a fresh remote scratch, verifies every uploaded file and
installed source pin, runs pytest under the existing CANN environment and
returns raw XML/logs. It does not initialize an NPU device. New labels are
required; existing run directories are never overwritten.

See [validation-summary.json](validation-summary.json) for per-module XML
counts, source identities, commands, current-at-freeze comparisons and raw
SHA256 values, and [the implementation contract](../dspark-target-shape-implementation-20261110.md)
for the retained-shape ownership chain. The original
[source audit](../dspark-graph-source-contract-20261110.md) and
[independent review](../dspark-target-shape-independent-review-20261110.md)
keep source/software findings separate from pending HCCL, Graph and full-model
DSpark qualification.
