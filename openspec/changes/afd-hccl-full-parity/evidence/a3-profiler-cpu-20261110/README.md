# C7 profiler ownership: actual-dependency CPU evidence

`profiler-owner-012` completed with **99 passed**, no failures, setup errors or
skips (JUnit 2.124 s). This independent A3 snapshot used actual installed runtime
packages, matched all ten upstream source pins, and started **no NPU probe**.
No packages or upstream files were changed.

| Selected module | Passed |
| --- | ---: |
| `tests/unit/compat/npu/test_profiler.py` | 26 |
| `tests/unit/v1/worker/test_npu_runtime.py` | 55 |
| `tests/unit/v1/worker/test_async_ffn_runtime.py` | 4 |
| `tests/unit/compat/patches/test_signatures.py` | 14 |

The profiler seams exercise the owner's scheduled start/stop/step lifecycle,
thread ownership checks, failure propagation and runner/worker integration.
Native profiler device calls are explicit CPU seam boundaries. This does not
prove torch-npu's C++ profiler is thread-local or global, qualify on-device
recording, or establish device performance. The separately collected installed
Python profiler source remains
[source-only evidence](../../../afd-hccl-v2-multistream/evidence/torch-npu-profiler-source-20261110.json).

The actual checkout HEAD was `8a6eeb2a81506621f8571784179a46e3f4680d95`; the immutable manifest digest is
`30363caa92f1d143a75fb8ca916dc432d94bdd3a69e90507bf8208e30af72f1b`. The [source identity](profiler-owner-012/identity.json)
contains the complete production, fake and mirrored-test SHA256s and all ten
expected upstream source hashes. The [index](validation-summary.json) records
per-module outcomes, the exact command and every raw artifact hash. Its later
comparison with current source bytes is advisory: an older run does not accept
changed or missing bytes.

Raw evidence: [summary](profiler-owner-012/results/summary.json),
[JUnit](profiler-owner-012/results/cpu.xml),
[stdout](profiler-owner-012/results/cpu.stdout.log),
[stderr](profiler-owner-012/results/cpu.stderr.log),
[delivery](profiler-owner-012/delivery.json),
[remote transcript](profiler-owner-012/remote-run.log),
[versions](profiler-owner-012/results/versions.json),
[environment](profiler-owner-012/results/effective-env.json) and
[source checks](profiler-owner-012/results/source-check.json).
Use a fresh label and the exact target list in the identity when reproducing
with `run_a3_graph_cpu.py`; never overwrite this attempt.

The [C6 index](../../../afd-hccl-v2-multistream/evidence/a3-pipeline-cpu-20261110/README.md)
retains the independently executed Graph/eager consumer attempts, including
44 owner fixture failures in 009, 13 session fixture failures in 010 and two
Graph worker setup errors in 013 with their separate follow-ups. Counts from
those attempts are not added to this 99-case run. Full-model and A5 acceptance
remain separate manual validation work.
