# cross_layer_v1 component evidence

The CPU snapshots below are immutable actual-dependency executions. A separate
device session records the three-layer TP1 controlled-arithmetic Graph
component. Neither scope establishes native model, Engram or performance
acceptance. The production model guard is unchanged.

| Snapshot | Result | Scope |
| --- | --- | --- |
| graph-cross-layer-025 | 647 passed, 5 failed, no skips/errors, 14 warnings, 7.68s | Eight complete mirrors: schedule, codec, bank, pipeline, A model, F model, coordinator and service |
| graph-cross-layer-026 | 167 passed, no failures/skips/errors, 14 warnings, 1.35s | Complete bank 43 and pipeline 124 mirrors after fixing the five test failures |

025's five failures were two old bank-error regex expectations and three new
pipeline-test comprehension variables shadowing `rows`. The follow-up keeps
the original bulk error text, gives cross-layer its distinct split-action
message, and fixes the test variables. Only those three files changed; the
other six mirrors passed at unchanged bytes: schedule 18, codec 153,
A Graph 64, F Graph 89, coordinator 88, service 73. This is two executions,
not one combined 652-pass run. Original failure XML/logs remain intact.

Both snapshots matched the installed upstream source pins. Each label retains
identity, full source manifest, stdout/stderr, JUnit XML, summary, source checks,
package versions and environment. `validation-summary.json` indexes those
raw hashes and the exact changed files. The final twenty-file Python source
freeze is `../graph-cross-layer-final-snapshot-20261110.json`.

CPU cases cover the shared three-layer paired action sequence, layer-major
native compute ordering, positive padding-only and static zero-row stages,
one-positive-stage degeneration, original FP32/IDs, native continuation/aux/MTP
outputs, per-stage output copies when consecutive fake MoE calls share a CPU
workspace, failure retention/reentry refusal, exact HELLO order negotiation,
and all-peer unsupported-policy preflight with empty eager retirement.
TP1/TP2 values in those tests are configuration/role seams; no device collective
or native quantization kernel is exercised.

The separate NPU component session below checks matched blocking warmup and
Graph capture/replay of the new wire order. Every physical HCCL operation
remains on its capture parent; this does not complete the physical
communication-stream or timeline requirements of C6 task 1.6.

## Three-layer probe preparation

`cross-prepare-001/identity.json` records a local prepare-only invocation with
`--stage-bank --exchange-order cross_layer_v1 --layer-count 3` and all five
Graph axes set to 1. It made no SSH connection and started no device work.
The snapshot contains 141 source entries, 31 locked upstream pins and 142
deduplicated archive members; its canonical source-manifest SHA256 is
`4dd1a582eeeefdfe5a785ca1ebf175cb92e0f315cb396a76614dfc077246229b`.

The extra probe and harness sources add a 2/3-layer choice, preserve two layers
and `stage_serial` as defaults, and execute the production shared action
schedule for cross-layer mode. The original five numerical/retirement helper
ASTs match HEAD exactly. Eight invalid-argument checks reject before remote
work. Both extra sources pass Ruff, Python 3.11 parsing and diff checks; all
twenty frozen implementation/test files still match the final CPU snapshot.
`../graph-cross-layer-probe-validation-20261110.json` records exact hashes and
the static checks. These preparation results are separate from device results.

## A3 three-layer TP1 device result

`cross-layer-tp1-001` passed on the first authorized execution: FFN device 0
and Attention device 2, three layers, `cross_layer_v1`, stage banks and all
five Graph axes set to 1. Both rank processes exited 0, with no timeout,
supervisor cleanup or owned-process kill. All 31 upstream source checks
matched; eight boards were idle before and after. The harness took 30.94s.
No package or upstream source was changed, and no retry was run.

Each rank completed 26 transactions: four warmups, four captures, twelve
replays and six agreed eager cache-miss fallbacks. The four fresh U2 resources
per rank each recorded six compute steps, two tail pairs and the production
three-layer action list:

```text
I00 I01 O00 I10 O01 I11 O10 I20 O11 I21 O20 O21
```

The unchanged independent oracle checked original FP32/router deltas, BF16,
raw IDs, exact outputs, fixed bank addresses, stage recurrence and the final
continuation. Replays used full, one-token and T-1 valid counts, including a
padding-only positive stage. Warmup/capture used valid count zero. Native TP
allreduce returned `[1]` before and after normal STOP, cache/group close.
U1 descriptors retained serial parent-stream exchange; U2 used compute-stream
arithmetic and parent-stream HCCL. The reported action list is a static plan,
not a measured device timeline.

The actual upload manifest is
`8539a7e899605d997f825c9fdcc7eee8084bd802928844aa190d6b1962516830`;
the raw harness summary SHA256 is
`8f996c28d6488ddf9b83b4b68cfd6e0ae416813973006ec1b9ebef26f80c40c7`.
The eight production files and two probe/harness files match their frozen
bytes. Three parallel runtime-config/F-worker files differ from prepare-only;
this arithmetic session does not qualify their model/DSpark behavior.

`../graph-cross-layer-device-validation-20261110.json` records per-rank
assertions, all 17 raw-file hashes and the preparation-to-device manifest
differences. `validation-summary.json` indexes CPU and device executions
separately. The device run used real base `CudaGraphManager`,
`NPUUBatchRunner`, NPUGraph and HCCL with controlled arithmetic. It did not
load checkpoint weights or execute native Attention, HC/pre_mix, Engram, MoE
or quantization. TP2 device, static zero-row stage device behavior, GE,
physical communication streams, overlap timelines and performance remain
outside this result.
