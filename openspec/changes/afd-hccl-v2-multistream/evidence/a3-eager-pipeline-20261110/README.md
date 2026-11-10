# A3 actual eager owner component evidence

This directory preserves actual ticket/loan owner runs, separately from the
earlier raw endpoint capability gate. No model weights or native MoE are loaded.
The probe checks two-stage posting and storage reuse across layers; it consumes
all outputs before beginning the next layer and therefore does not substitute
for the native model's delayed HC schedule or establish overlap/performance.

| Session | Scope | Result |
| --- | --- | --- |
| `prepare-tp1-001` | Local immutable snapshot preparation only | 33 installed-source expectations, archive manifest checked; no SSH/device |
| `owner-tp1-001` | Real two-rank TP1 owner | Passed; 6 target U2 parents, 16 positive layer/stage checks per rank, both exit 0, no timeout/kill, 8 idle before/after |
| `owner-tp2-001` | Independent four-rank paired TP2 owner | Passed; each rank checks 6 parents/16 positive steps, all exit 0, no timeout/kill, 8 idle before/after, native TP before/after `[2.0]` |

The full source manifest and package/source identity are in each immutable
`identity.json` and `results/`. `validation-summary.json` indexes raw SHA256s,
the exact owner/transport/probe bytes, rank results, and native TP checks. The
TP1 source snapshot manifest is
`ccf53188e398340b70e4975fc2b856e1ddce8f0a06056438f12b2199d591ea54`.
The independent TP2 manifest is
`4c58361ff41090bc58e5d87e1f69f124753be97c703a5c36102f540157ce27c7`.
Both sessions use the same owner `b758b9e8...`, transport `ae4c38de...`, probe
`d45ab505...`, and harness `5f549fc2...`; their full working-tree snapshots are
separate and never relabelled as each other.

The actual run invokes public matched warmup on both stage groups once, then
local count collection, accepted eager target U2 plans, actual owner posts and
consumer loans, per-stage reuse, synchronized COMPLETE, normal STOP, plugin
close, and a surviving native TP collective. BF16/original FP32/raw int32 IDs and
full returned padding values are exact against an independent CPU recurrence.
Zero stages and a wholly empty parent are explicit cases. Only the controlled
component scope is accepted for the paired TP1/TP2 sessions. Delayed native
HC/Engram, real MoE, failures,
allocator pressure and timelines remain separate validation gates.

The harness is `../run_a3_eager_pipeline.py`; its public `--prepare-only` mode
does not contact SSH. Device launches require reviewed artifacts and allocated
idle devices. It preserves the installed environment, checks 33 locked installed
source files, requires all eight boards idle, bounds execution, and supervises
only owned PID/start-tick identities. No package, FastAPI, checkpoint, or upstream
source is modified.
