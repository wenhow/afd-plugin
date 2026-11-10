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
| `diagnostic-archive-precheck-001` | Local opt-in diagnostic archive preparation | 33 expectations, 142 unique members and matching hashes; no SSH/device |
| `diagnostics-tp1-002` | Independent TP1 owner with actual stage diagnostics | Passed; each rank has 6 configuration records/146 scalar records, all payload checks and STOP/TP preserved, 8 idle before/after |
| `diagnostics-tp2-002` | Independent TP2 owner with actual stage diagnostics | Passed; all four ranks have 6 configuration records/146 scalar records, all payload checks and STOP/TP preserved, 8 idle before/after |

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

## Actual stage diagnostics

The two `diagnostics-*-002` sessions enabled the explicit
`--stage-diagnostics` flag. The real immutable `HcclV41DiagnosticBinding`
describes each accepted parent and binds progress to the real
`HcclEagerPipeline`. The sink only appends scalar records. No new device wait,
event, query or synchronization was added; the payload and arithmetic driver
functions retain the same AST as the previous probe.

Every rank passed its transaction/rank/stage identity check, actual stream-role
and alias check, and `actual.ffn_compute_sync=false` for every U2 record. All
three allocated side streams are distinct in these sessions; every alias is
false. Each rank recorded exactly one configuration per parent, including the
wholly empty parent, followed by real owner post/loan/reuse/finish host enqueue
milestones. Empty-parent records contain only configuration and finish entry
and return. The 146 records per rank remain in `diagnostic_records`; their
scalar verification is in `diagnostic_validation`.

The original six parents, sixteen positive layer/stage payload checks per rank,
independent FP32 and raw IDs, full padding output comparisons, normal STOP and
native TP checks all remain successful. TP1 and TP2 returned native TP values
`[1.0]` and `[2.0]` before and after plugin close. Their independent runs took
29.48 s and 31.49 s respectively, without timeout or any killed PID; these are
bounded execution observations and are not performance comparisons. Both
sessions matched all 33 installed source pins and observed eight idle boards
before and after. Packages, including FastAPI 0.123.10, remained unchanged.

Both new sessions froze the same working-tree source manifest
`b7d1964c67db1c6e5692ab363a1d90eb1480485ba290d4c4b2012e844384d20c`,
with separate invocation identities and raw output. The probe SHA256 is
`c8b4b0b2becb3fe11d958666b8e204e8757df09f418a32850bf76b12fd787fc6`;
the harness is
`e808603c73d5bd5b353db108d9dab63f902fd1dcc860469944efe60c46e1700e`.
The manifests include concurrent Graph source changes, which are not accepted
by these selected eager component runs. Historical `owner-*-001` artifacts
retain their earlier source bytes and results.

The [TP1 summary](diagnostics-tp1-002/results/summary.json) and
[TP2 summary](diagnostics-tp2-002/results/summary.json) link to per-session
`identity.json`, `delivery.json`, `remote-run.log`, source checks, standalone
metadata preflight, `npu-before.log`, `npu-after.log`, owned-process records,
probe `result.json` and every `attention-*.json` / `ffn-*.json` role file.
[The index](validation-summary.json) records all raw SHA256s; the independent
[integrity check](diagnostic-index-check-20261110.json) verifies all 76 raw
artifacts across the four device sessions. The
[design](../eager-ticket-diagnostics-design-20261110.json) and
[A5 owner CLI parse check](../a5-owner-diagnostic-cli-check-20261110.json) remain
source/CLI evidence without any device execution.

This acceptance covers actual owner host diagnostics with controlled arithmetic.
It does not exercise native A/F model consumers, Engram/MoE operators, an injected
sink failure on NPU, live serial-U1 compute synchronization, a device timeline,
effective communication/compute overlap or performance. The A5 batch's existing
two owner cases now include the diagnostic flag and require their role JSON
files; the batch remains fourteen cases.
