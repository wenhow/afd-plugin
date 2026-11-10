# q3 three-layer cross-layer Graph component result

The separately authorized fresh `target-query-cross-tp1-001` run passed in
the existing A3 venv. FFN used NPU 0; Attention used NPU 2. All 31 installed
source pins and the uploaded manifest matched, all eight boards were idle
before and after, both rank processes returned 0, and neither inner nor
outer supervision timed out or killed a process. Outer elapsed time was
30.93 seconds. No package or upstream file was changed.

This was actual native base-manager/UBatchRunner construction, NPUGraph,
stage-bank HCCL and controlled arithmetic with `target_query_len=3`, three
layers, `cross_layer_v1` and all five Graph axes enabled. It did not load a
checkpoint or construct a target/draft model. The artifact identity was:

- Probe SHA256
  `5fd406ecb6854c292e1fbe2639329959bebd738869c8b58dc15cea8f2e44c15a`.
- Harness SHA256
  `dd1e5c473a5bcd4bc2c897340a4b9551412b61cecd3a71c3176d76a12b73b537`.

Both roles recorded exactly 27 matching transaction IDs/phases/keys:
4 warmups, 4 captures, 13 replays and 6 eager fallbacks. The four native
capture buckets are T12/U1, T12/U2, T9/U1 and T9/U2. Their original replay
series retained full/1/T-1 valid counts, respectively 12/1/11 or 9/1/8.
The independent FP32 payload, BF16 recurrence, raw IDs, graph padding mask,
all-row outputs and fixed-bank address/byte-budget assertions all completed
without error across the three layers. The role files record bank bytes
and pipeline metadata; they do not contain a complete tensor-address dump.

The extra final replay used an actual native R2/T6 input batch and the
existing captured T9/U2 key. It verified the native prepared geometry:

| Stage | Requests | Tokens | Valid rows | Bank capacity |
| --- | --- | --- | ---: | ---: |
| 0 | `[0,2)` | `[0,4)` | 4 | 4 |
| 1 | `[1,3)` | `[4,9)` | 2 | 5 |

Thus a shorter actual batch can retain the captured request geometry.
The three distinct request-layout misses used actual R1/T3, whose padded
preview clamps the second request start to 0. Both peers selected eager
before preparation; the resulting fresh eager U2 rows were 1 and 2.
Three other eager fallbacks were real F generation misses. No private cache
mutation or forced dispatch result was injected.

Normal STOP and plugin-group close completed. The independent native TP1
allreduce value was `[1.0]` before and after close on each role. Device
ownership was released only after the final eight-board idle sample.

The immutable raw files remain under
[target-query-cross-tp1-001](target-query-cross-tp1-001/identity.json).
[Validation summary](target-query-cross-tp1-001-validation.json) records
the exact commands, manifest/archive hashes, all 31 source checks, both
role transaction tables, bank/pipeline records, selected source hashes,
package versions and every raw artifact SHA256.

This qualifies the paired TP1 q3 Graph/HCCL component and native request
geometry. It does not qualify full native V4.1 target/draft execution,
DSpark proposed/accepted/rejected tokens, native sampling, HC/Engram/MoE,
KV/history, TP2 device behavior, A5, GE compilation, measured overlap or
performance. Production admission remains closed. The geometry CPU
attempts and their fixture repair retain separate evidence and counts.
