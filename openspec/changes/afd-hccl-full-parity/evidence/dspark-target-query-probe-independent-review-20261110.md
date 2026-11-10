# Target-query Graph probe: independent artifact review

Read-only review of the final q1/q3 probe increment. No source blocker found
for a separately authorized bounded component run. This review did not start
SSH, CPU tests, graph capture, or NPU work.

| Artifact | SHA256 |
| --- | --- |
| `tests/npu/hccl_graph_manager_probe.py` | `5fd406ecb6854c292e1fbe2639329959bebd738869c8b58dc15cea8f2e44c15a` |
| `openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/run_a3_graph_manager.py` | `dd1e5c473a5bcd4bc2c897340a4b9551412b61cecd3a71c3176d76a12b73b537` |
| `a3-target-query-graph-20261110/probe-static-validation-20261110.json` | `5c389b336241178b350fff14f87bf9a07ba0acc590ac2fad1994e672eb370029` |

The new explicit CLI accepts q=1 or q=3 and keeps q=1 as the default. Both
supervisor and remote harness pass that value to each worker. The real native
manager's decode width, negotiated capabilities, native dummy batch request
count `T/q`, scheduled-token vector, and public geometry/control calls use the
same q. The q3 buckets are T=9/12 with R=3/4; q1 keeps its original T=4/8 buckets.
This is controlled target geometry with arithmetic collaborators, without a
native draft model or native DSpark sampling.

For T9/U2 the source explicitly checks the captured native slices
`request[0:2], token[0:4]` and `request[1:3], token[4:9]`. The original twelve
captured-key replays retain full/1/T-1 valid-count coverage. A separate final
q3 replay constructs the actual R2/T6 native batch and proves its preview key
equals the existing captured key before actual preparation and reservation
verification. Its valid/padded counts are `(4,2)/(4,5)`. The distinct R1/T3
request-clamp miss checks the changed second request slice `[0:3]` and must
agree on eager before the controlled body executes.

The q1 schedule remains 26 parents per rank: four warmup, four capture, twelve
replay, and six eager. The additional q3 short batch makes 27: thirteen replay,
with all other phases unchanged. The F generation miss still covers only the
three original replays after the twelve captured-key replays; generation
returns to zero for the request-layout miss and extra same-key replay. Its
transaction parsing, refill, device synchronization, exact verification and
COMPLETE remain outside the tensor body. Existing parent-stream HCCL and
retained bank/graph ownership are preserved.

The static artifact records exact AST equality to the prior accepted artifact
for `payload`, `verify`, `observe_bank`, `native_check`, and `complete_active`.
Its two final file hashes were independently matched to the current files.
Reported Ruff/format/3.11 grammar/invalid-CLI checks and local prepare-only
archives are static preparation evidence. The q3 prepared manifest is
`280493e27b3dee708a40ad0cf24954d7a9d6646a38cab0c7c1e380374e2ab2b2`,
with 31 expected installed-source pins; those expected pins have not yet been
verified against a running device environment by this artifact review.

Numerical, address-lifetime and failure/STOP assertions remain in the probe;
their presence is not an execution result. A later run must retain fresh raw
results and matching installed/source identities. Even a successful run would
qualify only this paired arithmetic Graph component and q geometry, without
native V4.1/DSpark model inference, Engram history, GE lowering, or performance.
