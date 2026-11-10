# Target query capability contract

This is the implementer's protocol record, not an independent review or a
DSpark model acceptance report. The change is limited to `hccl_plan.py`, the
staged transport's Graph-query validation, their existing mirrored tests, and
the existing shared HCCL fake. Exact identities and local results are in
`dspark-target-query-protocol-local-20261110.json`.

`HcclPlanCapabilities.target_query_len: int = 1` is appended to the immutable
capability so existing positional constructors keep their meaning. The value
must be a non-bool integer from 1 through the advertised `max_tokens`. A value
above one requires Graph v1, local-count collection v1 and one native TP1/TP2
world per role (`A == F == native TP`). It describes the target's query width;
the local draft width and HF draft block size are separate concepts. It does
not require every child stage's token count to be divisible by that width.

The encoder omits `target_query_len` when it is one. The existing q=1 HELLO
golden bytes remain exactly equal, and absence decodes to one. An explicit
q>1 field is parsed strictly and compared exactly during symmetric capability
negotiation. Capacity/stage limits may be reduced, but query width cannot be
silently reduced or changed. An old strict reader rejects the unknown field
rather than treating q>1 as q=1.

The existing initialization sequence is control-group creation, complete
HELLO negotiation, then stage data-group creation. The coordinator consumes
every peer HELLO before broadcasting a semantic error. New public-initialize
tests cover both A and both F rank positions: a query-width mismatch closes
after the host control-group boundary and never creates a data group. Matching
q=1/q=3 peers proceed to the two stage-group factory boundaries. Only those
external PG/device boundaries are replaced; the actual HELLO codec,
negotiation and failure state transitions run. This proves CPU lifecycle
ordering, not NPU process-group construction.

Every keyed warmup/capture/replay query must also have
`key.uniform_token_count == key.max_query_len == negotiated target_query_len`.
A coordinator rejects an incompatible key before native readiness resolution;
other A/F peers report their rejection through the existing drain/global-error
path. Eager candidates have no static key and retain their previous behavior.
The tests reject an incompatible key before the readiness/native-preparation
callback, without fabricating an input batch or entering a tensor body.

The local pure codec mirror passed 208 cases; the staged transport mirror
skipped as a whole because this host has no torch. The later immutable A3
`protocol-001` snapshot passed both complete mirrors at these same five
source hashes: 208 codec plus 113 transport cases, 321 total, no failures,
errors or skips, with 21 installed-source checks matched. Its raw files and
hashes are indexed in `a3-dspark-graph-cpu-20261110/protocol-README.md` and
`protocol-validation-summary.json`.

No device, weights, native DSpark proposal/acceptance, Graph model capture,
KV/history, useful overlap or performance result is claimed. Production model
admission remains closed. The other phase2 owners separately implement
Graph-key geometry, native target shape and worker/runner binding; these
capability checks alone do not enable that runtime.
