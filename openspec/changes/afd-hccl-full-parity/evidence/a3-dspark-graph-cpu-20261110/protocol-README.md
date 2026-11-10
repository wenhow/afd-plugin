# DSpark target-query protocol CPU evidence

`protocol-001` passed both complete mirrors in the installed A3 environment:
208 plan/codec tests and 113 staged-transport tests, 321 total, no failures,
errors or skips, 14 upstream deprecation warnings, 0.708s. No device probe was
started and no dependency was installed. The 21 installed-source checks and
the complete uploaded source manifest matched.

The immutable snapshot HEAD is
`06b9d620379944eb30ec41e99746f8e6f5ef6637`; its manifest SHA256 is
`555b1619cdb4dfffdbbc53e825f04b00dc0900c131acbbaeccb98c69bad6654e`.
The raw `protocol-001/results/summary.json` SHA256 is
`7f5e62f3d5033c2a506d5e8583feb16aeb023896f8b8f4ad93878af52ed5d890`.
`protocol-validation-summary.json` records all five source hashes, JUnit
module counts, the command, remote directory and ten raw-file hashes.

The codec cases protect the exact old q=1 HELLO bytes, strict bounded non-bool
query widths, paired native TP1/TP2/counts/Graph prerequisites, optional-field
round trips, symmetric capacity negotiation and rejection of different query
widths. Public `initialize()` tests cover all four A/F rank positions. They
run real HELLO encoding/decoding and negotiation while replacing only the
external PG factory, PG destruction and selected-device boundary. A mismatch
creates only a host control-group record, drains the coordinator's HELLOs,
broadcasts ERROR and closes without creating a data group. Matching q=1/q=3
sessions reach the host stage-group factory boundaries.

All keyed warmup/capture/replay paths reject a key with a different target
width before its native-readiness callback. These are observable CPU protocol
and lifecycle checks. Host factory records do not establish real NPU groups,
HCCL tensor transport, NPUGraph capture/replay or native DSpark target/draft
execution. Native HC/Engram/MoE, full-model acceptance and performance remain
pending; production admission stays closed.

The earlier local host run passed the 208 pure tests and skipped the entire
transport module before import because torch was absent. This actual-
dependency snapshot closes that specific CPU coverage gap, without promoting
the local skip or CPU results to device acceptance. Other admission/geometry
snapshots in this directory have separate owners and result scopes.
