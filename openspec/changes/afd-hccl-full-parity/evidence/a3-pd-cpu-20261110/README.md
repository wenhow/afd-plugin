# A3 native-PD eligibility CPU evidence, 2026-11-10

`pd-eligibility-017` is an immutable working-tree snapshot based on HEAD
`24c8d51df5d43d57dc8b4d3de0833fe592b42aa5`. Its manifest is
`f19716bfa9931cbce30a25e3a2fc316c56cbc8f65bc6f1898ad55a9a97d41ff8`.
The existing CPU harness uploaded that snapshot to an isolated scratch directory;
it did not modify installed packages or upstream workspaces.

Only these two test modules ran:

- `tests/unit/compat/npu/test_runtime_config.py`
- `tests/unit/connectors/npu/test_hccl_config.py`

Actual-dependency result: **215 passed, 0 failures/errors/skips**, exit 0.
`results/cpu.xml` records 215 cases and 0.385 seconds; stdout reports 0.38 seconds.
This includes eight producer/consumer × TP1/TP2 × U1/U2 cases constructing the
real native `KVTransferConfig` and checking equal A/private-F capabilities.
The five implementation/test/fake files still match the uploaded manifest.

The installed environment was Python 3.12.9, torch 2.10.0,
torch-npu 2.10.0.post4, vLLM 0.30.0+empty, vLLM-Ascend
0.1.dev1+g7e37e29a7, and deployed FastAPI 0.123.10. The user requested keeping
that deployed FastAPI version. These package versions do not resolve or qualify
its separately recorded declared dependency mismatch.

Raw identity, delivery, command, environment, XML, stdout/stderr and the ten
baseline native source checks are in
[`pd-eligibility-017`](pd-eligibility-017/identity.json). A separate stdlib-only
read of the two installed PD source files is recorded in
[`pd-native-source-check.json`](pd-eligibility-017/pd-native-source-check.json):
native `vllm/config/kv_transfer.py` and Ascend Mooncake Hybrid connector hashes
both match the pinned Git sources. Its raw stdout is preserved and
`torch_imported=false` / `torch_npu_imported=false` refer to that source collector,
not to the actual-dependency pytest process.

The initial local sandbox attempt could not read the SSH config and uploaded
nothing. After removing its empty output directory, the authorized escalated
CPU harness produced this single successful snapshot. That local permission
failure is not a failed A3 test result.

No selector/native-executor modules were collected in this run; those await a
separate integrated freeze because concurrent diagnostics/Graph work was in
progress. In particular, these tests do not execute native PD transfer engines,
request/history handoff, KV-only scheduler steps, device kernels, full-model
inference or a P/D deployment. No device probe was started. Configured equal
P/D TP/DP/PP remains an eligibility prerequisite, not remote physical-layout
verification. The global V4.1 model guard, Graph-PD and DSpark-PD restrictions
remain in place.

See the source contract and exact implementation hashes in
[`eager-pd-implementation-contract-20261110.md`](../eager-pd-implementation-contract-20261110.md).
