# C6 controlled arithmetic pipeline probe

The existing `tests/npu/hccl_graph_manager_probe.py` has an explicit
`--stage-bank --compute-pipeline --pipeline-hybrid 0|1` path. Omitting the new
option keeps the prior synchronous behavior. The child CLI validates the option
dependencies, and the immutable manager harness passes both options unchanged.
The initial design was reviewed before device execution; the subsequent two
controlled sessions are recorded in the separate
[device evidence](a3-pipeline-device-20261110/README.md).

Each native `create_forward_fn` invocation allocates its bank, pipeline, producer
buffers and final continuation outputs before returning the captured callback.
The pinned native manager calls this factory outside capture, once for warmup
and once for formal capture. Formal cache resources retain the entire bundle,
including its exact pipeline. Replay enters only the recorded graph.

Attention first enqueues every positive stage's side compute for a layer, then
records the original parent HCCL exchanges in stage order. The next layer's BF16
and independent FP32 producer values depend on the preceding returned output.
The last continuation adds the final returned output and final producer hidden
state on the side stream, masks padded rows, and joins before parent consumption.
FFN uses the real pipeline receive/compute/send scopes around the existing
controlled arithmetic and bank masking. Separate per-layer observation buffers
preserve every exchanged payload/output before cross-layer bank reuse.

The CPU oracle recursively constructs the same mathematical inputs and expected
outputs, including independent FP32 deltas and sanitized IDs. It checks all
layer/stage BF16/FP32/IDs/output observations and the final Attention continuation
after actual device synchronization. Native TP before/after checks, the real base
manager/UBatchRunner/InputBatch slicing/GraphControl, the original transaction
schedule and HCCL wire order are preserved.

Only in compute-pipeline mode, each exact captured key's three replays vary the
controlled total valid count across `full`, `1` and `tokens - 1`. The native dummy
request/token geometry remains fixed; this tests device-valid scalars and oracle
payloads rather than native request metadata packing. Each replay asserts the
same key and descriptor, and checks the accepted plan's per-stage valid counts.
U2 physical `(2, 2)` rows therefore use valid `(2, 2)`, `(1, 0)`, `(2, 1)`;
physical `(4, 4)` rows use `(4, 4)`, `(1, 0)`, `(4, 3)`. U1 uses the corresponding
single-stage counts. Warmup/formal capture retain zero valid rows. The two eager
fallback cases and all default non-pipeline valid schedules remain unchanged.
Both rank reports include the actual key, per-stage valid/padded counts and total
controlled valid count. Refill and every layer/final-continuation oracle derive
their masks from that accepted plan.

The reused manager harness supports a new evidence directory for this C6 result.
Pipeline mode additionally pins locked `vllm/utils/torch_utils.py` and the
previously collected installed `torch_npu/npu/streams.py` and `utils.py` source
hashes. It retains the existing timeout, owned-PID supervision and eight-board
before/after checks. No packages are changed.
Both `source-expected.json` and the installed API raw JSON are included in the
archive's source manifest. A local archive-only self-check verified all 133
manifest entries, 134 unique tar members and 31 expected installed source pins;
it stopped before network access. Its first runpy setup error and corrected
absolute-path retry are retained separately under the device evidence directory.

Static checks passed for the new probe/harness: Ruff check/format, Python 3.11 AST,
`--help`, default/0/1 CLI parsing and invalid option dependencies. These checks
did not import torch or launch SSH/device work. Root review and the final shared
component/consumer CPU seams must complete before a serialized A3 NPU attempt.
Even a successful future probe establishes only controlled arithmetic component
behavior; it does not execute the native Attention/HC/indexer/Engram/MoE model.

Current probe candidate SHA256:
`d2e780a99f080236a85fd3cbba1844d1b7077fac07c0dca62c52754433cccb87`.
The immutable harness candidate remains
`746be9cebc107cf5b57957fe8af407a502b1a929b7b3654d7c2bd13c5e81d963`
after adding those two manifest members (superseding `a52f56a6`).
These final bytes were subsequently bound to both session manifests in the
linked device evidence; that component result does not widen the model scope.
