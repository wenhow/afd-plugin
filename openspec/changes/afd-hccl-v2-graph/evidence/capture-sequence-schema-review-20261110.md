# Capture-sequence protocol independent review

Reviewed by the root agent on 2026-11-10. The neutral schema and FFN
preflight were implemented by the state-audit agent; this records an
independent source review, not device qualification.

The reviewed neutral source is `afd_plugin/distributed/hccl_graph.py`, SHA256
`ed9f425e0fd5a486a7d66adce77a17e57c7dd5ea2103b78a3d53740188804742`.
No unresolved blocker was found in this explicit component contract.

- `capture_sequence_digest` is optional and omitted when absent. Its presence
  requires exactly 64 lowercase hex characters and the empty warmup preflight.
  Explicit JSON null, wrong types, additional fields and malformed digests are
  rejected. Existing candidate/decision bytes are preserved without the field.
- The digest binds an ordered tuple of complete graph-key digests and the
  native profiling prefix. Order, count, later keys, metadata geometry and
  prefix all affect agreement. Empty schedules have an honest digest and can
  propose eager with no fabricated model key.
- Candidate digests already bind GRAPH_REPORT replies. This new control
  identity stays outside model cache keys and metadata shapes. A mismatching
  Attention schedule reports unsupported before native factory input writes.
  The resulting empty eager parent is completed by every role before skipping
  capture; a shorter loop cannot silently leave another rank expecting a key.
- FFN preflight checks the actual model/generation and a completely fresh
  cache, including pending or stale entries. It reports missing for a genuinely
  uncaptured real key and unsupported for the keyless empty proposal. It does
  not advertise ready or execute tensor/model work for this preflight.

Old strict graph-one decoders reject the additional field. This component
schema therefore requires matching A/F source delivery; graph-zero and ordinary
graph-one payloads without the optional field retain their earlier bytes.
The bounded schedule and digest add CPU hashing and one empty startup parent;
they add no device payload or per-layer graph operation. Native descriptor
generation remains upstream. Replace the preflight extension when native
capture exposes a common, agreed schedule before any peer prepares inputs.

Actual-dependency CPU evidence is recorded separately in
[capture-seams-013](a3-control-cpu-20261109/capture-seams-013/results/summary.json).
That snapshot passed the neutral, transport, coordinator and replay modules;
its two FFN test assertions referenced a nonexistent plan convenience field.
The original failures remain, and corrected FFN lifecycle cases passed in
[service-close-014](a3-control-cpu-20261109/service-close-014/results/summary.json).
These CPU records do not execute native Attention, HC, Engram or MoE on an NPU.
