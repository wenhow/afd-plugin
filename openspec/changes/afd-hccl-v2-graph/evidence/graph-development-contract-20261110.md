# C5 Graph development boundary, 2026-11-10

This increment implements the phase protocol, static tensor storage, native
geometry/dispatch hooks and controlled model callbacks. The production V4.1
Graph guard remains closed. Full ModelAclGraphManager execution, native model
operators, Engram history and A5 packed-cache acceptance are separate tasks.

## Control before preparation

Graph/counts capability version one is explicit and requires paired native TP.
Version zero omits the new fields and retains the previous wire representation.
QUERY, all-peer REPORT, DECISION and the complete ACK frontier precede native
input preparation. The later accepted parent digest includes the reservation.
Replay miss selects a fresh native eager descriptor before Engram preparation;
an unsupported producer or prefill is never converted to FULL afterwards.

The controller uses the actual ordered native BatchReqState and U-specific
manager lookup. It checks both token and request slices: equal padded token
totals do not prove equal native request geometry. A reservation is an owned
handle, not merely equal fields; retired or foreign handles are rejected.

The scoped capture wrapper intercepts the base manager's factory before native
prepare_dummy_inputs while preserving its original graph pools, output storage
and registration. The persistent session binds one phase owner at a time.
The exact native signature checks remain in the single upstream-alignment table.

Capture coverage consumes the real native manager's descriptor list, including
its U2 twins and profiling prefix. It validates query-one, LoRA-off layouts and
the negotiated token/stage capacity before factories write inputs. It preserves
native order and does not independently round scalar token totals on FFN. In
paired TP each FFN rank receives one Attention peer; its capacity and storage
budget are per rank, rather than multiplied by Attention size. Query-length
aliases for the same static layout cannot create duplicate cache entries.

## Tensor storage and native model ownership

HcclGraphStageBank resolves the actual independent AFD group rank. It retains
one exchange per stage and reuses it across layers on one compute stream.
BF16 activation/output, original FP32 router input and int32 IDs occupy distinct
storage. Count refresh occurs outside capture; fixed device scalars and masks
sanitize padded IDs and FFN output inside the graph. The returned output is a
borrowed view and the Attention continuation consumes it before reuse.

For hidden dimension H, padded parent rows T and U stages, bank storage per rank
is `(8*H + 9)*T + 4*U` bytes. It does not multiply by layer count. Each layer's
wire traffic is `(8*H + 4)*T` bytes. These figures exclude native HC/Attention,
Engram, graph pools and output storage. The device probe has separate per-layer
oracle buffers; their memory is not included in the production bank figure.
The bank adds three input copies and an ID mask on A, plus an output copy and
mask on F. A positive bucket with zero valid rows still executes the static
sequence; no throughput improvement is claimed from these component checks.

The graph model driver retains native delayed pre_mix, HC continuations,
pre-Engram auxiliary observations and parent MTP/collapse order. Stage-specific
shared-index views and metadata builders remain independently retained. One
native captured Engram producer receives unsliced parent request coordinates;
children consume lookup/mask views and ordinary readiness events. ExternalEvent
producer layouts require early eager until their separate contract is qualified.

The native ModelWithContext wrapper runs once around the parent metadata
frontier. The HCCL U2 runner returns an owning-thread synchronous callback instead
of native full-model handoff threads. The parent owner verifies current session,
endpoint group identity, wrapper/model/metadata identity, actual prepared geometry
and final storage generation before producer work. It does not call
finish_execution; native Ascend remains the sole runtime retirement owner.

Cache entries retain the whole resource bundle. Publish and close synchronize
before readiness or graph reset. Failure marks the cache/session terminal while
retaining dependencies for bounded external supervision; it is not a claimed
abort of a blocked HCCL operation.

## Evidence and remaining integration

- [Protocol independent review](graph-phase-independent-review-20261110.md)
  checks strict parsing, reservation/STOP boundaries and version-zero wire bytes.
- [Native geometry contract](native-graph-geometry-contract-20261109.md) and
  [early dispatch contract](early-native-dispatch-scope-contract-20261110.md)
  describe actual native slicing, U-specific lookup and pre-preparation hooks.
- [A3 CPU records](a3-control-cpu-20261109/README.md) identify each immutable
  snapshot and targeted regression, including preserved failed attempts. The
  stable integration-012 snapshot passed 2609 tests, with two CUDA skips and
  four device cases deselected. The still-developing outer coordinator was
  explicitly excluded. In service-010 the U1 preparation, runner, eager owner
  and signature groups all passed; the FFN fake's missing global ACCEPT caused
  19 failures. After that fake alone was corrected, service-fix-011 passed all
  29 FFN-service cases. Both attempts remain available.
- [Native manager records](a3-native-manager-20261109/README.md) qualify only
  real base-manager/NPUGraph/HCCL operations with controlled arithmetic, native
  InputBatch slicing and early control. They do not execute a checkpoint,
  native Attention, HC, Engram or MoE.
- [Model routing audit](model-graph-stage-routing-audit-20261110.md) distinguishes
  the V4.1 SFA/DSA no-op updater from generic UpdatableGraph providers and records
  the one-parent A5 metadata-task build contract.
- [U1 and coverage review](u1-preparation-coverage-independent-review-20261110.md)
  checks the exact preparation signature, native deferred metadata lifecycle,
  unchanged slot views, capacity validation and per-rank storage accounting.

The explicit FFN service now owns graph phases and eager fallback. The complete
A/F service assembly, native U1 capture/replay handoff, actual
ModelAclGraphManager lifecycle and A5 model/state comparisons remain required.
The centralized [A5 instructions](../../../../docs/npu/DSV41_A5_BATCH_VALIDATION_ZH.md)
collect independently runnable component checks without opening the model guard.
