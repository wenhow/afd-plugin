# C6 explicit stage diagnostics and actual execution

This is the implementer's contract and self-review record. The separate
state-audit reviewer read the frozen helper, eager/Graph FFN consumers and
root-owned service/coordinator wiring and reported no runtime blocker within
the stated static scope. Installed-dependency FFN CPU validation is pending.
There is no new NPU experiment, performance result or model admission here.

## Native reference and boundary

The donor is `4ddd69a391dfb5b3e362c7601399e538a31de702`.
`v1/worker/npu/ffn_model_runner.py:957-1218` defines first/last-layer
receive/compute/send host progress, non-cross-layer send-event frontiers and
the final send-event join. Its original blob SHA256 is
`f95532370a9a5157b16817c5879d78af7405c321d5739d975cd6e2785fe54ccb`.
Lines 960-972 gate optional compute synchronization on live serial U1, outside
Graph warmup/capture. Lines 1138-1155 surround that actual synchronization with
begin/complete logs. The ordinary per-step final fence is separate behavior.

`connectors/npu/hccl_diagnostics.py` is a new stdlib composition component,
not an upstream function replacement. It imports no tensor/device runtime,
reads no environment, adds no control metadata or protocol exchange and owns
no global logger, registry, device stream or communicator. Its default sink
obtains the standard logging logger at emission time; callers can instead
supply a structured sink. Both diagnostic settings default false in the
existing immutable settings. Owners construct the helper only when requested.

The immutable API is:

```python
HcclStageDiagnostics(
    selection,
    world_rank=...,
    role_rank=...,
    emit=...,          # optional structured record sink
    synchronize=...,  # optional actual device fence supplied by the owner
)
helper.bind_execution(
    layer_indices=...,
    actual_compute_stream="parent" | "compute",
    in_profile_run=False,
    is_warmup=False,
)
```

Binding returns a retained `HcclStageDiagnosticRun` and emits one requested
configuration record before the body. Its progress API is
`record(event, layer_idx, stage_idx)` and
`synchronize_compute(layer_idx, stage_idx)`. The latter adds a fence only when
the explicit sync diagnostic, role, U1/live-eager selection, actual serial
execution and profile/warmup gates all permit it. A missing synchronizer is
rejected before body entry. A throwing synchronizer emits no false completion.
The owner must bind the synchronizer to its actual runner device.

## Configured, eligible and actual values

Configuration preserves the selection's `configured.<ENV>`,
`effective.<ENV>`, `logical.*`, and `physical.*` plan values. It additionally
records actual mode, compute/send/receive streams, whether the compute
pipeline is enabled, explicit profile/warmup markers, actual sync diagnostic,
world/role ranks, role, phase and U.

An eager U2 plan can be logically eligible for overlap while the current
consumer is serial. Its record therefore retains that configured/effective
eligibility but reports `actual.mode=serial`, all actual streams as parent and
`actual.compute_pipeline_enabled=False`. This helper rejects an eager actual
side-compute mode because the present actual schema does not describe a true
eager three-stream transport. No eligibility value is an admission claim.

Graph compute off or U1 settings report the same Graph phase with actual
parent streams. A selected Graph compute pipeline reports compute on its side
stream and communication on parent. Explicit legacy U1 side compute remains
representable: pure U2 gates can be false while the actual legacy pipeline is
true. No field is named `overlap_active`; the implementation makes no measured
overlap assertion.

All ordinary progress records carry `boundary=host_enqueue`. A receive or
compute "returned" record describes Python enqueue return, not kernel or
communication device completion. Only the explicit synchronization begin and
complete pair carries `boundary=explicit_synchronize`. Native graph replay
does not re-enter the captured Python progress body, so these records do not
claim per-operation replay progress.

## Eager and Graph FFN ownership

`execute_hccl_v41_parent(..., *, diagnostics=None)` validates diagnostic role,
rank, phase and actual stage count before tensor work. It binds actual serial
parent execution; profile, warmup and synthetic dummy parents suppress compute
sync. For each active stage it records receive, compute and send boundaries;
only live target U1 can request the additional device fence after native MoE
and before send. Native FFN TP reduction/output semantics, parent completion
and terminal failure handling remain unchanged. Empty stages emit no tensor
progress; an empty parent can report configuration then complete directly.

`HcclV41FFNGraph(..., diagnostics=None)` validates role/U/rank/Graph phase,
and matches the complete selection to a selected pipeline when present. It
binds the actual stream and emits configuration in `__init__`, outside capture,
then retains that run. `run()` adds only requested host progress around the
unchanged tensor body. Non-cross-layer and final joins have begin/returned
records; all captured Graph paths suppress compute synchronization. Native
context and bank ownership, physical wire order and final join are unchanged.

Root-owned service/coordinator code supplies the immutable selections and
sink and records Attention configuration per descriptor. Its individual
changes are not authored or independently approved by this report. The new
helper and mirror fall under the existing connector validation paths, and the
two FFN mirrors remain under the existing FFN runtime validation paths.

## Frozen sources and checks

| Path | SHA256 |
| --- | --- |
| `afd_plugin/connectors/npu/hccl_diagnostics.py` | `96ae0f376770a4d425c3c8876f028f2fad24747f517aaea051d63220ac61c0e0` |
| `afd_plugin/v1/worker/npu/hccl_v41_ffn_runner.py` | `c3f3c434efcaa5cc533e5b4550534119140a894113a272e6aa96ca277ef565a8` |
| `afd_plugin/v1/worker/npu/hccl_v41_ffn_graph.py` | `b82fc1a39979e9d5e206fca565d5f1c83e7b11d5557958d08a74e80df1c096cf` |
| `tests/unit/fakes/hccl_diagnostics.py` | `0a1814984f486dcf980755552d1fd2a8b43dac3163755a4827bbc2c666b4cec6` |
| `tests/unit/fakes/hccl_v41_ffn_graph.py` | `9561d0a60e47c3f9e5b6aba989c5aff7858c8fd39b4ea62bf4d4423e3333336c` |
| `tests/unit/connectors/npu/test_hccl_diagnostics.py` | `218cf5652d9dea4838f0b7f4ab27bbcc345d4369b87e454ec575e6e36f6c4aac` |
| `tests/unit/v1/worker/npu/test_hccl_v41_ffn_runner.py` | `08e9fad715c464f1dc20344937ace2570f0bda4f3aa1c51beb81ac0dc609b7b1` |
| `tests/unit/v1/worker/npu/test_hccl_v41_ffn_graph.py` | `6fc07a2cb5d1215c24637c75f9dec68e258964bb0a3afbbf7fb0413c09765f75` |

Ruff check/format and Python 3.11 AST grammar parsing passed for all eight
files. Local Python 3.14.3 ran the actual standalone helper mirror:
35 passed, zero skips/failures/errors, in 0.31 seconds. It checks default
silence, first/last-layer progress, configured versus actual execution,
role/mode/U/profile gates, invalid inputs, the requested synchronization
protocol, failure without false complete and the default logging sink.
The sync fake records the explicit callback; it does not execute an NPU fence.

The FFN mirrors additionally cover public records with unchanged CPU payload
outputs, pre-body Graph configuration, join boundaries, wrong diagnostic
ownership and terminal sync failure. Their actual installed-dependency CPU
results are pending and must not be inferred from the standalone 35 tests.
Normal disabled operation adds no device synchronization. Requested logging
adds host dictionary construction and sink IO at selected boundaries; requested
live-U1 synchronization intentionally serializes compute before send. A later
performance report must state whether either diagnostic was enabled.

Removal can move these records to native structured callbacks only when their
rank provenance, actual execution description, pre-capture configuration and
explicit-sync failure semantics are preserved. No upstream checkout was edited.
