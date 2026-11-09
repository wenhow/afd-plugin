# C6 donor overlap switches and immutable policy contract

This began as a read-only source audit and an implementation proposal. The
final section records a subsequent separately authorized pure settings seam.
It enables no runtime schedule, changes no defaults, and accepts no NPU, model,
overlap or performance behavior. A switch existing in the donor is not target support.
All donor references below use Git objects at
`4ddd69a391dfb5b3e362c7601399e538a31de702`; no donor checkout files were changed.

## Source identity

SHA256 values hash original Git blob bytes, obtained with `git show`.

| Alias / donor path | Git blob | SHA256 |
| --- | --- | --- |
| D01 `afd_plugin/connectors/npu/p2p_hccl.py` | `9863e2475c1867a80605af264f1551ab47bef59e` | `594ac53c8c8008ba05da4d12db1c38511e94fdcb8e13340322a0587e940df0ef` |
| D02 `afd_plugin/model_executor/models/deepseek_v4.py` | `2d33402b611e15d3437ecc93435a7c2bfcaaeb59` | `d513a463d159a0a2c9dae18d566805d1f30a8bb9504909c29c7f1d04bb920d3b` |
| D04 `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `0543da98bd8dc81ddcb3d8d257dce4360c6a2662` | `f95532370a9a5157b16817c5879d78af7405c321d5739d975cd6e2785fe54ccb` |
| DW `afd_plugin/v1/worker/npu/ffn_worker.py` | `94b4dfc74a084df1451acc944e44541ddb86ed86` | `4f033170e28ef0563b4346323df75db45f697aa162a339c92b746d25e94f69ff` |
| DA `afd_plugin/v1/worker/npu/attention_model_runner.py` | `6dc513455667c2ebd8258cb398fc31cd152e6e0b` | `ba7713356f7ecfbefde289004ea911ba68150cc46d34c1bb0555a995907af89e` |
| DR `recipe/npu/P2pHcclAFDConnector/deepseek_v4/common.sh` | `e836ce57ac811edda976a74caf940ed9dd199e5d` | `edca3e976602ff5c3e20323916409d296673357deb796055096ae0385ff7f6b8` |

Current target source snapshots used for the mapping, before acceptance:

| Target module | SHA256 |
| --- | --- |
| `connectors/npu/hccl_graph_pipeline.py` | `6473cc7ad538ddeaf45a5a9233eef7d5bae79b1cb1e6c95b883953ef05db73cd` |
| `model_executor/models/npu/deepseek_v41_graph.py` | `3c45a37de08093a113533a3c63dd2be5212a193a7cc643cdf8f7817f58f03099` |
| `v1/worker/npu/hccl_v41_ffn_graph.py` | `3c708b4912fd8e72c6e16fa3ef59d9769440629c9673dfeacf1c5e26aaa84f37` |
| `v1/worker/npu/hccl_v41_graph_coordinator.py` | `33e2e84ca5cddfd85c26a3ba88aaaa129e2262dacd3f82ecc765b5a3394e0e20` |
| `v1/worker/npu/hccl_v41_graph_service.py` | `7fc9f30544143f61321feda08141d9bdabffb2d01543a39c44c0cd8bdaab6531` |

## Exact donor parsing and selection

D01:239-295 defines six overlap names and two diagnostics names. Its shared
parser is `_strict_binary_env_enabled(name: str, *, default: str = "1") -> bool`
at lines 249-253: read `os.environ.get(name, default).strip()`, accept only
`"0"` or `"1"`, otherwise raise `RuntimeError` naming the variable and bad
value. Thus whitespace around a binary digit is accepted; empty strings,
`true`, `false`, `yes`, `01`, and signed/other numeric values are rejected by
the Python parser. An explicit zero remains false.

All six overlap defaults are 1. Both diagnostics default to 0. Construction
samples values into connector fields at D01:370-417. There is one subtle parse
boundary: eager overlap calls its parser only when `num_stages > 1`, due to
the short-circuit at lines 377-379. The five Graph fields and diagnostics are
parsed at construction regardless of the selected execution mode. Parsing
does not by itself activate a schedule.

DR:307-360 supplies a separate launcher boundary. `eager:2` defaults eager
overlap using `${VAR:-1}` and validates the shell value against exactly `0|1`;
the shell does not strip whitespace and treats an empty value as unset.
`full-decode-only:2` unconditionally exports eager overlap as 0, even if the
caller supplied 1, and defaults each of the five Graph values with `${VAR:-1}`.
Graph values are then validated by connector construction. Do not conflate
these shell defaults with the Python parser's behavior.

| Environment name | Default | Donor consumer and effective scope |
| --- | --- | --- |
| `AFD_HCCL_EAGER_U2_STREAM_OVERLAP` | 1 | D01:595-623 enables Attention eager stream pipeline only when initialized, not Dynamo compiling, a forward context has DBO and U>1, and `afd_graph_ubatching` is false. D04:918-929 enables F eager compute/recv/send scheduling only for U>1, no active Graph-overlap selection, and runtime mode NONE/None. |
| `AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP` | 1 | D01:668-694 selects A side compute only with initialized Graph resources, Attention role, Graph + layer-major + DBO + U>1 context, outside Dynamo lowering. D04:900-916 gates F side compute for U>1 during capture or configured Graph warmup. Graph replay uses the recorded schedule; it does not re-read the switch inside a replay. |
| `AFD_HCCL_GRAPH_U2_HYBRID_DAG` | 1 | D01:695-700 and D02:1114-1244 alter A's compute dependencies only when the Graph compute pipeline is active. True waits this stage's preceding receive; false uses the parent-ready frontier and explicit receive join. It is not a second eager-overlap switch. |
| `AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM` | 1 | D01:629-646 chooses logical A send/recv stream entries, while compute has its own stream. D01:893-978 and 1780-1789 override physical communication to the parent during Graph warmup/capture. Therefore 1 does not authorize side-stream HCCL capture. |
| `AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM` | 1 | D04:930-937, 1002-1035 chooses F streamed receive only under Graph compute overlap. It waits the same stage's previous send before stage-bank reuse; its compute waits receive completion. D01:1241-1287 keeps Graph warmup/capture HCCL on the parent even through the streamed helper. |
| `AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER` | 1 | D04:938-946, 1178-1218 removes the optional per-layer send-event join and joins every last-layer stage send at return. Effective only with Graph compute and Graph recv-stream overlap. D01:391-398 rejects raw CROSS_LAYER=1/RECV_STREAM=0 at connector construction, including a mode where those Graph features would otherwise be inactive. |

Eager Attention uses connector-owned send/recv streams, pending transfer and
deferred-receive owners (D01:985-1099). That is a distinct implementation from
the Graph event DAG, even though both share layer/stage concepts. The Graph
parent-stream exception also covers warmup: warmup uses ordinary send/recv and
must leave a quiescent communicator before capture (D01:1780-1789).

## Diagnostics and synchronization provenance

| Environment name | Donor behavior | Target gap at the snapshot above |
| --- | --- | --- |
| `AFD_HCCL_STAGE_DIAGNOSTICS` (default 0) | D01:399-417 logs effective role/rank/stage/overlap values when either diagnostics option is enabled. DA:394-445 logs execute/sample boundaries; DA:529-554 logs IDs send boundaries. D04:276-325 logs pre-model IDs receipt; D04:957-1218 logs first/last layer recv/compute/send and send-wait boundaries. DW:90-114 reports MXFP AllToAllV patch status; DW:215-255 logs execute and final device-sync boundaries. | No parsing or these donor-compatible diagnostic consumers in the new C6 modules. A bounded probe JSON report does not replace service diagnostics. |
| `AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS` (default 0) | D04:960-972 enables extra synchronization only for live serial U1, outside capture/warmup, without compute overlap. D04:1138-1155 calls `torch.npu.synchronize()` at line 1147 after native F compute and before send, with begin/complete markers. | No matching opt-in live serial F hook. It must stay outside captured tensor bodies and be excluded from performance comparison unless explicitly reported. |

DW:250 calls the final `torch.npu.synchronize()` unconditionally; the stage
diagnostic option only wraps its logging. Consequently turning stage logging
off does not remove that existing step-completion fence. The compute-sync
option adds per-layer fences and does not create the baseline final fence.
These two binary options are diagnostics, not members of the six overlap
controls and not profiler start/stop RPC controls.

## Mapping to the current C6 composition

The current `HcclGraphPipelinePolicy` is frozen/slots and carries only `hybrid`;
None selects the existing serial Graph body. A coordinator/service retains
the policy for its cache/session lifetime, creates fresh stream/event owners
before each factory callback, and validates the retained bank/policy on replay.
No six-name environment parser currently configures those owners.

| Donor capability | Present software building block | Still missing before switch parity can be claimed |
| --- | --- | --- |
| Eager U2 overlap | C4 serial eager execution and stage/count protocol remain available. | Separate eager A send/recv pending/lifetime schedule, F recv/compute/send pipeline, explicit on/off selection and equal-output/timeline validation. The Graph pipeline must not be reused automatically for an eager fallback. |
| Graph compute overlap | Actual pipeline can fork A preparation/continuations and native F compute, join before parent HCCL and final outputs; A/F consumer code accepts it. None uses serial Graph. | Donor env/default/mode selection and all-off service integration tests; actual model/device acceptance. Current explicit policy can also exercise U1, but a `GRAPH_U2` env contract should not silently change U1 behavior. |
| Graph hybrid DAG | `hybrid=True/False` selects own-receive versus parent frontier, including typed final ready/done tails; A consumer queues all stages' wire before the next layer. | Immutable env-to-policy selection and per-rank effective-value reporting; actual model differential/timeline proof. |
| A logical three-stream choice | Parent HCCL restriction is explicit; side compute and graph-visible joins exist. This matches the donor warmup/capture safety rule. | Configurable logical send/recv plan and observable on/off contract. Merely creating extra unused streams or aliasing this flag to `hybrid` would not implement the choice. |
| F recv-stream choice | Same-stage previous-send reuse dependency and receive-to-compute event are implemented, with HCCL on parent. | Independent recv scheduling choice, off/on behavior and its immutable effective policy. Current F body queues receive→compute→parent-wait/send for each stage, so a side compute stream alone does not prove next-stage receive/compute overlap. |
| F cross-layer choice | All final stage sends join before F return; static bank reuse has the same-stage prior-send fence. | Independently selected per-layer versus final-only join schedule, dependency validation and on/off behavior. An unconditional final join is necessary but insufficient to claim the CROSS_LAYER switch is implemented. |

Graph all-off must remain Graph/U2: retain the manager, descriptor, bank, wire
layout and Graph phase agreement; select the serial C5 tensor body. It must
not turn into an eager decision. Eager fallback selection is a separate phase
with its own schedule, and the public donor recipe intentionally disables
eager overlap for its Graph configuration.

## Proposed immutable configuration API

Recommended first API, in a small plugin-owned module with only stdlib imports:

```python
@dataclass(frozen=True, slots=True)
class HcclOverlapSettings:
    eager_u2_stream_overlap: bool = True
    graph_u2_compute_overlap: bool = True
    graph_u2_hybrid_dag: bool = True
    graph_u2_attention_three_stream: bool = True
    graph_u2_ffn_recv_stream: bool = True
    graph_u2_ffn_cross_layer: bool = True
    stage_diagnostics: bool = False
    ffn_compute_sync_diagnostics: bool = False

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str], *, num_stages: int
    ) -> HcclOverlapSettings: ...

    def effective_settings(self) -> dict[str, bool]: ...
```

Pass an environment snapshot explicitly; do not store or read `os.environ`
from compute/replay or add a mutable registry. Reject non-bool direct fields,
apply the donor whitespace/binary parser and named RuntimeError, and reject
CROSS_LAYER=1/RECV_STREAM=0 at construction. Preserve the donor's eager U1
short-circuit: return false without reading that eager environment value,
including when the ignored value is invalid. Parse all five Graph fields and
both diagnostics regardless of U1/U2. The public diagnostic record preserves
parsed Graph values even in an eager session; it does not claim active modes.

A second pure projection can produce a frozen `HcclOverlapSelection` from
settings plus explicit role, actual phase, U, and execution-family inputs.
Keep configured values and effective values distinct. Graph fields apply to
Graph warmup/capture/replay U2; eager fields apply only to eager U2. Derive
hybrid/recv/cross-layer effectiveness under the compute gate, without erasing
the original settings. Serial U1 compute-sync diagnostics must require live
non-warmup/non-capture F execution and no overlap. An effective-values record
can be logged once per rank outside the body; it must not claim availability.

Do not silently project unsupported requested fields into the present
`HcclGraphPipelinePolicy(hybrid=...)`. That adapter is valid only for a reviewed
subset whose selected behavior is implemented. Keep normal production
selection closed while implementing the other axes. No new wire codec is
needed merely to parse flags. Before admission, a session must establish a
compatible A/F schedule and retain its immutable selection with graph/cache
owners; policy changes need fresh owners/session, not just a generation miss.
The graph key currently excludes schedule policy, and existing keys cannot be
recaptured in the same cache.

## Required observable checks and costs

Pure CPU checks should cover unset/default, explicit 0, stripped 0/1, invalid
values, dependency conflict, configured/effective mode isolation, and all-off
remaining Graph/U2. Consumer seam checks must then observe actual payloads,
stage/dependency order and final joins for each independently implemented
axis; parsing tests alone cannot close C6 1.7. Every rank's effective values
belong in the evidence. All-on/all-off plus one-axis differentials are required.

This configuration layer costs eight one-time parses and immutable records,
with no device allocation, control frames, per-layer environment reads or
synchronization. Stream/event/bank costs belong to the selected runtime
implementation. Stage diagnostics add warning-level host logging; compute-sync
diagnostics intentionally serialize live U1. Removal can consolidate parsing
into a future native immutable execution policy only after it preserves the
same defaults, explicit-zero behavior, dependency and mode isolation.

## Authorized pure settings implementation

`afd_plugin/connectors/npu/hccl_overlap.py` now implements the frozen settings,
explicit Mapping parser and fresh `dict[str, bool]` diagnostic record described
above. It imports only stdlib modules and neither reads nor changes process
environment. It does not configure the pipeline, coordinator, service, factory
or model guard. `effective_settings()` uses the eight exact environment names
as diagnostic keys; the returned dictionary can be modified without changing
the frozen settings. Strictly positive integer stage counts exclude bools.

The mirrored test loads the actual standalone module because parent connector
namespaces import device runtimes. No simulated device modules, shared fake
or new registry are required. Existing connector `validation_paths` owns
`tests/unit/connectors/npu/test_hccl_overlap.py`. Pure host checks cover defaults,
every option's explicit zero/one and invalid syntax, U1 ignored eager values,
the dependency under U1/U2, direct strict bool construction, frozen fields and
independent environment/diagnostic snapshots. This is configuration behavior
only; it does not complete C6 1.7 mode or scheduling parity.

Final pure-settings source SHA256 is
`ce0ff4b993dce0af6d0332ca296f18be84a9323a4f4b431f1a6312ca07513fc4`;
mirror SHA256 is
`29862c66e04e23644c17f6b9c685acbe2e501d26fa85aa26cd50097f3225ca90`.
Local Python 3.14 host verification ran
`.venv/Scripts/python.exe -m pytest -o addopts= -q tests/unit/connectors/npu/test_hccl_overlap.py`:
77 passed, zero skips, in 0.27 seconds. This includes seven explicit U1 invalid
Graph/diagnostic cases, confirming only the eager field short-circuits.
Ruff check/format and Python 3.11 grammar
parsing passed. No torch/vLLM dependency or device job was used for these tests.
The completed command's tool-result transcription, exit code, stdout, versions
and source hashes are recorded in `local-settings-20261110.json`; it explicitly
does not pretend to be an original raw log, and creating it did not rerun tests.

No existing runtime/tests, design/tasks or upstream files were edited, and no
device operation was run. The separate component and model-consumer evidence
remains authoritative for its exact bytes and scope.
