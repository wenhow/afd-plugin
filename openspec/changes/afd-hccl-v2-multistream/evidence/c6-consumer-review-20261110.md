# C6 consumer wiring: independent source review

Scope: the current C6 diff in `hccl_v41_ffn_graph.py`,
`hccl_v41_graph_execution.py`, `hccl_v41_graph_coordinator.py` and
`hccl_v41_graph_service.py`. This review reads source only. It does not execute
native MoE, model preparation, NPU streams, HCCL or graph capture/replay. Source
identities and checked method coordinates are in
[the review manifest](c6-consumer-review-20261110.json).

One capture-resource identity finding was raised and corrected in source.
Initially replay accepted any pipeline whose bank and hybrid policy matched.
A replacement body/execution could therefore advertise a fresh pipeline while
the graph still recorded the original stream and events. Mutable body references
alone did not enforce retention of the original pipeline until graph reset.

The corrected A and F frozen resource dataclasses retain the capture-time
pipeline explicitly. Attention's `_retain` stores `execution.pipeline`; replay
requires both execution and captured stage body to refer to that same object.
FFN `_prepare_resources` stores its newly allocated pipeline beside the bank,
contexts and body; `_validate_resources` requires the body to retain that exact
pipeline. The existing bank, session and hybrid-policy checks remain. A fresh
same-bank/same-policy replacement is now rejected before graph launch, while the
frozen bundle still owns the original stream/event objects. This finding is
**closed by source inspection and the subsequent actual-dependency CPU
replacement regressions** described below.

The reviewed wiring preserves these boundaries:

- Pipeline policy is a frozen value exposed through a read-only session
  property. The default remains `None`. Existing-key recapture remains forbidden;
  a generation miss does not change an already recorded graph's stream policy.
- Each warmup/formal preparation allocates a fresh pipeline outside the tensor
  callback. Attention retains it through the current execution and then the
  cache's frozen resource bundle. FFN retains partially prepared/failed resources
  through its active owners and publishes the formal graph with its resources.
- FFN records parent receive, side compute, parent output-copy/mask/send and a
  final parent join in layer-major/stage-minor order. The forward context and
  layer metadata surround these tensor operations. JSON agreement, count
  updates, allocation, graph creation/publication and COMPLETE remain outside
  the captured tensor body.
- Attention preparation passes the same bank/pipeline to its stage body. Replay
  reuses the original captured pipeline; it does not allocate or call `run` on
  that already recorded owner. Native replay metadata handoff and completion
  remain in the existing outer hooks.
- Failure paths keep the execution or service resources reachable and make the
  cache/session terminal. The changes add no device abort, retry-as-eager or
  production cancellation promise. Cache close still belongs after acknowledged
  STOP and synchronization/reset.

No additional blocking finding was identified in these four consumers after the
identity correction. The Attention model-stage implementation and shared NPU/TLS
stream helper have separate owners/reviews; their state was not treated as
accepted by this four-file source review.

The actual-dependency CPU follow-up used two immutable snapshots, both with
matching installed upstream pins, no device probe, and no failures/errors/skips:

- [consumers-002](a3-pipeline-cpu-20261110/consumers-002/results/cpu.xml):
  **130 passed**, JUnit 3.785 s, manifest
  `90a6099351021b0b4ba5a1cd8bc1ecaab610d09f44c2da3c32c0c7b5455760f8`.
  The stream helper contributed 10 cases, base pipeline 31, FFN graph 43 and FFN
  service 46. The service tests rejected same-bank pipeline replacement under
  both hybrid policies before replay and checked retention through acknowledged
  STOP. These are CPU protocol/lifetime seams with substituted device boundaries.
- [native-consumers-003](a3-pipeline-cpu-20261110/native-consumers-003/results/cpu.xml):
  **240 passed**, JUnit 4.113 s, manifest
  `362b7dd86680baabb90fbb66e383da391fb1b01b1ca0cdfe1a29755946e7e266`.
  This includes Attention model body 42, GraphExecution 16, coordinator 60,
  session 19, main entry 11, pre-KV bridge 27, eager execution 22, dispatch 23,
  signatures 14 and six new pipeline-policy cases. The coordinator rejected
  same-bank pipeline and policy replacement before a new FORWARD and checked
  original capture resources through preparation/replay. FFN graph and
  GraphExecution match the initial review hashes. Coordinator/Service received
  subsequent explicit local-policy/None guards; a supplementary source read
  confirmed their original-pipeline identity checks remain and the revised bytes
  match the CPU snapshots. Both initial and executed hashes are kept in the
  review JSON.

Later ParentTLS model source `a3140845` is a separate source revision from the
`dcae2bc4` model body used by 003; the 003 result must not be applied to that later
revision without its own affected regression. The two
[controlled NPU sessions](a3-pipeline-device-20261110/README.md) exercise the real
pipeline/bank/HCCL graph components with arithmetic, not these four production
native-model consumers. No CPU or controlled-device result here is full-model
acceptance.
