# Five fixed Graph axes in the controlled native manager probe

Status: all four authorized initial component sessions passed; see
`a3-graph-axes-20261110/README.md` and `validation-summary.json`. Previous
hybrid-only device evidence remains bound to its original immutable snapshots.
Only the existing probe and harness change; no runtime/source package is edited.

## Explicit CLI and immutable session schema

Both `tests/npu/hccl_graph_manager_probe.py` and its existing native-manager
harness accept five new optional binary arguments:

```text
--graph-compute 0|1 --graph-hybrid 0|1 --attention-three 0|1
--ffn-recv 0|1 --ffn-cross 0|1
```

Supplying one requires all five, plus `--stage-bank`. The new arguments exclude
legacy `--compute-pipeline` / `--pipeline-hybrid`; those flags and the default
path keep their prior meaning. `--ffn-cross 1 --ffn-recv 0` is rejected locally
before a torch import, SSH, directory creation or device allocation.

Each isolated rank constructs one `HcclOverlapSettings` from these explicit
values. Eager overlap and both diagnostics are explicitly false, independent
of ambient environment. Settings remain fixed across the native manager,
cache, all warmup/capture phases and replays; changing a setting requires a new
process/session/cache. The wire Graph key and parent plan are unchanged.

Each actual descriptor's fresh warmup/formal allocation constructs
`HcclOverlapSelection(settings, actual_role, graph_warmup|graph_capture, U)`.
The actual native warmup callback still receives mode NONE, while its logical
selection remains Graph warmup. `HcclGraphPipelinePolicy.from_selection` decides
whether to allocate. U1 or compute=0 gives **pipeline None**, while native
manager capture and HCCL Graph execution continue. Active U2 compute calls
`HcclGraphComputePipeline.allocate(bank, selection=selection)` with the public
production selection contract, never a manually equivalent policy.

The per-rank report preserves configured settings, each role/phase/U diagnostic
projection, allocated/not-allocated state, active policy, compute stream and
event counts. A captured bundle retains its selection and exact pipeline.
Replay retrieves that original bundle; it does not reconstruct or allocate a
pipeline, stream/event bank, production stage bank or Graph body. CPU oracle
and input refill remain outside the captured body, as in prior probe evidence.

## Equivalent controlled arithmetic with pipeline None

New matrix mode always executes the same two-layer recurrence and final
Attention continuation, regardless of whether compute is enabled. The former
controlled pipeline body now has parent-stream null contexts when the selected
pipeline is None. This prevents all-off or U1 from silently becoming a different
arithmetic workload or falling back to eager. Its output oracle applies exactly
the same recurrence as active U2 compute.

When a pipeline is active, Attention compute/exchange/final joins still use its
real public contexts. F receive/compute/send use their production public
contexts; **cross-layer=false calls `join_layer(layer)` after every complete
layer**, followed by final `join`. All HCCL remains on the actual parent stream.
No CPU control/wire/cursor is inserted in the tensor callback or replay.

The existing four native Graph keys and 26-transaction session remain:
four warmups, four formal captures, twelve replays and six honest eager
fallbacks. Fixed-key replays vary controlled valid counts full/1/T-1, including
U2 valid-zero and partial stages. Physical native dummy geometry is unchanged.
Every layer's BF16, original FP32, IDs and output is compared to CPU references,
and final Attention continuation is checked. Dirty padded IDs, device output
masking, bank address/byte checks, both original fallback reasons, STOP/native
TP retention and bounded supervision stay in scope.

These are real stream/event/HCCL/Graph components with controlled arithmetic.
They do not execute native V4.1 Attention, HC/indexer, Engram, MoE, ModelState or
ModelAclGraphManager, and do not establish hardware overlap or performance.

## Proposed independent device sessions

The tuple order below is compute, hybrid, Attention-three, F-recv, F-cross.
The parent requested at least these four sessions after source review:

| Proposed label | Tuple | Distinguishing behavior |
| --- | --- | --- |
| `all-on-001` | `1,1,1,1,1` | Active U2 side compute; every configured scheduling axis enabled. U1 remains pipeline None. |
| `all-off-001` | `0,0,0,0,0` | Same Graph/U2 workload with all pipeline work on parent, not eager fallback. |
| `compute-parent-wire-001` | `1,0,0,0,0` | U2 compute, canonical parent frontier, no logical A-three/F-recv or F-cross, layer joins. |
| `recv-layer-join-001` | `1,0,0,1,0` | Same preceding tuple with F-recv enabled and F layer joins retained. |

Each would use a new immutable evidence directory under
`a3-graph-axes-20261110`, one independent session at a time, devices 0/2 and
31 installed-source checks. No additional device combinations are implied.
Fine-grained axis-isolation sessions can be proposed separately if findings
justify them; this initial matrix is not exhaustive combination coverage.

The harness now supports `--prepare-only`, which validates unique archive
members and each manifest hash without SSH. It passes all five arguments to
the remote harness and each worker, preserving configured values in identity
and summary JSON. Generated source-expected and installed stream API raw JSON
are manifest members. No prior raw evidence is rewritten.

## Local candidate identity and checks

- Probe SHA256: `1d7f7b1ef445dd852026d66919bc75c20e62b429e95c66e5ac47e2e286213365`.
- Harness SHA256: `4a3e0b65a33562c9e63735f5dc0baa335522cbe468a58e53cde89b4f6773ac56`.
- Production pipeline expected SHA256: `2105627f04b4693c44d5fc12c4ffac318cf540f6709fbb86228b60576540c066`.
- `a3-graph-axes-20261110/static-check-001.json`: Ruff/format/Python 3.11 AST,
  two CLI help commands and eight parse-only scenarios pass; no torch/device.
- `a3-graph-axes-20261110/archive-precheck-001/identity.json`: prepare-only
  135 unique archive members, 134 manifest entries, 31 installed-source pins.
  Manifest SHA `5d570fef24725af73f06b75be46998a715f9250593e116fb935c89ff65fed787`.

The new production overlap/consumer CPU suite is parent-owned and has already
passed 460 actual-dependency cases. It is not repeated by this probe work.
Root and independent source review accepted the exact candidate before the
four serial sessions. The device results preserve each original snapshot;
preparation evidence is not rewritten or promoted to device execution.
