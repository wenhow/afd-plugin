# A3 local-count collection and FFN service evidence — 2026-11-09

The new cap1 protocol passed actual-dependency CPU regression and real HCCL
component probes on A3. The environment reused Python 3.12.9, torch 2.10.0,
torch-npu 2.10.0.post4, vLLM 0.30.0+empty, Ascend g7e37e29a7, CANN 9.0.1/ATB,
and FastAPI 0.123.10. No packages were installed or upstream sources changed.

## Actual results and preserved attempts

| Attempt | Result | Scope |
| --- | --- | --- |
| `counts-cpu-001` | pytest exit 4, zero cases | Initial uploader used a worker-test path without its `npu` directory. Raw output is retained. |
| `counts-cpu-002` | **324 passed**, exit 0 | Eight protocol/FFN/runtime CPU modules with installed dependencies; no device probe. |
| `service-1a1f-001` | exit 1 | Probe setup omitted `init_ascend_config`; F failed before receiving its first parent, and A saw Gloo peer closure. No timeout or process kill; all boards idle afterward. |
| `service-1a1f-002` | **passed**, exit 0 | Same-session local-count collection and FFN service on chips 0, 2. |
| `service-2a2f-tp2-001` | **passed**, exit 0 | Two A reports and two F ranks, each role one native TP2 world, on chips 0, 2, 4, 6. |
| `binding-cpu-003` | 3 failed, 123 passed | Shared-policy/FFN binding checks rejected the three unsupported configurations correctly; worker tests expected uppercase `HCCL` while the established policy message contained `Hccl`. Preserved for the test owner to correct. No device probe. |
| `binding-cpu-004` | **126 passed**, exit 0 | 108 shared-policy and 18 FFN worker cases after correcting only the test's message matching. No device probe. |
| `attention-cpu-005` | **112 passed**, exit 0 | New A execution scope 19, U2 runner 44, model-wrapper seams 41, and the unique upstream signature table 8. No device probe. |
| `bootstrap-cpu-006` | **98 passed**, exit 0 | New staged adapter 14, configuration validation 45, and NPU runtime configuration 39. No device probe. |
| `identity-cpu-007` | **39 passed**, exit 0 | A execution 20 and FFN worker 19, including rejection of a second transport distinct from the adapter's own transport. No device probe. |
| `integration-cpu-008` | 4 failed, 2238 passed, 2 skipped, 4 deselected | Broad CPU regression exposed four old tests patching a removed runner-module `AFDConnectorFactory` import. The test owner updated them to the public `create_npu_runtime_connector` seam. Raw pre-fix output retained; no device probe. |
| `integration-cpu-009` | **2242 passed, 2 skipped, 4 deselected**, exit 0 | Final frozen production/tests across model executor, worker, connectors and distributed, plus feature/config/runtime/signature modules. Includes C5's 22 CPU graph-component cases. No device probe. |

Both successful device snapshots use the same final probe bytes and protocol/F
executor bytes. All ten pinned installed-source hashes matched. The supervisor
recorded eight idle boards before and after each probe, no timeout, and no
process kill. Each rank's native TP all-reduce passed before initialization,
after initialization, and after closing only the plugin-owned groups (TP1 sum
1; TP2 sum 3).

`counts-cpu-002` contains 50 transport, 116 codec, 13 NPU FFN worker, 9 NPU FFN
model runner, 32 existing FFN model runner, 55 NPU runtime, 4 async FFN runtime,
and 45 staged FFN executor cases. Its F worker bytes precede the subsequent
shared-policy changes; this result is not the final shared-policy/A-entry CPU
qualification. The per-attempt identity manifests define its exact source.
`binding-cpu-004` separately qualifies the updated shared-policy/FFN binding
boundary. Its HEAD is `6d43f0390eb1bb846c4109d82f56b661bdb21aab`, canonical
manifest hash is
`fa0a1ed00d7bb591b247a7750717620d08261d20064ba3a129248b4e4bd23176`,
and tested production hashes are `hccl_config.py=1a44e0f61cab9b96a643e1321681509649375f27f34a7bec9854400ff663f9c7`
and `ffn_worker.py=fa94b1314217ccc9a1a13a37ddaf4e7e16813582fb58ffff1c4d2804cf7afd18`.
This pair does not qualify the later bootstrap tests or A-side public entry.
`attention-cpu-005` separately qualifies those A-entry CPU seams with actual
installed dependencies; it is not an A-side NPU execution probe. Its canonical
manifest hash is
`83e1a8af08deb8b2af998d4a82352544d2f3209e197272fe3cddfe71fa0b3099`,
and `hccl_v41_execution.py` has SHA256
`98914b3375fe4d9de884651761d438d907f14400285c042adb9a5c39ab741cc3`.
The two separate 112-case results in the broader C4 evidence cover different
module selections; their raw identities, rather than the case totals, identify
each qualification.
`bootstrap-cpu-006` qualifies the staged adapter/factory payload seams and
configuration selection. The later independent-review session-identity guards
are tracked as a separate incremental validation rather than attributed to
these earlier snapshots.

The final integrated CPU result is `integration-cpu-009`. Its HEAD is
`6d43f0390eb1bb846c4109d82f56b661bdb21aab`, canonical source-manifest hash is
`3de45ebd5a236ccc07a9dac5d4bbc0bbe72320e6b15e46d8c78bd9eb63024b2d`,
and all ten installed-source checks matched. It includes the final copied
native wrapper body (`deepseek_v41.py=5e1690b161674180faca8407eba806c8ad642418655b1e7eb4d43e8bc27f6007`),
the final execution scope (`hccl_v41_execution.py=bcd29066f59e7c053b183b77a6c249b92d22ec2f634f56ef87ac8a6548a85af4`),
the A/F transport identity guards, and the corrected existing constructor
fixtures. The two skips are existing fused-top-k cases requiring CUDA. The
four device-marked cases were deselected. The new C5 source and 22 CPU tests
were included because the connector scope discovers them; this is not C5
Graph hardware acceptance. Its separately executed hardware evidence remains
under the C5 change.
All tested C4 production bytes still matched when this summary was written.
The independent C5 graph module changed after this snapshot, so its 22 included
CPU cases qualify the snapshotted `d1f876...` version only; later C5 changes
need that change's own incremental evidence.

## Device behavior exercised

All A ranks call the actual
`HcclStagedTransport.submit_forward(HcclLocalForward(...))`. Only A0 supplies
the transaction ID; the probe supplies rank-local extents and never constructs
the global parent plan. It verifies the accepted plan's collected counts,
digest, and returned BF16 output against an independent arithmetic oracle.

Each F rank calls the actual
`AFDNPUFFNModelRunner.execute_hccl_v41_step(transport)`, which receives the
parent, enters the actual staged FFN executor and native forward context, and
completes the parent. The probe supplies a weightless runner shell and a
`torch.nn.Module` performing controlled arithmetic. The production FFN runner
constructor, admission, loader and native MoE are not exercised. No production
guard is patched or opened; the shell supplies only executor dependencies.
The model checks the actual payload, transaction/stage metadata, valid and
padded extents, context model/IDs, and layer-major order, then performs real NPU
arithmetic and a native TP all-reduce. HCCL and its process groups are real.

Each session executes the same five transactions:

| Transaction | Local valid rows | Local padded rows | Active layer/stage exchanges per rank |
| --- | --- | --- | --- |
| `counts-u1-first` | `(1,)` | `(2,)` | 2 |
| `counts-u2-first` | `(0, 3)` | `(0, 4)` | 2; first stage skipped globally |
| `counts-u1-fallback` | `(3,)` | `(4,)` | 2 |
| `counts-u2-second` | `(1, 0)` | `(2, 2)` | 4; second child contains padding only |
| `counts-empty` | `(0, 0)` | `(0, 0)` | 0; service still completes the parent |

Both layers `(0, 2)` retain BF16 hidden states, independently constructed FP32
router values (including differences not representable in BF16), and explicit
INT32 IDs. Every Attention rank finally calls `stop_attention`; F's real
service returns `False` after acknowledged STOP. U1/U2 here describes protocol
stage counts. The transaction name `fallback` does not establish production
native A-side fallback dispatch.

This evidence does not validate A-side native request preparation or public
model dispatch, full worker construction, model weights, native HC/indexer/MoE,
Engram producer/history, successful full-model inference, Graph, overlap or
performance. The cap0 legacy topologies were not rerun in this round.

## Source identity and raw files

| File | SHA256 in both successful device snapshots |
| --- | --- |
| `tests/npu/hccl_staged_probe.py` | `38f49aa99df791358b174081f2ba20d778099b174f969cd46dbe53a0fd88b28a` |
| `afd_plugin/distributed/hccl_plan.py` | `b3da12187d1355b05630f9167b1da198bbb9b8a18b25e7fc7c13e675983128b2` |
| `afd_plugin/connectors/npu/hccl_staged_transport.py` | `76878b3103a31338cb88033f4315b99ca33371183cc1ea4942c7b2c6d4d6d344` |
| `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `e1603fd984109293e73afb3af0ffd5ab062f1b86ad5d33c44b76a1aa1973e4f2` |
| `afd_plugin/v1/worker/npu/hccl_v41_ffn_runner.py` | `9628ce7861526c503d0aeebe6254f3a5c37147bb7335e811b7d36a3fc0d90eec` |

The TP2 snapshot records actual HEAD
`6d43f0390eb1bb846c4109d82f56b661bdb21aab` and canonical manifest hash
`77e508c4249db1753f0ef963282964c1c0205fb0e6c7c8132cf16fabf6ac027d`.
HEAD is the base revision; the source manifest identifies the tested working
tree bytes. Earlier attempts predate these extra uploader fields and retain
their full file manifests and archive hashes.

Each attempt directory retains `identity.json`, uploader delivery/remote logs,
`results/summary.json`, source checks and environment versions. Device results
are in `results/probe/result.json`, per-role JSON files and per-role logs.
CPU output and JUnit XML are in `counts-cpu-002/results/cpu-forward-counts.*`.
The final full command/result is in
`integration-cpu-009/results/summary.json`, `cpu-integration.log`, and
`cpu-integration.xml`. `validation-summary.json` indexes the successful and
superseded attempts without replacing their immutable raw files.

After this validation, the operational CLI added `--model` to let A5 select
its own local config path. The default remains the original A3 path. The
supervisor forwards the value to every child, and `build_config` consumes it.
This final probe SHA256 is
`96112b8192e2f1ff1a186b46f3f4d2f71910e1c15fd5ea4c7fe9821c744b796d`.
The successful hardware snapshots retain the earlier `38f49...` bytes; their
compute/control implementation was not changed by this CLI-only delta. Local
`--help`, Python 3.11 AST parsing, Ruff check and Ruff format checks passed.
The hardware matrix was not repeated for this config-path parameter.

## Reproduce

Choose a fresh label for every immutable snapshot:

```powershell
.venv/Scripts/python.exe -X utf8 -B openspec/changes/afd-hccl-v2-microbatch/evidence/a3-forward-counts-20261109/run_a3_forward_counts.py --label counts-cpu-003 --cpu-scope forward-counts --cpu-only
.venv/Scripts/python.exe -X utf8 -B openspec/changes/afd-hccl-v2-microbatch/evidence/a3-forward-counts-20261109/run_a3_forward_counts.py --label service-1a1f-003 --topology 1a1f
.venv/Scripts/python.exe -X utf8 -B openspec/changes/afd-hccl-v2-microbatch/evidence/a3-forward-counts-20261109/run_a3_forward_counts.py --label service-2a2f-tp2-002 --topology 2a2f-tp2
```

For direct A5 component validation, pass its local model path explicitly:

```bash
"$afd_python" tests/npu/hccl_staged_probe.py --model "$afd_model" --topology 1a1f --devices 0,2 --local-counts-service --output-dir /tmp/afd-counts-component-001
```

The uploader preserves the existing CANN/ATB paths and validates installed
sources. The device supervisor bounds each run to 240 seconds and only reaps
its own children. The outer operational deadline is 280 seconds, allowing that
supervisor to finish its own cleanup first. CPU-only runs never start the
device probe.
