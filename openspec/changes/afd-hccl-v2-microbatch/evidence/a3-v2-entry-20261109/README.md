# A3 native V2 U2 constructor evidence — 2026-11-09

`constructor-001` passed on A3 chip 0 using the existing Python 3.12.9,
torch/torch-npu 2.10, CANN 9.0.1 and FastAPI 0.123.10 environment. No dependencies
were installed. The supervisor recorded all eight boards idle before and after
the probe, verified all ten pinned installed-source hashes, and returned zero
without a timeout or process kill.

The immutable working-tree snapshot is identified by
[`constructor-001/identity.json`](constructor-001/identity.json). The remote
scratch is
`/mnt/workspace/tmp/afd-c0-20261104.hVgsIf/c4-v2-entry-20261109-ym17saug`.
The actual environment, command, process result and device snapshots are in
[`constructor-001/results/summary.json`](constructor-001/results/summary.json),
`versions.json`, `effective-env.json`, `npu-before.log`, and `npu-after.log`.

## Verified boundary

The standalone probe uses actual `ModelConfig`, `VllmConfig` and config-only V4.1
metadata generated inside its output directory. Only native **model shells**
bypass model construction; `EngramModelState` and all three inherited runner
constructors execute normally. Engram is disabled (`token_lookback_depth == 0`),
and attention groups are empty.

After normal Ascend platform configuration, the probe sets `ubatch_size=2` on
this explicit component's configuration. Ascend's normal platform configuration
resets the CLI value; this probe does not open production U2/model admission.

The actual native `torch_cuda_wrapper()` supplies NPU stream aliases. Inside
`use_npu_ubatch_factory(..., hccl_transport=transport)`, the native
`model_runner.maybe_build_ubatch_runner` constructs `HcclV41UBatchRunner`, including
`NPUUBatchRunner.__init__` and native `UBatchRunner.__init__`. The probe checks:

- Two distinct actual NPU streams.
- Two separately allocated, zero-initialized INT32 query-start buffers of shape
  `[3]` and sequence-length buffers of shape `[2]` on `npu:0`.
- Restoration of the native factory and SM-control aliases when the AFD scope
  exits. The native CUDA wrapper has its own upstream process-scoped behavior.
- The public native `run` entry rejects a real uninitialized
  `HcclStagedTransport` with `native staged run requires an active HCCL transport`.
  Construction leaves the caller-owned transport uninitialized and open.

The raw result is
[`constructor-001/results/probe/result.json`](constructor-001/results/probe/result.json);
the full process log is
[`constructor-001/results/probe.log`](constructor-001/results/probe.log).

This does **not** validate attention metadata builders or `prepare`, an accepted
HCCL parent, successful model execution, Engram producer/history, native HC,
indexer/MoE, checkpoint weights, Graph, DBO overlap, inference correctness or
performance. No distributed groups were initialized.

## Source identity and complementary CPU results

| File | SHA256 |
| --- | --- |
| `tests/npu/v41_ubatch_constructor_probe.py` | `49d6a407788f95d7d700a6683d3b8205e786ed8cf82a54690c16f5ce3d9c97e1` |
| `afd_plugin/compat/patches/npu/mrv2_ubatching.py` | `1e53b24475946f0cc83d6a05b5a83af5d4746287bd9ba04f357021df27905ecb` |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `7d9ec420eb7d15192819e47587e37116e846dde7967998ea615c284b95ea3435` |
| `run_a3_v2_entry.py` | `1c1113cfda774c6d1fe83e819c20c750d261a0bda9f01da2fa94e2ba2b20747d` |

The separate CPU snapshot
[`../a3-stage-state-20261109/native-entry-cpu-008/identity.json`](../a3-stage-state-20261109/native-entry-cpu-008/identity.json)
contains the same final production factory/runner bytes and unit-test hash
`ba7ac5e1af8e325a7d404f8a1ebae635e31a27f479d7ac1d5f3edc075e414d9d`.
Its actual-dependency result is **112 passed**: 41 stage-state cases, 28 staged
model seam cases, and 43 U2 runner cases. It did not start a device probe. See
[`cpu-seams.log`](../a3-stage-state-20261109/native-entry-cpu-008/results/cpu-seams.log)
and its JUnit XML for the exact cases. These CPU seams and the actual NPU
constructor probe are separate validation levels.

## Reproduce

From the Windows checkout with the configured SSH host and the existing A3 venv:

```powershell
.venv/Scripts/python.exe -X utf8 -B openspec/changes/afd-hccl-v2-microbatch/evidence/a3-v2-entry-20261109/run_a3_v2_entry.py --label constructor-002
```

Choose a new label for each attempt. The runner creates a new remote scratch,
captures bytes and hashes before upload, preserves CANN/ATB library paths,
checks device idleness, bounds the probe to 120 seconds, and reaps only its own
process group if the deadline expires. `--prepare-only` uploads an immutable
snapshot without running tests or allocating a device.
