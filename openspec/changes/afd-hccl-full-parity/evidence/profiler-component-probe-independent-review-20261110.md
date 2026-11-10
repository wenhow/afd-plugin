# Independent review: bounded role-profiler probe

The reviewer inspected the complete probe and delivery harness at:

| Artifact | SHA256 |
| --- | --- |
| `tests/npu/afd_profiler_probe.py` | `2bed9af7772764447222464478bc542c80a03f0946c1cfad28487b198d8ff57d` |
| `evidence/run_a3_profiler.py` | `ad1050550381625d95247a3fab3ac07d0b3ce2f9d76cffc34b3d3d24265854e7` |

No static blocker was found for a separately authorized, bounded A3 component
run. The reviewer did not launch a device test. The harness freezes plugin
profiler `4dcd664757a788f26efbdf1a3dfaf8acbf7a0a007f26bd2e53c1b6a896d1c117`
and installed torch-npu profiler
`1a04902572383b74a401449e73929b2c4ce728d6c81bb59341876b03c7a82168`.
Its prepare-only archive check is not device evidence, and this pinned
installed source gate makes it A3-specific rather than an A5-ready script.

Attention executes on the role subprocess main thread and consumes the actual
deferred execute context. F creates its NPU tensors and profiler backend in
one background owner thread. A Queue handshake prevents that owner from
advancing before management has observed stop/restart requested-versus-applied
state; a sleep or elapsed-time assertion does not stand in for that boundary.
Each role uses two fresh profiling rounds of three controlled matrix multiplies
plus an unprofiled stop-boundary multiply. Exact integer-valued FP32 CPU
oracles are checked independently. Six profiler steps are distinguished from
the seven actual arithmetic checks.

The installed Python source (retained in the existing source-collection JSON)
shows `tensorboard_trace_handler(..., worker_name=..., async_mode=False)` at
180–196 and synchronous `analyse(async_mode=async_mode)`. The probe does not
accept stop returning as proof of export. It inventories actual files and
parses `trace_view.json`, requiring nonempty NPU MatMul kernel events in both
role/rank and restart paths. Missing or differently shaped trace evidence is
`pending-or-unproven`, not a successful timeline. This conservative filename/
event classification can still require refinement after seeing actual output;
the first raw attempt must be retained if it does.

The CPU supervisor checks source identity without importing torch, requires
eight idle boards before and after, runs roles serially, bounds deadlines, and
retains exact PID/start-tick ownership for cleanup. The outer harness checks
archive bytes and installed source before launch and also requires idle
boards afterward. No package, upstream source, HCCL group or model checkpoint
is changed. Failures and incomplete traces remain explicit failures; timeout
cleanup is supervision of owned processes, not a network abort claim.

Even a passing device run qualifies only the real plugin profiler session,
owner-thread lifecycle and controlled NPU arithmetic/export. It does not run
native A/F worker RPC or HTTP servers, ModelRunner V2, a model, Graph/HCCL, or
the full serving schedule. It does not prove complete native-model timeline
coverage or infer an overlap/performance result.

## Original attempt and bounded detector follow-up

The reviewer independently read the original result archive in memory using
`tarfile`, without extracting its paths or contacting the device:
`.tmp/profiler-role-owner-001-results.tar.gz`, 79,564 bytes, SHA256
`fee716ee670a5108b53b15388f766a9823e46957f0ab167aea9923c000bd5a8a`.
Its original pending/failure report remains unchanged; F was not started after
the Attention trace check failed. This is not a passing A/F attempt.

Both actual Attention trace files contain 115 events. Their metadata maps pid
4292955552 to `process_name="Ascend Hardware"` and `process_labels="NPU 0"`.
Each contains two duration events on that hardware process with `ph="X"`,
`args["Task Type"]="AI_CORE"`, and name
`aclnnMatmul_MatMulCommon_MatMulV2`. These hardware events have no `cat` field,
so the original `cat == "kernel"` classifier rejected valid hardware evidence.
The independent `kernel_details.csv` files each contain the same two tasks:
initial Task IDs 4/5, restart 8/9, Stream ID 46, FLOAT 64×64 inputs and output.
Their task IDs, stream IDs and names match the corresponding trace events.

| Original raw artifact | SHA256 |
| --- | --- |
| Initial `trace_view.json` | `c9b2a0f35c3c60204ef0a0c6e28edac02c387c4b9e9372433f57d934efe221ab` |
| Initial `kernel_details.csv` | `480c73099e7e9e9dcf69c8958c808a8b215e83915af9c9ffe8b7aa36678a5d02` |
| Restart `trace_view.json` | `b5bd488bc4a5ae3d49836561bf0ee1f0ae9ab0ac6b2d84e0b44fd330aba1a420` |
| Restart `kernel_details.csv` | `175441553584a4f77fc024f6b17bea393bf87af82559a3049762a50ec68cc0cb` |

The reviewer then inspected the complete changed detector and harness paths:

| Follow-up artifact | SHA256 |
| --- | --- |
| `tests/npu/afd_profiler_probe.py` | `d85e58662c08a813f39e2c5a0c9af8e2a990710c19b5a162831640ca532cacf9` |
| `evidence/run_a3_profiler.py` | `a86df96db3c004e929f78a429e0c2ec6c93186c263a04bf0db5be0e536b42a0b` |

The new detector requires the hardware-name/NPU-label pid intersection,
duration AI_CORE MatMul events, exactly two active tasks per round, and exact
`Counter(name, task_id, stream_id)` equality with the separately exported
kernel CSV. Missing files, malformed data, a missing initial/restart prefix or
inconsistent task evidence remain pending. CPU/operator enqueue events cannot
meet this hardware-process condition. Offline reinterpretation of original
bytes does not change the original run outcome or supply the unexecuted F case.

The harness now uses the retained CANN environment for its final `npu-smi`
check and stores the complete raw archive before mapping selected trace files
to short local names. It rejects duplicate/unsafe/nonregular archive members
and refuses to replace different recovered bytes. The original missing-library
postcheck and Windows path-retrieval failures remain part of attempt 001.
The owned-PID supervision and real owner-thread arithmetic/lifecycle paths are
unchanged. No static blocker was found for a new separately authorized bounded
run at these exact hashes. No such new device run was launched by this reviewer.
