# Native Attention runner graph entry independent review

Static review of the root-owned main entry, graph policy and dispatcher delta
on 2026-11-10. The reviewer implemented the separate pre-KV bridge; this file
records the integration boundary and root's independent dispatcher finding,
and does not self-certify the bridge's architectural acceptance. No upstream
edits, NPU experiments or production Graph admission belong to this review.

## Exact reviewed files and sources

| File | SHA256 |
| --- | --- |
| `afd_plugin/v1/worker/npu/attention_model_runner_v2.py` | `ec5785cf81989002ed7c64070563a5ab2d2e5b48c11bd55067e181ba72729598` |
| `afd_plugin/compat/patches/npu/hccl_graph_dispatch.py` | `a845dcc6490f9a4a58afedf95c6f6fbd19ff43445df9fd1de4079064d9969d7f` |
| `afd_plugin/connectors/npu/hccl_config.py` | `8390a5a49633b60c452206f804963bd97ca7c88572dc0fabb0a52095dc7fba99` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph_eager.py` | `cd0f31fdc74ec1e3acb23d17d790e221015f617fe95d8f2ff35c3fc7b83961c3` |

Locked vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`:

- `vllm/v1/worker/gpu/dp_utils.py:214`,
  `dispatch_cg_and_sync_dp(cudagraph_manager, num_reqs, num_tokens,
  uniform_token_count, dp_size, dp_rank, max_query_len=None, need_eager=False,
  num_active_loras=0, parallel_config=None, allow_ubatching=False,
  uniform_decode=False, dp_sync=None) ->
  tuple[BatchExecutionDescriptor, DPSyncState | None]`.
- `vllm/v1/worker/gpu/model_runner.py:998`,
  `capture_model(self, *, profile_only: bool = False) -> int`;
  `execute_model` at line 1631 and `shutdown(self) -> None` at line 2238.

Locked Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`:

- `vllm_ascend/worker/v2/model_runner.py:352`,
  `initialize_kv_cache(self, kv_cache_config: KVCacheConfig,
  kv_cache_allocation_context: AbstractContextManager | None = None) -> None`.
- The inference-mode `execute_model` at line 486 has parameters
  `(self, scheduler_output: SchedulerOutput,
  intermediate_tensors: IntermediateTensors | None = None, dummy_run: bool =
  False, skip_attn_for_dummy_run: bool = False, is_profile: bool = False,
  context_len: int = 0, valid_dummy_state_slots: bool = False)` and no return
  annotation. Its `finally` calls `model_state.finish_execution` at line 530.
- Native `profile_run(self) -> None` at line 584 preserves Ascend's MC2/profile
  dummy calls before the later real-KV graph lifecycle. Native capture remains
  the inherited GPU entry above.

The main overrides preserve these signatures and forward all native arguments.
Their comments explain delegation of the large native execute, KV/capture and
cleanup bodies; the new host ownership logic remains in marked AFD sections.

## Finding and resolution

Root's independent review identified a real pre-KV dispatch ordering bug in
the first integration. The hook calls the original native dispatcher before
its controller selector. For a real gathered batch, `manager=None` and
`need_eager=False` therefore hit native's manager assertion before the
selector's `reserve_batch(need_eager=True)` could run. Merely forcing the later
controller candidate to eager would not fix it.

The reviewed dispatcher now includes `cudagraph_manager is None` in
`effective_eager` before calling the original method. The hook first requires
the passed manager to be the selected owner's actual manager. The manager-free
bridge requires both owner/control managers to be None, while the post-KV
coordinator rejects loss of its retained actual manager. This establishes a
genuine native NONE descriptor before input preparation, preserves native DP1
synchronization, and still performs the graph-protocol reservation from the
actual gathered batch. Tests must exercise the default non-profile real-batch
call without adding `need_eager=True` in their setup to hide this ordering bug.

## Ownership and lifecycle

The explicit pre-KV binder checks the owning runner, transport/control/session,
manager-free and no-ubatch startup, idle initialized wire and active cache. KV
initialization passes that same session into the selected native U2 factory.
The real-KV binder retains the same cache, generation callback, metadata-shape
callback, transport and session, and matches the selected native manager.
Per-execution storage checks remain authoritative after binding.

The main execute wrapper selects the bridge only while its actual manager is
None, and the coordinator after initialization. It derives target/dummy/profile
mode and valid tokens, then delegates the exact selected Ascend execute once.
For real batches the bridge consumes the actual ordered gather result. Native
dummy/profile gathering may honestly return None; the bridge projects the
original native eager descriptor without fabricating request state. A genuine
zero-work return reserves an explicit zero descriptor and completes its empty
counts/ACCEPT/COMPLETE handshake. Nonempty returns without dispatch or without
parent completion reject the session. Neither bridge nor main wrapper calls
Engram retirement; the selected native Ascend `finally` remains its sole owner.

Capture goes through the retained coordinator's all-peer preflight and owning
native capture entry. The legacy capture/provider path does not surround that
selected call. Normal connector construction still omits the new policy's
`graph` keyword, whose default is False; it retains eager-only admission. An
injected graph1 transport receives the separate strict NONE-compilation,
FULL_DECODE_ONLY, BF16, spec-off, DP1/PP1/CP1, paired native-TP checks. The model
guard and ordinary Graph factory admission remain closed.

Successful shutdown orders all-peer STOP/ACK before cache synchronization and
graph reset, native abort/cleanup and finally AFD group destruction. A failed
STOP or cache reset escapes before native storage release. Failed native
cleanup retains graph/native owners and process groups for supervision. The
cache and native dependencies remain alive while reset executes; a local
terminal flag is not claimed to cancel network or device work.

No further hard blocker was found in the reviewed explicit entry/policy delta
after the dispatcher fix. This is a host ownership conclusion, not proof of
V4.1 model weights, Attention kernels, offset views, Engram graph production,
native graph replay correctness or hardware performance.

Costs/removal: each pre-KV parent adds a graph-external eager reservation and
the existing actual-count/COMPLETE control and device-completion frontiers.
The bridge adds no bank or graph allocation and retains references until
handoff/cleanup. Removing scoped gather/dispatch and main wrappers requires
native owning pre-preparation and lifecycle callbacks; no performance result
or upstream acceptance is asserted.

## Validation boundary

Local Ruff check/format and Python 3.11 grammar checks passed for the bridge's
three files. The CPU-only local test gate reports one module skip because
NumPy is unavailable; that is not a runtime pass.

The immutable A3 `startup-bridge-016` snapshot retains its original result:
441 passed and 40 failed. Its failures included the new shared fixture's
missing worker-initialized native `parallel_config.rank` and two older fake
execution cases missing `cudagraph_manager`. The shared pre-KV fixture was
corrected to set rank zero without changing production validation:
`tests/unit/fakes/hccl_v41_graph_eager.py` SHA256
`896ca34d1db21df5d09cb2bfa8538822c005e7641641dcb6e7e96cf668c2de08`.
The historical 016/017 bridge mirror was SHA256
`739dd76fcb8337356cec85f5bfdd55303e44d31d2fa47a72fd8b82436c61a7ee`.

The immutable `startup-fix-017` affected-module rerun reports 53 passed and
7 failed: main entry 11/11 and existing execution 22/22 passed, while the
bridge had 20 passes and seven assertion failures. Six expected-transcript
cases omitted the actual global GRAPH_ACK; one post-scope restoration check
called native dispatch with manager None but without its required explicit
eager flag. Only mirrored assertions were corrected; production bridge and
dispatcher bytes above were unchanged. The intermediate 018 mirror SHA256 was
`79ff77054e3ff609af4b20d2d8ba4c40cebeae80c102c224f15d176b21127c09`.
That bridge-only run passed 21 and failed the same six full transcript cases:
the assertion incorrectly expected COUNTS_REQUEST in a 1A1F topology, and
omitted the global ACCEPT frontier. Actual `submit_forward` solicits only
other A ranks, of which that topology has none. This was another mirrored
expectation error, without a production change. The final test now asserts
the bridge seam: graph QUERY/DECISION/ACK order, ACK before FORWARD before
COMPLETE, accepted plan and model outputs. It retrieves the plan by message
kind rather than numeric index and leaves the complete transport transcript
contract to transport-owned tests.

The final mirror SHA256 is
`d783f450b76dbeb6bbf3d633016dad2c436a079ec2bb409c58296d737d7f1b1f`.
Immutable `eager-protocol-019` runs only those six affected parameter cases:
6 passed, zero failures/errors/skips, with `source_files_match=true`. Its raw
artifacts are
`a3-control-cpu-20261109/eager-protocol-019/results/{cpu.xml,cpu.stdout.log,summary.json}`.
Production bridge and shared fixture hashes are unchanged from 018. The prior
21 passing cases remain qualified by their 018 snapshot; these runs are not
presented as one new 27-case run of the final mirror. The prior failed/skipped
snapshots remain intact. These CPU host entry tests do not establish actual
native request/KV execution or NPU model acceptance.
