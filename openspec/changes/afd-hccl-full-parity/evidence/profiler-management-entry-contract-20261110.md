# Profiler external management entry: source contract

This source audit now has a narrow plugin-owned endpoint implementation.
It adds no native patch, external headless management client or socket. The
A/F worker session APIs and their evidence remain separate from HTTP/API
behavior, a deployed server and actual NPU trace acceptance. Directed actual
dependency CPU validation for the new endpoint is recorded separately below.

## Existing call chain and owner

All native entries below are locked vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` Git objects.

| Source / symbol | Exact source SHA256 | Contract |
| --- | --- | --- |
| `vllm/entrypoints/serve/profile/api_router.py:37`, `def attach_router(app: FastAPI):` | `b943c3a2201134e2d0b38bfa86d00e8b5685f5be6a54f6c7178dee784c1b78e5` | Includes `/start_profile` and `/stop_profile` only when native `args.profiler_config.profiler` is set. Role-env alone does not attach them. This native function has no return annotation; no patch is needed. |
| `vllm/entrypoints/serve/__init__.py:19`, `register_vllm_serve_api_routers` | `bac4ffa8440759d9a244acc25bf28ce6700d7106a2b35eff90c35e70a7519f52` | Lazily imports the native profile module's `attach_router`. |
| `vllm/entrypoints/launchers/api_server/routers.py:12`, `register_api_routers` | `c2f1bb2708f44f0fe24f3d8ab8f884c7262e551c6557ebc606263abc4d7b1db1` | Registers serve routers for the existing FastAPI app. |
| `vllm/plugins/endpoint_plugins/interface.py:44`, `EndpointPlugin` | `9293616ccd3cdf725fa087d3f76a518dba871ef2c796446990fa2a0a5fc159ee` | Public zero-argument factory protocol: `name`, `required_tasks`, `attach_router(self, app: FastAPI) -> None`, and `async init_state(self, engine_client: EngineClient \| None, state: State, args: Namespace) -> None`. Phase A calls each router at 91–105; Phase B awaits each initializer at 108–123. |
| `vllm/plugins/__init__.py:93`, `load_endpoint_plugins` | `fb6e6ee432c5a4ae207aaffe4b546aca9381f56ee8498b0a3e0d9af2d289d023` | Endpoint group is `vllm.endpoint_plugins` (30). Explicit `VLLM_PLUGINS` allowlisting is mandatory (121–132), independently of general-plugin registration; task intersection gates each plugin (142–153). Factory failures are logged/skipped (135–140). |
| `vllm/entrypoints/launchers/app.py:19`, `build_app` | `1f1ea5b79688cc042d9418435e84a6d04c17a46f6de6a98e560fe2f2ac5eeca4` | Calls native core routers at 45, then `attach_endpoint_plugins(app, supported_tasks)` at 51. No global monkeypatch is needed. |
| `vllm/entrypoints/launchers/api_server/app_state.py:24`, `init_app_state` | `7a00aa48a9b4d62961e64b33d78c3016e6eccd684c7f8bf5ff221b2ac27eeb10` | Retains the actual `state.engine_client` at 63 and `state.vllm_config` at 65; awaits endpoint initialization at 157. Routes can bind the existing client and actual typed configuration. |
| `vllm/entrypoints/launchers/api_server/entry.py:136`, `build_and_serve` | `db689f8486e5d2c9eb59d347e03a27ce837d014f8d3d7f84d55a0e382813ca84` | `build_app` is followed by awaited `init_app_state` at 137 before `serve_http` at 141. Initialization errors can fail startup before serving. |
| `vllm/engine/arg_utils.py:795`, `EngineArgs.__post_init__` | `ead004ca266419cd25f54a7aefc0e20e1eb046f363b7f5cc977e14f52ee44bb1` | Calls `load_general_plugins` at 839; `AsyncEngineArgs.add_cli_args` also calls it at 3009. The existing `afd` general entry installs worker behavior independently of the endpoint entry. |
| `vllm/v1/engine/async_llm.py:1112`, `collective_rpc` | `52cbf404f1dbc99c6bebf4a50a7eb18ff842a1f53106a45f365523192d816be1` | Awaits `engine_core.collective_rpc_async(method, timeout, args, kwargs)`. Native start/stop at 1044/1050 additionally manage an API-side profiler if configured. |
| `vllm/v1/engine/core.py:964`, `collective_rpc` | `268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5` | Forwards to this EngineCore's model executor. UTILITY requests at `_handle_client_request` resolve the named core method, returning either its result or a failure message. |
| `vllm/entrypoints/cli/serve.py:179`, `run_headless` | `c182c88347f1eb3440a816986f1c2025a2ab2a13236cd68b0ad0146250d031ff` | Headless mode has no API server. It does not gain HTTP reachability merely by adding a worker method. |

In this plugin, `afd_plugin/compat/patches/engine_core.py`
(`dca55ad18ad65416d968c550605cc868e948ec6f1238f1a8a992904a4497d292`)
constructs the F executor in `_initialize_ffn_engine_core:582`. The HCCL path
in `_run_ffn_busy_loop:670` starts resident worker threads through
`collective_rpc("start_ffn_server_loop")`. It keeps native utility management
alive through `_process_input_queue:691`, then polls the existing F service.
The worker's blocking tensor receive lives in its background thread, while a
management RPC only updates the deferred session request/status under its
small lock. No new HCCL control message or receive interruption is needed.

Consequently, a real client belonging to an F EngineCore can reach that core's
F executor using its existing utility/collective path. An Attention client's
collective RPC targets its own executor; it cannot discover another independent
F process. This source proof does not establish that a particular deployed F
recipe currently exposes an API server or that a headless endpoint exists.

The native developer `/collective_rpc` router is behind `VLLM_SERVER_DEV_MODE`
and accepts an arbitrary method name. Enabling that broad endpoint solely for
profiling is unnecessary. A narrow plugin request/status router can use the
same engine-client mechanism with fixed worker method names.

## Implemented endpoint boundary

Use the native `EndpointPlugin` contract. The implementation owner is
`afd_plugin/compat/npu/profiler_api.py`, with a mirrored
`tests/unit/compat/npu/test_profiler_api.py`, a shared engine-client seam in
`tests/unit/fakes/profiler_api.py`, and one `pyproject.toml` entry under
`[project.entry-points."vllm.endpoint_plugins"]`:

```toml
afd_npu_profiler = "afd_plugin.compat.npu.profiler_api:AFDNPUProfilerEndpointPlugin"
```

`AFDNPUProfilerEndpointPlugin` has a zero-argument lightweight constructor,
`name="afd_npu_profiler"`, and `required_tasks=("generate",)`. It follows the
two exact native protocol method signatures above. Native protocol conformance
is an inheritance/type contract, not a new copied-function signature test.
The existing compatibility module validation paths own the mirrored test;
the module-design document must explicitly record the new HTTP boundary.

The deployment explicitly adds this name to the existing allowlist, for
example `VLLM_PLUGINS=afd,afd_npu_profiler`. General and endpoint plugins are
independent: allowing only the endpoint does not install AFD workers; allowing
only `afd` does not expose the endpoint. Existing other entries, including an
opt-in quantization fix when required by the actual recipe, must be retained.
An editable reinstall is needed to publish new entry-point metadata.

`attach_router` attaches unique `/afd/profiler` paths and per-app state only.
`init_state` consumes the passed actual client and `state.vllm_config`, parses
the existing AFD config, verifies its NPU role and corresponding explicit
role-env profiler enable setting, and rejects native/role-env conflict. This
validation belongs in `init_state`, because native factory exceptions are
logged and skipped; silently losing an explicitly requested management
endpoint would hide a configuration error. A missing engine is rejected
explicitly; the `generate` task requirement also excludes the render server.
Default applications that do not allowlist the endpoint retain native behavior.

The router exposes a fixed JSON contract:

- `GET /afd/profiler`: await the existing engine client's
  `collective_rpc("afd_profiler_status", timeout=<bounded timeout>)` and return
  each actual worker's status.
- `POST /afd/profiler`: strictly validate `is_start: bool` and optional
  `profile_prefix: str`; call only `collective_rpc("profile", kwargs=...)`.
  Its response is `{"accepted": true, "role": <actual role>}`. A successful
  reply means accepted desired state, with applied/pending status
  still determined by the execution owner. It does not promise a captured
  timeline or interrupt a currently executing A model/blocked F receive.

Use the existing app and engine client. A bounded RPC timeout is a named
constant (30 seconds). GET reports the actual worker status list, including
requested/applied/pending and each worker's `step_unit`; it must not turn a
pending request into a claim of trace completion. A second POST can request a
new desired state while a previous request is pending; the existing session
owns coalescing and application, rather than a new HTTP queue.

No new profiler backend import, profiler instance, device operation, mutable
global request registry or general RPC escape hatch is needed in the API
process. No native function patch or new alignment-table row is needed.
CPU HTTP tests should exercise actual routes with an explicit
engine-client RPC boundary: registration, strict bool/prefix validation,
explicit allowlisting/task eligibility, native/default behavior, per-worker
pending status, independent app bindings and RPC failure propagation.
These tests would establish HTTP/API behavior only.

This covers ordinary A/F API-server processes. A headless F service needs a
separate, explicitly owned management entry, such as a bounded local control
socket retained by the F process and forwarding fixed operations to its
already-owned executor. The existing native EngineCore utility socket is not
a documented attach-by-SSH client API; inventing an engine handle or rebuilding
the engine cannot safely manage the running one. A socket implementation must
freeze startup path, endpoint ownership, request ordering, error/STOP lifecycle
and external caller contract before development. It should not be smuggled
into the HCCL data protocol.

The A5 profiler guide documents the endpoint allowlist, editable metadata
refresh and requests against an already-running server, with its executor-local
scope. It also retains the already-existing in-process `LLM.collective_rpc`
API. No command is supplied for the unimplemented headless management entry.
Device acceptance still requires actual A/F traces and does not follow from
HTTP 200, requested/applied state or CPU test success.

The adapter adds one JSON request/response and one collective RPC per HTTP
operation, with a fixed timeout. It allocates no device resource, introduces
no per-request background task and stores one immutable client/role binding
per app. There is no source copy or alignment-table expansion. Remove this
adapter when native profiling routes can safely select the AFD execution-owned
role session and expose its actual status without requiring a second profiler.

## Directed CPU validation

`a3-profiler-cpu-20261110/profiler-api-025` passed both full selected modules:
HTTP/API 31 and existing profiler 31, **62 passed**, no failures, errors or
skips, JUnit 0.254 s. It checked 15 actual installed upstream source hashes;
the five additional contracts come from the locked native plugins loader,
EndpointPlugin interface, EngineClient protocol, app-state initialization and
DeviceConfig. No dependency was installed and no device probe was started.

The snapshot used HEAD `36f52994217f4caeb4ae20d5b4783892da4b555d` and manifest
`3fd88c1f933395e50d21d801e369db6377ef4255f38ceb2c0916b725918b5d26`.
Final runtime is `d75cbba9548607feedd65f0c739f6ede827dfdb34bf3b5a1090ee4004cbd5841`,
shared fake `a276848da915d0b584d0b4c28c8762d22544b53e86f3c7ca89bb4b5f55e6000d`,
mirror `0dc97c2253626d2f5f62aa646b94cbeb089ee6404ead6141a520f0106225c98a`.
The freeze file `profiler-api-final-snapshot-20261110.json` records local Ruff,
format and Python 3.11 grammar checks and proves the three already-frozen
profiler/Attention runner/worker production hashes were unchanged.

Actual requests pass through FastAPI/ASGI and strict Pydantic validation.
Their explicit EngineClient seam consumes the real CPU deferred Session API;
it does not emulate device execution or claim real worker RPC/server startup.
Native loader tests exercise the actual locked loader. Local FastAPI absence
caused a preliminary collection skip, not a passing HTTP validation; the
existing A3 dependencies supplied the behavior evidence.
