# DeepSeek-V4.1-Flash A5 HCCL P2P Recipe

Run a DeepSeek-V4.1-Flash Attention/FFN service with vLLM ModelRunner V2 and
`P2pHcclAFDConnector`. The default uses one eight-device Atlas A5 host with
A4/F4 placement, TP1 / DP4 / EP4 per role, eager/U1 and DSpark disabled.
Attention serves the OpenAI-compatible API and owns Engram and KV state.
FFN uses the headless AFD launcher and owns the MoE computation.

## Runtime baseline

| Component | Baseline |
| --- | --- |
| Hardware | One eight-device Atlas A5 host per A/F instance |
| Model | Original DeepSeek-V4.1-Flash checkpoint, native FP8 load path |
| vLLM | `v0.30.0` at `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `rfc/vllm_cann` at `7e37e29a7803d6897a7113088e2b771f83205bc0` |
| CANN/HCCL | A5 runtime with the matching transformer custom operator package |
| afd-plugin | This delivery checkout. Record its SHA with `git rev-parse HEAD`. |

Use the Python interpreter and Ascend environment in which the pinned packages
and this AFD plugin are installed. The transformer vendor directory must
contain `op_api/lib/libcust_opapi.so`. With native CPU-offloaded Engram shared
across DP ranks, plan for approximately 190 GiB of available host RAM and
sufficient container shared memory. The default limits each rank to two
sequences, a 1,024-token model length and 128 batched tokens.

## Standalone A/F

Set the same configuration in the Attention and FFN launch shells.

```bash
export PYTHON_BIN=python3
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export TRANSFORMER_VENDOR=/path/to/installed/transformer-vendor
export ATTENTION_DEVICES=0,1,2,3
export FFN_DEVICES=4,5,6,7
export AFD_HOST=127.0.0.1
export ATTENTION_HOST=127.0.0.1
export DSV41_MODE=eager-u1
export DSV41_SPEC_TOKENS=0
```

Use the interface settings of the A5 runtime. To select a specific interface,
export `HCCL_SOCKET_IFNAME` and `HCCL_IF_IP` in both shells. The default
Attention HTTP port is 18031 and the HCCL bootstrap port is 29600.
`DSV41_PORT_OFFSET` increments the configured HTTP, A/F, native rendezvous
and KV handshake ports together. The same offset must be used by both roles.

From the repository root, inspect the arguments with `--dry-run`. This uses
only Bash and the Python standard library. Start FFN in one shell and
Attention in another. FFN can wait at the startup handshake for Attention.

```bash
recipe=recipe/npu/P2pHcclAFDConnector/deepseek_v41
bash "$recipe/afd_ffn.sh" --dry-run
bash "$recipe/afd_attention.sh" --dry-run
```

```bash
bash recipe/npu/P2pHcclAFDConnector/deepseek_v41/afd_ffn.sh
```

```bash
bash recipe/npu/P2pHcclAFDConnector/deepseek_v41/afd_attention.sh
```

```bash
curl -fsS http://127.0.0.1:18031/health
curl -fsS http://127.0.0.1:18031/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-afd","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

Check for HTTP 200 and a completion. To inspect Graph and U2, export
`AFD_HCCL_STAGE_DIAGNOSTICS=1` in both launch shells. Look for Graph replay
and two-stage transactions in the Attention and FFN logs for eligible batches. When DSpark is enabled, check proposal,
acceptance and rejection over several requests.

For a clean stop, set `ATTENTION_PID` to the Attention launcher PID printed at
startup and send `SIGUSR2`. The vLLM v0.30 wrapper calls the native multi-API
shutdown callback.
Wait for Attention and FFN to exit and for FFN to log peer STOP and successful
engine cleanup.

```bash
kill -USR2 "$ATTENTION_PID"
```

## Execution modes

Set `DSV41_MODE` and `DSV41_SPEC_TOKENS` identically in both A/F shells.

| `DSV41_MODE` | Execution |
| --- | --- |
| `eager-u1` | Eager with a single microbatch |
| `graph-u1` | `FULL_DECODE_ONLY` with a single microbatch |
| `eager-u2` | DBO with two serial HCCL stages |
| `graph-u2` | `FULL_DECODE_ONLY` with two serial HCCL stages when the batch admits U2 |

`DSV41_SPEC_TOKENS` accepts `0` for disabled or a fixed `1`, `2` or `5`.
DSpark uses eager draft execution and leaves the draft model on Attention.
The EP recipe uses serial U2 with stream overlap disabled.
The A5 functional evidence for the development snapshot covers the four
standalone modes at `N=0`. Verify DSpark and PD combinations on the target
image and this delivery commit before relying on them. Device utilization,
full-model precision and performance require separate measurements.

## Prefill/Decode separation

Use two A5 hosts with one A4/F4 instance per host and the same checkpoint.
The P instance uses `eager-u1`. The D instance can select any documented
execution mode. Start with DSpark disabled on both instances. Replace the
documentation addresses below with each host's local address reachable
from its peer and set it as `KV_IP` and allow the Mooncake
handshake ports and the transfer engine endpoint returned in that handshake.

In both P shells, keep the standalone environment and set these values. Run
FFN and Attention in separate shells.

```bash
export DSV41_PD_ROLE=producer DSV41_MODE=eager-u1 DSV41_SPEC_TOKENS=0
export DSV41_PORT_OFFSET=0
export ATTENTION_HOST=0.0.0.0
export KV_IP=192.0.2.10
bash recipe/npu/P2pHcclAFDConnector/deepseek_v41/afd_ffn.sh
# In the other P shell with the same environment:
bash recipe/npu/P2pHcclAFDConnector/deepseek_v41/afd_prefill.sh
```

On D, choose a distinct engine ID if running more than one instance with the
same role. Start FFN and Attention with the same D environment.

```bash
export DSV41_PD_ROLE=consumer DSV41_MODE=eager-u1 DSV41_SPEC_TOKENS=0
export DSV41_PORT_OFFSET=1000
export ATTENTION_HOST=0.0.0.0
export KV_IP=192.0.2.20
bash recipe/npu/P2pHcclAFDConnector/deepseek_v41/afd_ffn.sh
# In the other D shell with the same environment:
bash recipe/npu/P2pHcclAFDConnector/deepseek_v41/afd_attention.sh
```

When both Attention APIs are healthy, start the repository PD proxy. The
default P/D HTTP ports in this example are 18031 and 19031. The proxy listens
at `127.0.0.1:18030` and forwards the producer's KV transfer metadata to D.

```bash
export PREFILL_HOST=192.0.2.10 DECODE_HOST=192.0.2.20
bash recipe/npu/P2pHcclAFDConnector/deepseek_v41/afd_proxy.sh
curl -fsS http://127.0.0.1:18030/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-afd","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```
