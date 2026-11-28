# DeepSeek-V4.1-Flash Window AFD Recipe

This recipe starts asynchronous Window Attention and FFN services using
vLLM ModelRunner V2 on Ascend. It uses TP1 and static EP. The default settings
select eager/U1, MXFP transport, Attention-side gating and DSpark off.
See the [Window connector guide](../../../../docs/npu/WINDOW_AFD_CONNECTOR_USER_GUIDE.md)
for rank placement, checkpoint constraints and required CANN operators.

## Runtime baseline

| Component | Revision |
| --- | --- |
| vLLM | v0.30.0, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `rfc/vllm_cann`, `7e37e29a7803d6897a7113088e2b771f83205bc0` |
| AFD Plugin | Record the checkout with `git rev-parse HEAD` |
| CANN and torch-npu | Use a compatible Window build with the operators and scheduling API in the connector guide |

Install the same runtime and mount the same checkpoint at `MODEL_PATH` on
each participating node. Use the image's Transformer vendor path or set
`TRANSFORMER_VENDOR` to its installed vendor directory.

## Configure ranks

Use one local NPU per role rank. Set `ATTENTION_RANKS` and `FFN_RANKS`
to the complete role sizes on all nodes. FFN rank 0 owns the shared expert.
For a model with `L` decoder layers and `E` routed experts, choose
`FFN_RANKS` so that `L * ceil(E / (FFN_RANKS - 1)) <= 1024`. The largest
routed rank determines this limit. Choose a device layout that also meets the
per-role memory requirements.

Set these variables on each node using its own network interface and IP.
This example configures two Attention ranks and a 16-rank FFN role running
on two eight-device hosts. Adapt the counts to your checkpoint and hosts.

```bash
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export NIC_NAME=eth0
export HCCL_IF_IP=192.0.2.10
export AFD_HOST=192.0.2.20
export ATTENTION_RANKS=2
export FFN_RANKS=16
export ATTENTION_DP_ADDRESS=192.0.2.10
export FFN_DP_ADDRESS=192.0.2.20
```

Set `ATTENTION_DEVICES=0,1` on the Attention host. On each FFN host set
`FFN_DEVICES=0,1,2,3,4,5,6,7`. Set `DP_START_RANK=0` on the first FFN host
and `DP_START_RANK=8` on the second. Role rank ranges are contiguous and
must cover every global rank exactly once. The launcher uses vLLM's headless
DP startup for the secondary node.

| Variable | Default |
| --- | --- |
| `AFD_PORT` | `29761` |
| `ATTENTION_API_HOST` and `ATTENTION_API_PORT` | `127.0.0.1:8910` |
| `FFN_API_PORT` | `8911`, bound to loopback on the FFN head |
| `ATTENTION_DP_RPC_PORT` and `FFN_DP_RPC_PORT` | `29550` and `29560` |
| `ATTENTION_MASTER_PORT` and `FFN_MASTER_PORT` | `29501` and `29511` |
| `MAX_NUM_BATCHED_TOKENS` and `MAX_NUM_SEQS` | `128` and `2` |
| `MAX_MODEL_LEN` and `GPU_MEMORY_UTILIZATION` | `4096` and `0.90` |

The token capacity, rank counts, connector address, mode and slot count must
agree across all ranks. Run the respective role script with `--dry-run` to
inspect its command.

## Start and stop

Start `bash afd_ffn.sh` on both FFN hosts and `bash afd_attention.sh` on
the Attention host back-to-back so the rendezvous can see all ranks. Verify a
`[Window][init]` entry for every rank and send requests to the Attention API.
For a remote client, set `ATTENTION_API_HOST` to a reachable bind address.

```bash
curl -fsS http://127.0.0.1:8910/v1/models
curl -fsS http://127.0.0.1:8910/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/DeepSeek-V4.1-Flash","prompt":"What is the capital of France?","max_tokens":16,"temperature":0}'
```

Replace the model ID with the ID returned by `/v1/models`. Stop Attention
first, then stop the FFN services. Restart all ranks together.

## Execution options

- Set `EXECUTION_MODE=graph` on both roles for `FULL_DECODE_ONLY`.
- Set `U_BATCHES=2 DISABLE_ENGRAM=1` on both roles for U2. This disables Engram
  in the model config and changes the checkpoint's execution. Attention uses
  native DBO with two ubatches. FFN uses the same transport slots with its
  connector-driven execution.
  Both roles need at least two sequence slots and two batched tokens.
- Set `DSPARK_TOKENS` to the checkpoint's `dspark_block_size` on both roles
  to select the common Graph target width. The Attention service constructs
  an eager DSpark draft with fixed verification width.
  Graph/U2 needs capacity for two complete verification sequences. Set
  `MAX_NUM_BATCHED_TOKENS` to at least `2 * (DSPARK_TOKENS + 1)`.
