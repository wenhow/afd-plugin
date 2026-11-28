# Window AFD Connector User Guide

## Runtime

`WindowAFDConnector` runs text generation for DeepSeek-V4.1-Flash with
asynchronous Attention ranks and connector-driven FFN ranks. Attention owns
HC, attention, KV/index state, the MoE gate and top-k routing. FFN consumes
ready Window records and returns the shared and routed expert outputs for
Attention to combine.

| Component | Revision |
| --- | --- |
| vLLM | v0.30.0, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `rfc/vllm_cann`, `7e37e29a7803d6897a7113088e2b771f83205bc0` |

Use a single CANN installation and a compatible torch-npu build. The runtime
must provide `cann_ops_transformer` with
`get_buffer_for_attention_to_ffn`, `get_buffer_for_ffn_to_attention`,
`attention_to_ffn`, `ffn_worker_batching` and `ffn_to_attention`. It also
needs the HCCL communicator-name API, the
`torch_npu._afd.create_schedule_context_holder` scheduler and the checkpoint's
MXFP grouped matrix multiply and SwiGLU kernels.

## Deployment

Use the [DeepSeek-V4.1-Flash Window recipe](../../recipe/npu/WindowAFDConnector/deepseek_v41/README.md).
It starts in eager/U1 with MXFP transport. Role configurations must share
the rendezvous, A/F rank counts, transport settings and token capacity.
Configure `async=true` and `compute_gate_on_attention=true`.

Window's HCCL world orders FFN ranks before Attention ranks. FFN rank 0
holds the one shared expert for all decoder layers. The remaining FFN ranks
own contiguous, balanced ranges of routed experts. Each must own at least
one routed expert. For `E` routed experts, `L` decoder layers and
`N` FFN ranks, keep `L * ceil(E / (N - 1)) <= 1024` so each rank's
packed global weights stay within the grouped-matmul limit. A batching
transaction can carry up to 8192 ready expert groups.

| Area | Configuration |
| --- | --- |
| Parallelism | TP1, static EP, PP/PCP/DCP each 1. MoE sequence parallelism, EPLB and mixed placement are disabled. |
| Routing | Exactly one shared expert and `num_experts_per_tok <= 16`. |
| Weights | W4A8MXFP routed experts and W8A8MXFP8 shared experts across all decoder layers. |
| Transport | The recipe uses `quant_mode=4` and allows up to 512 batched tokens per role. |
| U1 | One Window slot with DBO disabled. |
| U2 | Two Window slots. Attention uses native DBO with two ubatches, text-only mode and Engram disabled. FFN keeps native model-forward ubatching disabled. |
| Graph | `FULL_DECODE_ONLY`. |
| DSpark | Attention-only native draft EP. Set the checkpoint's draft block size and ordered target layer IDs. Graph verification uses fixed width. |

Async DP allows FFN to combine ready records from separate Attention sessions
and layers. Each request still waits for its matching FFN output before the
dependent Attention layer advances. The default Window FFN multistream setting
places F2A on a separate stream when the two-slot pipeline is active.
Its local pipeline slots are separate from the two transport slots.

Use the recipe's startup checks before sending traffic to Attention. A
missing symbol points to the runtime build. A stalled rendezvous requires
checking all rank starts, network reachability and port ownership.
A failure during expert setup requires inspecting the checkpoint types and
the largest rank's layer/expert group count. Stop the complete Window world
before restarting.
