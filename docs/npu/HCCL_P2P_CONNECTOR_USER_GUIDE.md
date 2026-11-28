# Ascend HCCL P2P Connector

`P2pHcclAFDConnector` uses HCCL point-to-point communication between
Attention and FFN ranks. The model recipe documents runtime prerequisites, role launch
commands and a test request. [DeepSeek-V4.1-Flash A5 recipe](../../recipe/npu/P2pHcclAFDConnector/deepseek_v41/README.md)

## Role configuration

Use the `afd` object in `--additional-config` on both roles. Set `role`
to `attention` for the API process and `ffn` for the headless FFN process.
Set `connector` to `P2pHcclAFDConnector`. Both roles use the same HCCL
bootstrap `host` and `port`, rank counts and `compute_gate_on_attention`
setting. The V4.1 recipe sets `compute_gate_on_attention` to `false` so
native expert routing runs on FFN.

For the A4/F4 EP recipe, `num_attention_ranks` and `num_ffn_ranks` are
both four. The HCCL A/F group is separate from each role's native DP/EP
groups. HCCL carries the BF16 hidden states, FP32 router input and original
token IDs to FFN, then returns the MoE result. Attention owns the API,
sampling and native Engram and KV state.

## Graph, DBO and PD

- Use the same eager or Graph mode on both ends. The recipe configures
  `FULL_DECODE_ONLY` for Graph with eager execution where required.
- With DBO on the EP recipe, Attention and FFN process the two HCCL
  stages serially. Both roles use the same stage settings.
- Prefill/Decode transfers KV between the two Attention services using
  the AFD `MooncakeHybridConnector` adapter. Each of P and D has its
  own A/F HCCL group.
