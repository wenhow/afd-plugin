# DeepSeek-V4 Mooncake PD + AFD M9

For a manual two-node deployment, use the unified role manager instead of
re-entering all launcher environment variables:

```bash
bash tools/dsv4/mooncake_pd_manual/pd.sh init /mnt/workspace/pd-prefill.env
# Edit the generated role/IP/path/commit values, then:
bash tools/dsv4/mooncake_pd_manual/pd.sh install /mnt/workspace/pd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check /mnt/workspace/pd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh start /mnt/workspace/pd-prefill.env
```

Create equivalent `decode` and `proxy` configs. Start in prefill, decode,
proxy order; validate through the proxy; stop in proxy, decode, prefill order.
Run `collect` for all three roles to generate support archives capped at 2 MiB
each. See `tools/dsv4/mooncake_pd_manual/README_ZH.md` for exact commands.

The first M9 boundary composes two independent data paths:

- Prefill to Decode Attention: `MooncakeHybridConnector` KV transfer;
- Decode Attention to Decode FFN: `P2pHcclAFDConnector` hidden-state transfer.

Decode FFN never receives `--kv-transfer-config`. The initial functional gate
is TP1, eager/U1, MTP off. Graph, U2, MTP, and TP2 remain fail-fast until this
baseline passes the real-model token and lifecycle gates.

On the paired A3 host, install the same frozen runtime dependencies and local
Ascend wheel before running the launchers:

```bash
sudo apt install -y libgoogle-glog0v6t64 libjsoncpp25 libjemalloc2
/mnt/workspace/code/.venvs/afd-v023-vllm-cann/bin/python -m pip install \
  --no-deps --force-reinstall \
  /mnt/workspace/validation/dsv4_afd_v023_mooncake_pd_m9_contract_20260821_181148/mooncake_transfer_engine-0.3.9-cp312-cp312-manylinux_2_39_aarch64.whl
```

The wheel SHA256 is
`0f9964801b24fd683d6016e1196cc0606fc87b0285b45d89c433650b9477ca12`.
It is an A3 functional artifact; rebuild and revalidate Mooncake for A5 instead
of reusing this binary there.

Start the services in this order:

```bash
# A5 Prefill node, NPU 0-7, DP8/TP1 by default.
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/prefill.sh

# Decode node: FFN first, then Attention with Mooncake enabled.
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh
ENABLE_PD=1 bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh

# Either node, after both HTTP backends are healthy.
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/proxy.sh
```

Set `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`, API hosts/ports,
and the prefill/decode host lists to the actual two-node topology. The launchers
source `tools/dsv4/check_mooncake_runtime.sh` before model loading. This exports
the jemalloc preload required by the combined torch-npu/Mooncake process and
rejects missing Mooncake libraries or any CANN 9.1.0 dependency.

Before loading the model, validate a real two-process Ascend transfer on two
idle NPUs:

```bash
source tools/dsv4/check_mooncake_runtime.sh
python tools/dsv4/check_mooncake_npu_roundtrip.py \
  --producer-device 0 --consumer-device 1
```
