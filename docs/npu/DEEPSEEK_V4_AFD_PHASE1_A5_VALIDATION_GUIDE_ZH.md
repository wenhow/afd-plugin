# DeepSeek-V4 AFD 第一期 A5 dSpark 与 PD 验证指导书

## 1. 范围和结论口径

A5 单机可以验证 dSpark。本指导书先在一台 8 卡 A5 上验证 dSpark 本身和 AFD 组合，再在两台 A5 上验证 PD 数据面集成：

| 阶段 | 验证点 | 机器 | 目的 |
|---|---|---|---|
| S1 | no-AFD DP4 + dSpark eager | 单 A5，NPU 0-3 | 排除 AFD，确认 dSpark 权重、proposer 和指标有效 |
| S2 | A4F4 + dSpark eager/U1 | 单 A5，A=0-3、F=4-7 | 确认 dSpark 与 AF 分离基础组合 |
| S3 | A4F4 + dSpark Graph/U2 | 单 A5，A=0-3、F=4-7 | 确认 dSpark、Graph、microbatch 和多流组合 |
| D1 | PD + A4F4 Graph/U2 | 双 A5 | 确认 Mooncake KV 与 AFD Graph/U2 数据面 |
| D2 | PD + A4F4 + dSpark Graph/U2 | 双 A5 | 确认一期最大功能组合 |

单 A5 已完成的 no-AFD DP4、A4F4 eager/U1、A4F4 Graph/U2 和 A2F4 Graph/U2 MTP-off 结果不重跑。A4F2 保持外部 HCCL 阻塞 `A5-HCCL-RS-001`，不改记为通过。

本阶段只做功能门禁，不生成 golden、不逐 token 比对、不做精度、性能或 dSpark 加速比结论。最终精度测试在上述功能组合通过后单独执行。

## 2. 固定环境和前置条件

| 项目 | 固定值 |
|---|---|
| vLLM | `releases/v0.23.0`，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`，`3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | `feat/dsv4-afd-graph-u2-multistream-all-on-v1` 的当前交付提交 |
| CANN | 只指定本机实际绝对 `CANN_ROOT`，不强校验版本字符串 |
| Python | 使用已安装的 `/root/dsv4-afd-hccl/venv`，不重新安装环境 |

每台机器执行：

```bash
export AFD_PLUGIN_ROOT="/root/dsv4-afd-hccl/src/afd-plugin-phase1-a5-dspark-66ec72f"
export VENV_ROOT="/root/dsv4-afd-hccl/venv"

git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD
git -C "$AFD_PLUGIN_ROOT" status --short
git -C /root/dsv4-afd-hccl/src/vllm-release-v0.23.0 rev-parse HEAD
git -C /root/dsv4-afd-hccl/src/vllm-release-v0.23.0 status --short
git -C /root/dsv4-afd-hccl/src/vllm-ascend-rfc-vllm-cann rev-parse HEAD
git -C /root/dsv4-afd-hccl/src/vllm-ascend-rfc-vllm-cann status --short
npu-smi info
```

三个工作树必须干净，两个上游提交必须与表中一致，NPU 健康且没有残留模型进程。不要屏蔽 dirty 检查。

dSpark 使用独立权重，普通 `DeepSeek-V4-Flash` 不能通过启动参数变成 dSpark：

```bash
export MODEL_PATH="/home/models/DeepSeek-V4-Flash-DSpark"
export MODEL_TOOL="$AFD_PLUGIN_ROOT/tools/dsv4/hccl_manual_install/bin/model_launch_args.py"

"$VENV_ROOT/bin/python" "$MODEL_TOOL" \
  --model-path "$MODEL_PATH" --describe
```

输出必须包含正整数 `dspark_block_size` 和非空 `dspark_target_layer_ids`。脚本从权重读取准确 block size，不按旧 MTP 的 N1/N2/N3 手工填写。固定 v0.23 栈内部可能显示 `method=mtp`，实际 proposer 由上述 dSpark 字段选择，最终以 drafter 标记和 speculative metrics 为准。

## 3. 单 A5 dSpark 验证

以下命令在一台 8 卡 A5 上执行。每次使用新的证据目录。

### 3.1 no-AFD DP4 dSpark 基线

```bash
cd "$AFD_PLUGIN_ROOT"
export MODEL_PATH="/home/models/DeepSeek-V4-Flash-DSpark"
export A5_DSPARK_ROOT="/data/validation/dsv4-phase1-a5-dspark-$(date +%Y%m%d_%H%M%S)"
export PHASE1_NATIVE_OUTPUT_ROOT="$A5_DSPARK_ROOT/native-dp4-dspark"
mkdir -p "$A5_DSPARK_ROOT"

bash tools/dsv4/run_phase1_a5_native_smoke.sh run-dspark \
  2>&1 | tee "$A5_DSPARK_ROOT/native-dp4-dspark.console.log"
```

该项固定使用 NPU 0-3、DP4/TP1、eager target 和 eager draft，且不加载 AFD 插件。通过时以下文件均存在：

```text
native-dp4-dspark/functional_smoke.json       passed=true
native-dp4-dspark/dspark_gate.json            passed=true，draft/accepted token 均 > 0
native-dp4-dspark/summary.env                  passed=1，forced_stop=0，npu_cleanup_passed=1
```

### 3.2 AFD A4F4 dSpark 两点

```bash
export PHASE1_OUTPUT_BASE="$A5_DSPARK_ROOT/afd"

bash tools/dsv4/run_phase1_a5_matrix.sh list-dspark
bash tools/dsv4/run_phase1_a5_matrix.sh dspark \
  a4f4_dspark_eager_u1 \
  a4f4_dspark_graph_u2 \
  2>&1 | tee "$A5_DSPARK_ROOT/afd-dspark.console.log"
```

每个点自动执行两个独立冷启动 cycle，并运行 batch 1/8/32、取消恢复、日志、U2、dSpark 和 NPU 清理门禁。Graph 点固定为 `FULL_DECODE_ONLY/U2`、async off、draft Graph 和五个多流开关全开。

检查结果：

```bash
find "$A5_DSPARK_ROOT/afd/dspark" -name validation_summary.json -print \
  -exec jq '{passed,execution_mode,u_batches,enable_dspark,dspark_num_speculative_tokens,dspark_draft_execution}' {} \;
find "$A5_DSPARK_ROOT/afd/dspark" -name dspark_gate.json -print \
  -exec jq '{passed,draft_tokens,accepted_tokens,drafter_markers,expected_attention_ranks}' {} \;
```

两份 `validation_summary.json` 必须 `passed=true`；四个 cycle 的 `dspark_gate.json` 必须 `passed=true`、draft/accepted token 均大于 0，且 `drafter_markers >= expected_attention_ranks=4`。Graph/U2 cycle 的 `ubatch_gate.passed` 和 `observed_two_stages` 还必须为 `true`。

S1、S2、S3 全部通过后，才能进入双机 PD。任何一点失败都停止，不用 PD 结果覆盖单机失败。

## 4. 双 A5 PD 集成配置

PD 不能在一台 8 卡 A5 上完整验证：P 节点使用 8 卡 Prefill DP2/TP4，D 节点使用 8 卡 A4F4 Decode。网络路径为：

```text
client -> Proxy(P) -> Prefill(P8) --Mooncake KV--> Attention(D0-3)
                                             Attention --HCCL/U2--> FFN(D4-7)
```

两台机器分别创建现场配置；除本机 CANN 路径和网卡名可不同外，P/D IP、模型路径、运行根目录和代码提交必须一致：

```bash
export PD_DIR="$AFD_PLUGIN_ROOT/tools/dsv4/mooncake_pd_manual"
export SITE="/root/dsv4-afd-hccl/a5-pd/site-r1.env"
export CFG="/root/dsv4-afd-hccl/a5-pd/config-r1"

mkdir -p "$(dirname "$SITE")"
cp "$PD_DIR/a5_site.env.example" "$SITE"
vi "$SITE"
```

至少填写：

```text
CANN_ROOT=<本机实际绝对路径>
PREFILL_IP=<P节点业务/HCCL地址>
DECODE_IP=<D节点业务/HCCL地址>
NIC_NAME=<上述IP所在网卡>
FLASH_MODEL_PATH=/home/models/DeepSeek-V4-Flash
DSPARK_MODEL_PATH=/home/models/DeepSeek-V4-Flash-DSpark
A5_PD_RUN_BASE=/data/validation/dsv4-phase1-a5-pd-r1-<时间戳>
```

`CANN_VERSION=""` 保持为空。生成两个点、共 6 份角色配置：

```bash
bash "$PD_DIR/init_a5_pd_validation.sh" list
bash "$PD_DIR/init_a5_pd_validation.sh" init "$CFG" "$SITE"

for config in "$CFG"/*.env; do
  bash "$PD_DIR/pd.sh" print-config "$config" \
    >"${config%.env}.effective.txt"
done
sha256sum "$CFG"/*.env
```

两台机器的 6 个 env 文件 SHA256 必须相同。生成器拒绝覆盖已有目录；第二轮必须使用新的 `SITE`、`CFG` 和 `A5_PD_RUN_BASE`。

## 5. 双机预检

`install` 只审计既有 Python/CANN/vLLM/Mooncake，并以 editable 方式安装当前 afd-plugin，不重装整套环境。

P 节点：

```bash
export PD="$PD_DIR/pd.sh"
bash "$PD" install "$CFG/pd_afd_graph_u2-prefill.env"
bash "$PD" check "$CFG/pd_afd_graph_u2-prefill.env"
bash "$PD" check "$CFG/pd_afd_dspark_graph_u2-prefill.env"
bash "$PD" check "$CFG/pd_afd_graph_u2-proxy.env"
```

D 节点：

```bash
export PD="$PD_DIR/pd.sh"
bash "$PD" install "$CFG/pd_afd_graph_u2-decode.env"
bash "$PD" check "$CFG/pd_afd_graph_u2-decode.env"
bash "$PD" check "$CFG/pd_afd_dspark_graph_u2-decode.env"
```

预检不依赖 `ss`，也不校验 A5 的 CANN 版本字符串。任一检查失败都停止。

## 6. 执行双机验证点

第一轮按顺序执行 `pd_afd_graph_u2`、`pd_afd_dspark_graph_u2`。每个点都使用下面的完整流程。

P 节点先启动 Prefill：

```bash
export POINT="pd_afd_graph_u2"
bash "$PD" start "$CFG/$POINT-prefill.env"
bash "$PD" status "$CFG/$POINT-prefill.env"
```

D 节点启动共置 A4F4 Decode：

```bash
export POINT="pd_afd_graph_u2"
bash "$PD" start "$CFG/$POINT-decode.env"
bash "$PD" status "$CFG/$POINT-decode.env"
```

P 节点最后启动 Proxy 并发请求：

```bash
export POINT="pd_afd_graph_u2"
bash "$PD" start "$CFG/$POINT-proxy.env"
bash "$PD" status "$CFG/$POINT-proxy.env"
bash "$PD" smoke "$CFG/$POINT-proxy.env"
```

D 节点在服务仍运行时验证真实数据面并收集：

```bash
bash "$PD" verify-data-path "$CFG/$POINT-decode.env"
bash "$PD" collect "$CFG/$POINT-decode.env"
```

P 节点收集：

```bash
bash "$PD" collect "$CFG/$POINT-prefill.env"
bash "$PD" collect "$CFG/$POINT-proxy.env"
```

按 Proxy、Decode、Prefill 顺序停服：

```bash
# P 节点
bash "$PD" stop "$CFG/$POINT-proxy.env"

# D 节点
bash "$PD" stop "$CFG/$POINT-decode.env"
npu-smi info

# P 节点
bash "$PD" stop "$CFG/$POINT-prefill.env"
npu-smi info
```

两台 NPU 进程表为空后再执行下一个点。第一轮两点均通过后，以新的 `site-r2.env`、`config-r2` 和运行根目录重复生成配置，再完整执行 D1、D2 一次；不得复用第一轮进程或日志。

## 7. 通过标准

一期 A5 剩余功能目标关闭必须同时满足：

1. 单机 S1、S2、S3 通过；AFD 两点各有两个成功冷启动 cycle。
2. 双机 D1、D2 各连续完成两轮独立冷启动。
3. PD 点的 Prefill、Decode Attention、Decode FFN、Proxy 全部 ready，batch 1/8/32、取消和恢复请求成功。
4. `verify-data-path` 返回 0；Mooncake KV transfer 和在线 U2 marker 均大于 0。
5. dSpark 点的 4 个 Attention rank 均加载完整 drafter，使用权重的准确 block size，draft/accepted token 均大于 0。
6. 所有日志无 fatal marker，正常停服后 NPU 进程表为空。
7. 证据明确 `golden_checked=0`；不得据此声明精度通过。

## 8. 失败回传

单机失败时回传整个 `$A5_DSPARK_ROOT`；双机失败时，在服务仍存活时尽量执行三个角色的 `collect`，再正常停服，回传：

```text
P节点 prefill collect tar.gz 及 sha256
P节点 proxy collect tar.gz 及 sha256
D节点 decode collect tar.gz 及 sha256
<POINT>-decode-data-path 目录
三个角色的完整终端输出
两台机器停服后的 npu-smi info
```

不要 reset 工作树、屏蔽 dirty 检查、修改权重配置或降低门禁。最终精度测试在本文功能目标关闭后另行执行。
