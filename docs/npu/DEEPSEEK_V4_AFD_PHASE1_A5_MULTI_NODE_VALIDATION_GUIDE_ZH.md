# DeepSeek-V4 AFD 第一期 A5 与双机验证指导书

## 1. 目标与固定口径

本文只关闭第一阶段功能门禁，不用于发布 U3 或性能结论。固定栈如下：

| 组件 | 固定值 |
|---|---|
| CANN | `9.0.0`，整个 shell 只能加载一个绝对 toolkit root |
| vLLM | `releases/v0.23.0`，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`，`3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | 补丁包 `manifest/versions.env` 中的 `AFD_TARGET_COMMIT`/`AFD_TARGET_TREE` |
| 模型 | 同一份 DeepSeek-V4-Flash W8A8，`num_nextn_predict_layers=1` |
| 并行 | TP1；A/F 只验收 `A=kF`、`A=F`、`F=kA` 的整数比例 |
| MTP | `num_speculative_tokens=1/2/3`，最大值 3 |

第一阶段仍需完成的外部硬门禁：

1. A5 平台审计、独立安装和同栈 native golden。
2. A5 standalone A8F8、A4F8、A8F4 的 eager/Graph、U1/U2、MTP N=1/2/3 代表矩阵。
3. A8F4 在高 HBM A5 上的真实模型加载和端到端请求。
4. 双机 PD Graph/U2 下的 A8F8 N2/N3、A4F8 N3、A8F4 N3。
5. 每个 PD 点的真实双 stage、FFN Graph 动态路由、取消后恢复、优雅退出、NPU 清理和第二次冷启动。
6. 路径匹配的 PD no-AFD control，以及每次冷启动 10 条 prompt x 3 轮的 30/30 token exact。

不属于第一阶段：U3、正式 P2 性能收益、A5 参数调优、非整数 A/F、TP/SP/CP/DCP/PP、TP3、非等量 TP2 和 TP2 full-draft Graph/U2/MTP 最大组合。

## 2. 补丁包安装

在每台机器使用同一个 tar 包和 SHA256 文件：

```bash
sha256sum -c dsv4-afd-hccl-manual-install-slim-*.tar.gz.sha256
tar -xzf dsv4-afd-hccl-manual-install-slim-*.tar.gz
cd dsv4-afd-hccl-manual-install-slim-*
sha256sum -c manifest/SHA256SUMS
vi config.env
bash bin/00_print_config.sh
bash bin/install_all.sh
```

`config.env` 至少修改 `CANN_ROOT`、`MODEL_PATH`、`SOC_VERSION`、`NIC_NAME`、`HCCL_IF_IP`、安装目录和镜像地址。`CANN_ROOT` 必须指向 CANN 9.0.0 的真实根目录；不得先 source 其他版本再继续。

安装完成后保存包目录和源码目录：

```bash
export BUNDLE_ROOT="$PWD"
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
export AFD_PLUGIN_ROOT
```

轻量包会从分支基线提交应用二进制 patch，核对目标 tree，并生成内容确定的本地交付 commit。各节点的交付 commit 必须相同，且工作树必须干净。

## 3. H0 平台审计

每台 NPU 节点都执行，任何一项不一致就停止：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
bash "$BUNDLE_ROOT/bin/06_verify_install.sh"
git -C "$VLLM_ROOT" rev-parse HEAD
git -C "$VLLM_ASCEND_ROOT" rev-parse HEAD
git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD
git -C "$VLLM_ROOT" status --short
git -C "$VLLM_ASCEND_ROOT" status --short
git -C "$AFD_PLUGIN_ROOT" status --short
readlink -f "$CANN_ROOT"
npu-smi info
```

通过条件：两个上游 commit 与第 1 节完全一致；三个 `status --short` 均为空；CANN 只有 9.0.0；NPU 健康；开始验证前没有其他 NPU 进程。把输出保存为文本，后续随证据包回传。

## 4. A5 同栈 native golden

只在 A5 生成一次 native no-AFD golden。以下服务使用 8 卡，示例端口需保持空闲：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
export MODEL_PATH HCCL_IF_IP GLOO_SOCKET_IFNAME HCCL_SOCKET_IFNAME
export NATIVE_ROOT="/data/validation/dsv4-phase1-a5-native"
mkdir -p "$NATIVE_ROOT"

setsid env \
  API_HOST=127.0.0.1 API_PORT=8900 \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ENABLE_MTP=0 TENSOR_PARALLEL_SIZE=1 \
  bash tools/dsv4/run_v023_native_baseline.sh \
  >"$NATIVE_ROOT/server.log" 2>&1 &
export NATIVE_PID=$!
printf '%s\n' "$NATIVE_PID" >"$NATIVE_ROOT/server.pid"
```

等待 `curl -fsS http://127.0.0.1:8900/health` 成功，然后生成 3 轮稳定 golden：

```bash
python tools/dsv4/generate_golden.py \
  --endpoint http://127.0.0.1:8900/v1/completions \
  --model dsv4-v023-native \
  --prompt-source tools/dsv4/phase1_prompts.json \
  --output "$NATIVE_ROOT/golden_results.json" \
  --rounds 3 \
  --metadata baseline_kind=native_no_afd \
  --metadata cann_version=9.0.0 \
  --metadata vllm_commit=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 \
  --metadata vllm_ascend_commit=3da28f9414583d2d0b672a8f06d1fae142404bda

kill -TERM -- "-$NATIVE_PID"
wait "$NATIVE_PID" || true
npu-smi info >"$NATIVE_ROOT/npu-after-stop.txt"
jq '{passed,rounds,prompt_count,mismatched_prompt_indices}' \
  "$NATIVE_ROOT/golden_results.json"
```

必须得到 `passed=true`、`rounds=3`、`prompt_count=10`、空 mismatch，并确认停止后没有 NPU 进程。后续所有 A5 standalone 与 PD control 都使用这一个文件作为 prompt/token 参考源。

## 5. A5 standalone 门禁

矩阵脚本固定了 9 个代表点：

| 点 | 目的 |
|---|---|
| `a8f8_eager_u1_mtp_off` | A5 基础回归 |
| `a8f8_eager_u1_n1` | N1 兼容性 |
| `a8f8_eager_u2_n2` | eager U2、多 token |
| `a8f8_graph_u1_n2` | target Graph、draft eager |
| `a8f8_graph_u2_n3` | A8F8 最大一期组合、full draft Graph |
| `a4f8_eager_u1_n2` | `F=2A` fan-out 基础路径 |
| `a4f8_graph_u2_n3` | `F=2A` 最大一期组合 |
| `a8f4_eager_u1_n2` | `A=2F` 高 HBM 基础路径 |
| `a8f4_graph_u2_n3` | `A=2F` 最大一期组合 |

先执行 F0。F0 每点一次冷启动、1 轮、batch 1/8/32，不等待 30 分钟 idle：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
export MODEL_PATH
export PHASE1_GOLDEN="$NATIVE_ROOT/golden_results.json"
export PHASE1_OUTPUT_BASE="/data/validation/dsv4-phase1-a5-$(date +%Y%m%d_%H%M%S)"
bash tools/dsv4/run_phase1_a5_matrix.sh list
bash tools/dsv4/run_phase1_a5_matrix.sh preflight
bash tools/dsv4/run_phase1_a5_matrix.sh f0
```

F0 全部通过后执行 F1。F1 每点两次冷启动、10 条 prompt x 3 轮、batch 1/8/32，并在第一轮加入 1800 秒 idle-resume：

```bash
bash tools/dsv4/run_phase1_a5_matrix.sh f1
```

定位失败时可只重跑一个点，但必须使用新的 `PHASE1_OUTPUT_BASE`：

```bash
export PHASE1_OUTPUT_BASE="/data/validation/dsv4-phase1-a5-retry-$(date +%Y%m%d_%H%M%S)"
bash tools/dsv4/run_phase1_a5_matrix.sh f0 a8f4_graph_u2_n3
```

每个 `validation_summary.json` 必须满足：`passed=true`；topology/rank/capacity 与点名一致；U2 点观察到真实双 stage；两个 role return code 为 0；fatal marker 为空；每轮停止后 NPU cleanup 通过。不得使用 `ALLOW_NPU_PROCESSES` 绕过清理门禁。

## 6. 双机 PD 配置

以下将 Prefill/FFN 机记为 P/F，将 Attention 机记为 A；Proxy 可放在 P/F。两台机器安装同一个补丁包，模型、CANN、上游 commit 和 afd-plugin 交付 commit 必须一致。

在两台机器各自执行一次 `init`，然后将两个 `common.env` 修改为完全相同的固定 IP、NIC、模型、Mooncake 和 CANN 配置：

```bash
export MATRIX="$AFD_PLUGIN_ROOT/tools/dsv4/mooncake_pd_manual/pd_graph_matrix.sh"
export CFG="/data/config/dsv4-phase1-pd"
bash "$MATRIX" init "$CFG"
vi "$CFG/common.env"
bash "$MATRIX" list "$CFG"
```

不要修改生成的 `*-role.env`。`init` 会把本机 afd-plugin HEAD 写入 `common.env`；两台机器的 `AFD_PD_COMMIT` 必须相同。每一轮在两台机器设置同一个逻辑运行根：

```bash
export MATRIX_RUN_BASE="/data/run/dsv4-phase1-pd-r1"
```

一期 F1 使用 3 个路径匹配 control 和 4 个 AFD 点：

| control 点 | 对应 AFD 点 | 物理角色 |
|---|---|---|
| `control_graph_u2_mtp2_a8` | `afd_graph_u2_mtp2` | P8 + D8；P8 + A8F8 |
| `control_graph_u2_mtp3_a8` | `afd_graph_u2_mtp3`、`afd_graph_u2_split_a8f4_mtp3` | P8 + D8；P8 + A8F8；P8F4 + A8 |
| `control_graph_u2_mtp3_a4` | `afd_graph_u2_split_a4f8_mtp3` | P8 + D4；P8F8 + A4 |

把第 4 节的 native golden 放到 `common.env` 的 `NATIVE_GOLDEN_PATH`。三个 control golden 分别存入 `${MATRIX_RUN_BASE}/f1-control/<路径键>/golden_results.json`；不要跨路径键复制。

## 7. 生成路径匹配 control

以 `control_graph_u2_mtp2_a8` 为例，启动前先在对应节点执行 `check`：

```bash
# P/F 机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 prefill
# A 机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 decode
# Proxy 所在机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 proxy
```

按 Prefill、Decode、Proxy 顺序启动，在 Proxy 生成 control golden，再逆序停止和收集：

```bash
# P/F 机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 prefill
# A 机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 decode
# Proxy 所在机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 proxy
bash "$MATRIX" record-control "$CFG" control_graph_u2_mtp2_a8 proxy
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 proxy
# A 机
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 decode
bash "$MATRIX" collect "$CFG" control_graph_u2_mtp2_a8 decode
# P/F 机
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 prefill
bash "$MATRIX" collect "$CFG" control_graph_u2_mtp2_a8 prefill
```

将点名依次替换为 `control_graph_u2_mtp3_a8` 和 `control_graph_u2_mtp3_a4`，重复本节。每次 control 必须自身 30/30 稳定；若 native 与 control 不同，保留差异但不得用 native golden 直接替代 PD control。

## 8. 双机 AFD F0/F1

共置 A8F8 点 `afd_graph_u2_mtp2`、`afd_graph_u2_mtp3` 使用 P/F 机的 `prefill` 和 A 机的 `decode`。启动顺序与 control 相同。在 Proxy 先执行 `smoke`，再执行 `validate`：

```bash
# 两台 NPU 节点和 Proxy：启动前分别执行 check，然后按 prefill/decode/proxy 启动。
bash "$MATRIX" smoke "$CFG" afd_graph_u2_mtp2 proxy
bash "$MATRIX" validate "$CFG" afd_graph_u2_mtp2 proxy
bash "$MATRIX" evidence "$CFG" afd_graph_u2_mtp2 decode
```

split 点 `afd_graph_u2_split_a4f8_mtp3`、`afd_graph_u2_split_a8f4_mtp3` 使用 P/F 机的 `prefill_ffn` 和 A 机的 `attention`：

```bash
# P/F 机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
# A 机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
# Proxy 所在机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy

# 三个 check 都通过后再启动。
# P/F 机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
# A 机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
# Proxy 所在机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
bash "$MATRIX" smoke "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
bash "$MATRIX" validate "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
# A 机
bash "$MATRIX" evidence "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
```

四个 AFD 点都按逆序 `proxy -> decode/attention -> prefill/prefill_ffn` 停止。每个角色停止后执行对应的 `collect`。第一次全部通过后，把 `MATRIX_RUN_BASE` 改成 `...-r2`，重新生成 3 份 control golden，再完整执行 4 个 AFD 点，形成第二次独立冷启动证据。

每个 AFD 点必须同时满足：

1. Attention health 和全部 FFN connector loop ready。
2. `smoke`、取消请求后的 health recovery、batch 1/8/32 和 `validate` 通过。
3. 路径匹配 control 的 10 条 prompt x 3 轮达到 30/30 token exact。
4. `evidence` 覆盖全部 Attention rank，并观测到真实 U2 两 stage。
5. A4F8/A8F4 日志中的 rank、peer 和 per-peer token count 符合各自 fan-out/fan-in 方向。
6. 无 OOM、timeout、`Communication_Error`、`507015`、Python traceback 或 EngineCore fatal。
7. Attention 先退出、FFN 随后正常退出；停止后无残留端口和 NPU 进程。
8. 第二次冷启动结果与第一次一致。

## 9. 证据打包与回传

Standalone 验证使用统一收集器；它保留 JSON、环境、NPU 快照和截断日志，不包含 profiler 原始目录：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
bash tools/dsv4/collect_phase1_validation.sh \
  /data/artifacts/dsv4-phase1-a5-evidence.tar.gz \
  "$NATIVE_ROOT" \
  "$PHASE1_OUTPUT_BASE/f0" \
  "$PHASE1_OUTPUT_BASE/f1"
sha256sum -c /data/artifacts/dsv4-phase1-a5-evidence.tar.gz.sha256
```

PD 每个 `collect` 会打印一个小型归档及 `.sha256`。回传以下内容：

1. A5 evidence tar 和 `.sha256`。
2. 两轮所有 control/AFD 角色的 `collect` 归档和 `.sha256`。
3. 两台机器 H0 审计文本。
4. 三份路径匹配 control golden。
5. 任何失败点的完整 `validation_summary.json`、role 日志尾部、首次 fatal 前后至少 200 行，以及当时的 `npu-smi info`。

不要只回传成功截图，也不要在失败后覆盖原 `RUN_ROOT`。收到上述材料后，可按 stack、启动、数据面、Graph 动态路由、token exact、生命周期和清理六类门禁逐项分析。
