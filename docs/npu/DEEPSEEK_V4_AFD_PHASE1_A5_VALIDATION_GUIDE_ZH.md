# DeepSeek-V4 AFD 第一期 A5 PD 与 dSpark 验证指导书

## 1. 目标和当前状态

本文只描述一期剩余的 A5 功能验证：

1. 双 A5 的 Prefill/Decode 分离，确认 Mooncake KV 从 Prefill 节点传到 Decode Attention。
2. Decode 节点内部 A4F4 Attention/FFN 分离，确认 HCCL、Graph 和 U2 同时成立。
3. 在同一 PD + AFD 拓扑上叠加 dSpark，确认 draft 模型实际加载并产生、接受 draft token。

截至 2026-09-16，单 A5 已完成 no-AFD DP4、A4F4 eager/U1、A4F4 Graph/U2 和
A2F4 Graph/U2。A4F2 仍标记为外部 HCCL 阻塞 `A5-HCCL-RS-001`。这些结果不需要
重跑，也不能代替本文的 PD 和 dSpark 证据。

本阶段不生成 golden，不进行逐 token 比对，不做精度、性能或 dSpark 加速比结论。
最终精度测试仍在全部功能组合通过后单独执行。

## 2. 固定环境和拓扑

| 项目 | 固定值 |
|---|---|
| vLLM | `releases/v0.23.0`，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`，`3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | `feat/dsv4-afd-graph-u2-multistream-all-on-v1` 的当前交付提交 |
| CANN | 只固定两台机器各自的绝对 `CANN_ROOT`，`CANN_VERSION` 保持为空 |
| P 节点 | 8 卡 Prefill，DP2/TP4；同时运行 Proxy |
| D 节点 | NPU 0-3 为 Decode Attention，NPU 4-7 为 Decode FFN；A4F4、DP4/TP1 |
| Graph/U2 | `FULL_DECODE_ONLY`、U2、`AFD_ASYNC_SCHEDULING=off`、五个多流开关全开 |

两台 A5 各有 8 张 NPU。P 节点使用全部 8 卡运行 Prefill，D 节点使用全部 8 卡运行
A4F4，因此不能把三个角色放到一台 A5，也不执行 A8F4/A8F8。

网络数据面如下：

```text
client -> Proxy(P) -> Prefill(P8) --Mooncake KV--> Attention(D0-3)
                                             Attention --HCCL/U2--> FFN(D4-7)
```

执行四个点：

| 点名 | 权重 | target | microbatch | draft | 作用 |
|---|---|---|---|---|---|
| `pd_afd_eager_u1` | Flash | eager | U1 | off | PD + AFD 基础定位点 |
| `pd_afd_graph_u2` | Flash | Graph | U2 | off | PD + AFD 一期必过点 |
| `pd_afd_dspark_eager_u1` | dSpark | eager | U1 | eager | dSpark 基础定位点 |
| `pd_afd_dspark_graph_u2` | dSpark | Graph | U2 | Graph | 最大组合，一期必过点 |

先完整执行一轮四点。第一轮全部通过后，使用新的运行目录，对两个 Graph/U2 必过点
再做一轮独立冷启动。

## 3. 运行前检查

### 3.1 两台机器使用同一代码提交

P、D 两台机器分别执行：

```bash
export AFD_PLUGIN_ROOT="/root/dsv4-afd-hccl/src/afd-plugin-phase1-a5-native"
git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD
git -C "$AFD_PLUGIN_ROOT" status --short
git -C /root/dsv4-afd-hccl/src/vllm-release-v0.23.0 rev-parse HEAD
git -C /root/dsv4-afd-hccl/src/vllm-release-v0.23.0 status --short
git -C /root/dsv4-afd-hccl/src/vllm-ascend-rfc-vllm-cann rev-parse HEAD
git -C /root/dsv4-afd-hccl/src/vllm-ascend-rfc-vllm-cann status --short
npu-smi info
```

通过条件：两台 afd-plugin HEAD 完全相同，三个工作树都干净，两个上游提交与第 2 节
一致，NPU 健康且没有其他模型进程。不要临时屏蔽 dirty 检查。

### 3.2 确认两套权重

普通 `DeepSeek-V4-Flash` 权重不能通过一个启动参数变成 dSpark。两台机器都必须存在：

```text
/home/models/DeepSeek-V4-Flash
/home/models/DeepSeek-V4-Flash-DSpark
```

dSpark 权重的 `config.json` 必须包含有效的 `dspark_block_size` 和
`dspark_target_layer_ids`。在两台机器分别执行：

```bash
export VENV_ROOT="/root/dsv4-afd-hccl/venv"
export MODEL_TOOL="$AFD_PLUGIN_ROOT/tools/dsv4/hccl_manual_install/bin/model_launch_args.py"

"$VENV_ROOT/bin/python" "$MODEL_TOOL" \
  --model-path /home/models/DeepSeek-V4-Flash \
  --describe

"$VENV_ROOT/bin/python" "$MODEL_TOOL" \
  --model-path /home/models/DeepSeek-V4-Flash-DSpark \
  --describe
```

普通权重应显示 `dspark_block_size: null`；dSpark 权重应显示正整数
`dspark_block_size` 和非空 `dspark_target_layer_ids`。当前官方 dSpark 权重的 block
size 由权重配置决定，脚本会自动读取，不能再按旧 MTP 的 N1/N2/N3 手工选择。

固定 v0.23 栈在内部将 speculative method 归一为 `mtp`，但会根据上述 dSpark 字段
构造 dSpark proposer。这是当前固定栈的预期行为，不能仅凭日志中的 `method=mtp`
判定 dSpark 未启用。

## 4. 生成双机配置

以下操作在两台机器上执行，`SITE` 和 `CFG` 必须使用相同绝对路径。先创建第一轮现场
配置：

```bash
export PD_DIR="$AFD_PLUGIN_ROOT/tools/dsv4/mooncake_pd_manual"
export SITE="/root/dsv4-afd-hccl/a5-pd/site-r1.env"
export CFG="/root/dsv4-afd-hccl/a5-pd/config-r1"

mkdir -p "$(dirname "$SITE")"
cp "$PD_DIR/a5_site.env.example" "$SITE"
vi "$SITE"
```

只需修改 `SITE` 中这些现场值：

```text
CANN_ROOT=<本机实际绝对路径>
PREFILL_IP=<P节点业务/HCCL地址>
DECODE_IP=<D节点业务/HCCL地址>
NIC_NAME=<上述IP所在网卡>
A5_PD_RUN_BASE=/data/validation/dsv4-phase1-a5-pd-r1-<时间戳>
```

两台机器的 CANN 安装路径和网卡名可以不同，但 `CANN_VERSION=""` 必须保持为空。
其余路径与现场不一致时一并修正。两台机器必须填写相同的 Prefill/Decode IP、模型
路径和 `A5_PD_RUN_BASE`。

生成并检查 12 份角色配置：

```bash
bash "$PD_DIR/init_a5_pd_validation.sh" list
bash "$PD_DIR/init_a5_pd_validation.sh" init "$CFG" "$SITE"

for config in "$CFG"/*.env; do
  bash "$PD_DIR/pd.sh" print-config "$config" \
    >"${config%.env}.effective.txt"
done
sha256sum "$CFG"/*.env
```

两台机器的 12 个 env 文件 SHA256 必须相同。生成器会把当前 afd-plugin HEAD 固定到
每个角色配置，并拒绝覆盖已有目录；需要重建时使用新的 `CFG`，不要修改生成的角色
文件。

## 5. 安装审计和预检

`install` 不重装 Python、CANN、vLLM 或 vLLM-Ascend，只确认现有 Mooncake，并将当前
afd-plugin 以 editable 方式安装到既有 venv。

P 节点：

```bash
export PD="$AFD_PLUGIN_ROOT/tools/dsv4/mooncake_pd_manual/pd.sh"
bash "$PD" install "$CFG/pd_afd_eager_u1-prefill.env"
bash "$PD" check "$CFG/pd_afd_eager_u1-prefill.env"
bash "$PD" check "$CFG/pd_afd_dspark_eager_u1-prefill.env"
bash "$PD" check "$CFG/pd_afd_eager_u1-proxy.env"
```

D 节点：

```bash
export PD="$AFD_PLUGIN_ROOT/tools/dsv4/mooncake_pd_manual/pd.sh"
bash "$PD" install "$CFG/pd_afd_eager_u1-decode.env"
bash "$PD" check "$CFG/pd_afd_eager_u1-decode.env"
bash "$PD" check "$CFG/pd_afd_graph_u2-decode.env"
bash "$PD" check "$CFG/pd_afd_dspark_eager_u1-decode.env"
bash "$PD" check "$CFG/pd_afd_dspark_graph_u2-decode.env"
```

预检会核对提交、工作树、模型契约、CANN 路径、Mooncake、本机端口、NPU 数量和本地
NPU round-trip。它不依赖 `ss`，也不校验 A5 的 CANN 版本字符串。任一检查失败都先
停止，不进入模型启动。

## 6. 执行单个验证点

每个点都按本节完整执行。以下把点名放在 `POINT` 中；第一轮依次替换为第 2 节四个
点名。建议使用 P 节点两个终端和 D 节点一个终端。

### 6.1 启动

P 节点终端 1，先启动 Prefill：

```bash
export POINT="pd_afd_eager_u1"
bash "$PD" start "$CFG/$POINT-prefill.env"
bash "$PD" status "$CFG/$POINT-prefill.env"
```

D 节点，Prefill ready 后启动共置的 A4F4 Decode：

```bash
export POINT="pd_afd_eager_u1"
bash "$PD" start "$CFG/$POINT-decode.env"
bash "$PD" status "$CFG/$POINT-decode.env"
```

P 节点终端 2，最后启动 Proxy：

```bash
export POINT="pd_afd_eager_u1"
bash "$PD" start "$CFG/$POINT-proxy.env"
bash "$PD" status "$CFG/$POINT-proxy.env"
```

三个 `status` 都必须返回 0。dSpark 点的 Decode status 还必须显示：

```text
DSpark Attention drafter markers: 4/4
```

### 6.2 功能请求和数据面门禁

P 节点 Proxy 终端执行 batch 1/8/32、取消和恢复请求：

```bash
bash "$PD" smoke "$CFG/$POINT-proxy.env"
```

D 节点随后执行数据面门禁：

```bash
bash "$PD" verify-data-path "$CFG/$POINT-decode.env"
```

`verify-data-path` 必须输出 `Data-path gate passed`，并检查：

- Attention 日志至少有一次成功的 Mooncake KV transfer；
- U2 点至少有一次在线 `stage_count=2`，不能只看到 Graph capture/warmup；
- dSpark 点的 draft token 与 accepted token 指标都大于 0；
- Attention/FFN 进程仍存活，health 正常，日志没有 fatal marker。

证据写入：

```text
<RUN_ROOT>/output/<POINT>-decode-data-path/summary.env
<RUN_ROOT>/output/<POINT>-decode-data-path/mooncake-kv.log
<RUN_ROOT>/output/<POINT>-decode-data-path/online-u2.log       # U2 点
<RUN_ROOT>/output/<POINT>-decode-data-path/dspark.metrics     # dSpark 点
<RUN_ROOT>/output/<POINT>-decode-data-path/dspark-gate.env    # dSpark 点
```

### 6.3 收集和停服

服务仍在运行时收集证据，以便保存 metrics。P 节点执行：

```bash
bash "$PD" collect "$CFG/$POINT-prefill.env"
bash "$PD" collect "$CFG/$POINT-proxy.env"
```

D 节点执行：

```bash
bash "$PD" collect "$CFG/$POINT-decode.env"
```

按 Proxy、Decode、Prefill 顺序停服。

P 节点：

```bash
bash "$PD" stop "$CFG/$POINT-proxy.env"
```

D 节点：

```bash
bash "$PD" stop "$CFG/$POINT-decode.env"
npu-smi info
```

P 节点：

```bash
bash "$PD" stop "$CFG/$POINT-prefill.env"
npu-smi info
```

两台机器的 NPU 进程表必须为空，才能进入下一个点。每个点都是独立冷启动，不能复用
上一点的模型进程。

## 7. 执行顺序和第二轮

第一轮严格按以下顺序重复第 6 节：

```text
pd_afd_eager_u1
pd_afd_graph_u2
pd_afd_dspark_eager_u1
pd_afd_dspark_graph_u2
```

第一轮四点全部通过后，在两台机器分别复制一份新的 `site-r2.env`，设置新的
`A5_PD_RUN_BASE` 和 `config-r2`，重新运行生成器。第二轮只执行：

```text
pd_afd_graph_u2
pd_afd_dspark_graph_u2
```

第二轮仍需完整执行启动、smoke、`verify-data-path`、collect、停服和 NPU 清理，不能
直接复用第一轮日志。

## 8. 通过标准

每个点同时满足以下条件才记为通过：

1. Prefill、Decode Attention、Decode FFN 和 Proxy 全部 ready，health 正常。
2. batch 1/8/32 全部请求成功；取消请求得到预期超时，随后 recovery 请求成功。
3. Decode 的 `verify-data-path` 返回 0，Mooncake KV marker 大于 0。
4. Graph/U2 点记录在线 two-stage marker，配置为 `full-decode-only/U2`、async off、五个多流开关全开。
5. dSpark 点加载四个 Attention rank 的完整 dSpark drafter，使用权重的准确 block size，并且 drafted/accepted token 都大于 0。
6. 三个角色日志无 fatal marker；正常停服后两台机器的 NPU 进程表为空。
7. `summary.env` 明确 `golden_checked=0`，不得据此声明精度通过。

一期 A5 PD + dSpark 功能目标的关闭条件是：第一轮四点通过，第二轮两个 Graph/U2 点
再次通过。A4F2 的 `A5-HCCL-RS-001` 继续单列，不影响 A4F4 PD + dSpark 组合的功能
结论，但不能把它改记为 A4F2 已通过。

## 9. 失败处理和回传内容

任一点失败后不要继续下一个点。先在服务仍存活时尽量执行三个角色的 `collect`，再按
第 6.3 节停服。不要 reset 工作树、屏蔽 dirty 检查、修改权重或降低门禁。

每个 `collect` 会输出 `ARTIFACT` 和对应 SHA256。回传失败点的以下内容：

```text
P节点 prefill collect tar.gz 及 sha256
P节点 proxy collect tar.gz 及 sha256
D节点 decode collect tar.gz 及 sha256
<POINT>-decode-data-path 目录
三个角色的完整终端输出
两台机器停服后的 npu-smi info
```

按失败位置初步分类：

| 失败位置 | 优先检查 |
|---|---|
| `check` | 提交/dirty、CANN 路径、模型契约、Mooncake、本机 round-trip |
| Prefill 启动 | dSpark 权重、P8 容量、CANN/算子加载 |
| Decode 启动 | A4F4 HCCL、Graph capture、dSpark drafter、FFN connector loop |
| Proxy 启动 | P/D health、IP/端口连通性 |
| `smoke` | 请求路由、取消恢复、运行期 fatal |
| `verify-data-path` | Mooncake KV、在线 U2 或 dSpark 指标中具体缺失的一项 |

本阶段完成后，再安排最终精度测试；本文不包含性能 Profile、dSpark 加速比、A8F4、
A8F8、U3 或逐 token exact。
