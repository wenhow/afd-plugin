# DSV4.1 A5：完整权重功能矩阵验证

2026-11-20 新增 [TP1 / DP4 / EP4 验证入口](DSV41_A5_EP_VALIDATION_ZH.md)。
按当前用户方向先运行该入口的原生 TP1 对照和 A4/F4 EP 候选；下文 TP4 保留为
既有 EP-off 诊断配置。两条路径的参考结果不可混用，EP 的 A5 验收仍待实际执行。

2026-11-19 开发入口扩展为 `tools/diagnostics/run_dsv41_full_checkpoint.py`。
使用原始量化 checkpoint、CPU offload Engram、TP4 A4/F4。新入口默认只生成命令计划；
`--run` 才执行预检、载权、请求和必要清理。CPU/配置通过不能替代下面逐格 A5 验收。
历史 eager/U1 首 token 差异仍待本轮同 TP4 原生对照收口。

## 新批次准备与原生对照

从 `origin/feat/dsv41-flash-hccl-v030` 更新 A5 的独立干净 checkout；确认存在新脚本及
`afd_plugin/model_executor/models/npu/deepseek_v41_engram.py`，旧 U1 提交没有本批功能。
使用已有容器与算子 vendor，不安装或切换 vLLM、Ascend、torch、CANN 等运行依赖。
当前 checkout 需要在同一个 Python 中注册为 editable AFD 插件，见下述准备步骤。
源码锁仍为 vLLM `ced6857a`、
Ascend `7e37e29a`。各个 AFD 实例使用八张卡，原生对照使用四张；PD 的 P/D
必须同时驻留，设备安排尚未指定，由下述配置项声明。
普通矩阵候选固定为文本、DP1/PP1/CP1、EP/SP 关闭、最大两请求、上下文 1024 token、
调度块 128 token；真实 checkpoint 的层数、专家数和 Engram 保持原值。
当前先用 `--repeatability` 将最大活动请求数及客户端并发都限制为 1，定位 eager/U1
输出差异；该诊断不改变量化权重、TP4、原生 Engram 或既有 A/F 角色。

```bash
DSV41_PY=/usr/local/python3.11.10/bin/python
DSV41_MODEL=/home/models/DeepSeek-V4.1-Flash
DSV41_VENDOR=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer
```

在当前插件仓库根目录执行一次安装注册，沿用已有构建工具、跳过 Ascend 算子编译，
不解析运行依赖或下载包。仅更新代码、能 import `afd_plugin` 或设置
`VLLM_PLUGINS=...,afd` 都不会创建插件入口；vLLM 通过安装元数据发现 `afd`。

```bash
AFD_BUILD_ASCEND_OPS=0 "$DSV41_PY" -m pip install \
  --no-deps --no-build-isolation --no-index -e .

"$DSV41_PY" -B - <<'PY'
from importlib.metadata import entry_points, version
from importlib.util import find_spec
from pathlib import Path

print("vllm-afd-plugin:", version("vllm-afd-plugin"))
entries = {entry.value for entry in entry_points(group="vllm.general_plugins")
           if entry.name == "afd"}
assert entries == {"afd_plugin:register_afd"}, entries
root = Path(next(iter(find_spec("afd_plugin").submodule_search_locations))).resolve()
assert root == Path.cwd().resolve() / "afd_plugin", root
print("afd entry:", entries, "source:", root)
PY
```

上述检查只读取安装元数据和顶层模块位置，不导入 vLLM/torch，不初始化 NPU。
2026-11-19 回传的原生和 AFD 两次运行都因缺少该元数据/入口而停在预检；
25 份锁源和 checkpoint 结构检查通过，但尚未载权、发送请求或生成可用原生参考。
详见[归档审阅](../../openspec/changes/afd-hccl-full-parity/evidence/a5-full-checkpoint-plugin-registration-20261119/README.md)。

注册检查通过后，定义入口并先运行原生 TP4 对照。2026-11-20 起，该入口显式选择
`DSV41NativeTP4ControlWorker`：保留原生模型、ModelRunner、MoE 和 Engram，
仅在载权后为本实例的 packed Attention 接入与 AFD 相同的 16→64 query-head 适配。
当前 A5 wheel 不接受 TP4 的 16 heads；此对照不表示完全未适配的上游支持 TP4。
`launch-plan.json` 中的 `attention_head_adapter` 应为 `tp4-query-16-to-64-v1`，
AFD 参考准入同时要求该标记，避免混用旧对照。该 worker 的锁源加入本入口预检。

运行前用 `npu-smi info` 确认选定卡的进程表为空。默认原生使用 0–3，AFD 使用
0–7；若已有服务占用，先通过该服务自己的启动/管理终端正常停止，再核对资源。
本脚本只清理自己启动的进程，不会停止已有服务；也不能把旧快照中的 PID
当作当前停止目标。

```bash
full() {
  "$DSV41_PY" -B tools/diagnostics/run_dsv41_full_checkpoint.py \
    --model "$DSV41_MODEL" --transformer-vendor "$DSV41_VENDOR" \
    --output-root results/full-checkpoint "$@"
}
full --native-control --repeatability --run
```

等待就绪期间，启动器每 30 秒或状态变化时打印 `waiting for readiness`，并将最近一次
`/health` 的 URL、curl 退出码、HTTP 状态或连接/超时错误写入本次目录的 `health.json`。
就绪检查直接调用 `curl --disable --noproxy '*'`，禁用 `.curlrc` 并忽略代理；只有 curl
退出码为 0 且 HTTP 为 200 才就绪。环境需提供 curl；无法执行时立即报错并进入本轮清理。
生成、取消和 PD 的 HTTP 请求使用 Python 的无代理 opener，同样直连指定服务。
可以保留用于更新代码的代理配置，无需手动 unset 或修改系统代理。
模型首次载权需要时间；日志出现 `Application startup complete` 后仍未进入请求时，
在另一个终端查看本次 `health.json`，并对默认原生端口做只读检查：

```bash
curl --noproxy '*' --max-time 3 -i http://127.0.0.1:18031/health
```

2026-11-19 的 `full-eager-u1-yj67qgqv` 已通过预检、packed Attention probe，
完成 TP4 真实权重及 CPU Engram 载入，并在 17:08:02 完成 HTTP 应用启动；
但旧启动器未记录就绪或发送生成请求，17:20:13 被手动中断。
旧包没有健康检查错误记录，无法确认具体 HTTP 原因；代理干扰仅为本机已复现的失败路径。
此包缺少 `summary.json` 和最终资源快照，仍不能作为参考，见
[就绪等待审阅](../../openspec/changes/afd-hccl-full-parity/evidence/a5-native-readiness-20261119/README.md)。

2026-11-20 的 `full-eager-u1-yqxpk_w6` 已通过 curl 健康检查并发送首波双请求，
均因原生 packed Attention 缺少上述适配而返回 HTTP 500：`N1 must be 64 ... got 16`。
修复只补齐对照实例的 backend 选择，不改变 TP 数、checkpoint 或 Engram；
当时使用 `full --native-control --run` 生成普通矩阵参考；当前先运行上面的串行复测。
该包触发 owned 强制清理，
最终设备空闲，但请求/正常退出均未通过，见
[本轮审阅](../../openspec/changes/afd-hccl-full-parity/evidence/a5-native-tp4-heads-20261120/README.md)。

先确认原生目录的 `summary.json` 中 `request_passed=true`、`clean_shutdown=true`，
以及 `repeatability.json` 中 `completed=true`、`passed=true`，
再将本次打印的串行原生目录赋给 `DSV41_NATIVE_REPEAT`。若为 false，先处理其中的 `error`；
打印了 `results:` 仅表示创建目录。2026-11-19 的 `full-eager-u1-1g5_98ms`
因选定设备已被现存 vLLM 服务占用而在启动前退出，不能作为参考，详见
[资源检查回传](../../openspec/changes/afd-hccl-full-parity/evidence/a5-native-resource-gate-20261119/README.md)。
原生也保留 TP4、量化和 Engram，
避免继续用不同 DP/EP 拓扑的旧结果作为等价基线。

```bash
DSV41_NATIVE_REPEAT=/absolute/path/to/results/full-checkpoint/full-eager-u1-XXXXXXXX
full --mode eager-u1 --repeatability --reference-run "$DSV41_NATIVE_REPEAT" --run
```

eager/U1 原生及 AFD 串行对照通过后，可以沿用该稳定原生目录定位 Graph/U1：

```bash
full --mode graph-u1 --repeatability --reference-run "$DSV41_NATIVE_REPEAT" --run
```

该诊断保留 Graph capture/replay，单活动请求，采集预热及取消前/后各轮输出；
不因一次 token 不同提前结束全部轮次。U2 仍使用普通双请求矩阵，单活动请求的
串行协议不能证明 U2 stage 执行。原生自身不稳定时先分析原生，不能将全部差异
唯一归因 AFD 或 Graph。该扩展只增加诊断入口，不宣称修复了数值差异。

原生不通过时，先回传原生自动生成的归档，不启动 AFD；原生通过后再运行并回传 AFD。
串行诊断与旧双请求参考不能混用，脚本会在启动 AFD 前核对请求套件与完整诊断参数。

### 当前 eager/U1 串行复测如何判读

2026-11-20 的 `full-eager-u1-pqdgncjt` 已完成两个 HTTP 200、各 32 token 的 AFD
生成请求，并正常退出、回收八卡。输出对照分别从第 1、第 13 个 token 分歧；此前
原生 `full-eager-u1-7z0p69ts` 的相同请求重复执行也在这两个位置分歧。它们仍不足以
确认输出等价或将根因归到 AFD/Engram，不能用旧参考直接放行后续矩阵。

`--repeatability` 默认按以下顺序运行两个相同短 prompt，全部请求串行：

1. 预热一轮：`warmup-00/request-00`、`request-01`。
2. 取消前重复三轮：`before-cancel-00` 至 `before-cancel-02`。
3. 流式请求收到首个非空文本块后关闭客户端，保存 `cancel-first-chunk.json`。
4. 取消后重复三轮：`after-cancel-00` 至 `after-cancel-02`。

共 14 次非流式请求，每次保留 32 个输出 token。默认不请求 logprobs，先复用此前
已经成功返回 token 的采样路径。同一 prompt 的预热与实测请求参数相同。`--warmup-rounds` 默认 1、
至少 1；`--repeat-rounds` 默认 3、至少 2；原生与 AFD 必须使用相同参数。
客户端关闭连接不单独证明引擎完成取消；后续请求、服务日志和正常清理仍需一起审阅。
该入口允许 standalone eager/U1 或 Graph/U1、DSpark 关闭，
不将两个并发 HTTP 请求误称为固定调度批。

`repeatability.json` 保存进度、各次请求和最终判定。纯 token 不一致会继续采集所有轮次，
最终仍返回失败；HTTP、响应结构或缺失 logprobs 则立即停止并保存未完成报告和原始响应。
普通矩阵的逐请求严格失败行为保持不变。

| 字段 | 含义 |
| --- | --- |
| `completed` | 全部请求及比较完成，不等于通过 |
| `warmup_matches_measured_baseline` | 预热输出是否与首个取消前实测一致，仅作诊断，不计入通过条件 |
| `before_cancel_stable` | 取消前三轮同 prompt 的输入/输出 token 是否与首轮精确一致 |
| `after_cancel_matches_before` | 取消后三轮是否与取消前首轮精确一致 |
| `native_tokens_match` | AFD 实测是否与原生对应轮次的输入/输出精确一致；原生自身为 null |
| `passed` | 取消前稳定、取消后保持一致，且提供原生参考时实测对照也全部一致 |

发生输出分歧时，各请求目录的 `within_run-comparison.json` 和
`native_reference-comparison.json`（AFD）保存首个分歧。默认 `first_divergence_logprobs=null`。

需要候选分数时，另建一组原生/AFD 归档，两侧均增加 `--repeat-logprobs 5`，例如
`full --native-control --repeatability --repeat-logprobs 5 --run`。响应保存用 token ID
标识的 top-5 `raw_logprobs`，分歧报告另含两侧候选与 top1−top2 差距。
不能复用未请求分数的参考；缺失或非有限分数会保留响应并明确失败，不静默降级。
2026-11-20 的 `full-eager-u1-5zwijbpu` 已在 A5 采集全部 14 次响应的 top-5 分数。
数列 prompt 的输入一致，七次首 token 都为 1，首步候选分数已有变化。
取消前第 2、4 个输出 token 在相同生成前缀下复现候选排名翻转。

logprobs 不同于原始 logits；小差距不能单独证明误差可忽略，后续分歧步也因生成
前缀不同而不能直接用于同输入数值比较。重复性通过同样不代表完整模型准确率通过。

如果取消前就不稳定，先排查原生数值/状态与预热路径；如果只有取消后变化，再定位
取消及槽位/历史复用。AFD 自身稳定、与稳定原生仍不同，才进一步采集层间边界定位。
观察到的变化不能单独证明某个模块是根因。所有完整模型任务仍按原验收条件保留。

### 原生首步 logits 定位

使用包含 `--trace-native-prefill` 的提交，从同一个 A5 checkout 执行：

```bash
full --native-control --repeatability --repeat-logprobs 5 --trace-native-prefill --run
```

继续使用两条 prompt、串行顺序及取消前后流程。设备使用原生 TP4 的四张卡。
每个 TP rank 在请求的前两个生成步记录 `AFD native prefill trace` JSON 行。
`attention-server.log` 中的 `request_id` 可用各 `response.json` 的 completion ID 关联。
日志包含输入 token、位置、logits 行索引、传入 logits processor 的 hidden state
摘要以及采样前 logits 的摘要、top-5 和固定候选值。

回传本次 `diagnostics: ...tar.gz`，即使严格 token 对照使脚本退出 1 也回传。
先比较同一 TP rank、相同输入和位置的首步。hidden state 摘要不同，则向模型前向
逐层定位。hidden state 相同而原始 logits 不同，则检查输出头与 TP 汇聚。
两者都相同但 HTTP 分数不同，则检查采样器和评分路径。
第二步还需核对首个生成 token 一致。张量复制会增加耗时，带 trace 的归档用于定位，
脚本拒绝将其用作 AFD 参考。

### 原生首个 prefill 逐层定位

`full-eager-u1-4mrsrmjy` 已采齐 120 条原生 trace。数列 prompt 的首步 hidden state
在输入 token、位置和采样行相同时仍然变化，取消前已复现。用同一个原生 TP4 环境执行：

```bash
AFD_DSV41_NATIVE_LAYER_TRACE=1 full --native-control --repeatability \
  --repeat-logprobs 5 --trace-native-prefill --run
```

采集限于单个新请求的首个完整 prefill，最多 32 个 prompt token，覆盖现有两条
测试 prompt。逐层记录位于 `attention-server.log` 的 `AFD native layer trace` 行。
用已有首步 trace 的 `request_id`、`rank` 关联 embedding、每层入口和出口及最后
norm 的前后。decoder 边界分别记录有效 token 的 `hidden_state` 和 `pre_mix`
摘要。层间 Engram 更新发生在下一层入口前，比较上一层出口和下一层入口可将其与
decoder 内部变化区分开。prefill 后会把临时快照复制到 CPU 并计算摘要，
采集时的耗时及数值表现应按诊断运行解释。

即使严格对照退出 1，也回传本次 `.tar.gz`。

## standalone 十二种组合

串行定位收口后，先用普通矩阵流程重建双请求原生参考；两种套件的目录格式和调度上限
不同，不能混用。普通矩阵参考及首格命令如下，后续本节与 PD 的 `DSV41_NATIVE` 均指此目录；
确认新原生目录 `summary.json` 的 `request_passed=true`、`clean_shutdown=true` 后再使用：

```bash
full --native-control --run
DSV41_NATIVE=/absolute/path/to/new/matrix/full-eager-u1-XXXXXXXX
full --mode eager-u1 --reference-run "$DSV41_NATIVE" --run
```

`--mode` 取 `eager-u1`、`eager-u2`、`graph-u1`、`graph-u2`，
`--spec-tokens` 取 `0`、`1`、`2`，分别对应 DSpark 关、N=1、N=2。
同一命令按表替换参数即可启动每格；默认多流显式开启。需要串行对照时加
`--stream-overlap 0`，Graph/U2 同时改用 `stage_serial` 交换顺序。

```bash
full --mode eager-u1 --spec-tokens 1 --reference-run "$DSV41_NATIVE" --run
full --mode eager-u2 --spec-tokens 0 --reference-run "$DSV41_NATIVE" --run
full --mode graph-u1 --spec-tokens 0 --reference-run "$DSV41_NATIVE" --run
full --mode graph-u2 --spec-tokens 2 --reference-run "$DSV41_NATIVE" --run
```

| 格子 | 启动参数 | 额外通过条件 |
| --- | --- | --- |
| eager/U1，无 DSpark | `--mode eager-u1 --spec-tokens 0` | 与原生五个对应请求的 prompt/output IDs 全等，包含首 token |
| DSpark/U1，N=1/2 | `--mode eager-u1 --spec-tokens 1` 或 `2` | 真实详细 metrics 同时观察接受、拒绝，且拒绝后还有同请求的验证轮次 |
| eager/U2，无 DSpark及 N=1/2 | `--mode eager-u2 --spec-tokens 0`、`1`、`2` | 真实 U2 parent、多流计算/发送/接收、同会话单请求 U1 回退；与 overlap=0 对照 |
| Graph/U1，无 DSpark及 N=1/2 | `--mode graph-u1 --spec-tokens 0`、`1`、`2` | A/F capture、实际 replay 返回、长 prefill→eager 回退；父 Engram lookup 每步更新 |
| Graph/U2，无 DSpark及 N=1/2 | `--mode graph-u2 --spec-tokens 0`、`1`、`2` | 真实双 stage、跨层/多流及图回放，bucket/请求切换后仍与原生输出全等 |

每格自动执行两波双并发短请求，中间取消一个流式请求，随后复用槽位；最后发送一个
超过 128-token 调度块的长 prompt，验证 chunked prefill 与单请求回退。正常请求均生成
32 token，固定 temperature=0/seed=0。取消在首个非空文本 chunk 后关闭连接。
DSpark 无有效提案、只接受/只拒绝、或没有拒绝后继续轮次，均不能通过脚本检查。

`summary.request_passed=true` 只表示本次请求检查完成。U2/Graph 不得只看启动参数：
核对 A/F 日志的 `AFD HCCL stage diagnostic`、实际 `num_stages`、
`actual.mode`、非 warmup/profile 的业务 parent，以及 `graph_parent_replay_returned`。
长 prompt 的 eager 业务 parent 和后继 replay 必须处于同一会话；缺失任一项时该格待验，
不能由纯 U1 或 eager 回退结果替代。

## 可配置 PD：先确定地址，设备稍后分配

P 与 D 各启动一个 A4/F4 实例。P 当前使用 eager/U1；DSpark PD 时 P 和 D 使用相同 N。
两节点可使用相同卡号与默认端口；同节点部署时选择互不重叠的八卡集合，并给 D 设置
`--port-offset`。`--kv-ip` 必须是对端实际可达的本节点地址，`--bind-host` 控制 HTTP
监听。单个实例的 A/F 本轮仍为同节点 mp/TP4，未把 A/F 拆到两台机器。

在 P 节点（先不加 `--run` 可检查计划）：

```bash
full --pd-role producer --mode eager-u1 --spec-tokens 1 \
  --engine-id dsv41-prefill --bind-host 0.0.0.0 --kv-ip "$DSV41_P_IP" \
  --attention-devices "$DSV41_P_A_DEVICES" --ffn-devices "$DSV41_P_F_DEVICES" \
  --producer-stop-file /absolute/path/to/p-finished.flag --run
```

等待打印 `P ready`，在 D 节点执行：

```bash
full --pd-role consumer --mode graph-u2 --spec-tokens 1 \
  --engine-id dsv41-decode --bind-host 0.0.0.0 --kv-ip "$DSV41_D_IP" \
  --attention-devices "$DSV41_D_A_DEVICES" --ffn-devices "$DSV41_D_F_DEVICES" \
  --port-offset "${DSV41_D_PORT_OFFSET:-0}" \
  --prefill-url "http://$DSV41_P_IP:18031" \
  --reference-run "$DSV41_NATIVE" --run
```

上述 IP/卡号变量由实际设备安排填写，例如每节点 A=`0,1,2,3`、F=`4,5,6,7`。
P 也使用端口偏移时，`--prefill-url` 的 HTTP 端口改为 `18031+P偏移`。
D 的参考目录须从本批原生对照复制过来。D 全部请求及清理结束后，在 P 节点创建
`/absolute/path/to/p-finished.flag`；P 才执行正常退出。该文件启动前必须不存在。

| PD 必测格子 | D 参数 | P 参数 |
| --- | --- | --- |
| eager/U1 | `--mode eager-u1 --spec-tokens 0` | eager/U1，N=0 |
| DSpark/U1 | `--mode eager-u1 --spec-tokens 1`，再 N=2 | eager/U1，相同 N |
| Graph/U2 | `--mode graph-u2 --spec-tokens 0` | eager/U1，N=0 |
| DSpark eager/U2 | `--mode eager-u2 --spec-tokens 1`，再 N=2 | eager/U1，相同 N |
| DSpark Graph/U2 | `--mode graph-u2 --spec-tokens 1`，再 N=2 | eager/U1，相同 N |

PD 额外要求：每次 P 响应包含 `afd_engram_history_v1`；D 在原生 KV pull 前校验完整
原始 prompt 的长度、摘要、源 request/engine ID，使用自己的原生 token history 重建
slotless Engram lookback。CPU offload 不免除历史检查。DSpark 继续核对既有 source、
Ready/读取/DONE 协议，不把 history 声明当作 KV ready。P/D 的 request/前缀/退出日志及
全部 D 输出均需对应；取消后新请求不得沿用旧历史。F 不接收 Engram 表或请求历史。

## 回传与验收边界

每个结果目录旁生成 `.tar.gz`，回传原生、各模式及 PD 两端的归档即可。
所有模式均要求正常请求与原生精确匹配、必要清理成功、选定卡无本轮残留；强杀、超时、
错误退出不转为通过。归档保留命令计划、预检、每请求 JSON/metrics、取消片段、日志及
清理结果。`hardware_acceptance=false` / `full_model_qualification=false` 保留为待审阅标志，
不因一次 HTTP 成功自动宣布完整支持。profiler、性能与服务启停/重启专项仍按原排期后置。

下面是旧单次 U1 流程，保留用于历史报告复现；新批次使用上面的矩阵入口。

## 历史：完整权重 eager/U1 单次验证

本轮使用原始完整 checkpoint，实际读取 safetensors 参数并执行推理。
先验证载权、一个固定短请求及正常退出，再根据保存的原生结果核对输出。
此前七场景的 dummy/C2 组件测试不需要重跑。

## 1. 本轮范围和准备

- 使用现有 A5 容器、CANN/ATB 环境及 Python 3.11，不安装或切换依赖。
- 源码锁为 vLLM `ced6857a`、Ascend `7e37e29a`。完整权重预检的 21 份原生源码
  已按锁定 Git 对象核对；旧 `4ec6dc77` 的四个不同哈希不再作为本轮期望值。
- 八张卡须可用：A 使用可见卡 0–3，F 使用 4–7，两侧各自 TP4；
  该入口没有 `--devices` 参数，不能沿用四卡组件命令。
- BF16 激活、原始 checkpoint 的 `deepseek_v4_fp8` 量化和 `safetensors` 加载；
  不改 config、层数、专家数或权重。A 保留 Attention/HC/pre_mix、KV/index 和
  原生 Engram，F 加载其 MoE 参数；Engram 表按既有配置卸载至 CPU。
  因而完整权重参与推理不等于所有参数都常驻 NPU。
- DSpark、PD、Graph、U2/DBO 和 EP 关闭；单请求，最大上下文/批次 1024 token。
  原生 Engram U1 保持已有行为，本轮不开发新的 Engram/U2/Graph 支持。
- Engram host 表下限约 188.8 GiB，启动器检查可用主机内存；这只是下限，
  载入临时内存、其他参数和运行开销仍需额外余量。

在 AFD 仓库根目录更新：

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030
```

## 2. 先预检

将模型路径改为原有完整模型目录。结果目录自动生成唯一名字，可保留旧结果。

```bash
/usr/local/python3.11.10/bin/python -B tools/diagnostics/run_dsv41_a5_afd.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --output-root results
```

要求退出码 0、打印目录中的 `preflight.json` 为 `passed=true`。
预检检查包/插件来源、源码、原始 config/index、48 个 shard 的结构及索引范围，
不读取 tensor payload、不启动服务、不接触 NPU，不能作为实际载权成功的证据。
若失败，回传该 `preflight.json`，先分析具体差异；不修改哈希或模型来绕过。

## 3. 执行一次完整权重请求

预检通过后执行：

```bash
/usr/local/python3.11.10/bin/python -B tools/diagnostics/run_dsv41_a5_afd.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --output-root results \
  --run
```

启动器先检查真实 packed-Attention 小算子，再启动 A4/F4，实际加载完整权重，
自动发送请求，随后执行 A→F STOP 和进程回收并生成归档；无需手工 curl 或 Ctrl+C。
默认加载/启动上限 3600 秒，可用 `--startup-timeout` 调整；单请求上限仍是 300 秒。

固定请求为 `Explain how to keep a simple daily work log.`，`temperature=0`、
`seed=0`、`max_tokens=32`、`ignore_eos=true`，同时保存输入及输出 token IDs。

如果已有对应原生 `response.json`，在本次命令末尾追加：

```bash
  --reference-response /absolute/path/to/native/response.json
```

这是上一条命令的可选参数，不是再启动第二轮。原生参考必须是完整 completion JSON，
包含一个 choice、32 个输出 token 和非空的 `prompt_token_ids`，不能传 summary。
无现成参考时可以先完成载权短请求；回传归档后可离线补对照，无需仅为比较再次载权。

## 4. 如何判定

| 检查 | 判定 |
| --- | --- |
| 实际载权/请求 | A/F 加载日志及启动完成；`response.json` 返回 32 个 token、`finish_reason=length`，`summary.request_passed=true` |
| 正常退出 | `clean_shutdown=true`，probe/A/F 均退出 0，四个 F rank 清理标记完整，无强杀/残留，退出后八卡空闲 |
| 未提供原生参考 | `token-comparison.json.status=not_checked`，只验证执行，不判输出一致 |
| 参考及输入/输出一致 | `status=tokens_match`、`prompt_token_ids_match=true`、`reference_token_ids_match=true`；这是该请求的 token 对照结果 |
| 输入或输出不同 | `status=tokens_differ`，记录原生/AFD 输入与输出及 `first_output_mismatch` 的位置和两个 token；进程返回失败，原始响应保留 |
| 参考缺字段/格式不正确 | `status=invalid_reference`，保留诊断并失败，不当作“未提供参考” |

对照信息也复制到最终 `summary.token_comparison`，失败时同样保留。
原生参考缺失或差异失败时，不用为了得到绿色状态移除参考后重新执行。

历史原生参考是 DP8/TP1/EP8、CPU Engram 共享，历史 AFD 是 A4/F4 TP4、
DP1/EP off、CPU Engram 非共享；原生版本也可能不同。该历史对照用于定位，
不能将差异直接归因为 AFD 缺陷或可接受浮点误差。此前输出首 token 不同仍是未解决问题。
同 TP4 原生完整模型的内存容量尚未确认，不直接要求启动这个对照组。

即使本次短请求及 token 对照成功，`hardware_acceptance` 和
`model_equivalence_accepted` 仍保持 false：单条短请求尚不能关闭完整模型验收。
完整数值收口需要核对同 checkpoint/tokenizer、源码、输入/采样参数和可比较的并行/
Engram 条件，再覆盖多条输入和连续请求。当前没有自动 logits 或逐层差异采集。

## 5. 回传和后续

执行完回传控制台打印的 `diagnostics: ...tar.gz`，失败也回传同一归档。
它包含 preflight、launch-plan、A/F 载权和执行日志、request/response、token 对照、
生命周期及资源记录，不包含模型权重。预检失败且未加 `--run` 时只需回传 preflight。

收到这轮报告后，先定位首个数值分歧并完成真实权重 eager/U1 对照，
再安排连续请求和后续 DSpark/PD/U2/Graph；本轮不开展性能、profiler 或服务专项。
本地源码/标准库检查见
[交付证据](../../openspec/changes/afd-hccl-full-parity/evidence/a5-full-checkpoint-guide-20261119/README.md)，
不将这些检查计作新 A5 完整权重验收。
