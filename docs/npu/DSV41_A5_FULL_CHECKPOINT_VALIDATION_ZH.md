# DSV4.1 A5：完整权重 eager/U1 验证

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
