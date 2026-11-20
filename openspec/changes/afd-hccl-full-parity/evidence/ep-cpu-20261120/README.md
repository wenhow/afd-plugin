# TP1 / DP / EP：实现与 CPU 证据

2026-11-20，用户授权“按此分析开发支持 EP”。实现限于本 afd-plugin checkout，
首个完整权重资格配置为 A4/F4、两侧 TP1/DP4/EP4、eager/U1。
本目录是架构审查、冻结源码和 CPU 合约证据；没有启动模型服务或初始化 NPU。
当前任务状态以 [tasks.md](../../tasks.md) 的 EP1–EP7 为准。

## 源码依据与职责

只读检查 `.agents/dsv41/sources.lock.json` 中的 Git 对象，没有切换上游工作区：

| 源码 | 锁定版本 | 本次参考边界 |
| --- | --- | --- |
| DSV4 供体 | `4ddd69a391dfb5b3e362c7601399e538a31de702` | `afd_plugin/compat/npu/feature_validation.py` 的 DP×TP role 坐标、FFN-side gate；`afd_plugin/v1/worker/npu/ffn_model_runner.py` 的完整 DP token 向量与原生 forward context |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | `vllm/config/parallel.py` 的 engine rank、DP rank 和 world size；`vllm/v1/engine/utils.py:256` 的首个退出后关闭 manager 行为 |
| vLLM-Ascend | `7e37e29a7803d6897a7113088e2b771f83205bc0` | `vllm_ascend/models/deepseek_v41/model.py:163` 的原生 `DeepseekV41MoE`，EP 专家分片、FP32 gate、shared experts、通信与输出合并 |

供体展示了 HCCL 与 DP/EP 可以组合，不能据此直接复用旧 V1 runner。
本次接到当前 V2 / v3 staged 边界：

1. `hccl_config.py` / `hccl_plan.py` 协商显式 EP 能力。EP 使用 TP1、
   DP=角色数、对称 A/F、U1；engine-local `parallel.rank=0`，角色与 WORLD rank
   包含 DP。EP-off 的 HELLO 与诊断记录保持原有字节/字段。
2. `hccl_staged_transport.py` 从所有 A 的真实 counts 生成共同 parent plan。
   原生同步 DP 允许 target 与 idle dummy 同一 wave，A0 空闲也能协调；
   dummy 的 valid count 必须为零，profile/warmup 仍严格一致。
3. `hccl_v41_ffn_runner.py` 从共同 plan 推导 F 的完整 padded DP token 向量，
   每 stage 构造一次，送入原生 Ascend forward context。局部无有效 token 的
   F 仍计算 padding 并进入原生 MoE；只有全局空 stage 整体跳过。
4. A 保留 native HC/pre_mix、Engram ModelState/history、KV/index 和 U1
   DPMetadata。A/F 只交换 BF16 hidden、独立 FP32 router input、原始 IDs 和
   完整 MoE 输出；没有新增专家算子、gate 近似或重复 all-reduce。
5. F launcher 启动全部本地 DP engines，等待所有 engines 完成 STOP 清理。
   第一个成功退出后设置剩余清理期限；异常退出和超时交回已有 client shutdown。
   未把原生 manager 的“首个退出即关闭全部 engines”用于 EP 正常退出。

原生 EP group 与 A/F activation HCCL group 分开，各自保持原有 owner。
本次没有新增上游函数复制/猴补丁，保留已存在的 AFD role/model 接口。
代价是每个 parent 既有 counts/control 同步，以及原生 EP 的通信和共享专家成本；
HBM/host RAM 峰值及性能必须另测。若原生提供可直接消费的 A/F MoE 服务边界，
应迁移并删除对应 glue，而不是继续扩张本地 runner。

## 验证结果

| 层级 | 结果 | 原始证据 |
| --- | --- | --- |
| EP 新合约对旧实现 | 35 failed、1 error、7 passed | `baseline-ep/results/junit.xml`，旧生产快照和当时新测试，预期失败，未作为最终结果 |
| 既存三个过期拒绝断言复现 | 3 failed、5 passed | `baseline-stale-tests/results/junit.xml`，EP 前生产快照已支持对应 TP4 stream / PD Graph 行为 |
| 最终冻结 CPU 回归 | **1859 passed，0 skipped** | `cpu-final/results/junit.xml`、`pytest.log`、`state.json` |
| 真实原生 EngineArgs | **native、AFD A、AFD F 三组通过** | `cpu-final/results/native-config.json`，TP1/DP4/EP、local DP4、CPU Engram DP sharing、原始量化、SP off |
| 本机标准库诊断入口 | **79 passed** | `host-tests.xml`、`host-tests.log` |
| 静态检查 | 28 份 Python（含证据 runner）语法、Ruff check / format 通过 | `static.json` |
| 本地规划与源码保护 | guard / check 通过 | `guard.json`、`planning-check.json`，不是官方 OpenSpec 或设备验收 |

三个过期断言在 EP 前的生产快照也失败，现改为验证已经存在的正向配置/协议结果；
其他真正不支持的 producer/native PD/full/bulk 负例保留。
它们未导致本次扩大 EP-off 的运行时准入。

最终 CPU 环境为现有 A3 隔离 venv：Torch `2.10.0`、Torch-NPU `2.10.0.post4`、
vLLM `0.30.0+empty`、Ascend `0.1.dev1+g7e37e29a7`。已核对远端原生 parallel
和 V4.1 model 两个关键文件 pin；不据此声称整个已安装环境与本地锁完全一致。
测试和配置检查均禁止 NPU lazy init，前后 `npu_initialized=false`；未载权、
未构造模型、未安装包或修改共享源。配置使用原始 checkpoint config，SHA256 为
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`。

`cpu-final/source.tar.gz`、`identity.json`、`remote-runner.py`、`remote.json` 保存开发工作区实际
执行快照、417 项 manifest、远端运行脚本/路径与归档哈希。测试后远端 manifest
全部保持，最终本地源码也与该快照一致。上游/供体保护独立由 guard 核对。
基线失败快照与原始失败日志保留，不改写为通过。

## 本次提交范围的独立复核

用户随后授权提交并推送。以远端已存在的 `f29b74d9` 为基线，在独立 worktree
只装入本轮 EP 改动；同文件内此前未提交的其他测试也按增量分开，保留在原工作区。
开发工作区的 1859 项结果仍保留，不能替代本次实际交付源码的检查。

交付源码从 Git index 读取规范字节重新冻结，**1761 项 CPU 回归、79 项本机诊断入口、
native/A/F 三组真实 EP EngineArgs 全部通过**，全程未初始化 NPU。详见
`delivery-validation.json`、`cpu-delivery/results/` 与 `delivery-host.xml`。
该快照不依赖本次未交付的本地实现或测试；最终提交的 Python blob 必须与其 manifest
逐项相符。开发与交付两个快照分别保留，不把测试数量差异解释为跳过失败。

复现 helper 改用同目录已归档的 `checkpoint-config-original.json`，其 SHA256 与前述
原始配置相同；不再依赖另一个尚未提交的证据目录。检查目标均为仓库已有模块。
外层文本统一 LF，压缩归档保留实际冻结源码和运行结果。

复现入口（沿用已有 SSH 配置、隔离临时目录和 venv，不安装依赖）：

```powershell
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-full-parity/evidence/ep-cpu-20261120/run_cpu.py --output .tmp/ep-development-20261120/recheck
.venv/Scripts/python.exe -B -m pytest tests/e2e/diagnostics/test_dsv41_full_checkpoint.py tests/e2e/diagnostics/test_dsv41_a5_afd.py -q -p no:cacheprovider
.venv/Scripts/python.exe -B .agents/dsv41/harness.py guard
.venv/Scripts/python.exe -B .agents/dsv41/harness.py check
```

## 验收边界

按 [A5 EP 指导](../../../../../docs/npu/DSV41_A5_EP_VALIDATION_ZH.md) 先运行
原生 TP1/DP4/EP4，再运行 A4/F4；参考记录校验 TP、DP、EP 和 head adapter，
拒绝混用历史 TP4/EP-off 输出。TP1 使用原生 64 query heads，不走 TP4 16→64
适配或其专项 probe。

A5 完整权重载入、真实 HCCL/EP、专家分片/内存、原生数值对照、连续/取消后请求、
idle DP 与全部正常退出仍待用户执行。配置/CPU 通过不计 NPU 验收。
EP+Graph/U2/DBO/DSpark/PD、非对称拓扑和性能未闭合；首版显式拒绝组合，
不借用 EP-off 的组件证据，不据本轮勾选完整 HCCL 迁移任务。
