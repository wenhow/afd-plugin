# DSV4.1-Flash v0.30 HCCL：代码交付与 A5 操作入口

更新：2026-11-09。仓库为 `wenhow/afd-plugin`，开发分支为
`feat/dsv41-flash-hccl-v030`。本分支交付适配源码、回归测试、组件采集工具和
A5 操作说明；更新分支后即可使用当前源码，无需再上传旧的源码快照包。

**当前完成 C2 核心实现及组件验证，完整 checkpoint 的原生/AFD 文本推理仍待验收。
V4.1 AFD 模型 guard 保持关闭；C3 DSpark 契约和 C4 多 stage 传输组件已并行推进，
完整模型和模型侧 U2 尚未硬件验收。**

## 在 A5 上更新

新建容器或尚无 Git 仓库时，在可写的父目录下选择一个不存在的目标目录：

```bash
git clone --single-branch --branch feat/dsv41-flash-hccl-v030 \
  https://github.com/wenhow/afd-plugin.git \
  /mnt/workspace/code/afd-plugin-dsv41-a5
cd /mnt/workspace/code/afd-plugin-dsv41-a5
git log -1 --format='%H %s'
```

已有仓库的切换、后续更新及 SHA 核对见
[切换和更新 afd-plugin](../../openspec/changes/afd-v41-hccl-text-eager/evidence/MANUAL_QUALIFICATION_ZH.md#在-a5-上切换和更新-afd-plugin)。
已有目录须先检查本地修改和 origin，随后 fetch、switch、`merge --ff-only`。
保留有改动或分叉的目录，用新目录 clone。

A5 使用 `/usr/local/python3.11.10/bin/python`，无 venv，也不要求 ATB 环境脚本。
组件工具将当前源码放在子进程的 `PYTHONPATH` 前端；确认导入位置的方法见上述说明。
更新 Git 不会替换已运行进程中的 Python 模块，需要重新启动本次验证命令。
本轮沿用容器部署的 FastAPI `0.123.10`；更新插件不需要重装 vLLM、Ascend 或 FastAPI。

## 当前下一步

用户回传的 `dsv41-target-001/002/003` 已通过 MXFP base、18 组原生 wheel norm、
缩小 A 模型构造和 TP2 HCCL 组件检查。更新源码本身不要求重跑这些检查。
组件退出码 0 不能代替完整权重加载、native FP4 MoE 或完整模型验收。

950DT 8 卡、每卡显示 98304 MB 的环境，下一步按
[A5 原生 TP8 完整模型控制组](../../openspec/changes/afd-v41-hccl-text-eager/evidence/A5_NATIVE_EAGER_ZH.md)
核对 checkpoint、空闲设备并手工验证首次文本请求。该说明使用原生 vLLM CLI，
保留 Engram，关闭 DSpark/graph/EP/SP/DBO，不加载 AFD 插件。
完整 checkpoint 控制组尚未执行，实际解析、算子和内存结果均以该轮日志为准。

按每卡 96 GiB 的容量估算，原生 TP8 约 78.790 GiB/rank；AFD 的容量候选为
A4/F4、两侧各 native TP4，约 A 69.135、F 68.006 GiB/rank。
[容量预算及假设](../../openspec/changes/afd-v41-hccl-text-eager/evidence/a5-topology-budget-20261109.json)
未包含全部复制参数、格式转换、KV、workspace、通信和加载峰值，不能保证可运行。
原生 TP8 与 AFD TP4 存在 TP 差异，不能直接归因性能提升或声称固定 TP 数值等价。

原生控制组通过后，继续完整 AFD 冷加载、路由数值、Engram 请求复用与结束、
prefill/decode、连续/并发请求及资源回收验收；通过后再进行 DSpark 硬件验收。
等待 A5 结果期间可并行准备 DSpark 代码：原生 draft 留 A，F 的私有配置视图移除
draft 和 Engram 绑定，loader 按实际请求的模型身份选择类，不改原生 sampler。
此准备工作不改变下方控制组命令，验证时记录实际插件 SHA。
当前不提供绕开模型 guard 的 AFD 完整服务命令。

C4 已实现独立 v3 网络握手、全体计划 ACK、按 stage 隔离的真实 HCCL 收发、聚合/fanout
及完成/退出协议。A3 的 1A1F、2A1F、1A2F、2A2F/TP1 和 2A2F/TP2 组件均完成 U1/U2、
两层、六个连续 parent；使用受控算术验证 BF16、独立 FP32、IDs 和返回去 padding。
组件探针为 [tests/npu/hccl_staged_probe.py](../../tests/npu/hccl_staged_probe.py)，
不构造 native MoE 或加载模型权重。它也验证提前 COMPLETE、错误 stage、重放和 active STOP 拒绝，
以及 HELLO 能力不匹配、坏 ACK 摘要时的全 rank 拒绝和正常回收。
本轮继续实现了显式调用的 V2 stage 输入/执行接口，以及 V4.1 HC continuation、
shared Attention buffer views 和父 Engram lookup/mask 切分；aux 保持层入口采集，
MTP hidden 按父 token 坐标归并。新增 A 侧 90 项、F 侧 45 项 CPU seam
已分组在 A3 实际依赖下通过；F 侧执行入口保留完整 native MoE 和归约所有权，
暂限 DP1/PP1/CP1、non-SP 及覆盖整个 F role 的 native TP world。
独立 [NPU view 探针](../../tests/npu/v41_stage_state_probe.py)也通过真实张量的偏移、隔离及恢复检查。
该探针没有执行 HC/indexer/Engram producer 或加载模型权重，不能证明对应算子接受 offset view。
生产 connector factory 仍为 v2/U1；新接口尚未接入原生 V2 的自动调度入口，
更新源码不会自动切换到 v3 或启用模型侧 U2，完整模型 guard 保持关闭。
本轮没有新增 A5 操作步骤，继续优先验证关闭 DSpark 的完整模型控制组。

## 交付的代码边界

- A 端保留 native forward、HC/pre_mix、Engram、shared-KV 和 V2 ModelState，
  仅替换 MoE 构造；F 端只构造完整 native MoE。
- loader 保持原生模型身份、权重映射及量化后处理，按完整参数命名空间划分 A/F。
- HCCL 传输 BF16 激活、独立 FP32 router 输入及 raw INT32 token IDs；
  等规模 A/F native TP 按 rank 配对，保留各自原生 TP 组。
- 首路径为 V2、text、eager、U1、F-side gate、DSpark off；
  PP/DP/PCP/DCP 均为 1，EP/SP/DBO/graph/speculation 关闭。

源码锁定见 [sources.lock.json](../../.agents/dsv41/sources.lock.json)：
vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，
Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`。
分支名 `rfc/vllm_cann` 不代表自动采用其最新提交。
插件基线是 `39017a137d7809bd0ab7cda4a86697e2a6b7a522`，不能用该基线 SHA 代替本轮交付 SHA。

本轮本地无 torch/vLLM 的 CPU unit 回归为 1037 passed、292 skipped、35 subtests passed；
新增 transport 构造/关闭模块在本机跳过，在 A3 实际依赖下 13 passed。
A3 恢复后已补跑此前跳过的 C3 loader、模型配置、target/draft context 等六文件：
456 passed，无 skip；首轮暴露并修复了共享测试 fake 缺少 TP 字段的问题。
上一轮 A3 connector/distributed 回归为 867 passed、4 deselected、无 skip，含新增 13 项构造用例；
此前 plan/state 129 项也单独全通过。各测试组存在重叠，不累加为唯一测试数。
本轮扩大 CPU 回归为 1983 passed、1 failed、2 skipped、4 deselected；该失败是新 F 测试
期望消息大小写不匹配，修正并收紧 CP/native world guard 后，上述 F 侧 45 项定向通过。
扩大组的 26 项旧失败已由固定基线复现后完成测试适配；实际验收记录保留各快照与失败原文，
未把定向补跑写成重新执行整组全通过。
四进程 eager/U1 TP2 HCCL 组件在设备 0、2、4、6 通过，覆盖两层、五事务、FP32/IDs、
STOP 和 native TP 组保留；验证后全部设备无运行进程。FastAPI 0.123.10 路由 smoke
通过，未启动 listener、引擎或模型。这些结果均不代替完整模型和 U2 验收。
A3 的 CANN 9.0.1 和 9.2.0 均拒绝所需 DynamicMxQuant；A5 base 调用通过，
仍需独立验证真实量化层。依赖声明中的 FastAPI 冲突也不因这些结果自动消除。

需要重新采集组件证据时，按
[A5 手工组件验证说明](../../openspec/changes/afd-v41-hccl-text-eager/evidence/MANUAL_QUALIFICATION_ZH.md)
选择检查项和新的输出目录。首次完整模型验证优先返回控制组说明末尾列出的少量文件，
完整日志和原始张量保留在目标环境中。
