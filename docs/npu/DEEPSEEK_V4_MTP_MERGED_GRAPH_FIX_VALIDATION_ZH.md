# MTP N 步 Graph 捕获修复与验证（2026-09-10）

## 问题和修复

双 A3 A4F8 / Graph U2 / MTP N3 的现场日志中，Attention 第一次 MTP header
捕获通信成功，第 2 次捕获通信资源建链超时。FFN 此时已经结束单步 MTP 捕获，
并持续等待设备任务。四个 Attention rank 和八个 FFN worker 的状态一致。

根因是两端图的范围不同：vLLM-Ascend 的 `_run_merged_draft` 在一个图内执行
全部 N 步；旧 FFN `_capture_mtp_graphs` 只捕获一步，之后重放 N-1 次。这无法
匹配 Attention 后续步骤在捕获期间创建的 HCCL 通信资源。

本次修改：

- FFN 在同一 Graph 捕获内完成 N 次 header 接收、计算、结果发送。
- 在线 Graph 路径每个 merged-draft 阶段重放一次完整图。
- duplicate-target 捕获路径同样只重放一次完整 MTP 图，避免 N×N 次执行。
- 保留 eager 路径的 N 次执行和 Graph header 内固定 step=0 的协议。
- 修正物理 HCCL 组件验证：Attention 捕获完整 N 步，FFN 直接调用生产 runner
  的捕获和重放方法，仅将模型计算替换为小型加法。每步输出作为下一步输入，
  每次重放更换输入，逐步检查结果，覆盖漏执行、多执行及旧缓冲复用问题。

修改位于 AFD FFN runner，不改变上游 proposer 的 merged-draft 接口。
图缓存 key 已包含 speculative 配置；新图按 N 步保留捕获资源。相较旧实现，
FFN 图包含更多节点和通信资源，需在目标模型上检查图显存与启动结果。

## 固定环境

| 项目 | 本次验证使用值 |
| --- | --- |
| CANN | 9.0.0 |
| torch / torch-npu | 2.10.0 / 2.10.0.post2 |
| vLLM | 0.23.0，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann` 固定提交 `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| 修改前 AFD | `40c07705458ffa8a1904246091d5c8aadc9f9e9d` |
| 模型 | 既有 `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 硬件 | 单台 A3，16 个可见 NPU；A4F8 使用 0–3 和 8–15 |

使用独立验证进程，未切换或重装上游环境。

## 已完成验证

1. `test_npu_runtime.py` 与 `test_p2p_hccl_connector.py`：322 项全部通过。
2. 将修改前的三个 runner 方法仅加载到独立 Python 进程，运行相关回归用例：
   N=2/3 的捕获、在线重放、重复捕获共 12 项按预期失败；证明用例能检出旧问题。
3. 真实 HCCL / NPUGraph 组件：A1F2 U2 N1、N2、N3，以及 A4F8 U2 N3 全部通过。
   均覆盖首次完整 N 步捕获、重复捕获、在线重放和逐步数值校验；所有 worker
   正常退出。A4F8 N3 为 12/12 rank 通过。
4. Ruff 检查、格式检查及 `git diff --check` 通过。

验证产物：`/mnt/workspace/validation/mtp_merged_graph_fix_20260910/`。

全模型验证 `full_model_a4f8_u2_n3/summary.json` 为 `passed=true`：

- 单机 A4F8，Graph U2，MTP N3，异步调度关闭；capture sizes 配置为 1/2/4，
  Attention/FFN token 上限为 4096/2048，max_num_seqs=8，PD 关闭。
- 冷启动到 health 就绪用时 468.251 秒。4 个 Attention worker 完成图捕获，
  8 个 FFN worker 均记录完整 N3 MTP 图捕获成功；同时有重复捕获和在线重放记录。
- batch 1、8、32 共 41 条请求，全部返回 16 个输出 token，共 656 token。
  请求后 health 正常，未出现 EI0006、107025 或 invalid MTP phase。
- 两个启动器退出码均为 0，NPU 清理检查通过，无残留设备进程。

| 请求 batch | 返回条数 | 每条输出 token | 本轮耗时 |
| --- | --- | --- | --- |
| 1 | 1 | 16 | 2.51 秒 |
| 8 | 8 | 16 | 13.91 秒 |
| 32 | 32 | 16 | 44.30 秒 |

这是启用诊断日志的功能验证，不是性能基准；请求报告的 `golden_checked=false`，
未执行 token-exact 精度比较。较大 batch 可能按现有 capture size 配置回退到
eager 路径，不能将所有请求都记为 U2 Graph replay。

停服阶段，FFN 在收到终止信号后打印 `KeyboardInterrupt: terminated`，同时有
Python resource_tracker 清理告警；它们出现在请求完成、主动停服之后。此处记录
事实，不将启动器返回 0 解读为日志完全没有告警。

## 双 A3 升级与验证

使用新生成的 `slim-dual-a3-reuse` 安装包。先停止旧 Attention，再停止
prefill_ffn；两台机器各自在新包目录执行 `bash bin/install_all.sh`。安装成功后，
两端各执行 `bash "$MATRIX" refresh-config "$CFG"`，再按主指导书第 8.2 节
启动 prefill_ffn、Attention 和 proxy，运行 health/smoke。

仅解压或 refresh-config 不会更新源码。运行中的进程也不会自动换成新的已捕获
Graph，必须重启。包内 `manifest/versions.env` 给出目标提交，两端应一致。

单机 A4F8 的结果不能代替双 A3 的跨机 HCCL 与 Mooncake PD 验收；组件数值校验
也不能代替真实模型的 token-exact 精度比较。
