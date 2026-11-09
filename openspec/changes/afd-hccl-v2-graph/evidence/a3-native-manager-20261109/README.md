# C5 原生 base manager / HCCL phase-cache 组件验证

北京时间 2026-11-10，最终 [native-manager-006](native-manager-006/results/summary.json) 在 A3 芯片 0/2、1A1F/双方独立原生 TP1 上通过，30.82 秒，退出码 `[0,0]`。运行前后八板均无设备进程；28 个已安装上游源码哈希全部一致，无内外超时或强制清理，未安装依赖、未修改上游。

本次真实执行 base `CudaGraphManager` 与 `NPUUBatchRunner` constructor，使用原生 `torch_cuda_wrapper`、Ascend config、communicator scope 和插件 SM-control scope；A 的原生 manager `capture(factory)` 完成四个 descriptor，F 使用实际 NPUGraph 与 HCCL graph exchange。A 调用真实 `HcclV41GraphControl.reserve_capture` / `reserve_batch`，每次 reservation 后用原生 `InputBatch.make_dummy`、`create_ubatch_slices` 构造受控几何并调用真实 `verify_prepared`。geometry 为 capture sizes 4/8 × U1/U2，两层。未构造原生模型、attention builder、真实 ModelState/Engram 或完整微批线程执行链，runner 的 model_state 为 unused `None`，attention groups 为空。

双方各 26 个父事务，原始记录见 [A rank](native-manager-006/results/probe/rank-1.json) / [F rank](native-manager-006/results/probe/rank-0.json)：

| 阶段 | 事务数/每侧 | 校验 |
|---|---:|---|
| warmup | 4 | 真 HCCL + 受控算术，精确核对 BF16、独立 FP32、IDs 和输出 |
| capture | 4 | 原生 manager factory 与配对 F 捕获，资源从 active factory 生命周期交接至 cache；capture recording 不声称已执行 payload |
| replay | 12 | 每个 descriptor 3 次，A 原生 `manager.run_fullgraph` + F graph replay；不同输入，精确核对传输与输出 |
| F generation/cache miss → eager | 3 | 全局决策后进入真实 eager HCCL payload，精确输出 |
| A 同 bucket / 不同请求 geometry miss → eager | 3 | 全局决策后进入真实 eager HCCL payload，精确输出 |

最后正常 STOP，关闭 cache 与插件通信组，双方原生 TP1 all-reduce 仍成功。该结果是**受控算术组件验证**，不代表 `ModelAclGraphManager`、原生 V4.1 attention/HC/indexer/Engram/MoE、生产 graph selector、完整模型 graph/U2 overlap、性能或故障恢复通过。

最终 probe SHA256：`9002ffef4a84e9008c8012756956a6716f4bf68b23540a4cd49fe4e8b93a50cf`。HEAD 为 `45356b2ff0f81e1d3fcf9b660026bdc904738a19`，完整工作区字节快照 [identity.json](native-manager-006/identity.json) manifest SHA256 为 `9a68f201852f9711175728124247c2579e389eb67dec0f12b04d4f5b1761ffde`；不能用 HEAD 代替新增工作区模块的身份。

| 关键生产模块 | 最终 SHA256 |
|---|---|
| `connectors/npu/hccl_graph.py` | `95a9ae01aef550fd0dd5967aa35458455c74c898ff9909316d89003bedf34ab7` |
| `connectors/npu/hccl_graph_cache.py` | `07e45259d4c0ec034f4c7b21e9dcab8aec26eaab8377b328cfa5ec6839e60020` |
| `connectors/npu/hccl_staged_transport.py` | `9ae40a90dab184fc4f1d17f934520318c2bd50eea75b5d082ed284e9a4e79128` |
| `v1/worker/npu/hccl_v41_graph.py` | `bb8a73e4d1ea46dc012af7980cc4a5edcad1f915e7b9c62d6942b47c09da9777` |
| `compat/patches/npu/mrv2_ubatching.py` | `1e57de63190849a2d5e51f9d2dc9e33326878c1d54e1db1f86ef88c0e0400773` |
| `v1/worker/npu/hccl_v41_graph_control.py` | `d961c7ae0ce2ebde9ca80d488b74ee56052843d0e74bc1682404151db792ed54` |

运行前先以锁定 Git 对象验证新增的 9 个 native 文件：vLLM 的 `gpu/cudagraph_utils.py`、`gpu/ubatch_utils.py`、`gpu/model_runner.py`、`gpu/input_batch.py`、公共 `ubatch_utils.py` / `ubatching.py`，以及 Ascend `worker/v2/aclgraph_utils.py` / `utils.py`、`compilation/acl_graph.py`。锁为 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`；新增各文件的 SHA256 和锁对象来源均在 `locked_git_object_sources`。加上此前锁定/实采的 19 文件共 28 项，实际安装字节全匹配，见 `results/source-check.json` / `upstream-identity.json`。

环境沿用 Python 3.12.9、torch 2.10.0、torch-npu 2.10.0.post4、vLLM 0.30.0+empty、Ascend 0.1.dev1+g7e37e29a7、FastAPI 0.123.10 与现有 CANN/ATB；版本、环境、命令、原始日志、八板 before/after、自己进程身份分别在最终 `results/` 中。未重复 CPU 大组；配套 CPU 实际依赖证据为相邻 `a3-control-cpu-20261109/control-003` 的 519 passed、`endpoint-004` 的 endpoint/cache 38 passed、`consumer-006` 的最终 controller/dispatch/cache/signature 132 passed。

此前 `native-manager-005` 的直接 phase/cache 版本也通过（27 pins、30.17 秒、probe `4a2de225...`），原始记录和其 manifest `0488fd25...` 保留。006 取快照时已包含最终 controller 归属校验修复与 transport 只读 property；按实际 manifest 确认后不重复同字节的 007 设备运行。

## Stage bank 007

新增 `--stage-bank` 的 `native-manager-bank-007` 已通过设备计算与内部监督：双方 `[0,0]`、31.43 秒、同 26 事务，无 timeout/cleanup；实际 bank 跨 layer 复用、地址保持、图内 padding IDs/F output mask、精确 payload/output 均由 probe 断言。四种 geometry 的 bank 逻辑张量存储分别为 552、556、1100、1104 字节，不是整进程显存或 allocator 峰值。最终 probe 为 `a17e799558e628503e00424a8a2588879558abe455361ba81694024aa920c530`、bank 为 `b55e8dce0a72109bfcceaca52163ac2177ca0d0d6697664af64ed3dd4460a00f`，完整 manifest 为 `b2534487ea30eef23579a500388b54ecaab62af1bce8b71ab07189f6a94de9c3`。

该次外层 [首样本 summary](native-manager-bank-007/results/summary.json) 保留 `success=false`：probe return0，但 `npu-after.log` 为七板空闲，board0 另列 PID4115288，不在观察到的 bank 自有 PID 记录中。没有因此杀任何进程；同期另有 owner CPU 回归，但没有证据确证该 PID 的来源，不能将其归因为 bank allocator 或 CPU。首次只读 postcheck 显示所有观察 PID 已不存在，但未准备 CANN 导致 npu-smi 缺 `libc_sec.so` 返回127，此原始错误也保留；随后使用既有 CANN/ATB 的 [postcheck-002](native-manager-bank-007/postcheck-002.json) 显示八板空闲、全部观察 PID 不存在，npu-smi return0。没有推断精确恢复时长、没有重写首次 summary、没有重复已通过的 body。结构化解释见 [interpretation.json](native-manager-bank-007/interpretation.json)。之后 CPU 和 NPU 收尾采样串行安排。

保留的历史失败：

| 尝试 | 原始结果 / 修复归属 |
|---|---|
| native-manager-001 | 本地 uploader 以 GBK 读 UTF-8 source lock 失败，未执行 SSH/设备；uploader 改显式 UTF-8。见 `local-attempt.json`。 |
| native-manager-002 | A constructor 报 `torch.cuda.Stream requires CUDA support`；root probe 补原生 alias/config/communicator/SM scope。F 自己监督 SIGTERM，退出 `[-15,1]`。 |
| native-manager-003 | 首 warmup endpoint 报 `graph exchange requires a different peer`；发现组件把独立 AFD/native rank domain 混用，root 修正 `group.rank()` 并补 CPU 回归。F 自己监督 SIGTERM/SIGKILL，退出 `[-9,1]`。 |
| native-manager-004 | 四个 capture 记录完成，首 replay refill 报 inference tensor 在 InferenceMode 外原地修改；root probe 补原生执行所需 inference lifecycle。该次不能声称 replay 输出通过。F 自己监督 SIGTERM/SIGKILL，退出 `[-9,1]`。 |

002/003/004 均非 timeout，外层未强杀，原始 worker log/json、各自 manifest、cleanup signals 完整保留，退出后八板均 idle。005/006 均无 cleanup signals。自动权限复核曾对 005 启动超时，工具确认命令未执行；一次正常重试后才创建 005 快照，未绕过审核。

本地 PowerShell 使用新 label 复现：

```powershell
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/run_a3_graph_manager.py --label native-manager-new
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/run_a3_graph_manager.py --label native-manager-bank-new --stage-bank
```

执行脚本内层 300 秒、外层 340 秒；外层只观察自身启动的 descendants，并在强杀前核对 PID start ticks。失败尝试的进程清理由内层 supervisor 仅操作自己启动的 PID；未使用全局 pkill。`--model` 默认为已安装 A3 的配置目录，可指定其他本地模型配置路径，始终不加载权重。uploader 的 Ruff/format 与 Python 3.11 AST 通过。
