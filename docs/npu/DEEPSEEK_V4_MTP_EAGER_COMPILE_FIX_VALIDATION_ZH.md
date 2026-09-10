# MTP eager 多步启动修复与验证（2026-09-10）

## 故障与原因

现场 Attention 日志来自 `826bb4c53209470fb9f080ca6b192780.zip`，
FFN 日志为 `d1c513b9b48041f7a9e41b618c618ea1.txt`。
两侧均为 A8F8、target Graph U2、MTP 2、draft `enforce_eager=True`。

FFN 八个 rank 在 09:24:11 进入工作循环；09:26:05 首先报
`invalid MTP phase: speculative_step=0, expected_step=1`。
Attention 在 09:26:06 的启动 profile 阶段发送下一轮控制元数据时，才报
`Connection closed by peer`。Prefill 已启动成功。

固定的 vLLM-Ascend proposer 在一次 merged draft 中重复调用单层 MTP，
没有显式传递 `spec_step_idx`。插件通过模型 `forward` 中的 Python 计数器
补足步号，但 eager draft 仅关闭 ACL Graph，仍会继承 target 的
torch.compile 配置。vLLM 复用编译结果时不重新检查 Python 状态 guards，
因此首次编译的步号 0 被重复使用，第二个 MTP token 即触发 FFN 的严格校验。

使用原函数与相同的 guard 策略在 CPU 上复现：普通调用步号为
`0,1,0,1`，编译复用后为 `0,0,0,0`。此次故障不属于 A8F4 的权重容量限制。

## 修改与影响

修改仅位于插件 `AFDDeepSeekV4MTP` 的 `support_torch_compile(enable_if=...)`：
当 draft 配置为 eager 且 `num_speculative_tokens > 1` 时，直接执行模型，
使每次调用都更新 Python 步号。使用上游提供的编译开关，不覆盖上游源码。

- target 模型继续按原配置编译和捕获 Graph，U2 配置不变。
- graph draft、单 token eager draft 保留原编译路径。
- FFN 的 phase/step 校验、MTP 放置方式、权重和通信协议不变。
- eager 多步草稿不再获得整模型 torch.compile 融合，可能影响该模式性能；
  本次首先恢复功能，性能差异未做专项基准。

后续若需要恢复 eager 多步草稿的编译，应将可变的协议步号移出编译边界，
或以显式运行时数据传递，并保留本次回归覆盖。

## 验证记录

验证目录：`/mnt/workspace/validation/mtp2_eager_compile_fix_20260910/`。

固定运行栈：vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，
vLLM-Ascend `rfc/vllm_cann` 的 `3da28f9414583d2d0b672a8f06d1fae142404bda`，
torch 2.10.0，torch_npu 2.10.0.post2，CANN 9.0.0。
本机工具链为 `/mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0`，
虚拟环境为 `/mnt/workspace/code/.venvs/afd-v023-vllm-cann`。

`regression-before.log`：通过真实模型装饰器及调用入口，以 CPU backend
复用编译图。旧代码的 eager MTP 2/3 两项失败，graph draft 两项通过。
新增测试另覆盖 MTP 1 的编译路径以及同一模型连续两轮 proposal。

`unit-host-after.log`：模型构造、代理、NPU runner 和两个 P2P 连接器共
460 项测试全部通过，用时 20.30 秒。首次隔离环境运行在导入 CANN 分配器时
中止，记录保留在 `unit-after.log`；最终结果来自可访问驱动的宿主环境重跑。

整模型验证使用已有 `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp`，
本机 A8F8，Attention 0–7、FFN 8–15，target Graph U2，MTP 2 eager。
max model len 4096，max sequences 16，两侧 max batched tokens 4096，
capture sizes 为 1/2/4/8。

`a8f8-mtp2-eager/summary.json` 整体 `passed=true`：

| 检查 | 结果 |
| --- | --- |
| 启动 | 448.299 秒；8 个 Attention rank 完成 Graph capture 和 engine 初始化，8 个 FFN rank 就绪 |
| 短 prompt | 1/2/3/4 token 输入均返回 4 个 token，耗时分别为 2.632/2.802/2.107/1.889 秒 |
| `batches.json` | batch 1/8/32，共 41 个请求，均返回 4 个 token |
| 取消后恢复 | 首个流式输出后断开连接，恢复请求成功，health 正常 |
| 停止与清理 | Attention、FFN 退出码均为 0，NPU 无残留进程 |

时间为本机 Asia/Shanghai 时区：18:32:47 全部 Attention 完成图捕获，
18:32:51–52 全部 engine 完成初始化；批量 smoke 为 18:33:06–21。
没有重现 `invalid MTP phase`。18:33:23 主动发送 SIGTERM 后，FFN 的
DEBUG 日志包含 `receive loop stopped during shutdown` 及连接关闭堆栈，
属于停止阶段，与现场启动阶段 FFN 先报非法步号不同。

验证基于 `f70289f` 加本次修改执行，`summary.json` 保留六个运行文件的
SHA256，用于对照交付代码。Ruff、格式检查和 `git diff --check` 均通过。

本机只有一台 A3，验证关闭跨机 PD，不作为双 A3 Mooncake KV 传输或
token-exact 精度验收。现场仍需执行主指导书第 8.2 节的 PD smoke。

## 现场升级

先停止旧服务，两台机器分别在新版安装包目录执行
`bash bin/install_all.sh`，再执行 `bash "$MATRIX" refresh-config "$CFG"`。
仅解压或仅 refresh-config 不会安装插件修复。

保持点名 `afd_graph_u2_mtp2`：P/F 机用 `prefill`，A 机用 `decode`，
代理用 `proxy`，按第 8.2 节执行 check、冷启动、status、smoke、逆序停止和 collect。
F0 不依赖第 4、5、7 节；F1 精度比较仍需第 7 节的对应 PD control。
