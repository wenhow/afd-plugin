# PD 首轮 Decode 短输入通信修复与验证（2026-09-10）

## 当前状态

已修复编译后的 A4F8 单 token 分发错误，本机真实模型的短请求、批量请求、
取消后恢复全部通过。双 A3 的跨机 PD F0 和 token-exact 验收仍需现场执行。

## 现场现象

来源：`def2b782cb384196b83bc82abd977d68.zip`。两端 AFD 均为 `84dd069`，
vLLM 为 `0fc695fc`，vLLM-Ascend 为 `rfc/vllm_cann` 的 `3da28f941`，
CANN 为 `/usr/local/Ascend/cann-9.0.0`。两端 Mooncake 库校验值一致。

以下时间沿用现场日志时区：

- 06:06：Attention 的四个 worker 完成图捕获，HTTP health 返回 200。
- 06:07:31：首个 smoke 请求完成跨机 KV 传输，耗时 131.70 ms。
- 06:12:32：Decode DP0 报 `RPC call to sample_tokens timed out`；Proxy 的
  Decode 请求随后收到 500，三次尝试均失败。
- 06:14:47：FFN 收到主动停服的 SIGTERM，随后出现 KeyboardInterrupt/EOF。

这次已越过之前的 MTP N 步图捕获故障。Proxy 已开始流式响应时打印的 HTTP 200，
不能证明后续 Decode 成功。随包的 local roundtrip 是 9 月 9 日的旧记录；本次
跨机 KV 成功以 9 月 10 日的 Attention transfer 日志为证据。

## 第一层 target 通信错误

本机使用同一 A4F8、Graph U2、MTP N3 配置，以 1-token prompt 复现阻塞。
逐层同步诊断显示，FFN DP0 和 DP2–7 完成第一层输入接收，FFN DP1 未完成；
其他 FFN 随后等待第一层 MoE。活跃 Attention DP0 的主线程停在
`_bookkeeping_sync -> _to_list -> event.synchronize`，MTP 尚未进入执行。

直接检查这次服务生成的 `backbone/computation_graph.py` 和
`artifact_compile_range_1_4096_subgraph_0`，发现两个问题：

1. 较大 batch 预热时，Python 条件判断省略了 fanout 补齐分支。vLLM 随后
   不重新检查形状 guards，直接复用范围图；在线 1-token 输入仍按两个 FFN
   切片，第二个切片为空，而 FFN DP1 的控制消息要求接收 1 个 token。
2. 原分片公式中的 `offset < remainder` 在偶数 batch 预热时被固定为 false。
   同一图处理奇数输入时，发出的分片长度与 FFN 按真实 token 数计算的长度不符。

| Attention token 数 | 两个 FFN 应接收 | 旧编译图发送 |
| --- | --- | --- |
| 1 | 1 + 1，第二个为补齐 | 1 + 0 |
| 3 | 2 + 1 | 1 + 1 |
| 5 | 3 + 2 | 2 + 2 |

这里 target 的 `CUDAGraphMode.NONE` 只表示本轮不重放 ACL Graph，模型仍可
运行经 torch.compile 编译的范围图，因此仅把 FFN 改为 eager 不足以修复。

连接器修改如下：编译时始终保留足够的发送补齐和接收缓冲容量，按每轮真实
分片裁剪；分片长度改用不依赖条件分支的整数公式。线上消息数量和布局定义不变，
补齐 token 不进入最终输出。普通 eager 路径保留原有按需补齐方式。编译
fanout 路径引入额外补齐缓冲，可能增加设备内存复制；性能影响尚未专项量化。

## 同时修复的 target/MTP 状态配对

PD 首轮仅有 1 个 Decode token，空闲 DP 则运行 MTP N3 的 4-token dummy。
还需保证以下状态由两侧共同遵循：

- target 控制消息传递 `target_graph_replay`。FFN 不能只因形状命中缓存图，
  就在 Attention 已选 eager 时单独重放 target 或隐式 dummy MTP 图。
- target 的 DP padding 使用同步后的全局 Graph 模式，避免活跃 DP 发送
  `[1,4,4,4]`，空闲 DP 却发送 `[4,4,4,4]`。SP、o-proj TP、embedding TP
  等独立要求 padding 的条件保留。
- 在线空闲 DP 也参加已有的 MTP phase handshake。FFN 收到开始信号后才进入
  draft；启动期 profile、warmup、capture 按原配对流程执行。
- 在线 draft runnable 前，按实际 draft DP metadata 更新稳定头部，避免把
  1-token target 的头部沿用到补齐后的 4-token draft。布局不变时不重复写入。

`target_graph_replay` 随既有消息传递，不增加 target 控制消息。旧消息缺少
字段时保留 `None` 的原行为；现场仍要求 Attention、FFN 两端一起升级。
修改位于 AFD 插件，不修改 vLLM/vLLM-Ascend 源码。

## 本机验证

环境：单台 A3，Attention 0–3，FFN 8–15，A4F8，Graph U2、MTP N3；
vLLM `0fc695fc`，vLLM-Ascend `3da28f941`，torch 2.10.0，torch_npu
2.10.0.post2，CANN 9.0.0。工具链路径为
`/mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0`，运行环境为
`/mnt/workspace/code/.venvs/afd-v023-vllm-cann`。使用已有
`/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp`，未下载或链接权重。

验证目录：`/mnt/workspace/validation/pd_first_token_smoke_20260910/`。

| 证据 | 结果 |
| --- | --- |
| `before-fix/` | 原版真实模型 health 就绪，1-token 请求超时 |
| `target-layer-diagnostics/` | 定位第一层 FFN DP1 输入接收未完成 |
| `unit-final.log` | 415 项相关单测通过 |
| `old-runner-final.log` | 原版 runner 对照 20 项失败，10 项通过，符合预期 |
| `old-fanout-final.log` | 原版连接器图复用对照 18 项失败，6 项通过，符合预期 |
| `after-fanout-compile-fix/summary.json` | 真实模型回归整体 `passed=true`；启动 466.269 秒 |
| `after-fanout-compile-fix/batches.json` | batch 1/8/32，共 41 个请求，均返回 4 个输出 token |
| `after-fanout-compile-fix/cancellation-recovery.json` | 收到首个流式输出后关闭连接，后续恢复请求成功 |

短首轮结果：

| Prompt token 数 | 输出 token 数 | 请求耗时（秒） |
| --- | --- | --- |
| 1 | 4 | 1.161 |
| 2 | 4 | 1.308 |
| 3 | 4 | 1.089 |
| 4 | 4 | 3.068 |

Attention 四个 DP 完成图捕获，八个 FFN 进入 connector loop；请求后 health
正常，两侧日志无致命错误。按 Attention → FFN 停止后，两侧退出码均为 0，
`npu-smi` 确认无残留 NPU 进程。Ruff、格式检查、`git diff --check` 均通过。
测试以 `84dd069` 加本次修改执行，`summary.json` 中记录了六个运行文件的
SHA256，用于核对交付代码与实际测试内容一致。

图复用回归先以 8-token 输入编译，再直接调用所得范围图处理
1/2/3/5/8/9-token 输入，覆盖一对二、一对四的发送和接收。这与普通
`torch.compile` 在形状改变后允许重新编译的测试不同，能检出本次故障。
临时逐层同步和打印已经从交付代码移除。

本机短 prompt 覆盖首轮执行形状，不伪造远端 KV，不作为双 A3 跨机 PD 或
token-exact 精度验收。此前完整 prompt 的 batch 1/8/32 结果没有覆盖这个入口。

## 把 MTP 全部放在 Attention 侧是否可以解决

这样能消除 MTP 专属的跨 Attention/FFN 头部、握手和图配对问题，但上述第一层
target 仍需跨 Attention/FFN 传输，短输入编译分片错误仍需修复。
此外 Attention 侧需要承担 MTP 专家权重、MoE 计算及相应 EP 通信，显存和
吞吐需要单独评估。本次修复不改变 MTP 的放置方式。

## 现场使用

停止旧服务，两台机器各自在本次新包目录执行
`bash bin/install_all.sh`。成功后分别执行
`bash "$MATRIX" refresh-config "$CFG"`，再按主指导书第 8.2 节冷启动、
status、smoke。只解压或只 refresh-config 不会安装修复。
复用安装包保留已有上游环境；不需要重新下载模型，也不需要先补做第 4、5、7 节。
