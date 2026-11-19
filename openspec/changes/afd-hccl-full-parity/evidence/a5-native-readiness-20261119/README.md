# A5 原生 TP4：应用启动完成，启动器仍等待就绪

输入 `01052bc8ff084a74965b94e01b68f9e9.gz`，SHA256：
`1e72adcd3e76f3e89c1267471a15888a618ec9c584fba46b45374e2eaec61a88`。
仅离线审阅，没有连接 A5/A3、启动推理或发送设备信号。
归档 11 个成员中的 10 个文件保持字节不变，逐文件哈希与结论见 [review.json](review.json)。

运行目录 `full-eager-u1-yj67qgqv`，源码 HEAD `5087fa0`、受检 tracked source 无改动，
Python 3.11.10；卡 0–3、TP4、真实量化 checkpoint、原生 CPU offload Engram，
DSpark/PD 关闭。服务地址 `http://127.0.0.1:18031`。
安装包版本后缀与之前相同，但本次 25 份锁定文件哈希全部匹配，预检无错误。

| 时间（2026-11-19，Asia/Shanghai） | 原始证据 |
| --- | --- |
| 17:02:31–17:05:00 | packed Attention probe 启动、exit 0 并通过；覆盖 prefill/decode × 压缩/非压缩四项，三种 max diff 均为 0、输入/cache 保持标志为 true；仅该 A5 算子组件范围 |
| 17:05:00 | 原生服务启动 |
| 17:06:08–17:07:35 | 四个 TP worker 的两张 Engram 表 offload 到 registered host memory，并加载来自 shard 47/48 的行区间 |
| 17:07:36 | 四个 worker 载权完成，各记录 71.64 GiB；safetensors 48/48 |
| 17:07:59 | KV cache 初始化与 Triton warmup 完成；记录可用 KV 10.20 GiB |
| 17:08:02 | API server 绑定默认地址，注册 `/health`，`Application startup complete` |
| 17:20:13 | lifecycle 记录 `KeyboardInterrupt`，随后向本轮 attention 发 SIGINT；服务开始 drain/shutdown |
| 17:20:25 | `EngineDeadError` 出现在 EngineCore 被停止之后；lifecycle 开始 owned SIGTERM 强制清理 |

服务完成应用启动后约 12 分钟，启动器没有记录 `ready` 或 `request_completed`。
包中没有生成请求、响应、请求矩阵、`summary.json` 或最终 `npu-after.log`。
因此可确认停留在就绪等待阶段，不能确认 `/health` 已返回 200、请求输出正确或最终资源清理完成。
停止前未见 OOM、HCCL 异常或未结束的载权/JIT 报错；停止后的 `EngineDeadError`
不能作为先前等待的根因。可选 op 加载 warning 后服务继续完成启动，也不能单独认定为本次阻塞原因。

旧 `health()` 吞掉 HTTP、连接和超时错误，只返回 false；`wait_ready()` 默认可持续等待
3600 秒且不报告中间状态。归档没有保存失败 HTTP 详情或继承代理环境，
故本包的具体 HTTP 失败原因尚未证实。锁定 vLLM 的 `/health` 会执行
`engine_client.check_health()`，应用启动日志本身不能替代这个响应。

本轮仅修复诊断启动器：

- 健康检查、生成、流式取消和 PD prefill 使用显式无代理 opener，直连命令声明的服务。
- 每次健康检查保存 `health.json`，包含时间、URL、状态码、原始异常类别/消息及是否通过。
- 等待时每 30 秒或状态变化时打印进度，并将 `waiting_for_ready` 写入 lifecycle。
- 不改变模型参数、原生 `/health` 判据、参考准入或清理通过条件；未修改 vLLM/Ascend。

主机上用实际 loopback HTTP 服务和返回 403 的代理服务复现：旧健康检查返回 false，
旧生成返回 `completion HTTP 403`；修复后健康、并发生成、流式取消及 PD HTTP 传输通过。
额外验证真实 503 仍返回 false，且保留 HTTP 状态/错误。**这证明代理干扰是一条实际失败路径，
不证明 A5 本次一定设置了代理，也不证明 Engram/PD 或模型输出通过。**

相关主机回归 **17/17 通过**，见 [host-tests.xml](host-tests.xml)。
同一测试集覆盖失败参考准入、原生/AFD 配置、输出比较及 DSpark 报告的既有主机契约。
Ruff、格式、规划一致性及上游保护检查另见 [checks.json](checks.json)；均不替代设备验收。

更新代码后只需先重新运行指导中的 `full --native-control --run`。
若应用已启动但继续等待，在另一个终端读取新包的 `health.json`，并执行
`curl --noproxy '*' --max-time 3 -i http://127.0.0.1:18031/health` 留下响应。
中断后让启动器完成 owned 清理和归档；这次缺少最终清理证据，重跑前仍须核对当前资源。
生成请求及 clean shutdown 成功之前，不将本次目录传给 AFD。
任务 2.2/1.8、历史首 token 差异和完整模型矩阵继续 pending。
