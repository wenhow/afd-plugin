# 005 后既有异常所有权修复：独立源码复核

结论：以下三份现有差分未见源码阻断，可作为正确闭包保留。这是已完成修复的源身份复核，不新增 profiler 功能或专项验收。

父包为 `a3-text-normal-worker-20261116/prepared-execute-005/source.tar.gz`。审查读取了实际 tar 成员及当前源，未修改生产、测试、旧包或设备状态。

## EngineCore.collective_rpc

锁定 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 的 `vllm/v1/engine/core.py:964`。已用 Git 原件实际解析 AST：补丁全部参数、默认值和返回注解与原生完全相等，`_R` 直接从原生模块导入，对应其 `:102` 的 TypeVar；没有另造同名变量。正常路径原样保留唯一 delegate return。

差分有 immediately-above 的来源、原因、功能、成本、精确签名及移除条件，并用 PATCH markers 标出异常处理与安装。只在指定 `profile` / `afd_profiler_status`、NPU、AFD HCCL、真实 MultiprocExecutor（含子类）同时成立且原 RPC 已抛异常时处理。MP 类仅在失败路径懒加载；非 AFD、非 NPU、非 HCCL、其它方法、UniProc/Ray 保留原结果或原异常，不改变 executor 状态。

失败时先设置原生 executor.is_failed，再消费并调用现有 failure_callback；F 另外禁止在失败 ordered channel 上发送 STOP。callback 或记录日志再次抛 BaseException 均不能替换原 RPC 异常。没有同步 shutdown、额外正常路径 RPC、device 操作或同步。

原生 `multiproc_executor.py:340/353/375–392` 证明 execute_model、sample_tokens 都重新进入 collective_rpc，后者在 enqueue 前以 is_failed 拒绝。因此它阻止后续模型/RPC复用可能污染的回复队列；并非仅拦 HTTP。`core.py:1041/1560` 的原生 fatal 消息继续拥有进程退出路径。

限制保留：已在途 FutureWrapper 的 `_wait_for_response` 不检查 is_failed，因此本补丁不取消先前 future、不 drain/恢复旧队列，也不证明 device quiescence。monitor 与失败回调可能并发，不承诺 exactly-once 通知。原始异常仍 bare raise。

## profiler Session 与 cleanup

相对005，仅将 Session 的保存类型和 factory/start/step/stop 异常捕获扩到 BaseException，保留同一异常实例。后续 request/require_owner/finish 先检测 sticky failure，因此不能在 KeyboardInterrupt/SystemExit 后重复 stop 或接受新管理请求。

两个 contextmanager 仅在原 serving 错误清理时抑制 cleanup 的 BaseException，并 best-effort 抑制 logging 的 BaseException，随后重新抛原 serving 错误；正常退出时清理错误仍向上传播。未改变 profiler 默认关闭、执行线程所有权或设备采样步骤。

## HTTP API 启动门禁

相对005，新增 NPU HCCL 限制，并要求显式 VLLM_PLUGINS allowlist 包含 `afd`。None 保留原生默认全发现行为；非 None 严格使用原生相同的逗号 split 语义，不 trim 或替换名字。endpoint 插件仅在前端加载，不能把其 import 当成 EngineCore owner patch 安装证明，因此该启动门禁与已声明使用方式一致。无 profiler endpoint 显式启用时不新增服务行为。

## 源身份

| 路径 | 005 SHA256 | 当前 SHA256 |
| --- | --- | --- |
| compat/patches/engine_core.py | ee5d31461487c8ca03fb08a0a725ce8f9fdb22612f0347d605f2deebf9726fad | c84b5ec7e2be6231da409b5ab3823b9e1f67eeaba021e8d446e27f858f925d4e |
| compat/npu/profiler.py | 4dcd664757a788f26efbdf1a3dfaf8acbf7a0a007f26bd2e53c1b6a896d1c117 | 92b418c949591c5f67f75c8ddfc4225944ce95446a48bf55697e3615f1516dc0 |
| compat/npu/profiler_api.py | d75cbba9548607feedd65f0c739f6ede827dfdb34bf3b5a1090ee4004cbd5841 | 107d4458d6caf63ff79cd3a89eee55dde13db364f459f3d455ef6976c324616e |

本轮只执行标准库源读取与签名 AST 比较，没有另跑 pytest、HTTP、SSH 或 NPU。最终完整 unit 结果由 root 的实际闭包运行记录，不能将本源码复核作为设备或完整服务验收。
