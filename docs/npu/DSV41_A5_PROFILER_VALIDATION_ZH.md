# DSV4.1 A5 profiler 验证说明

当前补齐的是 A/F profiler 的线程归属和管理 API。CPU 用例验证请求、状态、
执行边界和清理顺序；A5 上的真实 timeline、完整模型推理和性能仍需分别验收。
本页不要求立即启动新服务。原有 A5 模型/Graph 操作说明继续提供启动入口。

## 启用方式

使用 A5 已提供的 Python：`/usr/local/python3.11.10/bin/python`。以下环境变量
需放入各自角色的启动环境，实际启动命令仍沿用已审查的对应配置。

```bash
# Attention 进程
export AFD_NPU_ATTENTION_PROFILER_ENABLE=1
export AFD_NPU_ATTENTION_PROFILER_RANKS=0
export AFD_NPU_ATTENTION_PROFILER_DIR=/tmp/profile/attn
export AFD_NPU_ATTENTION_PROFILER_WAIT=0
export AFD_NPU_ATTENTION_PROFILER_WARMUP=1
export AFD_NPU_ATTENTION_PROFILER_ACTIVE=4
export AFD_NPU_ATTENTION_PROFILER_REPEAT=1
export AFD_NPU_ATTENTION_PROFILER_SKIP_FIRST=0

# FFN 进程
export AFD_NPU_FFN_PROFILER_ENABLE=1
export AFD_NPU_FFN_PROFILER_RANKS=0
export AFD_NPU_FFN_PROFILER_DIR=/tmp/profile/ffn
export AFD_NPU_FFN_PROFILER_WAIT=0
export AFD_NPU_FFN_PROFILER_WARMUP=1
export AFD_NPU_FFN_PROFILER_ACTIVE=4
export AFD_NPU_FFN_PROFILER_REPEAT=1
export AFD_NPU_FFN_PROFILER_SKIP_FIRST=0
```

`RANKS` 使用角色内 rank，可写 `0,1` 或 `all`；默认仅角色 rank 0。
`WITH_STACK=1` 可额外启用调用栈。上述短 schedule 适合后续专门采样，默认配置
仍是 disabled，且默认 `SKIP_FIRST=1500`。启用后按 schedule 自动采集，无需
先调用管理 API。构造和权重加载阶段不包含在新的 A 延迟会话中。

不要同时启用本插件的 role-env profiler 和原生 `--profiler-config`，即使该
rank 不在 `RANKS` 内也会明确报配置冲突。只使用原生 profiler 配置时，插件
保留原生 `profile` 行为；原生控制成功也不能证明 F 驻留线程已产生有效 trace。

## 控制接口与当前限制

已有进程内 Python `LLM` 对象时，锁定 vLLM 提供真实公开入口
`LLM.collective_rpc(method, timeout=None, args=(), kwargs=None)`。在**拥有现有
Attention 引擎对象 `llm` 的代码**中，可以调用：

```python
status = llm.collective_rpc("afd_profiler_status", timeout=30)
llm.collective_rpc("profile", timeout=30, kwargs={"is_start": False})
status_after_request = llm.collective_rpc("afd_profiler_status", timeout=30)
```

这些调用控制该 executor 实际管理的 workers；不会自动找到或控制另一组独立
F 服务进程。F worker 本身同样实现 `profile` 和 `afd_profiler_status`，但当前
交付没有可从外部终端连接已运行 F 进程的管理脚本，也没有仅靠 role-env 注册
的 HTTP 路由。因此不要在另一 Python 进程新造 `llm` 来控制已有服务，也不要
将上述片段当作独立终端命令。F 外部控制入口和原生 HTTP 路由接入仍待实现；
这部分暂不要求用户执行。

状态中 `requested/applied/pending` 分别表示请求、执行线程已应用的状态、尚待
应用的请求。A 在下次 execute 边界应用，F 在下次服务边界应用；阻塞中的 F
receive 不会被管理请求强行唤醒。A `steps` 的单位是成功的 profiled execute，
包括合法返回 None 和 KV-only 维护，`step_unit="execute"`；F 的步进边界是
成功完成的 parent。两者不能直接当作 HTTP 请求数或模型 forward 数比较。

正常退出按 peer STOP/ACK、Graph 清理、同线程 profiler 导出、原生 KV/模型和
通信组清理完成。导出失败或错误线程停机时保留资源并报错，需按既有进程监督
流程处理；不要通过销毁通信组伪装成正常 profiler 停止。

## 后续真实采集需要回传的内容

待完整模型启动条件满足并安排 profiler 专项验证后，优先保留：

- A/F 启动日志、实际角色/rank、分支 commit 和本次环境变量；
- A/F 各一个选中 rank 的 trace 输出目录及其文件清单，包含 profiler 元数据
  和可读取的 timeline 文件；
- 可用入口取得的 `afd_profiler_status` JSON，以及实际负载与开始/结束时间；
- 异常时完整第一处错误和退出日志，正常时 STOP/导出完成日志。

trace 能打开、覆盖了期望的 A/F 算子和通信、各 rank 时间线对应正确，才能
作为设备 profiler 证据。文件存在、CPU 管理 API 通过、status 为 applied 或
完整模型能启动，都不能单独替代这项验收。对开启 profiler 后的耗时也不作
无 profiler 性能结论。
