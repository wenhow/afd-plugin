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

## 可选 HTTP 管理入口

代码提供独立的原生 endpoint plugin `afd_npu_profiler`。只有显式列入
`VLLM_PLUGINS` 才会注册 `GET/POST /afd/profiler`；仅设置 role-env 不会注册
路由。该入口面向已经按现有说明启动的 API server，不负责启动或连接新的引擎。
A5 完整模型和真实 HTTP 服务仍需独立验证，以下是该入口的操作方式。

更新分支后，在插件 checkout 中使用现有 Python 更新 editable 安装元数据：

```bash
afd_python=/usr/local/python3.11.10/bin/python
"$afd_python" -m pip install --no-deps -e .
export VLLM_PLUGINS=afd,afd_npu_profiler
```

保留实际启动配置中已有的其他插件名。例如需要量化补丁时，应继续保留
`afd_dsv41_mxfp8` 及其对应环境开关；不要用上述示例覆盖整个现有插件列表。
HTTP entry 和 `afd` general plugin 分别加载，前者不会代替后者注册 AFD workers。
对应角色的 profiler role-env 必须已启用；非 NPU/非 AFD、role-env 未启用或与
原生 profiler 配置冲突时，入口初始化会明确报错。

在实际 API server 已启动的环境中，将 URL 指向该角色现有的 server 地址：

```bash
afd_api=http://127.0.0.1:8000
curl --fail-with-body "$afd_api/afd/profiler"
curl --fail-with-body -H 'Content-Type: application/json' \
  -d '{"is_start":false}' "$afd_api/afd/profiler"
curl --fail-with-body "$afd_api/afd/profiler"
curl --fail-with-body -H 'Content-Type: application/json' \
  -d '{"is_start":true,"profile_prefix":"sample2"}' "$afd_api/afd/profiler"
```

POST 只返回 `{"accepted":true,"role":"attention"}` 或对应 F 角色，表示已接受
期望状态。GET 返回 `role` 和实际 worker `workers` 状态列表。请求体中的
`is_start` 必须是 JSON 布尔值，不能写成字符串；只支持可选字符串
`profile_prefix`，不接受任意 RPC 方法名。RPC 超时返回 HTTP 504，其他 RPC
失败返回 HTTP 502；超时后应重新读取状态，不能假定请求一定未应用。

一个 API server 只能管理其既有 EngineClient 所属 executor。A 的 URL 不会
转发到另一独立 F 服务。headless F 可使用下一节的显式本地 socket 入口。
不要新建一个 Python 引擎来管理已运行的 F 进程。HTTP 线程只发送管理请求，
不会直接停止后台 profiler、唤醒阻塞 receive 或修改 HCCL wire。

## 可选 headless F 本地管理入口

在 **F 的启动环境**中启用上面的 F role-env，并指定一个简短的绝对路径。
父目录由运行 F 的用户拥有，仅供该用户访问；每个独立 F EngineCore 使用
不同路径。此入口默认关闭，不依赖 HTTP endpoint plugin。

```bash
install -d -m 700 /tmp/afd-profiler-f
export AFD_NPU_FFN_PROFILER_CONTROL_PATH=/tmp/afd-profiler-f/core.sock
# 然后执行已审查的 F 启动命令，保留其余启动配置。
```

路径所在目录必须已存在，Unix socket 路径需符合系统长度限制。若路径已有
文件或 socket，启动会失败；程序不会删除另一进程的入口。更新代码前启动的
F 不会自动得到该入口，需要在后续计划的服务重启时添加环境变量。

在同一容器、可访问该 socket 的终端执行：

```bash
afd_python=/usr/local/python3.11.10/bin/python
afd_socket=/tmp/afd-profiler-f/core.sock
"$afd_python" -m afd_plugin.compat.npu.profiler_control --socket "$afd_socket" status
"$afd_python" -m afd_plugin.compat.npu.profiler_control --socket "$afd_socket" stop
"$afd_python" -m afd_plugin.compat.npu.profiler_control --socket "$afd_socket" status
"$afd_python" -m afd_plugin.compat.npu.profiler_control --socket "$afd_socket" start \
  --profile-prefix sample2
```

`status` 返回 `ok/role/workers`；`start/stop` 成功返回
`{"ok":true,"role":"ffn","accepted":true}`。它只修改现有 executor 的期望
状态。是否应用仍需读取各 worker 的 `requested/applied/pending`，真实采集还需
检查 trace。阻塞 receive 中的 F 执行线程不会因管理请求被强行唤醒。

入口只接受固定命令；单请求最多 4096 字节、响应最多 65536 字节，最多四个
连接，每次 poll 最多转发一个同步 RPC。未完成的客户端 IO 有五秒期限，RPC
另有三十秒超时，CLI 等待最多三十五秒；这些限制不代表业务负载下的响应时延
验收。超时后重新查询状态。正常退出关闭连接并仅清理自身绑定的 socket。

此路径已通过 A3 实际依赖下的 CPU Unix socket/EngineCore 协议测试，尚未完成
A5 全模型运行期间的管理可达性、真实 trace 和压力测试；请与本页最后的采集
材料一起统一验证。

## 进程内控制与状态边界

已有进程内 Python `LLM` 对象时，锁定 vLLM 提供真实公开入口
`LLM.collective_rpc(method, timeout=None, args=(), kwargs=None)`。在**拥有现有
Attention 引擎对象 `llm` 的代码**中，可以调用：

```python
status = llm.collective_rpc("afd_profiler_status", timeout=30)
llm.collective_rpc("profile", timeout=30, kwargs={"is_start": False})
status_after_request = llm.collective_rpc("afd_profiler_status", timeout=30)
```

这些调用控制该 executor 实际管理的 workers；不会自动找到或控制另一组独立
F 服务进程。不要在另一 Python 进程新造 `llm` 来控制已有服务，也不要将上述
片段当作独立终端命令。已有 API server 可使用上面的独立 endpoint plugin；
headless F 可使用上面的本地 socket 入口。

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
