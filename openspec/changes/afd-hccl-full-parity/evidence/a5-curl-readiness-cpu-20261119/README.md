# 就绪检查直接使用 curl，无需清除环境代理

用户要求在脚本中直接使用 `curl --noproxy '*'`。本轮将 A5 原生/AFD 共用的
`health()` 切换为 curl；没有连接 A5/A3、启动推理或修改上游。
这不确认此前 A5 等待的具体 HTTP 原因，也不改变原生参考和完整模型验收状态。

`health()` 使用参数列表启动 curl，不经过 shell。首参数 `--disable` 禁用 `.curlrc`，
`--noproxy '*'` 绕过环境代理；单次 curl 最长 2 秒，外层进程最长 3 秒。
响应 body 丢弃，通过 `--write-out '%{http_code}'` 记录 HTTP 状态。
只有 curl exit 0 且 HTTP 200 才通过；curl 非零退出即使已经收到 HTTP 200 也不得就绪。
每次检查在 `health.json` 保留 client、exit_code、status_code、error 和 passed。
无法启动 curl 时先保存错误，再向外抛出异常，进入启动器已有的本轮清理流程。

生成、流式取消和 PD prefill 继续使用 `ProxyHandler({})` 的无代理 Python opener。
这与健康检查的无代理连接语义一致，不需要用户清除用于更新代码的代理配置。
就绪检查仍调用真实服务 `/health`，未用应用启动日志代替服务响应，未放宽请求、参考或清理条件。

Windows 主机使用 curl 8.22.0；已用本地 `curl --version` / `curl --help all` 核对选项。
实际 loopback HTTP 服务及返回 403 的代理保持启用，验证 curl 能取得目标的 200，
真实 503 仍失败并保留状态；既有并发生成、取消与 PD HTTP 传输也通过。
另以进程边界故障验证 curl 缺失立即失败、外层超时不就绪，以及非零退出不因 HTTP 200 误通过。
这仅验证主机 CLI/HTTP 行为，HTTP 测试中的 PD 历史值为不透明测试值，不证明 Engram/PD 运行时正确。

相关主机回归 **20 个测试方法全部通过**，JUnit 包含 20 个 testcase 元素及 20 个
通过的 unittest subtest 计数，见 [host-tests.xml](host-tests.xml)。
命令、退出码、Ruff、格式及规划/上游保护结果见 [checks.json](checks.json)。
A5 用户更新代码后继续运行 `full --native-control --run`；实际生成输出与清理仍待回传验收。
