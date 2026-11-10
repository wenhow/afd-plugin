# A5 原生 EP8：服务已启动，首请求缺少 CompressorV2 ACLNN 接口

2026-11-10 回传 `b0a1f0ab6acf4ebc993ca3cc9342f698.gz`。
实际 AFD SHA 为 `36f52994217f4caeb4ae20d5b4783892da4b555d`，
Ascend HEAD 为 `4ec6dc7722ec4f7f8005a86570238ac4377c5d43`；
没有声称后者与项目锁定 RFC 全部相同。

本次 `/dev/shm` 已扩大为 512 GiB。8 个 rank 完成完整模型加载，
DP8/TP1/EP8、Engram CPU offload/shared memory、eager、DSpark off，
未启用 AFD 或 TP8 量化兼容插件。API 启动完成，health HTTP 200。

15:25:52，最早的 DP0 worker 经 `execute_dummy_batch` 进入原生 Attention
压缩器。Python `cann_ops_transformer` wrapper 能导入，但 ACLNN loader 报：

```text
aclnnCompressorV2 or aclnnCompressorV2GetWorkspaceSize not in libopapi.so,
or libopapi.so, or libopapi_transformer.sonot found.
```

该消息不能区分库文件未找到、缺少符号、依赖加载失败或搜索到旧库。
[响应](response.json)为 HTTP 500 EngineCore 错误；[请求](request.json)本身已进入执行。
[诊断记录](diagnosis.json)保存配置、版本、归因边界和压缩包 SHA256。
原始 `native-server.log` 与 `environment.txt` 留在本地同目录供核对；
根异常原日志行 1554。保留 archive，不复制截断的终端日志替代证据。

下一步运行 [只读 opapi 采集工具](../../../../../tools/diagnostics/collect_dsv41_a5_opapi.sh)，
核对真实库、符号和依赖，再决定环境修正；尚未执行该环境检查。
完整首请求、连续推理、模型精度与 AFD 对照仍未验收，任务 1.10 保持未完成。
