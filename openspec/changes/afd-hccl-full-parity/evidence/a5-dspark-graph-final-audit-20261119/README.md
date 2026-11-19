# A5 Graph/U2：驱动完成标记已出现，最终失败尚缺内层报告

输入原件为用户提供的 `9d66ef97f9f748a89aeebd68d84b303d.zip`，保存为
`initial-input.zip`。`assess_report.py` 重建 `assessment.json`，不修改原始报告。

## 已确认与缺失证据

- 顶层 runtime_source_sha256 为
  `c576f79af0732bfcedc3deab35d8e0edac34a0f56e4475a0e61a5193d2212379`，
  匹配 51a0954；runtime preflight 成功。
- 顶层 reject 场景 functional_verification.accepted=true，process returncode=0，
  四角色 exit=0。该场景仍通过；不因后一个场景失败而更改其结果。
- Graph run_id `9ead09c9335949019bf137a913ff0774`；P/D 日志包含初始两请求和
  替换请求的第三次 KV 传输；P sender socket owner joined。提供的四份角色日志
  均未出现前轮计数 AssertionError 或 corrupted double-linked list。
- `consumer-verified.json` 已生成。对应的冻结 lifecycle driver 在 owner/shutdown
  和 receiver retirement 返回、decode/取消替换驱动的内部检查通过之后写入它，
  因而当前执行已越过之前第一轮 eager 计数断言。但它早于外层 observer 的最终
  检查和 supervisor 的双端协议审计，不能代替场景整体验收或最终进程退出证据。
- 提供的 ffn_producer.json success=true、native_daemon_retired=true；其余角色
  最终 JSON 未提供，不能把该单角色状态外推为四角色 exit=0。
- 顶层 Graph process returncode=1、无超时/强制终止/存活子进程。顶层 traceback
  只有入口主动抛出的 `DSpark/PD scene failed: graph-u2-cancel-replace`，缺少实际
  内层异常。ZIP 的 summary 位于顶层，不是场景内部 supervisor summary。
- 缺少场景内 `summary.json`、kv_consumer.json、kv_producer.json 和
  ffn_consumer.json。已向用户请求这些文件路径，无需再跑设备；未据此猜测或修改
  runtime/协议校验。当前完整 Graph/U2 结果保持失败，根因待内层报告。

## 独立补齐顶层失败诊断

`dsv41_a5_functional_validation.py` 原来在 process success=false 时立即抛错，
不会把内层 summary 的实际 traceback/角色退出码带到顶层。新增
`failure_summary`：保留内层路径、存在状态、原始字节 SHA、success、traceback、
returncodes、forced_killed_pids、timed_out 和内嵌角色 success/traceback。
省略大体积库清单；缺失/截断/非 UTF-8 的 summary 只记录读取诊断。

该读取只发生在原失败分支，不写入源报告、不将错误改为成功，也不重试设备。
普通成功分支、原有功能校验、Graph observer、146 份生产源码及 194 个原生 pins
保持。此提交补充诊断，不声称解决当前尚未提供的内层失败。

## 本地检查

- `summary-red.json`：新增两个标准库检查因旧入口缺少失败摘要 API 失败。
- 最终 `validation.json`：原有入口/观察器加失败摘要检查，共 **23 项通过**；
  Ruff、186 份 Python 编译、194 source pins、确定性重建、本地规划及上游保护
  检查通过。无新 A3/A5 推理或 NPU 初始化。
- `source-verification.json`：与 51a0954 冻结包相比，只改 launcher 和 manifest；
  146 份生产文件、全部 Graph/DSpark helper、native identity 字节保持。
- tasks 仅登记当前证据和待补报告，整项验收未勾选；保留其他本地修改。

当前需要的是既有运行目录中的
`results/dsv41-a5-dspark-stage-observer-20261119/dspark-pd-graph-u2-cancel-replace/summary.json`。
若该内层 summary 尚未载入 workers，再根据它指向的失败角色补对应 JSON。
无需为了新诊断入口重新执行本次场景。
