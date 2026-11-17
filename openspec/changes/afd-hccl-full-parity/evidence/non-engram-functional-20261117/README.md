# 非 Engram 功能交付：2026-11-17

本轮完成受控正常 Worker 组合资格入口，并保留已有跨层 PD、FFN daemon 和管理 RPC
异常归属修复。A5 执行使用 [统一指导](../../../../../docs/npu/DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md)，
默认七个顺序场景。**本目录是源码/CPU证据，A5设备验收待执行。**

最终 [汇总](validation-summary.json) 核对：提交清单373路径与真实CPU运行源码逐字节相同，
其中146生产模块与A5冻结包也逐字节相同。A5源码包SHA256：
`87391afb4559f73a07ddf2211f17e31d8dba40b3d0e6efbaeeb7bb01f7ca51c0`。

| 检查 | 实际结果 | 范围 |
| --- | --- | --- |
| Windows完整unit | 1805P / 698S / 0F / 0E | 无Torch/vLLM等依赖，跳过不计通过 |
| A3现有真实依赖venv CPU | 873P / 0S / 0F / 0E | feature/runtime/FFN/engine/profiler/API/signature seams；NPU初始化前后False |
| 首次EngineArgs | 原生1、A4、F4全部返回0 | eager/Graph、U1/U2；NPU初始化前后False |
| A5七场景metadata | 通过 | stdlib、不导入runtime，不代表设备执行 |
| 源码审查 | 无阻断 | 默认拒绝/边界/F隔离/跨层PD/daemon/RPC；锁定原生签名 |

真实CPU记录为 `native-cpu-*`；首构造记录为 `engine-args-*`；本地完整unit为
`unit-004.*`。`native-cpu-source.tar.gz` 含完整CPU快照和逐文件manifest；
`native-cpu-results.tar.gz` 保留原始日志、JUnit和NPU状态。
执行没有安装/修改共享包，也没有启动NPU推理。A3已有metadata设备失败保持原结论，
本轮不追加设备定位，不把环境差异当作已确认根因。

历史冻结005中已有未交付的11个ordinary-MTP签名项：本地因依赖缺失跳过，真实依赖CPU
发现它们引用本轮明确排除的生产模块。本轮仅从交付快照删除这些签名项和唯一对应分支，
保留工作区暂停开发文件，未用skip掩盖。差异及原件SHA见
[裁剪记录](paused-mtp-signature-pruning.json)。此前完整单测检查还发现历史生产/测试
异常归属修复未同步，最终闭包保留已完成的修复；没有新增profiler设备场景。

新增资格入口仍默认关闭，只开放明确的BF16/dummy三层C2/TP1/最多两请求候选。
旧资格入口、Engram边界与默认拒绝保留。当前P中途抢占按terminal处理，不新增恢复协议。
正常Worker验证、DSpark/PD受控target验证和完整训练检查点精度严格区分，不能相互替代。
