# A5 原生 vendor 前置后的生命周期通过（2026-11-18）

用户上传的两个 ZIP 以原文件名原样保存。`a095f8a7ad73413a9fa98abd3eceaf49.zip`
包含受控配置、加载参数、生命周期、六轮 hidden 及十八次路由输入；
`e19a3975e09d411a838711d3b1e7c245.zip`包含 summary、Worker/预检报告、日志及进程记录。
SHA 和独立核对见 [assessment.json](assessment.json)。占用观察按用户要求排除，原始文件保留。

## 已通过范围

此次运行使用 `3f2651d` 交付的冻结包，SHA 为
`d0a32a80a5737f04b968e176cc61f7afa3af444658ecdfc93f6f937b359da521`，
189 份源码、146 份生产模块；预检 194 份原生 Python 源码全部匹配，
对应 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 和
vLLM-Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`。
包版本字符串仍按原报告保留，不以包名中的开发版本号推断源码身份。

实际执行在 `npu:0`，原生 `NPUWorker` + V2 `NPUModelRunner`，eager/U1。
模型为三层 `[0,0,2]` C2、BF16 dummy 参数：H1024、64 专家、top-k2、
64 个 Attention 头、32 个 Indexer 头且维度 128、HC4、8 MiB KV 预算；Engram 关闭。
没有修改原生数学、替换算子或导入 AFD 模块。

- 正常加载、KV 初始化、warmup、实际执行和采样全部返回。
- 共六轮执行、十八次路由调用；每次输入捕获完整、有限且算子返回成功。
- A/B 两轮采样后取消 A，B 继续到长度上限；随后接入 C 并正常结束。
- Scheduler 清空、Worker 请求退役，新请求复用了已退役的请求槽位。
  本场景不要求证明物理 KV block 复用。
- Worker shutdown、Ascend/model/distributed 组销毁全部通过，无清理异常。
  原生进程退出码 0，无超时、强杀或所属残留进程；预检进程也正常退出。

| 请求 | 实际输出 token IDs | 结束方式 |
| --- | --- | --- |
| functional-A | 239, 14 | 两轮后取消 |
| functional-B | 1140, 382, 3962, 2736 | 长度上限 |
| functional-C | 1605, 3825 | 长度上限 |

六份 hidden 文件的长度和 SHA 与每轮报告匹配，加载记录和 lifecycle-progress
与最终报告一致，十八份输入/metadata 与最终 capture 记录逐项匹配。
这些 ID 是本次原生受控基线的观察结果，不作为后续 AFD 输出的硬编码预期。

## 导入时序调整的设备证据

`operator_environment_preparation.extension_already_imported=false`，
已安装的 `/opt/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer`
在扩展导入前排在首位，原外部 Transformer 与 FLA 目录保留。
首次路由输入 PT 的 SHA 与上轮原生失败及独立重放使用的输入完全相同：
`6bf7ca17a89d2e6dbcbc1c9878b567a5e2a3dca2357d1be2a84eaad37e93d6e0`。
这次该输入在原生上下文也执行成功，随后十八次调用及请求生命周期均通过，
验证了前置 vendor 准备后的这条原生诊断路径可以运行。

成功进程中仍加载 `opp/vendors/customize/op_api/lib/libcust_opapi.so`。
符号观察表显示已安装 custom_transformer 和 customize 库都导出两个 hash router API；
因此不能将先前故障简化为“只要加载 customize 库就会失败”，也不据此卸载或屏蔽它。
`dlsym/dladdr` 的导出来源仍不是 C++ 静态缓存指针的直接观测。
搜索表导入时序是有源码依据、且调整后设备运行通过的解释；未进一步证明故障库内部的细节。

## 验收边界和下一步

原始 summary 为 `success=true`、`diagnostic_only=true`、
`npu_acceptance=diagnostic_only`，只有 `native-eager-u1` 一个场景。
本次关闭该原生诊断路径的启动、首轮和必要清理阻塞；完整七场景仍待执行。
原生/AFD 权重与采样等价、AFD U2/Graph/多流、DSpark/PD 组合、完整训练检查点、
Engram、物理重叠及性能均没有由此次运行验收，原始标志保持不变。

下一步按[集中指引](../../../../../docs/npu/DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md)运行
完整七场景命令，使用全新目录；移除诊断参数，回传整个输出目录。

可在 CPU 主机复核本次原始产物，不导入 torch/vLLM 或设备运行时：

```powershell
python -B openspec/changes/afd-hccl-full-parity/evidence/a5-native-vendor-first-pass-20261118/assess_archives.py
```
