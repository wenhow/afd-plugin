# eager/U1 输出差异与串行重复性诊断

用户继续定位真实量化 checkpoint 的 eager/U1 输出差异，A5 由用户手工执行。
本轮仅审阅已回传证据、修改插件仓库内诊断工具/指导，并运行主机协议检查；未连接或启动 A5。

## 输入与结论

`6e06dda0ce0e4544bb6f383b9b75b004.gz` 的 SHA256 为
`ed578e580820a56c08baa7b47da4b22d2807e2f2ad985349f04768a858aeaf52`。
运行目录 `full-eager-u1-pqdgncjt`，HEAD `d3a6f27`，26 份受检上游源码匹配。
原始量化 safetensors、A4/F4 TP4、A 侧原生 CPU Engram、eager/U1、DSpark/PD 关闭。
两次 HTTP 200 均生成 32 token，输入 token 与原生完全相同，输出分别从第 1 和第 13 个
token 分歧，不同位置分别为 32/32、13/32。严格对照失败，不能记为完整模型通过。
A/F 均退出 0，F rank 0–3 peer STOP/清理齐全，无强制清理/残留，最终八卡 idle。
`EngineDeadError` 在主动清理后出现，不是本次输出差异的触发点。

参考确实对应上一上传 `3e76c4e71f12464b86e5b9a87470329a.zip` 中的
原生 `full-eager-u1-7z0p69ts` request-00/01。但该原生相同请求再次执行时，输入文件
逐字节相同、输入 token 相同，request-00→02 首 token 分歧、32/32 不同，
request-01→03 第 13 个 token 分歧、17/32 不同。中间有流式取消；该顺序不能单独证明
取消导致历史泄漏，也不能从当前结果唯一归因 AFD/Engram/HCCL/量化算子。
本次 AFD 第一轮后失败退出，未执行取消、第二轮及长输入，其他完整模式没有新增验收。
输入归档、逐文件哈希及结构化对照见 [received-results.json](received-results.json)。

## 本轮工具修改与边界

`run_dsv41_full_checkpoint.py --repeatability` 只允许无 DSpark/PD 的 eager/U1。
原生与 AFD 同时限制 `max-num-seqs=1`，客户端串行；预热一轮，取消前/后各三轮，
每轮两个短 prompt，共 14 个非流式响应。保留相同温度、seed、32 token、量化/Engram。
首个取消前实测是同 prompt 的本次基准；预热差异单列，不参与稳定性通过条件。
AFD 另与原生对应轮次比较。输入/输出均要求精确一致，不按文本相似放宽。

新增标准库模块 `dsv41_repeatability.py` 调用既有直连 HTTP 和清理入口。
纯 token 差异不会截断诊断采集，全部请求完成后仍抛错；HTTP/响应结构失败立即保留
原始响应和未完成报告。常规矩阵的严格逐请求失败行为不改。参考套件/参数/稳定性
在启动 AFD 之前检查，旧双请求结果不能作为串行参考，反向也不混用。
客户端关闭流不证明引擎已完成取消；实际行为仍需服务日志、后续请求和清理证据。

默认不请求 logprobs，复用此前已成功生成 token 的采样分支。
可选 `--repeat-logprobs 5` 两侧均须使用，并另建参考；要求有限、完整、以 token ID
标识的 top-5 raw logprobs，保存首分歧步两侧候选与 top1−top2 差距，不静默降级。
锁定 vLLM `ced6857a` 的 `gpu/sample/sampler.py` 调用 `compute_topk_scores`，
而锁定 Ascend `7e37e29a` 的 `patch_v2/patch_triton.py` 仍绑定旧
`compute_topk_logprobs`；未实测该分数采集分支，不能据此断言其设备失败。
本轮没有增加采样器/上游补丁。logprobs 不是原始 logits，也不据分数接近认定差异可忽略。

工具始终输出 `diagnostic_only=true`、`hardware_acceptance=false`、
`model_equivalence_accepted=false`。串行通过只代表本协议重复性及给定参考对照通过，
不替代双请求/取消复用/长输入矩阵、完整准确率或其他 HCCL 组合验收。

## 验证与后续

主机检查覆盖协议计划、参考拒绝、默认 token-only 和可选分数响应、预热差异记录、
取消前/后差异聚合、AFD 与稳定原生不一致、缺失 logprobs 保留失败证据，以及既有
curl/代理/HTTP/监督契约。合成 HTTP 响应只验证工具协议，不代表 NPU 数值行为。
31 个测试函数（含参数子项）通过，0 失败/错误/跳过；JUnit 的 tests 属性计入子项为 65。
结果见 [validation.json](validation.json)、[host-tests.xml](host-tests.xml)。
Ruff、格式、diff 检查及本地 harness 的规划/上游只读保护通过；后者不是官方 OpenSpec
验证或模型验收。未改 vLLM、Ascend、donor，未修改 Engram 或运行时代码，原任务勾选不变。

用户先按[指导](../../../../../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)
执行 `full --native-control --repeatability --run`。原生失败时先回传该归档；原生稳定且
正常退出后，再执行相同协议的 AFD。若 AFD 仍有差异，再按分歧步收集分数或层间输出。
