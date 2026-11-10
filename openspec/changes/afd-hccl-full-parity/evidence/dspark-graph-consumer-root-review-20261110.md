# 固定 DSpark target 查询宽度：消费者复核

2026-11-10。本次审查覆盖 candidate 中 8 个运行文件和 13 个镜像/共享夹具，
包括后续 target/draft 隔离用例以及 DP 调用夹具修复。复核者直接检查当前差异、
原生外层 capture 流程及原始 XML；未改上游，也未开放完整模型资格。

## 运行边界

协商宽度是 session 常量。A 原始配置、native decode_query_len、不可变 F
target_shape、FULL descriptor、cache key 与 bank 均须一致。FULL 使用
T=R*q；原生 U2 沿 token midpoint 切片，不要求每个 child 的行数整除 q。
真实短 batch 根据完整 request/token 布局选 replay 或前置 eager，不能只检查 T。
F 不恢复 speculative/Engram/KV owner；采样和请求历史继续由原生 A 持有。

原生 GPUModelRunner.capture_model 的 target → speculator.capture 顺序实际执行。
四个 CPU 用例覆盖 U1/U2 及草稿错误。真实 DFlash NONE-manager capture 之后，
用例额外执行真实 _run_model 验证 context 隔离；这一步不声称原生 NONE capture
本身执行了模型 forward。可观察断言覆盖草稿独立 metadata、没有 target sidecar、
没有额外 HCCL 消息、target 输出的对象/地址/值、aux、cache 可用性和失败资源保留。
设备 memory/workspace/factory 是明确的 CPU 边界，不能据此推断设备 Graph 正确。

本批没有新增上游 monkeypatch，也没有扩大既有 patch 的签名。宽度检查是 host
标量检查；capture/preview 沿现有 ordered request counts 投影，成本随请求数增长。
这些检查不进入模型 tensor 算子体；原有所有权核查成本仍存在，不宣称零开销。
未来原生 HCCL/模型适配提供相同固定宽度与 owner 接口时，可移除 plugin 的映射层。

## 独立核对结果

geometry-001：九个完整模块，551 passed / 4 failed / 0 error / 0 skipped。
四个 native target/draft 隔离用例均通过。失败是共享夹具把 q 作为第 5 个位置参数
传给原生 dp_size，q1 掩盖了这一错误；生产 DP1 guard 正确拒绝 q2/q3。

geometry-002：仅修复共享夹具，以真实 parallel_config 的 dp_size/dp_rank 显式
传参，完整 coordinator 模块 100 passed / 0 failed / 0 error / 0 skipped。
两个快照的 8 个运行文件逐字节一致，21 项 installed source checks 均匹配，
source_files_match=true；未启动 device probe。两次运行独立保留，不合并成
一次“555 全通过”。最终源码和证据 SHA 由交付清单另行冻结。

此外，H12/H13 审计中的 31 个锁定 Git 文件对象已在本地重新取出并核对 SHA256。
该审计说明旧 MTP 与 PD 的真实缺口，不构成对应功能实现或设备验收。

正常 connector factory 尚未自动选择和组装 Graph；本批是显式组件接口。真实
DSpark proposal、接受/拒绝、native HC/Engram/MoE、KV/history、完整 checkpoint、
A5、Graph PD 和性能仍各自待验证。q3 受控 NPU 探针另有独立证据，不能混入本批
CPU 结果。结论：本批接口和 CPU 行为可交付，C7 的完整组合任务继续 pending。
