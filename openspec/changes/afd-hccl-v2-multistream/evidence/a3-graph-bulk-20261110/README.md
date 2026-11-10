# Graph bulk_v1：CPU 回归与设备组件验证

本目录包含实际依赖 CPU 回归、`--prepare-only` artifact 和两个独立受监督的
TP1 bulk Graph NPU 组件会话。生产 guard 不变，不声称原生 V4.1 模型、实际
通信重叠或性能通过。

| 快照 | 结果 | 范围 |
| --- | --- | --- |
| graph-bulk-019 | SSH 初次建 remote 目录前 exit255；无测试执行 | 仅创建本地空 label，harness 拒绝同 label 重用；简短记录见 connection-failure.json |
| graph-bulk-019a | 552 passed / 1 failed / 0 skipped，14 warnings，7.49s | 七个真实依赖镜像，包含默认 serial 全回归；source_files_match=true |
| graph-bulk-019b | 64 passed / 0 failed / 0 skipped，14 warnings，2.80s | 仅 service 镜像，修正 oversized-peer 夹具后定向复跑；source_files_match=true |
| bulk-prepare-001 | 139 archive members / 138 manifest entries / 31 source pins | 本地准备，不 SSH、不使用设备；probe/harness 的显式 `--exchange-order bulk_v1` |
| bulk-tp1-001 | 双 rank exit0 / 31 pins match / 8 idle → 8 idle / 无 timeout、强杀 | 单 session、devices 0,2、axes11111；每 rank 4 warmup、4 capture、12 replay、6 eager fallback |
| graph-owned-output-024 | 114 passed / 0 failed / 0 skipped，14 warnings，0.87s | bank 41 + F Graph 73；在原 compute scope 内提前固定 stage 输出的独立 CPU 快照，source_files_match=true |
| bulk-owned-output-002 | 双 rank exit0 / 31 pins match / 8 idle → 8 idle / 无 timeout、强杀 | 固定 stage output copy 的独立 TP1 session，33.63s；每 rank 4 warmup、4 capture、12 replay、6 eager fallback |

019a 分模块为 codec 147、bank 31、pipeline 112、A Graph 55、F Graph 67、
coordinator 77 全通过；service 为 63 通过、一个旧 capacity 测试失败。
该旧测试在构造 service 后更改 negotiated capacity，与新增 session 能力固定
契约冲突，实际在读取 peer QUERY 前拒绝。019b 只改共享 fake 和该测试，改为
发送超出已协商容量的真实 key，验证预期的全 peer ERROR；生产源码没有为
夹具放宽。原失败 XML/log/identity 保留，不把两次回归拼成一次总通过。

每个实际回归保留 identity.json、CPU stdout/stderr、JUnit XML、summary.json、
installed source-check 与包版本；validation-summary.json 记录各模块结果、
完整 source manifest 和所有 raw 文件 SHA256。019a 的 coordinator 对应 bulk
冻结字节，随后其他并行增量不能从这次结果推导通过。

设备准备保留既有 native base-manager 探针的 4 warmup、4 capture、12 replay、
6 eager fallback、独立 BF16/原始 FP32/IDs oracle、bank 地址、nativeTP 与正常
STOP；bulk 分支通过新 bank/pipeline 公开 split API 排入相配的 parent HCCL。
root 独立审查并授权后，bulk-tp1-001 首次执行通过，elapsed 30.43s。该会话的
manifest 为 `7c5007bcc046f0266f27eff70f65da476a42d7e6e5aca11e454644a5fad965a2`，
两个逐 rank JSON 均记录 bulk_v1 和各四个 U2 warmup/capture pipeline 资源，
native TP 前后 `[1.0]`。parent 约束由每个 HCCL scope 的 actual current_stream
与 run-bound parent 比较强制执行；没有输出 numeric parent handle 或 timeline。

该结果验证受控算术下的跨 stage group bulk 排入、实际 NPUGraph capture/
replay、full/1/T-1 有效 token、padding-only child、固定 bank 地址/复用和正常
STOP。静态零行 stage 目前只有 CPU 覆盖；TP2 bulk、长运行 allocator 压力、
native V4.1 HC/Engram/MoE、完整 checkpoint、物理收发侧流及性能仍待验证。
设备 snapshot 含后续 coordinator 诊断字节，但本探针不构造该 coordinator，
不能据此宣称该并行增量通过。该旧标签没有设备重试。

后续只读 MoE 输出审计发现，通用 `Tensor` 返回接口没有承诺输出不借用可复写
workspace；Python 引用和 `record_stream` 不能代替值的独立归属。当前默认共享
专家的 functional add 路径未发现确定覆盖，但 deferred bulk model seam 已保守
修正为在 compute_done 前验证并复制到既有 stage output bank。serial 仍只在
发送时复制一次；未增加 buffer、collective 或 wire。详见
`../native-moe-output-ownership-audit-20261110.md`。

024 包含两次 native 调用复用同一 workspace 的完整双层输出回归，以及拒绝
广播 shape、隐式 dtype 转换、非连续布局和异设备输出。源/hash 见
`../graph-owned-output-final-snapshot-20261110.json`。probe 的 bulk 分支也已采用
同一 compute 内 copy scope；独立 `bulk-owned-output-002` 会话随后通过，
manifest 为 `17b6cb1d0c5c92d60679ccb89e62da357aaeb90f08180d99d6a82929614a50cd`。
两个 rank 均完成 26 个事务，保留四个 U2 pipeline 资源，native TP 前后均为
`[1.0]`；31 个安装源 pin 匹配，前后八板空闲，无 timeout、cleanup 或强杀。
第一次外层审批基础设施失败发生在创建进程之前；该标签只有一个实际设备
session，没有失败后设备重试。原始 summary/rank/source/address-oracle 范围
以及 raw 文件 SHA256 已分别写入 validation-summary.json。旧 `bulk-tp1-001`
保持原源快照，不升级为新源结果。

024 的共享 workspace 回归验证真实 CPU Tensor 的覆盖反例；002 的算术模型
验证新增 copy 在实际 NPU compute/capture/replay 中的位置。它们不证明原生
MXFP4 MoE 会复用 workspace，也不构成完整模型、Engram 或重叠性能验收。
