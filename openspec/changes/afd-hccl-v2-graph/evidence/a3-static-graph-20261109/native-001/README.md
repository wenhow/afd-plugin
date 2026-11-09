# 首轮 A3 静态通信图组件（2026-11-09）

结论：首轮 bytes 的 **22 项 CPU seam 测试、两卡 native NPUGraph 组件通过**。
独立审查后补充了 payload overlap 检查与 graph/input bundle 生命周期；最终 bytes
请使用 [native-002](../native-002/README.md)。本目录保留原始首轮结果，不覆盖重写。

- 命令：`python openspec/changes/afd-hccl-v2-graph/evidence/run_a3_graph.py --label native-001 --devices 0,2`。
- 不可变 snapshot：`/mnt/workspace/tmp/afd-c0-20261104.hVgsIf/c5-graph-20261109-hw43n453`。
- rank0 FFN / `npu:0`，rank1 Attention / `npu:2`；调用者拥有 Gloo 与两个 stage HCCL groups。
- torch 2.10.0 / torch-npu 2.10.0.post4 / CANN 9.0.1，Ascend lock `7e37e29a`；
  安装文件 hash 在执行前与锁定/只读采集记录核对，详见 identity 与 upstream-identity。
- CPU XML：22 tests，0 failures/errors/skips。测试截取外部 operators，使用 CPU tensor
  验证布局、FP32/IDs payload、group-relative coordinate、角色和错误更新边界，不模拟设备。
- 两进程实际 `torch.npu.graph`：U1 rows `(4,)`、U2 `(3,2)`、含空 stage `(0,2)`，
  各自 warmup/capture 后 replay 三次；全空 `(0,)` 双方事先同意跳过。
- 每次 replay 原位改 FP32 增量、IDs、valid 行数；真实返回值与 CPU 参考精确相等，
  F 收到的 FP32/IDs 精确相等，四 payload buffer 地址不变。
- 双进程 exit `[0,0]`，完整探针 22.62 秒；前后 `npu-smi` 都显示八板无运行进程。

原始证据：`identity.json`、`results/cpu.xml`、`results/cpu.log`、
`results/native/rank-{0,1}.log/json`、`results/native/summary.json`、
`results/npu-before.log`、`results/npu-after.log`。四个测试源码 hash 均保留在 manifest。

范围严格为 **显式、同步、静态 shape、受控算术的通信图组件**。没有 weights/MoE/
V4.1 attention/Engram、native V2 manager、phase/key 协商、fallback、GE 或 Dynamo tracing。
真实组仅含 global ranks `[0,1]`，CPU `(10,20)` 坐标检查不等于实机偏移组验收。
耗时含进程启动/初始化，不是性能 benchmark。生产 Graph/model guard 未开放。
