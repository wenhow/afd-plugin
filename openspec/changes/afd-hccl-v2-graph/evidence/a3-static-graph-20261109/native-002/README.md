# 最终 A3 静态通信图组件（2026-11-09）

结论：**最终 bytes 的 23 项 CPU seam 测试与两卡 native NPUGraph 组件通过**。
独立审查见 [组件审查记录](../../graph-component-independent-review-20261109.md)。
本结果不代表 native V2 manager 或 V4.1 全模型 Graph 已接入。

本轮新增 `hccl_graph.py`，图外将 world peer 转成 group coordinate，注册实际
`npu_define._send/_recv`，固定 BF16 / 原始 FP32 / int32 IDs / BF16 output 四 buffers。
图内收发保留静态 shape，recv 原位写回。无 JSON/Gloo、IDs.item、host cursor、
buffer allocation 或 COMPLETE。调用者拥有 process groups；组件不销毁它们。

相对首轮，两项独立审查意见已修复：四 payload 连续字节区间不得重叠；缓存 graph
与 exchanges/valid-counts/row-indices 一起保留到同步并 reset。构造器只检验同 device
及 storage/layout；分配入口要求 indexed NPU。CPU 手工 tensors 仅在单测截取外部
operator 的 seam 使用，不能执行真实 HCCL，不使用伪 NPU tensor 或测试开关。

| 源码 | 最终 SHA256 |
|---|---|
| `afd_plugin/connectors/npu/hccl_graph.py` | `b98e5f0f8c115a2a1f6e5c1ddc25e4f532425eb02649a0d3d1c08602d99b4d37` |
| `tests/unit/connectors/npu/test_hccl_graph.py` | `a03e7f7b8635e683b567a9d40cd0648b9b7c3913f90f508cb00bb3bfb104f165` |
| `tests/unit/fakes/hccl_graph.py` | `7fd357da41dd902933c91c9e8954d691084a256ee493187b3337bf1ea55b7775` |
| `tests/npu/hccl_graph_probe.py` | `691d1bd3ed6e318f1f3d82f109b08b193a15bf78d80dab4f412e1a2610e24b35` |

命令：`python openspec/changes/afd-hccl-v2-graph/evidence/run_a3_graph.py --label native-002 --devices 0,2`。
snapshot 位于 `/mnt/workspace/tmp/afd-c0-20261104.hVgsIf/c5-graph-20261109-5v1u42s2`。
Python 3.12 现有 venv，torch 2.10.0、torch-npu 2.10.0.post4、CANN 9.0.1，
vLLM 0.30.0 / Ascend `7e37e29a`；未安装/修改包或上游。上传 manifest 与实装
source-expected 在执行前逐项比较；本轮 manifest 四 hash 与最终工作树一致。

实际结果：

- CPU：23 tests，0 failures/errors/skips，exit0；新增 overlap 拒绝用例。
- rank0 FFN / NPU0，rank1 Attention / NPU2。双方在通信前用 Gloo 核对完整 schedule。
- U1 `(4,)`、U2 `(3,2)`、含空 stage `(0,2)` 各捕获一次、各 replay 三次；
  全空 `(0,)` 双方一致跳过，不构造零行 exchange。
- replay 每次原位改变 BF16、FP32 独立增量、IDs 和有效行；F 收到的 FP32/IDs
  与 CPU 来源精确相等，双端 output 与 CPU 算术参考精确相等，padding 输出为零。
- 所有 replay 的四 payload buffer 地址保持不变；graph reset 后仍可在调用者 Gloo
  domain barrier，最后正常销毁调用者 groups。
- 两进程 exit `[0,0]`，完整探针 25.82 秒；前后八板无运行进程。

raw 位于 `results/`：`cpu.log/xml`、`native/rank-{0,1}.log/json`、
`native/summary.json`、`upstream-identity.json`、`effective-env.json`、
`npu-before.log`、`npu-after.log`、`run-summary.json`。本地静态 Ruff check/format
及 Python 3.11 语法检查通过；本地缺 torch 的 pytest skip 不计入上述 23 项。

成本：每 stage 的四持久 payload buffers 为 `rows*(8*hidden_size+4)` bytes，另有
graph pool 及本探针的 valid-count/index tensors；recv graph op 返回后有一次 `copy_`。
保留 FP32 独立通道。未测 cold capture HBM 或吞吐/延迟；上述秒数不是 benchmark。

仍待实现/验收：native manager fresh warmup/capture 与图外 phase/key/availability 共识、
host cursor/COMPLETE、双方 miss eager、真实 MoE/weights、Engram producer/events、
shared attention state、Dynamo/GE，以及 A5。没有开放生产 Graph/model guard，
没有修改 C4 协议或把通信组件结果提升为模型验收。
