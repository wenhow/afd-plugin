# C4/C5 稳定模块的实际依赖 CPU 回归

本批最终一次广范围回归 `integration-012`：**2609 passed、2 skipped、4 deselected，43.57 秒，exit 0**。两个 skip 均为 `test_deepseek_v2_proxy.py` 的 fused top-k 需要 CUDA；四个设备用例由 `-m 'not npu and not gpu'` 排除。范围为 `tests/unit/model_executor`、`v1/worker`、`connectors`、`distributed`，加 feature/config/runtime validation 与唯一 signatures table；没有启动 NPU probe、安装依赖或修改上游。原始 [summary](integration-012/results/summary.json)、[stdout](integration-012/results/cpu.stdout.log)、[JUnit](integration-012/results/cpu.xml)、[identity](integration-012/identity.json) 与 [delivery](integration-012/delivery.json) 均保留。

本次与修复定向 `service-fix-011` 使用相同稳定源码 manifest `92b905874fc2fc6f47080010ec7a100b17d3a5efa1961794ac57664fffa82b30`，HEAD 为 `45356b2ff0f81e1d3fcf9b660026bdc904738a19`。snapshot 是工作区实际字节，不能只用 HEAD 代表新增模块。现场复核全部 manifest 文件仍匹配；已安装的 10 个锁定上游文件匹配，FastAPI 保持 0.123.10。

明确未包含尚在开发的 coordinator，以下三路径从 tar/manifest 与测试 collection 排除，后续单独验证：

- `afd_plugin/v1/worker/npu/hccl_v41_graph_coordinator.py`
- `tests/unit/v1/worker/npu/test_hccl_v41_graph_coordinator.py`
- `tests/unit/fakes/hccl_v41_graph_coordinator.py`

本次已覆盖最新 geometry/coverage 54、cache 15、controller 39、F service 29、stage bank 29、F graph 23、A graph owner 12、runner 70、A execution 22 等模块；精确集合以 JUnit 为准。geometry 源码为 `38994c5b...`，cache 为 `82a8c52c...`，controller 公共 `key_for` 版本为 `918c6a03...`。CPU 结果不代替完整模型或设备 graph 验收。

新增服务定向首轮 `service-010` 为 **129 passed、19 failed，3.21 秒**；[原始 stdout](service-010/results/cpu.stdout.log)、[JUnit](service-010/results/cpu.xml) 和 [identity](service-010/identity.json) 完整保留。U1 preparation 14、runner 70、A execution 22、signature 13 全部通过；F graph service 为 10 passed/19 failed，均在 `receive_forward` 的 ACCEPT 握手处报 `ValueError: expected ACCEPT`，尚未据此声称服务通过。manifest 为 `84831f3470395f7368111bf42e39119e54707ce64836cca179df2245a6827076`。未启动设备验证，已交模块 owner 核查。

owner 确认独立 scripted peer 漏发全局 ACCEPT，仅修 fake 的 `queue_parent`，生产 service 与测试字节未变。新快照 `service-fix-011` 只重跑该文件：**29 passed、0 failed/error/skipped，1.96 秒**，见 [stdout](service-fix-011/results/cpu.stdout.log)、[JUnit](service-fix-011/results/cpu.xml)、[identity](service-fix-011/identity.json)。manifest 为 `92b905874fc2fc6f47080010ec7a100b17d3a5efa1961794ac57664fffa82b30`；service `6a4dcd6c...` / test `7ff41a85...` / fake `a7901699...`。该次开始显式将尚未冻结的 coordinator 生产模块、镜像测试与独立 fake 三路径排除出 tar/manifest，并对测试 collection 加 `--ignore`；完整路径记录在 identity/summary 的 `snapshot_exclusions` 中。

北京时间 2026-11-10，最终 `control-003` 在现有 A3 venv 执行七个模块：**519 passed，0 failed/error/skipped，4.63 秒，exit 0**。包含五个 C5 模块的 319 项，以及验证 SM-control factory 重构的两模块 200 项。只执行 CPU module seams，未启动 NPU probe、未占设备、未安装依赖、未修改上游。

| 模块 | JUnit 实际 case 数 |
|---|---:|
| `tests/unit/distributed/test_hccl_graph.py` | 40 |
| `tests/unit/distributed/test_hccl_plan.py` | 135 |
| `tests/unit/connectors/npu/test_hccl_staged_transport.py` | 84 |
| `tests/unit/connectors/npu/test_hccl_graph_cache.py` | 14 |
| `tests/unit/v1/worker/npu/test_hccl_v41_graph.py` | 46 |
| `tests/unit/v1/worker/test_npu_model_runner_v2.py` | 156 |
| `tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py` | 44 |

最终原始输出与精确命令：[summary.json](control-003/results/summary.json)、[stdout](control-003/results/cpu.stdout.log)、[stderr](control-003/results/cpu.stderr.log)、[JUnit](control-003/results/cpu.xml)。版本和实际环境另见 `results/versions.json` / `effective-env.json`。10 个已锁定上游文件哈希均一致，见 `results/source-check.json`。

快照 HEAD 为 `45356b2ff0f81e1d3fcf9b660026bdc904738a19`，包含当时工作区全部生产与 unit Python 字节，完整 [identity.json](control-003/identity.json) 的 canonical manifest SHA256 为 `c812931ee16be1418c234403f45fb2e702071d770b9010045dfb333765c9b690`。不能仅用 HEAD 代表这些新增工作区模块。

| 生产模块 | SHA256 |
|---|---|
| `distributed/hccl_graph.py` | `111e6f5ca34280ec3e1339f8b57e8d42dbc17edca11099b1e06ef0d1de50de3b` |
| `distributed/hccl_plan.py` | `fa4b08b0179d992732d1a5f7243ae635d27c5992281c04ce2abe84eaa6041186` |
| `connectors/npu/hccl_staged_transport.py` | `a8ce468558a82b577e2fad8119030aa8dea33008bc621618b50d8c717ccc9dd3` |
| `connectors/npu/hccl_graph_cache.py` | `14364ea20d398ece9218aa0e052144f49af75f5514b376a8cc2556f55995c88c` |
| `v1/worker/npu/hccl_v41_graph.py` | `bb8a73e4d1ea46dc012af7980cc4a5edcad1f915e7b9c62d6942b47c09da9777` |
| `compat/patches/npu/mrv2_ubatching.py` | `1e57de63190849a2d5e51f9d2dc9e33326878c1d54e1db1f86ef88c0e0400773` |

历史原始快照均保留：`control-001` 首轮 317 passed/2.42 秒（transport f728，尚无两条 STOP/count reservation 误序修复）；`control-002` 修复后五模块 319 passed/2.46 秒；`control-003` 加上 factory seams 后为上述 519 passed。三次均未运行设备。

后续真实 manager probe 暴露 endpoint 的独立 AFD/native rank domain 问题，修复后另跑 `endpoint-004 --endpoint-seams`：**38 passed，0 failed/error/skipped，1.70 秒**（endpoint 24 + cache 14），没有重复上述 519 项。该次 [identity.json](endpoint-004/identity.json) manifest 为 `c79d6754e7028122f3c12732042864a04f2cb5d6c1a89b115f1965255a5b8f2e`；endpoint 为 `95a9ae01aef550fd0dd5967aa35458455c74c898ff9909316d89003bedf34ab7`，新增 cache `contains` 接口为 `07e45259d4c0ec034f4c7b21e9dcab8aec26eaab8377b328cfa5ec6839e60020`。同时收进 transport docstring-only 更新 `666e8ffab77ab15f9b5a9d1e9c004b2450349c3465fa2fb9e1c6a3e733d391df`；与 519 项原快照分别记录。

新增 consumer/controller 与 scoped native dispatch 在 `consumer-005 --consumer-seams` 定向执行：**129 passed，0 failed/error/skipped，3.80 秒**。分项为 controller 36、geometry 46、dispatch 23、唯一 upstream signature table 10、cache 14；不是上述 519 项的重复大组。其 [identity.json](consumer-005/identity.json) manifest 为 `c0487f0a9f8fc9ee3a0720e1bdabcb0196a97f835c050cd25fa5a2ba64b651db`，生产 controller `1680e2922c0e3f1d23a6b060c34ac81b35ca15a875a8832b3e6385ef77b49955`、dispatch `4b8800f54dd6cde11a7c1bbf21dc12b8ac24dd70272bbbd4f0f7654693255e0c`。该结果仍为实际依赖 CPU seams，不是完整模型 graph hook 验收。

controller 归属校验与 stale handle 拒绝补齐后，最终 `consumer-006 --consumer-seams` 为 **132 passed，0 failed/error/skipped，3.85 秒**：controller 39，其余分项不变。其 [identity.json](consumer-006/identity.json) manifest 为 `f8a7751d54417617e84a9283a37b9d202323c06626c0233b72caa85dacb8cf23`，controller 为 `d961c7ae0ce2ebde9ca80d488b74ee56052843d0e74bc1682404151db792ed54`，transport 新只读 property 版本为 `9ae40a90dab184fc4f1d17f934520318c2bd50eea75b5d082ed284e9a4e79128`。这两个生产文件与 `a3-native-manager-20261109/native-manager-006` 设备快照字节一致，分别保留 CPU 和 NPU 原始结果。

新 stage bank / F graph / A tensor body / A execution / runner seams 首轮 `model-seams-007` 为 **187 passed、11 failed，2.48 秒**，原始失败完整保留，不能记为通过。10 项失败来自 `fakes/hccl_v41_graph_execution.py` 的重复 `batch` keyword，导致 execution fixture 构造即报错；另 1 项为 runner 测试期望 int64、实际原生 input_ids 派生输出 int32。各 owner 修复后再取新快照，禁止覆盖 007。该次 [identity.json](model-seams-007/identity.json) manifest 为 `8b7e9804c3db62d3382b41de6e167483dbf5392e4da5e3ae7f114bace9b972c1`；分项 bank 29、F 23、A model 19、signature 11、control 39 全部通过，runner 为 66 passed/1 failed，A execution 为 10 failed。仅 CPU，未占设备或修改依赖。

修复两个测试 fixture 后，`model-seams-008` 定向执行受影响的 runner 67 / A execution 10，加上新 capture patch 11 / graph session 19 / signature 12：**119 passed、0 failed/error/skipped，3.36 秒**。没有重跑 007 已通过的 bank/F/A model/control。该次 manifest 为 `f44d23b580a9db95f2911b26e91a84fd68852531adddee8aae6ffbf5afad861a`，capture patch 为 `b708402c9bce56c921141352c40d485e7a5f6a8843237b5ab54e971e7514beb9`，session 为 `98e816a495b47f49b46b641b42fec1f69fa11378862ce0646feab097b2a0a5fc`。

008 取快照后，A execution 新增 generation/metadata stale launch 校验，单独 `owner-009` 仅跑该模块，**12 passed、0 failed/error/skipped，1.96 秒**，manifest 为 `f09b162a5739b8b2a75655515c39499206098a001ff98004e671db41ce493c31`；execution 最终字节为 `c021fe036f8083a9bcbd7e17d21bcebbff36eeb46373035d0056a66e88be2f10`。此 12 项仍是 CPU seams，未把它当作模型/NPU graph 验收；本次与 stage-bank 设备任务独立并行，后续按设备采样串行安排。

CPU fake/control peers 只证明协议和模块行为，不作为 HCCL graph capture/replay、实际设备缓存或完整模型推理验收；root 的真实 native manager probe 独立记录。本证据未重复 C4 的 2242 项大回归或其正负设备矩阵。

使用新的 label 复现，不覆盖原始结果：

```powershell
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-control-cpu-20261109/run_a3_graph_cpu.py --label control-new --factory-seams
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-control-cpu-20261109/run_a3_graph_cpu.py --label endpoint-new --endpoint-seams
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-control-cpu-20261109/run_a3_graph_cpu.py --label consumer-new --consumer-seams
```

uploader 沿用现有 CANN/ATB 环境并保留原有 `PYTHONPATH`/`LD_LIBRARY_PATH`，只在前方加入新快照；pytest 以 `-m 'not npu and not gpu'` 执行默认五模块，`--factory-seams` 追加上述两模块，`--endpoint-seams` 单独选 endpoint/cache 两模块，`--consumer-seams` 选 consumer 五模块；`--targets` 可显式指定 `tests/unit` 下的其他定向路径。每次新建远端目录、按 manifest 校验上传字节、校验已安装上游文件后运行并回收原始日志。uploader 的 Ruff check/format check 通过。
