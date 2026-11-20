# 原生 TP4 首波请求失败：缺少已存在的 query-head 适配

输入 `34fc5dae2eb34a73bfb0fb722fa329f8.gz`，SHA256 为
`16ebc80ef9af01d8e78bc61a7f336245abb8c8864e6c2a217f2d26cc05a1c622`。
离线审阅 A5 归档，未连接 A5 或启动新的设备推理。17 份运行文件保持原始字节，见
[review.json](review.json)；编辑器 `.attention-server.log.swp` 仅保留在输入包/本地解包目录。
归档 PID/资源是当次快照，不代表当前设备状态。

运行目录 `full-eager-u1-yqxpk_w6`，HEAD `1dcd2f0`、受检 tracked source 无改动；
Python 3.11.10、卡 0–3、TP4、原始量化 checkpoint、CPU Engram、DSpark/PD 关闭。
25 份预检源码匹配，packed Attention 四项 A5 算子 probe 通过。

| 2026-11-20（Asia/Shanghai） | 原始证据 |
| --- | --- |
| 09:27:13–09:27:58 | packed probe exit 0，资源检查通过，原生服务启动 |
| 09:30:52 | 应用启动完成，curl exit 0、`GET /health` HTTP 200；lifecycle 记录 ready |
| 09:30:54 | `request-00/01` 均返回 HTTP 500；首个 Worker 异常为 `N1 must be 64 (the only supported query-head count), got 16` |
| 09:30:54–09:31:13 | 启动器捕获 HTTP 500，向本轮 API 发 SIGINT；API exit 0 后对残留 owned descendants 做 SIGTERM；最终 survivors 为空、八卡 idle |

错误调用链为原生 V2 ModelRunner → `DeepseekV41Attention` → `AscendDSAV41Impl`
→ 原生 `MixedQuantPackedCacheOps.qsmla` → `cann_ops_transformer` DSL wrapper
→ `ops/mixed_quant_sparse_flash_mla.py::_validate_inputs`。
HTTP 500 和 `EngineDeadError` 是该 worker 异常的传播结果。本次健康和生成请求已直连到服务，
不是前一次健康检查等待问题。`request_passed=false`、`clean_shutdown=false`，
即便最终资源释放也不能作为成功参考；后续请求、取消及连续性矩阵未执行。

## 架构审查与最小修复

已有 `AFDMixedQuantPackedCacheOps` 只被 `AFDDeepseekV41Attention` 选择。
probe 显式测试该 adapter，因此通过；本次 native 控制仍直接使用未适配 backend，
TP4 的本地 16 query heads 被只接受 64 heads 的 wheel 拒绝。

在插件内新增 `DSV41NativeTP4ControlWorker(NPUWorker)`，仅由显式 native TP4 对照
命令选择。其唯一 override 是 `load_model(self) -> None`，与锁定 Ascend
`7e37e29a` 的 `worker/worker.py:837` 签名（参数/default/注解/返回）相同。
先检查 TP4，再委托原生载权，随后遍历 model modules，只对 native
`DeepseekV41Attention` 且 backend 恰为 `MixedQuantPackedCacheOps` 的实例选择
已有 adapter。其他 backend 不覆盖；找不到目标时明确报错，避免假装适配成功。
worker 不继承 AFD runner/worker，不更改上游全局类或函数；原生模型、参数名、MoE、
Engram committed history、shared KV/index、HC/pre_mix、调度/采样和 shutdown 都沿用原生。

该 seam 位于原生载权结束、profiling/首次 forward 之前；父方法的 sleep allocator 与
weight transfer 保留，直接委托比复制不相关资源所有权更合适。override 有紧邻原因、
功能、delegation exception、PATCH 标记、成本和移除条件说明。
新增 `worker/worker.py` SHA256 预检，完整入口现检查 26 份 native 文件。
未修改 vLLM、Ascend 或 donor 工作区。

native 和 AFD 计划都记录 `attention_head_adapter=tp4-query-16-to-64-v1`，参考准入要求
标记一致；原生对照的 worker-class 参数也可审阅。**这是带共同 query-head 兼容层的
原生对照，不是完全未修改上游路径的 TP4 支持证明。** 不通过更改 TP/EP/DP、关闭 Engram、
替换权重或删除算子校验消除错误。

新增成本只有载权末的一次 module 遍历；运行时沿用既有 padding：16 heads 补到 64、
sink 补零、取前 16 输出，KV/index/metadata 不复制，原始输入不改写。独立 head 的
数学及组件证据见 [模块说明](../../../../../docs/design/module/dsv41_packed_cache.md)。
attention head 计算为原来的四倍，仍须设备性能验收。wheel 原生支持 16 heads 后移除
该 worker、adapter 及控制标签。默认 native Worker 与 GPU/A3 入口不选择此类。

## CPU 与交付边界

在此前授权的 A3 隔离临时目录、既有 venv 中检查真实依赖，只运行 CPU 张量/模块选择契约。
冻结快照来自 `1dcd2f0` 加三份明确源码增量，排除本地暂停的旧 MTP 与其他脏文件。
四份 native 文件哈希匹配；NPU `_lazy_init` 被禁止，前后 initialized 均为 false。
新增控制 worker 五项与原有 head adapter 十四项 **19/19 通过**。
包括仅修改本实例 backend、其他 backend 不覆盖、原生载权错误保留及非 TP4 拒绝；
既有 CPU headwise reference 检查数学/输入状态，不模拟 packed NPU kernel。
详见 `cpu-003/` 和 [checks.json](checks.json) 的命令、源码 manifest 与结果。

主机 CLI/HTTP 回归 **21 个测试方法通过**，覆盖 native worker-class、adapter 标记及不匹配
参考拒绝；见 [host-tests.xml](host-tests.xml)。这些不证明真实模型输出、DSpark/Graph/PD 或性能。
历史尝试和扩大范围时的问题分别保留在 `cpu-001/002/004-baseline/`，不混入通过计数。
001 因测试环境构造时覆盖了 CANN 的 PYTHONPATH，缺少 acl 而未收集；保留原路径后
002 的新增五项及 adapter 十四项均通过，但额外纳入的旧 attention worker 契约中
3 项失败、4 项 fixture error。004 用完全没有本轮改动的 `1dcd2f0` 快照复验，
同名七项以相同错误重现：旧 fixture 路径触发 CPU 禁止的 NPU 初始化等行为。
这七项保留为旧测试独立执行的未解决问题，不归因于本轮 worker，也不计作通过；
本次聚焦的新 worker/adapter 19 项在 003 中全部通过。
规划、Ruff、格式、signature/source 复核与上游保护见 checks；A5 完整模型验收保持 pending。

用户更新代码后先重新运行 `full --native-control --run`，无需清除代理。
日志应出现 `Native TP4 control: installed 16-to-64 query-head adapter ...`。
只在本轮 `summary.json` 的 request_passed 和 clean_shutdown 均为 true 时再作为 AFD 参考。
首 token 差异、完整组合与正常清理仍须实际 A5 回传确认，任务 2.2/1.8 不提前完成。
