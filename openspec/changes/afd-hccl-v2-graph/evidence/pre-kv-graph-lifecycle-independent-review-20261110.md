# Pre-KV Graph 协议 eager 桥接独立审查

日期：2026-11-10。审查人为主开发代理，桥接实现由状态审查代理完成；未将桥接作者的自检当作独立审查。

## 审查对象与原生边界

- `afd_plugin/v1/worker/npu/hccl_v41_graph_eager.py`：SHA256 `cd0f31fdc74ec1e3acb23d17d790e221015f617fe95d8f2ff35c3fc7b83961c3`。
- `afd_plugin/compat/patches/npu/hccl_graph_dispatch.py`：SHA256 `a845dcc6490f9a4a58afedf95c6f6fbd19ff43445df9fd1de4079064d9969d7f`。
- 复用 `HcclV41GraphControl.reserve_batch/reserve_eager`、`HcclV41Execution.scope(manage_dispatch=False)` 和公共 `eager_hccl_v41_geometry`；不复制 request、KV、Engram history 或 native execute。
- 原生对照为 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 的 `vllm/v1/worker/gpu/dp_utils.py::dispatch_cg_and_sync_dp`，以及 Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0` 的 V2 execute 生命周期。正常 NPU 启动链见[独立源码审查](native-profile-lifecycle-review-20261110.md)。

## 发现与闭环

审查发现一个真实的前置顺序缺陷：dispatch scope 原先先调用 native dispatch，再调用 AFD selector。若真实 batch 已 gather、manager 仍为 None 且没有显式 `need_eager=True`，原生会在 selector 有机会强制 eager 前触发 manager 非空断言。

修复位于 dispatch scope：已确认传入 manager 是当前 owner 的 manager 后，将 manager 为 None 纳入 `effective_eager`，在原生分派前选择 NONE。真实 batch 仍通过原生 gather 产生；selector 继续先完成 all-peer eager reservation，再进入已有 eager preparation scope。没有在测试中强行传 `need_eager=True` 隐藏该缺陷。

桥接构造与执行入口绑定同一 owner、model、ModelState、config、parallel config、transport 和 cache；其 manager 必须为 None，cache 必须为空且有效。真实 KV 初始化出现 manager 后，桥接拒绝继续使用，由后续 coordinator 接管。NONE-state dummy 使用原生已有 descriptor，whole-zero early return 使用诚实的零 parent；两者都不构造虚假 BatchReqState。mode/valid-count 校验先于协议。

真实 batch 决策后才安装 eager scope，保证 `_cg_mode=NONE`、input/Engram producer 准备与 F 端所等 phase 一致。异常保留 cache/transport 的 terminal 状态及资源，不声称取消阻塞的 HCCL 或设备操作。Engram finish_execution 始终由原生 execute 的 finally 负责。

## 成本、移除与验证范围

桥接不分配 Graph stage bank。每个 parent 在已有 counts/COMPLETE 前增加图协议的模式共识；真实 batch 的 CPU 几何计算仍使用 native scheduled-token 数据，dummy 不另造请求向量。它依赖单 owning thread 的临时 gather/dispatch/provider 绑定，禁止并发嵌套。

当上游提供 input preparation 前的分派回调及原生 A/F 生命周期扩展点时，可移除该临时 scope，保留同一 wire 共识与唯一 native retirement 约束。

本地无 NumPy/torch 的主机只验证依赖 gate。A3 的实际依赖 CPU 回归按不可变快照分别记录：[016—019 记录](a3-control-cpu-20261109/README.md)。016 暴露测试夹具缺少原生 Worker 写入的 rank，其他六模块 421 项通过。017 中主入口 11 项与既有 eager execution 22 项通过；018 中桥接 21 项通过；019 中修正后的最后六个参数组合通过。后两轮修正了桥接镜像对于全局 GRAPH_ACK、单 A counts solicitation 的期望，以及恢复原生 dispatch 后的调用条件；最终断言只覆盖桥接负责的共识和 parent 完成边界。本批 C5 相关生产源码保持一致；019 通用快照另含并行开发的 C6 pipeline 候选，该六参数测试未验证它。保留首轮结果，不跨快照合计为一次全绿。

当前源码审查无未解决的架构阻断。该结论不代表 NPU model Graph、真实 Engram/KV 算子、完整 checkpoint、性能或生产准入已经验收。
