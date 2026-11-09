# Native parent 接入独立复核（2026-11-09）

Root 对 Attention 实现者提交的实际 diff 与锁定 Git 对象进行了复核。
受控范围为 text、eager、DP1/PP1/CP1、non-SP、A=F=native TP、spec off。
没有发现该软件范围内尚未修正的静态阻断；模型门禁保持关闭，结论不包括
完整 checkpoint、原生 HC/indexer/MoE、Engram 历史或性能验收。

来源为 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 与 Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`。核对了
`gpu/model_runner.py` 的 execute/dummy 调用顺序、`gpu/ubatch_utils.py` 的
prepare/slice/child context，以及 `deepseek_v41/vl_model.py` 的 wrapper forward。
上游 checkout 保持只读。

新增 wrapper 与 native forward 的参数和返回注解一致。审查要求把短小的
non-AFD 主体直接复制为 `self.language_model(...)`，已经修正，AFD 分支有
PATCH 标记。scoped prepare 保留精确 native 签名；其委托例外是保留已选择的
HCCL/NPU 实例 override，而非复制 base 后绕过后端切分和 eager 校验。
它进入现有唯一签名对齐表。native execute 的大型 request/KV/采样生命周期仍由
父类执行，原生返回 `None` 的语义没有改变。

U2 报告来自实际 prepare 返回的 token slices，U1 padding 来自 parent descriptor。
在 parent context 产生时，所有 A 提交各自 valid/padded counts 并接受同一计划；
KV maintenance 不产生 parent，child context 不再次提交。U1 保留真实模块 pre-hook，
消费已经生产的 Engram kwargs；U2 跳过外层模块调用，在 parent context 消费一次
pending producer。两条路径共用 layer-major stage engine，原生 Ascend 仍唯一负责
`finish_execution`。CPU seam 不冒充真实 request/KV 或 Engram 算子验证。

正常 U2 配置由 backend selector 选取 staged adapter，并绑定它自己的唯一 transport。
default U1/C3 DSpark 保留 v2；配置 U2 的 profile/低阈值 U1 仍在同一 v3 session。
不能把一个 FULL U1 descriptor 仅改 U2 标签用于未来 Graph；该限制已交 C5 跟踪。
绑定核对实际设备、native rank、模型维度/完整层集、容量；HELLO 后再次核对 U2
协商容量。错误 transport 身份绑定已补拒绝。

正常退出顺序为 all-A STOP/ACK、native cleanup、关闭唯一绑定 transport。
active/terminal session 或 native cleanup 失败时保留通信资源交 worker supervision，
不在阻塞 HCCL 下提前销毁 group。native TP 组仍属于原生 owner。

最终实际依赖 CPU 快照见
[integration-cpu-009](a3-forward-counts-20261109/integration-cpu-009/results/summary.json)：
2242 passed、2 skipped、4 deselected；所有 C4 生产文件与该快照一致。该快照还含
并行 C5 初版的 22 项 CPU case，C5 最终修正另由自己的 23 项和 native-002 验证。
旧 Factory 夹具迁移失败及修正保留在原始记录中。
真实 HCCL 服务组件的 1A1F、2A2F/TP2 连续 U1/U2/empty/STOP 结果见
[A3 证据](a3-forward-counts-20261109/README.md)。它们使用受控算术，不构成
Attention 模型前向或 checkpoint 验收。

新增成本为 bounded host sidecar、一次本地 counts 回合及已有全 rank ACCEPT/COMPLETE；
未新增 native request/KV buffer 或 TP group。eager 路径仍同步执行并在 COMPLETE 前
同步设备，未声称 overlap 收益。上游提供 prepared-state/model-call callback 后可移除
scoped prepare、context provider 与 wrapper sidecar；Engram 的原生退休所有权应保留。
