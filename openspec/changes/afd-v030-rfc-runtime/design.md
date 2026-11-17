# v0.30 与 RFC 运行时契约对齐：技术设计

状态：C0 运行时契约与原生最小初始化完成，见 [最终记录](evidence/lifecycle-review.md)；完整模型数值/硬件验收留后续 C1/C2。下文保留设计决策及其边界。
源码标号见 [锁定阅读索引](../../../.agents/dsv41/sources.lock.json)。
## 源码事实与阅读入口

参照 sources.lock.json 中 A01/A02/A03/A07、U01/U02/U05/U06/U07。
当前 vLLM 与目标都是 ced6857a；Ascend 从 8d4409d6 到 7e37e29a 才是主要运行时变化。
首次实现前还须完成方法级差异表，不能把本设计当作补丁审查已经通过。

## 决策

- 以当前 V2 worker/runner 为骨架；按目标上游函数重新复制并应用 AFD 标记，不引入旧 V1 runner。
- 保留原始 DeepseekV41ForCausalLM 架构元数据。检查注册选择、ModelConfig 缓存和 loader 的 model_class 注入契约，保持三者一致。
- F 初始化在通信组和模型构造前隔离 Engram；检查通用与 Ascend 两个初始化入口。只创建 F 角色所需原生 MoE。
- F constructor 已在 super 之前创建顶层配置浅拷贝并清空 Engram 绑定，保留原始 ModelConfig；不克隆其他配置子对象。配置边界见 [边界审查](evidence/ffn-role-review.md)。独立 F-only 容器已通过真实 native CPU 构造和 A3 单逻辑 NPU 缩小配置构造，参数与模块清单见 [构造审查](evidence/ffn-construction-review.md)；后续已接入 F loader hook，完整设备组/引擎初始化仍待验收。
- 优先原生 loader 加角色模型类，避免重写全部权重格式与量化后处理。若接口不足，只在 afd-plugin 增加有源码依据的小补丁。
- loader 方案已确定并接入 NPU runtime：保留原生 Ascend 注册和 ModelConfig 身份，在插件的 create_model 小补丁中向 initialize_model 显式注入 F 类；非 AFD/普通模型/非 NPU 保留 native 解析。接口/缓存审查见 [loader 决策](evidence/loader-identity-review.md)，受控小配置完整 dummy 加载见 [接线记录](evidence/loader-wiring-review.md)。A 角色与完整推理尚未实现，运行拒绝门保持有效。
- 不修改上游目录、版本文件或依赖声明。按用户 2026-11-06 决定，沿用容器 FastAPI 0.123.10 继续开发和运行验证；声明冲突不再作为阻塞，保留历史 solver 结果并分别记录实际运行证据。
- 精确模型能力表代替 DeepseekV4 子串判断，因为该子串也命中 DeepseekV41。

## 补丁审查与退出

每个补丁记录目标 SHA/符号/签名、原因、PATCH START/END、性能影响、非 AFD 行为、上游化或移除条件。
初期使用 eager/U1，新增逻辑只对精确 V4.1 + HCCL 路径生效；非 AFD 和已有 connector 维持现有行为。
回退通过撤销本 change 的 afd-plugin diff 完成，不更改上游锁来掩盖问题。

## 验证

角色构造前的进程组成员和分配清单、真实模型选择与权重格式后处理、非 AFD 注册不变、CPU 测试可收集。
NPU 初始化/内存和实际环境安装另存硬件/环境证据。

## 2026-11-09 完整迁移范围修订

本 change 保留原有实现/验收范围和进度。最终目标是 [HCCL 全能力迁移](../../HCCL_CAPABILITY_MIGRATION_ZH.md)，
新增 C4 microbatch/拓扑、C5 Graph、C6 多流、C7 DSpark/PD 组合均为本项目必交付。
新增任务见 [tasks.md](tasks.md) 第 2 节；本次不把既有通过扩大为完整迁移通过。
