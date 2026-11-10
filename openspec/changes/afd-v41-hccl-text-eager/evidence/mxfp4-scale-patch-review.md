# DS MXFP4 MoE scale 配对与 K 对齐审查

日期：2026-11-10。范围：第三/四次 A5 TP8 启动及后续 MoE 探针暴露的量化形状问题。
本次审查在本地完成；补 K 对齐后的 12 项 A5 GMM 探针已通过，完整模型又在
融合 SwiGLU 的 576-wide 输入校验失败。新增 gate/up 对齐、完整模型与 AFD 对照仍待验收。

## 根异常与源码身份

原始回传及 hash 记录在 [diagnosis.json](a5-20261109-native-startup-003/diagnosis.json)。
`native-server.log:452–481` 为最早 worker 根 traceback，W2 scale 有
17,694,720 个元素，即 `(384, 5120, 9)`；上游尝试 reshape 为
`(384, 5120, 4, 2)`。Engram 参数已经解析，48/48 权重分片已读取，
后续 EngineCore 初始化失败和清理信息属于 worker 失败的传播。
已有 MXFP8 Linear 探针 6 项通过，不能证明 routed MoE 的 FP4 路径。

后续 [MoE 探针](a5-20261109-mxfp4-probe-001/probe.json) 在 K=256 的 W13/W2
及 intermediate=288 的 W13 共 6 项误差为 0；W2 K=288 第一项报
`161002/EZ0009`，要求 `K == 64 * perTokenScale.shape[1]`。
scale 的 ceil 配对并不证明该 WeightNz kernel 能直接接受原 K=288。
新 [诊断](a5-20261109-mxfp4-probe-001/diagnosis.json) 同时核实回传源文件 hash
与锁定 Git 对象一致；本次日志未记录 afd-plugin commit SHA。

目标为 `vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py` 中
`AscendW4A8MXFPDSDynamicFusedMoEMethod` 的四个方法，其中 GMM1/2 继承自原生基类：

| 方法 | 锁定行号 | 完整签名 | 变化 |
| --- | ---: | --- | --- |
| post-load | 584 | `process_weights_after_loading(self, layer)`，无返回标注 | W13/W2 scale 配对，W2 在 NZ 转换前补齐物理 K |
| EPLB 单专家 scale | 561 | 静态 `_process_scale_2d(scale_2d)`，无返回标注 | 与 packed tensor 使用相同规则 |
| 原生 GMM2，继承自非 DS 基类 | 499 | `apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale)`，无返回标注 | 量化激活补零到物理 K，保留 dtype、scale 和分组参数 |
| 原生融合 GMM1/activation，继承自非 DS 基类 | 398 | `apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput)`，无返回标注 | 仅在融合 SwiGLU 前分别补齐 gate/up 两半 |

锁定 SHA `7e37e29a7803d6897a7113088e2b771f83205bc0` 与回传实际 SHA
`2db8549c548731611baeb169fa1ef13aa55fc6aa` 的该文件逐字节相同。
四处补丁签名已与锁定 Git 对象进行 AST 比较，参数与返回标注均一致；
设备环境上的动态 signature 检查加入既有唯一签名表，当前本机缺少 Ascend 而跳过。
回传 vLLM 包版本为 `0.30.0+empty`，未提供可核实的安装源码 SHA。

## 所有权与最小切入点

错误归属量化 scheme 的权重后处理和 GMM2 消费边界；无需改变路由、模型切分或 loader。
补丁复制锁定 post-load 骨架，保留原生 FP4 NZ 转换、W13/W2 transpose、
dynamic EPLB 分支和 expert-list 构造。差异有紧邻函数的原因/功能/签名说明及 PATCH 标记。
无 `_original_*` 委托，无新增全局可变状态。

共享 helper 保留有效 E8M0/uint8 scale bytes，在奇数尾列补零。
W2 K=288 时输出 `(E, 5, N, 2)`，与
[官方 GMM MX 配对布局](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)
的 ceil(K/64) 元数据规则一致；实际 WeightNz 的 K 等式由此次目标日志确认。

W2 在原生 post-load、packed/list 分支及 NZ 转换之前，将物理 K 从 288 补零到 320。
W13 仍产生逻辑 intermediate=288；原生 `apply_moe_mlp` 的 fused/unfused activation
分支最终均调用 `quant_method.apply_gmm2`，这里补齐量化 FP8 激活。
新增的输入 scale 空位设为 E8M0 单位值，避免 NaN 空位乘零产生 NaN；有效字节不改，
scale copy 强制 contiguous，确保对未使用空位的写入作用于返回 buffer。
GMM2 除新增一个 alignment 赋值外，其余函数体 AST 与锁定上游完全一致。
逻辑 K、checkpoint、TP 归约、HC/pre_mix、Engram 和共享 KV 的所有权不变。

第四次回传已确认八个 rank 完成 post-load，profile 的 fused SwiGLU 输入为
`2*288=576`，而 MX 模式的 C++ 输出构造要求其能被 128 整除。
该约束在锁定 `csrc/torch_binding.cpp:1843` 已存在。
新 hook 在原生 GMM1 和分组计算之后、SwiGLU 调用之前，将每半各补 32 个零，
得到 `gate[320]+up[320]`。原有 BF16 有效字节、clamp、group_index、SITU 分支均保留；
禁止对拼接后的 tensor 简单尾部补零，那会打乱有效配对。
SwiGLU 输出直接为物理 K=320，与现有 W2 匹配，无需更改逻辑 intermediate 或 checkpoint。
移除唯一新增的 helper 赋值后，复制函数的完整 AST 与锁定上游一致。
本轮 Ascend 实际源码 HEAD 为 `4ec6dc77…`、包元数据仍为 `…g2db8549c5`；
本地没有前者对象，不能把此前两个旧提交的相同文件结论推广到新提交。
此次针对日志和锁定原生合同修复，目标新探针仍是独立验收步骤。

现有 `afd_dsv41_mxfp8` entry point 由原来的单项 Linear 修正扩展至两项量化修正，
仍需 `AFD_DSV41_MXFP8_SCALE_PATCH=1`。入口名称与环境变量沿用，避免已有启动命令漂移。
默认发现但未启用时不导入 NPU patch，不注册 AFD 模型/worker，不放开总 guard。

## 开销、回归与删除条件

E=384、N=5120 时，先前 scale 补齐每层增加 1.875 MiB。
W2 K 从 288 补齐至 320 再增加 384×5120×16 byte，即每层 30 MiB。
加载时旧/新 buffer 暂时共存。每次不对齐的 GMM2 还分配 FP8 激活和 scale copy，
执行 padding 和空位初始化；64 对齐的输入直接保留原 tensors。
没有新增 host 同步或分布式通信，但不能声称没有 forward 开销。
融合 SwiGLU 的未对齐输入新增一次 BF16 padding/copy，每 token 64 个元素，
1024 token 的新增零值为 128 KiB；完整新 buffer 为 1024×640×2=1.25 MiB，
分配期间旧 1.125 MiB buffer 仍可能存活。已有 W2 的 30 MiB/层不再次增加。
不承诺此前容量估算已经覆盖全部格式转换和加载峰值。

`test_mxfp_scales.py` 检查 2D/3D、奇/偶列、E8M0/byte 的真实 CPU 张量输出，
另覆盖 FP4/FP8 零填充、非连续 scale、空 token 行及独立反量化点积等价。
`test_mxfp4.py` 检查 packed/list 两条 post-load 输出；NZ 边界为 fake，
不会将这些测试作为设备证据。签名检查继续使用既有表。
`dsv41_mxfp4_moe_probe.py` 使用真实 NPU post-load 与 FP4 grouped matmul，
覆盖 2 个 experts、W13/W2、intermediate=256/288/320、两种 scale_alg，共 12 项。
新版 W2 调用生产的 `scheme.apply_gmm2`，记录逻辑/物理 K、hook 文件 hash 和失败 case。
它检查全 K 与最后一组输出，不覆盖 routing、SwiGLU、EPLB、MegaMoE 或 TP/Engram。

新增 `dsv41_mxfp4_activation_probe.py` 真实调用生产的 GMM1/SwiGLU 和 GMM2，
以两专家、三种宽度、clamp=0/6、count/cumulative 分组构成 12 项，
通过不同的 gate/up 值和最后一组 scale 检查配对与裁剪，再与独立 CPU 数值比较。
这新增的设备探针尚未在 A5 执行；CPU 的 gate/up 配对、空行/strided tensor 和
SwiGLU 算术测试已通过。启动器默认先执行新探针，失败时不再加载完整模型；
`--probe-only` 可单独收集新组件证据。控制台只打印目录，stdout/stderr 全部写文件。

最新本地检查结果见 [local-validation.json](a5-20261110-native-startup-004/local-validation.json)，
先前静态证据保留在第三次启动目录。
A5 探针通过后仍需完整 checkpoint 冷加载、HBM/KV、首请求和数值验收。
删除条件：锁定上游的 DS MoE packed/list 均支持奇数 scale 配对和 W2/激活 K 对齐，
或目标内核明确支持未对齐的原 K，同时支持未对齐的融合 SwiGLU 输入；
且本地数值回归、A5 FP4 组件和完整模型验收通过后，移除该复制补丁及其 installer 调用。
