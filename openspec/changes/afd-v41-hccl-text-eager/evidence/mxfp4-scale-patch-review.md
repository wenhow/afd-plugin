# DS MXFP4 MoE 奇数 scale 配对修正审查

日期：2026-11-09。范围：用户第三次 A5 TP8 启动失败的量化 post-load 修正。
本次审查在本地完成；A5 FP4/NZ/GMM、完整模型与 AFD 对照均未因此通过。

## 根异常与源码身份

原始回传及 hash 记录在 [diagnosis.json](a5-20261109-native-startup-003/diagnosis.json)。
`native-server.log:452–481` 为最早 worker 根 traceback，W2 scale 有
17,694,720 个元素，即 `(384, 5120, 9)`；上游尝试 reshape 为
`(384, 5120, 4, 2)`。Engram 参数已经解析，48/48 权重分片已读取，
后续 EngineCore 初始化失败和清理信息属于 worker 失败的传播。
已有 MXFP8 Linear 探针 6 项通过，不能证明 routed MoE 的 FP4 路径。

目标为 `vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py` 中
`AscendW4A8MXFPDSDynamicFusedMoEMethod` 的两个方法：

| 方法 | 锁定行号 | 完整签名 | 变化 |
| --- | ---: | --- | --- |
| post-load | 584 | `process_weights_after_loading(self, layer)`，无返回标注 | W13/W2 scale 配对前补齐奇数尾列 |
| EPLB 单专家 scale | 561 | 静态 `_process_scale_2d(scale_2d)`，无返回标注 | 与 packed tensor 使用相同规则 |

锁定 SHA `7e37e29a7803d6897a7113088e2b771f83205bc0` 与回传实际 SHA
`2db8549c548731611baeb169fa1ef13aa55fc6aa` 的该文件逐字节相同。
两处补丁签名已与锁定 Git 对象进行 AST 比较，参数与返回标注均一致；
设备环境上的动态 signature 检查加入既有唯一签名表，当前本机缺少 Ascend 而跳过。
回传 vLLM 包版本为 `0.30.0+empty`，未提供可核实的安装源码 SHA。

## 所有权与最小切入点

错误归属量化 scheme 的权重加载后处理；无需改变 MoE 路由、模型切分或 loader。
补丁复制锁定 post-load 骨架，保留原生 FP4 NZ 转换、W13/W2 transpose、
dynamic EPLB 分支和 expert-list 构造。差异有紧邻函数的原因/功能/签名说明及 PATCH 标记。
无 `_original_*` 委托，无新增全局可变状态。

共享 helper 只处理 E8M0/uint8 scale bytes，保留所有有效值，在奇数尾列补零；
逻辑 K、权重、TP 归约、HC/pre_mix、Engram 和共享 KV 的所有权不变。
W2 K=288 时输出 `(E, 5, N, 2)`，与
[官方 GMM MX 配对布局](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)
的 ceil(K/64) 规则一致；当前 wheel 的 FP4 GMM 支持仍待设备探针确认。

现有 `afd_dsv41_mxfp8` entry point 由原来的单项 Linear 修正扩展至两项量化修正，
仍需 `AFD_DSV41_MXFP8_SCALE_PATCH=1`。入口名称与环境变量沿用，避免已有启动命令漂移。
默认发现但未启用时不导入 NPU patch，不注册 AFD 模型/worker，不放开总 guard。

## 开销、回归与删除条件

只在加载时为奇数列分配新 scale buffer；偶数列使用原 view/reshape/transpose。
E=384、N=5120 的 W2 每层多存 1,966,080 byte，即 1.875 MiB。
补齐期间旧 buffer 与新 buffer 暂时共存；不增加 forward 通信、同步或算子调用。
不承诺此前容量估算已经覆盖全部格式转换和加载峰值。

`test_mxfp_scales.py` 检查 2D/3D、奇/偶列、E8M0/byte 的真实 CPU 张量输出。
`test_mxfp4.py` 检查 packed/list 两条 post-load 输出；NZ 边界为 fake，
不会将这些测试作为设备证据。签名检查继续使用既有表。
`dsv41_mxfp4_moe_probe.py` 使用真实 NPU post-load 与 FP4 grouped matmul，
覆盖 2 个 experts、W13/W2、intermediate=256/288/320、两种 scale_alg，共 12 项。
它检查全 K 与最后一组输出，不覆盖 routing、SwiGLU、EPLB 或 TP/Engram。

本地检查结果另见 [local-validation.json](a5-20261109-native-startup-003/local-validation.json)。
A5 探针通过后仍需完整 checkpoint 冷加载、HBM/KV、首请求和数值验收。
删除条件：锁定上游的 DS MoE packed/list 两处均支持奇数 scale 配对，
且本地数值回归、A5 FP4 组件和完整模型验收通过后，移除该复制补丁及其 installer 调用。
