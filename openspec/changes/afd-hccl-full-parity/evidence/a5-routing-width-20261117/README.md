# A5 同步栈定位 MoE 路由及受控专家数修正

输入为用户上传的`c4416cd61d6b4c05bd80ee69c0ea2662.zip`日志，以及
`8066337ffd29444893f2ad8b8c5ffd4b.zip`中的summary/native-runtime。
原字节以同名`.gz`保存，输入SHA见`inputs.json`。本轮父版本为`50b5702`。

`ASCEND_LAUNCH_BLOCKING=1`已生效，首轮原生forward的同步栈直接停在
`AscendFusedTopKRouter._compute_routing -> _C_ascend.moe_gating_top_k_hash`，
报507035/341、VEC访问UB越界。实际gate及FP32 gate为`[8,1024]`，
模型加载、Attention KV和公开warmup完成，但推理未完成、无采样结果；
Worker shutdown失败，native groups已销毁，最终八卡空闲且无残留进程。
预检194份Python源码匹配；原生扩展和vendor库的实际身份在原始native-runtime中保留。

## 源码判断与修改

检查锁定vLLM-Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`的
`csrc/moe/moe_gating_top_k_hash/op_kernel/arch35/`与对应tiling：

- `moe_gating_top_k_hash_tiling_arch35.h:243`将每组专家数补齐到32。
- `moe_gating_top_k_hash_regbase.h:1508`等按补齐数分配bias/FP32等UB缓冲区。
- `common.h:143`的FP32加载和`moe_gating_top_k_hash_regbase.h:1108`的
  排序结果加载不接收元素掩码；后续计算的掩码不能缩小已发出的读取。
- 官方[950向量寄存器定义](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/C-API/reg/reg_data_definition.md)
  给出256字节，即64个FP32元素。8专家补齐后的32元素缓冲仅128字节；
  64专家使该缓冲达到完整向量宽度。

这是静态分析得到的小形状风险，并非设备级根因证明。A5实际编译扩展的C++源码来源
未核对；不能把Python预检通过当成该C++源码与已安装二进制一致。
记录的文件Git blob/SHA及风险计算见`source-analysis.json`。
本次只将普通Worker受控配置在首次EngineArgs声明前统一设为64专家，记录并断言
实际专家数和top-k=2；原生/A/F共同使用。functional资格增加固定64候选，
保留严格整数检查、原8专家及其他限制。历史dummy资格继续拒绝64专家。
独立受控DSpark组件及其8专家draft元数据不变；未替换路由函数、评分公式或算子，
未修改任何上游工作区。

## 检查结果

- `run_cpu_validation.py red`：新测试对父版本准入为36失败/436通过，失败均为
  新增64专家的正向配置；进程正常记录pytest退出1，NPU初始化前后均False。
- 同一测试GREEN：472通过，无失败/错误/跳过；九组真实EngineArgs配置
  （native eager/U1及A/F各eager/Graph、U1/U2）全部成功，实际64专家、top-k=2、
  H1024和Index32/D128，NPU初始化前后均False。
- 本地及A3各六项stdlib CLI测试通过，包含单原生同步诊断与默认七场景metadata。
  A3 CLI测试未导入torch/NPU/vLLM/插件运行时。
- 185份Python编译、194份原生Git身份及Ruff、规划/上游保护检查通过。
  与`parent-bundle`相比，188份源码中的功能probe及资格校验变化；
  145份其他生产模块和全部原生身份保持原字节。锁/默认包已同步。

CPU结果分别在`cpu-red/results.tar.gz`、`cpu-green/results.tar.gz`；
CLI结果在`cpu-cli/`，复现使用上一轮已提交的`run_cpu_cli_validation.py`。
命令、退出码、审计结果见`local-validation.json`与`checks/`。

A5需先更新分支、使用新目录保持`--diagnose-native`重跑，再根据原生结果继续七场景。
本轮没有运行NPU算子，不宣称越界已消除、采样/原生AFD等价或任何完整模型验收。
