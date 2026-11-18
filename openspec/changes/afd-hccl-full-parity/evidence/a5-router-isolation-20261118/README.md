# 64专家复现失败与原生路由单算子隔离

输入为`59a4b91ce20c4c3aa755523e918e496e.zip`，只包含`native-eager-u1.log`。
原字节以gzip保留，ZIP SHA和关键字段在`failure-summary.json`。
父版本`d21332bd65718619c5684c65e7c70d8ce95c805f`。

## 已确认与未知

同步开关1、H1024、64专家/top-k2、Index32/D128均已生效。
模型加载、Attention KV和warmup返回，但首轮原生执行仍在
`fused_topk_router.py:204 -> _C_ascend.moe_gating_top_k_hash`报507035/341 UB越界。
相对kernel PC偏移仍为0xa860；没有完成推理或采样。Worker shutdown失败，
native groups已销毁。本次ZIP没有父summary或native-runtime，无法据此确认八卡最终空闲
或延用上一轮的194份实际安装源码匹配结论。

上一轮E64调整没有解决设备问题。继续核对锁定vLLM-Ascend `7e37e29a`后，发现
`op_kernel/moe_gating_top_k_hash.cpp`包含同目录的`moe_gating_top_k_hash_regbase.h`，
此前重点检查的`op_kernel/arch35/`同名实现不能直接代表该编译入口。
host在ASCEND950选择arch35 tiling，与kernel包含的文件路径是两个事实。
因此撤回将旧缓冲区推断当作故障解释的可能理解；保留它为历史静态风险记录。
实际已安装扩展、opapi和设备kernel的C++构建来源仍未知。
本轮源码blob/SHA记录在`source-analysis.json`；不修改上游。

## 交付

新增`tests/npu/dsv41_a5_router_probe.py`与集中入口`--diagnose-router`：

- 六个独立进程分别运行E64/K2/FP32、E384/K2/FP32、E384/K6/FP32、
  E64/K2/BF16、E384/K6/BF16及E384/K6/FP32+vision bias。每项8行固定种子输入；
  不启用Hash路由。最后一项是锁定原生测试形状/vision bias对照，不宣称视觉模型支持。
- 沿用正常A5 bootstrap和实际`torch.ops._C_ascend.moe_gating_top_k_hash`，
  不替换算子、路由公式、生产配置或资格限制。CPU参考遵循锁定原生
  `test_npu_moe_gating_top_k_hash.py`：bias用于选择ID，权重取未加bias的sqrtsoftplus评分，
  按选中项归一化并缩放1.5；比较精确ID及FP32/BF16权重容差。
- 输入、输入SHA、输出、失败阶段/同步栈及schema独立保存。
  `/proc/self/maps`采集实际已加载opapi、tiling和Ascend扩展的路径/SHA，
  同时记录bootstrap后的有效环境；这些证据仍不直接证明设备kernel构建来源。
- 无Worker、模型、checkpoint加载或collective。保留集中预检、八卡空闲和有界进程清理。
  一个case故障后，仅在无残留所属进程且八卡空闲时继续下一独立case。
  任何case失败或缺少有效算子比较结果，最终退出非零且保留已收集报告。
- 两种诊断模式互斥；默认七场景及原生同步诊断行为保留。
  成功诊断最多标记diagnostic_only，不能接受Worker、AFD等价或完整模型。

## 检查

本地及A3既有解释器各10项CPU测试通过，覆盖默认七场景metadata、六项单算子metadata、
同步环境、互斥模式、缺失结果拒绝和首个故障后其他五项的结果保留。
后两项测试模拟进程结果，只验证父进程控制契约，不是设备或故障恢复证据。
缺失结果测试先出现RED（退出0但无结果被误记成功），增加报告校验后GREEN。
最终A3源码及结果在`cpu-cli-delivery/`，测试进程未导入torch/NPU/vLLM/插件运行时。
复现使用已提交的`a5-native-sync-debug-20261117/run_cpu_cli_validation.py`。

186份Python编译、189份源码、194份原生Git身份检查通过。
相对`parent-bundle`仅更新launcher、新增probe及manifest，146份生产模块与原生身份原字节不变。
默认包及锁同步；入口SHA见`bundle-delta.json`。Ruff、规划与上游保护检查通过。
命令、退出码及原始本地检查日志在`local-validation.json`和`checks/`。

A5单算子尚未运行；本轮未解决启动，不接受七场景、数值等价或任何性能结论。
下一步按照更新后的中文指引运行`--diagnose-router`并回传完整诊断目录。
