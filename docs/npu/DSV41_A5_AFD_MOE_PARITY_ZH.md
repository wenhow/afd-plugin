# A5 同 TP4 单层 native / AFD F 对照

本次只需执行下面一个场景。此前原生 TP1/TP4 单层结果已经回传，gate 一致、MoE
相对 L2 误差约 0.44%；本次固定 TP4，进一步区分 F 层构造、载权和计算边界的影响，
不重复 TP1 或完整模型服务。

在已配置 CANN/ATB 的 shell 中，更新仓库后先解开独立冻结包，再从包根目录执行。
该包包含本次需要的完整插件源码；不依赖仓库内其他尚未交付的开发改动：

```bash
validation_dir="$(mktemp -d /mnt/workspace/tmp/dsv41-afd-moe.XXXXXX)"
tar -xzf reports/dsv41-a5-afd-moe-parity-20261113.tar.gz \
  -C "$validation_dir" --strip-components=1
cd "$validation_dir"

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
/usr/local/python3.11.10/bin/python -B tools/diagnostics/run_dsv41_a5_moe_parity.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --output-root /mnt/workspace/tmp \
  --comparison native-vs-afd \
  --run
```

该入口顺序启动 `native-tp4`、`afd-tp4`，每阶段使用四个全新进程，两阶段之间检查
八卡空闲。每阶段只载入相同 checkpoint 的第 0 层 FFN 权重，原始 40 层配置保持；
不需要安装依赖、修改模型或启动完整服务。默认 600 秒内部预算和 720 秒外部监督，
只清理本次拥有且 PID/start ticks 匹配的进程，完成或失败都会生成诊断归档。

每阶段自动覆盖 10-token prefill、1-token decode、独立 FP32 路由输入的 decode，
同时比较相同权重下的直接 MoE 与 F `compute_ffn_output`。每次调用使用全新输入，
原始 BF16、FP32、IDs、gate 和输出均随包保存。所有八个 rank 校验 checkpoint、
输入和源码身份；分别检查两个阶段四 rank 的归约一致性，再按同 rank 比较两阶段。
独立 FP32 输入用于定位路由路径，不预设硬件输出必须发生变化或可接受的误差阈值。

AFD 阶段执行真实 `AFDDeepseekV41FFNLayer` 构造、F `load_weights` 过滤入口、原生
权重映射/TP 切片/FP4 后处理和绑定的 F 计算方法。为控制显存，承载对象跳过完整
F 模型构造；输入权重限定为该层 FFN。完整 loader `create_model`、worker、runner、
AFD HCCL、Attention、Engram 和完整模型输出不属于本组件，报告会明确记录这些边界。

回传打印的 `dsv41-a5-moe-parity-*.tar.gz`，失败也回传同一个归档，无需追加场景。
`execution_success` 表示诊断流程、身份、finite 和 rank 一致性成立；数值误差及
前后处理参数指纹只作分析，`model_equivalence_accepted` 始终为 false。

原有 TP1/TP4 模式和[历史说明](DSV41_A5_MOE_PARITY_ZH.md)保留；本次必须指定
`--comparison native-vs-afd`。独立包的 `source-manifest.json` 记录逐文件 SHA256，
不包含暂停的旧 V4 MTP 或其他未选入的工作区改动。
