# A5 单层 MoE 数值定位：一次顺序 TP1/TP4

本轮只增加这一项 A5 手工验证。此前 AFD 32-token 请求已经完成，但与原生基线从
首 token 起不同；两者并行布局不同。先固定同一真实第 0 层权重和输入，比较原生
TP1 与 TP4 的 MoE 输出及 gate，隔离并行切片、量化和归约差异。完整原生 TP4 的
显存估算超出现有预算，因此采用单层组件；不重复此前三条原生请求或为清理日志重跑服务。

诊断入口和探针直接保存在仓库的 `tools/diagnostics/` 和 `tests/npu/`。
更新后在仓库根目录运行，不需要解压源码副本、安装包或切换分支。
`--output-root` 只存放日志、采集证据和结果归档。

在 A5 上该仓库的 `feat/dsv41-flash-hccl-v030` 分支根目录更新后，
在已配置 CANN/ATB 的 shell 中执行：

```bash
git pull --ff-only
/usr/local/python3.11.10/bin/python -B tools/diagnostics/run_dsv41_a5_moe_parity.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --output-root /mnt/workspace/tmp \
  --run
```

2026-11-13 回传的旧版运行在源码采集时退出：
`FileNotFoundError: installed source is required: vllm_ascend/models/deepseek_v41/model.py`。
当时尚未启动 TP1/TP4 子进程、载入权重或执行算子，没有数值对比结果。
旧定位逻辑只扫描 `sys.path` 和 distribution 目录，遗漏 editable 安装的源码根目录。
当前使用顶层包的 import spec 查找实际源码，不执行包的 `__init__`；
此前 A5 记录的 Ascend 源码位于 `/opt/vllm-ascend`，无需为此重装依赖。

入口会设置 vendor 的 OPP/OpAPI 路径和仓库 Python 路径。默认设备为 `0,1,2,3`，
先单卡 TP1，再四卡 TP4；开始、阶段结束和最终退出都要求八卡空闲。整个组件有
600 秒内部预算，额外监督负责中断、按 PID/start ticks 清理以及打包失败日志。
去掉 `--run` 仅收集 checkpoint header 和源码；`--run --setup-only` 是导入/配置诊断，
本轮无需另跑这两个前置场景。

完成或失败后均回传入口打印的 `dsv41-a5-moe-parity-*.tar.gz`。归档包含配置/index/
选中 tensor 的身份、原生加载完整性、算子库及实际源码 SHA、各 rank 输入/gate/输出
二进制、数值差异和进程退出/空闲快照。仅第 0 层 FFN 被构造；原始40层配置保持，
实际 `num_hash_layers` 默认0，因此使用普通 gate+bias，不人为改成 Hash MoE。
本模型该层逻辑 tensor payload 约6.76 GiB，实际 HBM 峰值由结果记录。

结果中的 `execution_success` 只表示诊断流程、相同权重/输入、finite 和 TP4 rank
归约输出一致等条件成立。TP1/TP4 的误差保留供分析，不预设可接受容忍度；
`model_equivalence_accepted` 始终为 false。该组件不经过 AFD HCCL，不替代完整模型
输出、Engram 或性能验收，AFD DSpark N1 门禁继续保留。

此前版本在 A3 已完成真实安装包导入/TP1与TP4配置检查，均exit0、前后八卡idle，
25份安装源码与锁定Git对象一致；独立入口的默认metadata/归档也通过。
仓库运行的结果记录插件 Git HEAD、入口/探针及依赖 SHA256，原始 A3 证据保留在开发工作区。
源码定位修复经过 CPU 检查，包含 editable 安装和不导入运行时的边界；
这没有执行载权、FP4算子或TP collective，数值部分仍需要 A5 回传。
