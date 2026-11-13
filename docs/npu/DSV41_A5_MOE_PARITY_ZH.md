# A5 单层 MoE 数值定位：一次顺序 TP1/TP4

本轮只增加这一项 A5 手工验证。此前 AFD 32-token 请求已经完成，但与原生基线从
首 token 起不同；两者并行布局不同。先固定同一真实第 0 层权重和输入，比较原生
TP1 与 TP4 的 MoE 输出及 gate，隔离并行切片、量化和归约差异。完整原生 TP4 的
显存估算超出现有预算，因此采用单层组件；不重复此前三条原生请求或为清理日志重跑服务。

独立验证包为 `reports/dsv41-a5-moe-parity-20261112.tar.gz`。它包含锁定插件基线的
源码及本次诊断入口，附逐文件 SHA256 清单；解压到新目录运行，不需要安装包或切换分支。
不包含 checkpoint 权重、其他开发中的改动或 upstream 源码修改。
验证包 SHA256：`5a24753b232b663e3a0d5b8c58233feec0480e02be5fcbc71cf27d4355ab334d`。

在 A5 上该仓库的 `feat/dsv41-flash-hccl-v030` 分支根目录更新后，
在已配置 CANN/ATB 的 shell 中执行。验证包已随仓库提交，无需另外传输：

```bash
git pull --ff-only
moe_bundle_path="$PWD/reports/dsv41-a5-moe-parity-20261112.tar.gz"
sha256sum "$moe_bundle_path"
mkdir -p /mnt/workspace/tmp/dsv41-moe-check
tar -xzf "$moe_bundle_path" -C /mnt/workspace/tmp/dsv41-moe-check
cd /mnt/workspace/tmp/dsv41-moe-check/afd-dsv41-moe-parity-20261112
/usr/local/python3.11.10/bin/python -B tools/diagnostics/run_dsv41_a5_moe_parity.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --output-root /mnt/workspace/tmp \
  --run
```

入口会设置 vendor 的 OPP/OpAPI 路径和本包 Python 路径。默认设备为 `0,1,2,3`，
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

A3 已完成该探针的真实安装包导入/TP1与TP4配置检查，均exit0、前后八卡idle，
25份安装源码与锁定Git对象一致；独立入口的默认metadata/归档也通过。
包内 `source-manifest.json` 保存诊断源码身份和准备验证记录的位置；原始 A3 证据保留在开发工作区。
这没有执行载权、FP4算子或TP collective，数值部分需要本轮 A5 回传。
