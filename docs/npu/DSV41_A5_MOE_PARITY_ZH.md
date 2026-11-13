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
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
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

随后 TP1 在 MoE 构造时报告 `RoutedExperts.__init__() got an unexpected
keyword argument 'n_shared_experts'`，尚未载权。探针绕过原生 worker 构造，
遗漏了 `register_ascend_customop(config)`；仅导入 ops 不会注册 Ascend 层替换。
当前补齐原生的 dummy fusion op 和 Ascend custom op 注册，并在载权前记录、
检查实际选择的 `AscendRoutedExperts` / `AscendMoERunner`。注册分发与 worker
源码也随结果归档；构造成功后另外记录实际 MoE 实现类。
A3 实际安装包的配置/注册检查确认：注册前通用类拒绝该参数，注册后选中
Ascend 实现并接受该参数，TP4 配置重复注册也正常。该检查没有构造 MoE、
加载权重或执行算子，修复后的 A5 数值部分仍待回传。

下一次 A5 已进入 MoE 前向，但 `_C_ascend.moe_gating_top_k_hash` 尚未注册。
Python 层替换注册与编译扩展加载是两个步骤；当前在子进程绑定设备后，先通过
原生 `import_kernels()` 准备路径，再加载 `vllm_ascend.vllm_ascend_C`。
在建立通信组和载权前检查路由算子的参数名与 PrivateUse1 kernel，记录扩展路径、
二进制 SHA256 和实际 schema。缺少扩展、算子或参数时保留原始异常并提前退出。
这个原生算子也服务普通 `sqrtsoftplus` 路由，模型的 `num_hash_layers` 保持原值。
A3 当前环境的扩展文件查找结果为缺失，不能验证实际扩展加载或 A5 算子执行；
源码接口检查与本地静态检查单独记录，A5 数值部分继续待验。

随后一次设备绑定失败为 `Invalid device ID`，运行 shell 确认设置了
`ASCEND_RT_VISIBLE_DEVICES=0`，只开放一张卡，不能完成 TP4。
探针继承该限制；`--devices` 指进程内可见编号，不是宿主物理卡号。
例如可见列表为 `2,3,4,5` 时，仍使用 `--devices 0,1,2,3`。
上面的命令显式为这次运行开放四卡；入口记录继承的可见列表，实际执行时在 TP1
绑定设备和载权前检查整个四卡列表与 `torch.npu.device_count()`，并记录选择结果。
`--setup-only` 保持只检查导入/配置，不执行这个设备数量检查。

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
