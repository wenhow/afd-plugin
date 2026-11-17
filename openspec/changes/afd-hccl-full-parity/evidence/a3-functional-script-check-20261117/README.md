# A3 共用脚本检查：2026-11-17

用户要求先在 A3 排查验证脚本，再交付 A5。此次使用真实 A3 环境及锁定源码，
没有安装共享包、改写上游、伪装 A5 或执行推理。结论范围为配置、进程监督、
参数观测和原生 Worker/KV 初始化；A5 专用 bootstrap、算子与七场景推理仍待验收。

## 冻结版本与结果

| 冻结版本 | 包 SHA256 前缀 | 已验证内容 |
| --- | --- | --- |
| `fcef50b` | `817d04ee81dc` | 串行完整 23 组检查，含 13 种真实配置 |
| `63c5a44` | `28472209418f` | 14 组补验，含 4 种 overlap 配置、32 头 indexer 的原生加载与 KV |
| `a4b3ab6` | `bd537ca3ce1c` | 最新 vendor 入口的 4 项真实 A3 标准库/子进程检查 |

[汇总审计](validation.json)及[初轮串行摘要](summary.json)保留各自版本，
不能把旧冻结包通过结果直接标为新包通过。`63c5a44` 的四种 overlap 配置，
结合[同源码的九种 CPU 首配置](../a5-qli-geometry-20261117/cpu-green/summary.json)，
共覆盖原生 1 种、A/F 各 6 种配置。复用前逐字节核对 functional worker、
feature_validation 和 A5 runtime 三份源码；该 CPU 轮另有 386 项通过、0 跳过。
配置通过不代表 Graph、U2 或多流已经执行。

`a4b3ab6` 相对 `63c5a44` 只修改集中 launcher 及包清单，其他运行源码保持。
最新入口在 A3 单独检查；没有把旧 launcher 的设备流程记成新版整轮设备执行。

## 真实 A3 检查范围

- 七场景 metadata 入口：不导入运行时，冻结包成员及 launcher 身份一致。
- 三类非法参数：重复设备、非正场景超时、非正预检超时均退出 2，未创建结果目录。
- Linux 真实子进程正常退出、退出 7、超时回收；仅故意超时的子进程被强杀，
  所有受监督 PID 均无残留。退出 7、参数拒绝和硬件拒绝是预期结果。
- A5 runtime 预检识别真实 `built_device_type=A3` 后拒绝，未伪装设备或执行 A5 内核。
- NPU 1 上真实 ND=2、带 padding 的 NZ=29、BF16 标量读回；ND/NZ SHA 一致，
  原张量格式、指针和 stride 保持，内部格式开关恢复，未导入 AFD 模块。
- 用冻结功能脚本的首次 EngineArgs 和真实原生 Worker 执行 init_device/load_model，
  在 NPU 1 采集 79 个有限参数，两次记录一致，随后销毁原生进程组，cleanup 无异常。
- 最新 QLI 几何为普通 Attention 64 头/512 维、Indexer 32 头/128 维；三层 C2
  `[0,0,2]`、HC=4、BF16、dummy 权重。仅以完整 checkpoint 的 config 为输入，
  不加载完整训练权重，不执行 Scheduler、warmup、forward 或采样。
- 原生 KV 使用冻结 `initialize_attention_kv`：8 MiB backing 预算，LBNHC，
  64 blocks、5 groups、6 bindings；分配后参数观测不变。不是 KV 读写/回放验收。
- 最终串行轮和 QLI 补验开始及结束均八卡空闲；194 份原生源码身份一致，
  1068 份运行环境文件、1404 份 CANN/ATB 文件及 5 份重点库校验保持。

## A3 发现的标准库测试问题及修复

最新 vendor CLI 检查首次在 A3 出现 3 失败、1 通过。测试父进程用了 `-S`，
测试启动的 Python 子进程没有；A3 venv 的 site 初始化额外前置 FLA vendor 路径，
干扰了继承路径的精确断言。生产启动路径没有因此判为失败。

只给这两个标准库测试子进程补 `-S`，使其与父进程的无 site 语义一致；
真实 A5 launcher 启动不变。[修复后 A3 四项全通过](latest-a4b3ab6/validation.json)，
包括显式 vendor 的七场景 metadata、无参数继承、缺库提前拒绝和真实子进程环境。
合成库文件只检查路径存在，未加载库或模拟设备。失败原件保留于
[attempt-001](latest-a4b3ab6/attempt-001/results.tar.gz)，不能将它改写为通过。

## 环境、命令与原始证据

实际 venv 为 `/mnt/workspace/code/.venvs/afd-v030-dsv41-c0`，
模型配置来自 `/mnt/workspace/models/DeepSeek-V4.1-Flash`，设备为 NPU 1，TP=1。
复用只读 CANN/ATB 9.2 与已存在的 7e37e29a shadow package；
vLLM 锁定 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，Ascend 锁定
`7e37e29a7803d6897a7113088e2b771f83205bc0`。实际安装路径、每份文件与库 SHA
保留在 source capsule 的 environment-record 以及结果的 native-source-identity 中。

- [检查器](check_scripts.py)、[首轮执行输入](execution-source.tar.gz)、
  [串行执行脚本](run-serialized.sh)、[串行执行身份](serialized-execution-identity.json)、
  [串行结果及前后保留检查](results.tar.gz)。输入 capsule 含冻结包、快照、环境与校验器；
  在原同类隔离环境解包后运行 `bash run-serialized.sh`。
- [原生 KV 补验差异](kv-extension.patch)、[补验输入](kv-extension-source.tar.gz)。
  [原始交错结果](interleaved-results.tar.gz)中 KV 独立子轮通过；主轮因补验并行占卡，
  最后空闲断言失败，整体保持失败。之后完整串行重跑 23 组通过，以 `results.tar.gz` 为准。
- [QLI 新版执行输入](latest-63c5a44/source.tar.gz)、
  [原始结果](latest-63c5a44/results.tar.gz)、[独立审计](latest-63c5a44/validation.json)。
  capsule 内含独立 runner、冻结快照、run.sh、环境及前后文件校验器，运行 `bash run.sh`。
- [最新 launcher 输入](latest-a4b3ab6/source.tar.gz)、
  [原始结果](latest-a4b3ab6/results.tar.gz)。在解包的 snapshot 运行
  `python -B -S tests/e2e/diagnostics/test_dsv41_a5_functional_validation.py -v`。

每轮使用新结果目录。JIT cache 位于本次自有 scratch；初轮杀编译后仅在确认所属
PID/start ticks 全部结束且八卡空闲时处理该私有 cache 的遗留锁，未处理共享 cache。
`attempt-001` 保留首次冷 JIT 的 120 秒配置超时；后续独立配置检查预算为 600 秒。
`attempt-002` 至 `attempt-004` 保留本轮 ND/NZ 测试夹具修正过程：默认禁用内部格式、
只写配置 getter 误用、1D backing reshape 不能证明 NZ。最终使用直接拥有二维存储的
样本，以真实格式 29 和逻辑值 SHA 证明读回，不将这些夹具失败称为生产回归。

保留既有任务勾选。A/F 数值对齐、完整 checkpoint 精度、DSpark/PD、Engram 与性能
均不因本次脚本检查而验收；后续先过共用脚本检查，再由 A5 执行设备场景。
