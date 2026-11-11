# V4.1 PD 实机组件验证

本探针验证[布局连接器](DSV41_PD_LAYOUT_CONNECTOR_ZH.md)的真实外部工厂、原生
Mooncake worker、内存注册、后台线程、请求完成与 DONE 协议。P/D 各一个 TP1
进程，默认使用 NPU 0、2。探针手工建立小型 V4.1 typed shared-page 视图，不加载
模型权重，不使用模型原生 KV allocator，也不验证请求历史或 Engram。生产模型门禁仍关闭。

2026-11-10 的首轮 A3 组件会话整体失败：四个请求的真实字节校验、冲突拒绝和 DONE 均完成，
但 D 进程在退出时收到 SIGABRT，末尾日志为 `terminate called without an active exception`。
八卡恢复空闲且监督器没有强制终止，该首轮失败原始结果保持不变。2026-11-11 使用独立新进程和
`--retire-receiver` 的 transfer、layout-mismatch 两个受控组件会话均通过，P/D 正常退出、
无超时或强制终止、运行前后八卡空闲。成功范围限下述小型 typed 页、DONE 和新退出路径，
不能推广为完整模型 PD、原生 allocator、历史或 Engram 验收。仍须以整体 summary 和退出码判断，
不能仅凭 role JSON 的 `success` 判断成功；失败时停止后续会话并保留结果。

`tools/diagnostics/run_a3_pd_layout.py` 面向当前锁定的 A3 环境；默认复用已有 venv、
CANN 和 ATB，不安装依赖。其他容器须先核对脚本中的源码锁点及实际二进制，不能直接
把本环境的身份校验当成 A5 验收。

## 先准备并复核快照

在本地 PowerShell 的 afd-plugin 根目录执行；`--model` 仅用于读取原生模型配置。
每次使用新的输出目录和 label。

```powershell
$afd_python = ".venv/Scripts/python.exe"
$afd_prepare = ".tmp/pd-component-prepare-001"
& $afd_python tools/diagnostics/run_a3_pd_layout.py prepare `
  --label prepare-001 --output-dir $afd_prepare `
  --model /mnt/workspace/models/DeepSeek-V4.1-Flash
```

SSH 默认使用现有 A3 地址和本地 AtomGit SSH config。需要时可显式指定
`--host`、`--ssh-config`、`--remote-parent`、`--venv`、`--cann-env`、`--atb-env`
及 `--extra-library-path`；这些路径应来自目标容器的实际安装。

`prepare` 只上传插件源码及探针依赖、读取 33 个已安装源码和两个 Mooncake `.so`
的哈希、配置文件与软件包元数据，并采集八卡空闲状态。它不导入 torch、torch_npu
或 Mooncake，不启动设备子进程。保留并复核：

- `prepare/summary.json`：`success=true`、`device_probe_started=false`、八卡空闲。
- `prepare/source-check.json` 与 `manifest.json`：全部源码匹配，包含两份复用 helper。
- `prepare/binary-identity.json`、`prepare/model-config-identity.json`：实际二进制和
  `config.json`、可选 `generation_config.json`、配置目录 Python 文件身份。
- `prepared.json` 与 `delivery-identity.json`：不可变远端目录及本次准备的完整哈希。

二进制身份只说明所用文件，不能代替运行验证。确认当前快照已完成源码审查、设备窗口
可用后，才进行下面两个独立会话。任何源码、二进制或模型配置变化都须重新准备新快照。

## 两个独立会话

```powershell
$afd_identity = Join-Path $afd_prepare "prepared.json"
$afd_identity_sha = (Get-FileHash -Algorithm SHA256 $afd_identity).Hash.ToLowerInvariant()
& $afd_python tools/diagnostics/run_a3_pd_layout.py run `
  --prepared $afd_identity --identity-sha256 $afd_identity_sha `
  --case transfer --devices 0,2 --label transfer-001 `
  --output-dir .tmp/pd-component-transfer-001
if ($LASTEXITCODE -ne 0) { throw "transfer 失败：保留结果并先诊断" }

& $afd_python tools/diagnostics/run_a3_pd_layout.py run `
  --prepared $afd_identity --identity-sha256 $afd_identity_sha `
  --case layout-mismatch --devices 0,2 --label mismatch-001 `
  --output-dir .tmp/pd-component-mismatch-001
```

两个会话各自创建新的 P/D 进程。`transfer` 顺序覆盖真正空请求、非连续 block
映射、源列表按原生窗口规则取尾部，以及跨逻辑组向同一物理页写入不同源 block 的
整批拒绝。接收端逐字节核对整个 slab，包括完整页、页内 padding、未写 block 和配置
页之外的 guard；guard 可能包含在原生实际注册范围内，不能据此声称越界访问已被硬件拦截。

`layout-mismatch` 在 P 端使用实际 BF16 scale 视图、D 端使用 FP16 scale 视图，
经真实网络握手验证布局不兼容在地址发布前被拒绝；后续失败请求仍须完成原生 DONE。
它不伪造引擎或网络返回，不表示完整 PD 服务正确性或故障恢复已通过。

## 结果与收尾

下载目录中的 `runs/<label>/summary.json` 和 `probe/result.json` 必须均成功，
且八卡在运行前后均空闲、`outer_timed_out=false`、强制终止列表为空。还须查看：

- `probe/producer.json`、`probe/consumer.json`：每个请求的完成、错误 block ID、
  全 slab 比较、原生 TP 前后结果、真实注册范围与线程启动返回。
- 两份 `*-layout.json`、`source-preflight.json`：实际 NPU typed 布局及本次源码、
  二进制、模型配置匹配结果。
- `producer.log`、`consumer.log`、`probe.log` 与 `npu-before.log`、`npu-after.log`：
  原始错误和设备收尾状态；原始归档保存在 `results.tar.gz`。

当前原生 worker / GlobalTE 包装层没有配套 stop、close 或 unregister 生命周期入口。
底层 `TransferEngine` 实际公开 `unregister_memory` 和 `batch_unregister_memory`，
本探针没有调用它们；这些方法也不等于引擎 close。探针分别记录请求排空、接收 executor
的 `shutdown(wait=True)` 返回，以及两个进程的实际退出状态；注册依赖保留到进程
finalization。不开启插件退休时，原生接收 daemon 在 executor 排空后仍运行；首轮退出故障
未归因到具体原生对象。开启新退休路径的两次成功观察不证明底层析构根因，也不能称为完整主动注销，
role 报告的成功不能替代进程正常退出。

内部监督上限 240 秒，外层上限 280 秒，回收最多共享 10 秒预算，另有只读空闲采样。
监督器只会终止本次记录且 PID start ticks 仍匹配的进程。超时、强制终止、未排空、
非零退出或非空闲均判失败，保留首份结果；诊断后用新 label，不覆盖或自动重试旧会话。

最小回传为两个会话各自的 summary、result、P/D role JSON、源码检查、布局快照和
首错原始日志；完整归档保留在环境或本地。运行结果、环境及二进制身份不随代码和说明推送。

## 可选 D 接收线程合作退出

`run --retire-receiver` 显式启用插件接收线程退休，默认关闭。它先关闭整个原生批次的准入，
等待每个已接纳任务的原生 DONE/ACK，再使用本地专用 sentinel 退出并 join 接收线程，最后
等待 executor 退出。原生 `None` 队列项仍表示继续运行。该参数不停止 P 发送线程，不调用
`unregister_memory`，不关闭引擎，也不重置 GlobalTE 的注册生命周期。

该路径于 2026-11-11 在锁定 A3 环境完成两个独立受控会话。transfer 的空请求、非连续映射、
窗口尾部和整页别名冲突均完成整 slab 字节核对及 DONE；layout-mismatch 的 BF16/FP16 scale
布局差异在地址发布前被拒绝，接收 slab 未变且失败请求仍完成 DONE。两次均确认 D 接收线程
退出、executor 返回、双端原生 TP 前后不变、P/D 退出码为零以及八卡空闲；不安装依赖、不注销引擎内存。
首轮 SIGABRT 仍保留为独立失败，未将其原始结果改成成功。

复现时须重新 prepare 并审核新源码快照，不能复用原先未包含该实现的 prepared identity。
使用新 label，并在第一个会话整体成功后再运行布局负例，例如：

```powershell
& $afd_python tools/diagnostics/run_a3_pd_layout.py run `
  --prepared $afd_identity --identity-sha256 $afd_identity_sha `
  --case transfer --devices 0,2 --retire-receiver `
  --label transfer-retire-001 --output-dir .tmp/pd-component-transfer-retire-001
if ($LASTEXITCODE -ne 0) { throw "transfer 失败：保留结果并先诊断" }

& $afd_python tools/diagnostics/run_a3_pd_layout.py run `
  --prepared $afd_identity --identity-sha256 $afd_identity_sha `
  --case layout-mismatch --devices 0,2 --retire-receiver `
  --label mismatch-retire-001 --output-dir .tmp/pd-component-mismatch-retire-001
```

`consumer.json` 应另外记录 `receiver_retirement_requested=true`、`receiver_retired=true`、
`receiver_input_thread_joined=true`。进程退出码、八卡空闲和原有全部字节/DONE 校验仍须独立通过。
API 的 timeout 覆盖准入、已接纳任务和接收线程 join；最后的 Python executor shutdown 没有
内部 timeout，由已有外层进程监督提供最终上限。超时或 ACK 错误保留资源并判失败，不能宣称硬件 abort。
