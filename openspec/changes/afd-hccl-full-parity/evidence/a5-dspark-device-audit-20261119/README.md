# A5 Graph/U2：四角色成功，修正审计卡号后既有报告复核通过

本次补充目录为 `C:/Users/zq/Downloads/dspark-pd-graph-u2-cancel-replace`，
包含场景内 summary 及 P/D/F-consumer 三份角色 JSON。四份原始文件未经改写，
压缩保存于 `inner-reports.tar.gz`，各自大小及 SHA 见 `input-manifest.json`。
场景 run_id 为 `9ead09c9335949019bf137a913ff0774`，对应上一轮
[顶层报告](../a5-dspark-graph-final-audit-20261119/README.md)及其原始 ZIP。

## 根因和原始运行结果

- 四角色 `success=true`，四个 returncode 均为 0；无 timeout、forced kill。
- Graph/U2 观察器已接受本次执行。有效批次依次为 U2 eager 两次 Python target
  stage、U2 replay 零次、U1 replay 零次、U1 eager 一次、U1 replay 零次；
  每次 eager stage 的 token_rows 均为 3。P/D/F 的正常回收已返回。
- 原始 `summary.success=false`，失败链为 supervisor → cross-layer → overlap →
  lifecycle → `verify_graph_host_records`，最终断言将卡号写死为 Attention=2、FFN=6。
- 顶层场景实际 `--devices 0,1,2,3`，角色顺序为 P、D、F-producer、F-consumer，
  因而本场景消费端 Attention=1、FFN=3。记录与启动参数一致。
- overlap 的两次 `verify_pipeline_storage` 同样把期望卡号写死为 2/6。
  本轮同时修正这些审计入口，没有跳过卡号、数值、KV、协议、取消或退出检查。

## 修复边界

从 supervisor 的实际 `args.devices` 和固定角色映射得到期望卡号，作为必填关键字
贯穿 Graph U1、U2、cancel/replace、compute overlap 和 cross-layer 五条审计分支。
host record 和 pipeline storage 均与独立的启动参数比较；不从待验证记录推导期望值。
supervisor 的新报告也保存 `local_devices`。

仅修改六份冻结诊断 helper，通过受 SHA 约束的 successor 生成默认包。
完整差异见 `graph-device-audit.patch`；146 份生产源和 native identity 字节保持。
六份审计 parent 均与本次实际执行的 `c576f79a...` 快照相同。
新包 SHA：`cca676b103a913917e64377ad1fb772270a328ba4ed77625fead496f69ed9d9d`。
`parent-runtime-source.tar.gz` 保留提交前默认包；源比较见 `source-verification.json`。

## 既有 NPU 报告的离线复核

`audit_existing.py` 在 `-S` 下导入真实冻结报告校验器，复核原 supervisor 的角色结果、
producer 完整报文校验，以及 consumer 的完整 cross-layer → overlap → lifecycle →
Graph 协议、host/schema、pipeline、逐层数值与请求状态审计。
输入档案先逐文件核对 SHA，再核对独立角色 JSON 与 summary 内嵌 worker 一致。
没有构造模拟 NPU 结果，也没有导入 torch/torch_npu/vLLM 或启动设备进程。

- `offline-red.json`：使用实际执行的旧冻结包复现同一个卡号断言，exit=1。
- `offline-green.json`：按启动参数 Attention=1、FFN=3 复核完整报告，exit=0。
- `offline-negative.json`：对同一报告传入错误期望 2/6，仍在 host 卡号校验失败，exit=1。
- 25 项标准库检查通过，其中新增两项直接校验本次实际 host/pipeline 记录，
  分别覆盖两个角色的正确期望通过、错误期望拒绝；不把这些 CPU 检查称作新设备执行。
- 186 份 Python 编译、194 个锁定原生源身份、三份默认交付物确定性重建和 Ruff 通过；
  本地规划及只读上游保护结果见 `validation.json`。

复核命令从仓库根目录运行：

```powershell
$evidence = 'openspec/changes/afd-hccl-full-parity/evidence/a5-dspark-device-audit-20261119'
python -B -S "$evidence/audit_existing.py" --output "$evidence/offline-green.json"
```

原始运行的失败 summary 保持原样，`offline-green.json` 是独立的事后复核。
现有真实 NPU 报告已支持本次受控非 Engram Graph/U2/PD/DSpark 组件组合的检查结果，
本项无需为定位或复核重跑 NPU；更新后入口自动生成成功顶层 summary 尚未经新设备运行。
本结果不构成完整训练权重推理、Engram、设备时间线/物理重叠或性能验收，
也不勾选 1.3/1.8 等包含更多组合的整项任务。
