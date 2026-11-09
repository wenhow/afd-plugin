# C4 有界异常与生命周期组件验证（2026-11-09）

两项均在 A3 实际依赖、真实 NPU 张量和 HCCL/Gloo 通信组上通过，使用芯片 0/2，拓扑 1A1F、双方独立原生 TP1。每次新建不可变快照，运行前后八板均无设备进程；无超时、无强制终止、未安装依赖，10 个已锁定上游源码哈希全部一致。

| 原始证据 | 实际验证 | 退出码 F/A | 耗时 |
|---|---|---|---|
| [compute-cancel-001](compute-cancel-001/results/probe/result.json) | A 成功发送首层首 stage 的 BF16、独立 FP32 和 IDs；实际 `execute_hccl_v41_step` 收到并精确校验后，受控计算抛出 ValueError。F 会话进入 terminal，监督进程通过独立 stdin 管道通知 A 协作取消；双方 close 后原生 TP1 all-reduce 仍成功。 | 0 / 0 | 27.30 秒 |
| [peer-exit-001](peer-exit-001/results/probe/result.json) | 实际 F 服务完成 U2 父事务（两层、两 stage、精确算术输出）后自行 `os._exit(17)`。监督进程确认退出后，A 才提交下一父事务；真实 Gloo 返回 `Connection closed by peer`，A 会话 terminal，后续 submit/STOP 均拒绝；A close 后原生 TP1 all-reduce 仍成功。 | 17（预期）/ 0 | 28.25 秒 |

边界：A 在外部协作取消用例中，故意停在输入已发送且 F 已接收的 quiescent gate，未进入无法匹配的 HCCL output recv。异常退出用例发生在前一父事务 COMPLETE 之后。这些证据**不覆盖阻塞 HCCL abort、设备故障恢复、原生 TP peer 失效恢复或生产 CANCEL 协议**。外部监督仅管理自己启动的 PID；两次均未触发强杀。

F 使用无权重 runner shell 调用真实公共服务入口和 Ascend forward context，明确未执行 F runner constructor、模型加载、原生 MoE、HC/indexer 或完整 V4.1 推理。输入与输出使用已有 positive probe 的确定性 oracle，未 mock 分布式后端或设备行为。正常 STOP 已由前轮 positive evidence 覆盖；本轮只补上述两个异常边界，不能据此勾完任务 1.9/1.10/1.11。

源版本：两次 HEAD 均为 `45356b2ff0f81e1d3fcf9b660026bdc904738a19`，快照包含当时工作区字节。完整 manifest 见各自 `identity.json`，不能只用 HEAD 代替工作区源码身份。

| 快照 | manifest SHA256 | staged transport SHA256 |
|---|---|---|
| compute-cancel-001 | `80996fcd5a8c581cfa9daa3cfe2eeabd1efad0e66e751e07c84ad81ff3994bc3` | `8f38cb4e43c33d4431d975b36890b2659c39cf9e02691f622b25ce050bb24c63` |
| peer-exit-001 | `f9366a6a0f1827210a870567bbf3f0b1929585c6cf9c1732a11928b63ce31c64` | `5ec889818e2291fa441335c50c22bdb8bb2b3ffd838354bdbed2827585fc15a3` |

两次 probe SHA256 均为 `7f844a39823624117a5cfe1ecdea9b3b2ac08ede27ce04fd238d9f8fef9f9b87`，positive oracle 为 `96112b8192e2f1ff1a186b46f3f4d2f71910e1c15fd5ea4c7fe9821c744b796d`。第二次快照同时收进并行 C5 的 transport/cache 更新及新增 geometry 模块；本轮未请求 graph。两个快照分别保留原始身份，不宣称相同整树或相同 transport 字节。

环境：Python 3.12.9、torch 2.10.0、torch-npu 2.10.0.post4、vLLM 0.30.0+empty、vLLM-Ascend 0.1.dev1+g7e37e29a7、FastAPI 0.123.10；沿用 CANN 9.0.1 和现有 ATB 环境。详见各自 `results/versions.json`、`effective-env.json`、`source-check.json`、`npu-before.log`、`npu-after.log`。本轮新增 probe/harness 的 Ruff check、format check、Python 3.11 AST 与本地 `--help` 均通过；未重复 CPU 大回归。

本地 PowerShell 复现（label 需使用新的值，禁止覆盖原始证据）：

```powershell
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-microbatch/evidence/a3-failure-20261109/run_a3_failure.py --label compute-cancel-new --case compute-cancel
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-microbatch/evidence/a3-failure-20261109/run_a3_failure.py --label peer-exit-new --case peer-exit
```

独立 Linux 环境也可在正确安装依赖并初始化 CANN 后运行：

```bash
python -B tests/npu/hccl_staged_failure_probe.py --case compute-cancel --devices 0,2 --model "$afd_model" --output-dir /tmp/afd-compute-cancel-new
python -B tests/npu/hccl_staged_failure_probe.py --case peer-exit --devices 0,2 --model "$afd_model" --output-dir /tmp/afd-peer-exit-new
```

上述直接 probe 由内部 240 秒监督约束；A3 uploader 额外提供 280 秒外部边界、自己的 PID ownership 记录和运行前后八板空闲检查。新进程为独立进程组；未使用全局 pkill。
