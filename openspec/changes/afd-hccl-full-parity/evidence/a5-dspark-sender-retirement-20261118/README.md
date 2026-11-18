# A5 前五场景通过及 DSpark 生产端发送线程退出

用户提供 `dsv41-u2-rope-report` 目录。本轮原始 JSON/log 原字节归档在
`uploaded-reports.tar.gz`，逐文件 SHA 和独立核对见 `assessment.json`；
可执行 `python -B <本目录>/assess_reports.py` 重查，不导入设备运行时。

## 本次 A5 观察

- 原始冻结包为 `c1107179`，原生预检 194 份源码匹配。
- 原生 eager/U1、AFD eager/U1/U2 多流、Graph/U1/U2 多流五项均退出 0；
  无监督超时、强杀、所属残留。四种 AFD 的 79 份 canonical 参数身份及采样
  输出与原生相同，取消后继续和追加请求通过。范围仍为 BF16 dummy C2。
- DSpark/PD 四份角色报告均成功，Ready、整池 KV 读取及拒绝后继续有实际记录。
  D 三轮拒绝数为 2/1/1，输出 `[7,7,8,7,8]`，达到自然长度上限。
- P 报告 owner shutdown 和通信组销毁返回后，日志结束于
  `corrupted double-linked list`，退出 -6；其他三角色退出 0。
  supervisor 正确失败，第七场景未执行。占用观察按用户要求排除。
- 没有原生崩溃栈，无法确定堆损坏发生位置或归因到 TE/HCCL 析构；不改写原始失败。

## 补丁与本地架构复核

锁定 Ascend `7e37e29a` 的
`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`：
`KVCacheSendingThread.run(self)` 持有 native `zmq_ctx`，调用
`run_busy_loop(self, sock: zmq.Socket)`；原生循环永久阻塞收包，没有停止接口。
原生 `MooncakeConnector` 沿用 base 的空 shutdown。插件此前仅关闭 P registry。

本轮在既有 DSpark sender 上增加实例级 Event/lock 和 `retire(timeout=...)`。
关闭顺序为 registry 停止 admission 并证明 source/lease 已完成，然后请求循环停止，
由原发送线程离开 native socket/context，调用者 join 后才能继续 runner 清理。
未完成租约时仍失败并保留 DONE 服务；失败不能通过后续重试变成成功。

`MooncakeHybridConnector.shutdown(self)` 和 `run_busy_loop(self, sock: zmq.Socket)`
的签名与锁定源一致，已有单表签名检查覆盖。AFD 差异有短 PATCH 标记；新功能
留在外部 connector 的实例生命周期，不修改 upstream、scheduler、MoE 或设备数学。
没有新增 mutable global、wire 消息、engine 注销/关闭或 registration reset。
100ms receive timeout 只唤醒空闲 host listener；消息到达立即处理，不增加设备同步。
局部 sender 退出契约审查通过；当 upstream 提供合作退出时删除该实现。
此审查不证明原始 native heap corruption 已修复，实际效果待 A5 重跑。

入口设置子进程 `PYTHONFAULTHANDLER=1`、`PYTHONUNBUFFERED=1`，异常终止仍失败；
若再次 SIGABRT，可回传 Python 各线程栈，不能把它当作 C/C++ 栈的替代。
新增 `--dspark-only` 保留同一预检及两个原始 DSpark/PD 场景（含完整 Graph/U2
参数），通过后仅记 `dspark_pd_components_passed`，不记七场景整体通过。

## 验证

- RED：8 个新用例因旧实现无 sender retirement API 在 setup 报错，未初始化 NPU。
- 最终 `cpu-green-component`：KV connector、DSpark PD、统一签名三模块
  **213 passed、0 failed/error/skip**，NPU 初始化前后均 false，冻结源码字节不变。
  新增用例使用真实 CPU 线程和 ZMQ socket；TE/NPU 注册/模型计算不在其证明范围。
- 入口标准库测试 **19 项通过**，Ruff、186 Python 编译、194 Git pin 校验及
  本地规划/上游保护检查另见 `validation.json`。
- 初次 green 因测试路径拼写错误未执行；第二次广回归虽 333 项断言通过，
  NPU 初始化状态变为 true，故该轮不计作 CPU 证明。随后广范围加设备初始化
  阻断的尝试未得到合格报告；最终移除相邻 runner 模块，仅上述三模块通过并
  保持设备初始化阻断。既有 runner 模块的 CPU 边界问题未在本轮修补或声称通过。
- 三个 pre-existing consumer 测试增量留在工作区；本轮交付测试仅纳入新增 sender
  用例，并删除与既有 consumer retirement 冲突的历史 noop 参数用例。
- `source-verification.json` 确认新冻结包只改一个生产文件，其他 145 份保持。
  NPU 堆损坏消除、DSpark 两项整体和完整训练检查点验收均待确认。

## A5 重跑

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030
python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 --preflight-timeout 900 --dspark-only \
  --output-dir results/dsv41-a5-dspark-sender-20261118 --run
```

保留原五项成功证据。此命令不重复它们，也不把跨版本的两批结果合并成同一冻结源
的七场景验收。回传顶层 JSON/log 和失败场景 JSON/log 即可。
