# A5 原生参考：预检通过，资源检查阻断启动

输入 `b583d09c3a0f484dbd6d2419075dbcfd.gz`，SHA256 为
`e35a780b44e644bbaac4e9a1bdcdc8903abab7f749d310219b2e85e0c8b278c1`。
本次仅离线审阅，没有连接或操作 A5/A3；进程/PID 仅描述归档时的快照。
原始八个成员中的七个文件保持字节不变，汇总见 [review.json](review.json)。

原生目录为 `full-eager-u1-1g5_98ms`，代码 HEAD 为 `5087fa0`，Python 3.11.10。
卡 0–3、TP4、量化 checkpoint、CPU Engram、DSpark/PD 关闭。
2026-11-19 14:47:36（Asia/Shanghai）资源检查报错，尚未启动任何 owned probe/服务。

| 检查 | 本包结果与边界 |
| --- | --- |
| Python、包元数据及插件入口 | 通过；`vllm-afd-plugin=0.1.dev750+g5087fa022`，`afd=afd_plugin:register_afd`，上次注册缺失已解决 |
| 插件源码目录/Git | 指向当前 checkout，受检 tracked source status 为空 |
| 原生锁源 | 25 份文件 SHA256 全匹配；不是整仓版本一致性证明 |
| checkpoint | 配置/索引身份及 48 shard、96,085 tensor 结构通过；没有验证 payload 哈希或加载权重 |
| host 内存记录 | 可用约 819.49 GiB，大于 Engram 表下界约 188.83 GiB；不代表实际加载峰值已验证 |
| 选定卡资源 | 未通过；卡 0–7 均有先前已存在的 vLLM 进程，每卡 HBM 占用约 83,134–83,176 MiB |
| 端口、算子库、packed Attention probe、服务/载权、请求 | 均尚未执行，不能据此报告通过或失败 |

`summary.json` 的原始失败为：

```text
RuntimeError: selected idle devices were not confirmed; see npu-before.log
```

`request_passed=false`、`clean_shutdown=false`、`exit_code=1`。
`exit_codes={}`、`forced_cleanup=false`、`survivors=[]`；本轮没有创建服务进程，
因此 `survivors=[]` 不能解读为原机器已空闲。前后 `npu-smi` 的现存进程一致，
每卡包含一个 Worker_DP 和一个 EngineCore，共 16 行。可用卡集合为空。

用户随后使用该失败目录作为 `--reference-run`，触发
`reference must be a successful native TP4 control run`。这是 AFD 启动前的参考资格检查，
不是 AFD eager/U1 模型执行异常。未放宽参考条件或设备资源门禁；仅增强
[CLI 报错](../../../../../tools/diagnostics/run_dsv41_full_checkpoint.py)，直接包含参考目录、
`native_control/request_passed/clean_shutdown` 和参考报告的原始错误，说明 AFD 服务未启动。

按用户“先忽略卡占用，其他检查还有问题么”的问题：**已执行的其他预检没有失败**。
端口和全部运行时阶段没有证据，不能由此宣布原生 TP4、载权或输出正常。
[指导](../../../../../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md) 已补充资源检查与
参考成功条件。若选定设备用于本批验证，应由现有服务的管理者通过其正常生命周期释放，
核对实时状态后重新生成成功原生参考；本次没有停止、发送信号或处理任何现存进程。

标准库 CLI/HTTP 主机回归 **14/14 通过**，含失败参考携带原始原因且不启动服务的反例，
见 [host-tests.xml](host-tests.xml)；这是诊断行为验证，不是 NPU 执行或设备释放证明。
eager/U1 输出等价、首 token 差异以及完整模型矩阵仍待验，任务勾选不变。
