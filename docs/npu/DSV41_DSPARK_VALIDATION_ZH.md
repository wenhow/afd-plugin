# V4.1 DSpark：开发范围与统一验证

当前优先推进 **V4.1 DSpark**。普通旧 V4 MTP 的后续迁移暂缓；已完成的本地
改动保留。DSpark 的开发和 CPU 接口回归可继续进行，A5 完整模型验收后续统一执行。

## 执行边界

DSpark 使用同一 checkpoint 中的原生 draft，完整 draft 留在 Attention（A）端。
FFN（F）端只处理 target 的远程 MoE；draft 不宣告 F 工作，也不复用 target
的 AFD 元数据、Graph sidecar 或 Engram producer。

运行路径以 `method="dspark"` 和原生 `DeepseekV41DSparkModel` 的选择为准；
checkpoint 中出现 `mtp.*` 参数名不表示启用了普通 MTP。

一次 proposal 先执行一次 draft backbone，再由 Markov head 顺序采样 N 个
token。它与普通 MTP 的逐步 draft 调用不同，不需要先完成旧 V4 MTP 的传输协议。
target 验证的查询宽度为 N+1；原生 draft 的查询宽度根据 `sample_from_anchor`
为 N 或 N+1。N 表示 speculative token 数，与 draft 层数和卡数分别配置。

插件保留锁定版本中的原生 sampler 和 ModelState。正常顺序为 target 执行、
接受/拒绝采样、更新请求历史、生成下一批 draft、原生 KV post-forward。
target 的 Engram 执行资源在 execute 的收尾中退役；下一轮 lookback 依据
`num_computed_tokens` 从历史中取值。已生成但尚未计算的新 bonus token 不应
提前出现在该窗口中。

## 开发时的 CPU 回归

在具有匹配依赖的环境中，于仓库根目录运行以下两个完整模块。A5 容器的 Python
路径沿用 `/usr/local/python3.11.10/bin/python`；不要求 venv 或 `atb-env`。
这些测试不启动推理服务，不加载 checkpoint 权重，也不使用 NPU。

```bash
afd_python=/usr/local/python3.11.10/bin/python
afd_result_dir=$(mktemp -d /tmp/afd-dspark-cpu.XXXXXX)
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

git rev-parse HEAD > "$afd_result_dir/git-head.txt"
"$afd_python" -B -m pytest -q \
  tests/unit/v1/worker/npu/test_hccl_v41_graph_coordinator.py \
  tests/unit/v1/worker/npu/test_attention_model_runner_v2.py \
  --junitxml="$afd_result_dir/pytest.xml" \
  > "$afd_result_dir/pytest.log" 2>&1
afd_test_status=$?
printf '%s\n' "$afd_test_status" > "$afd_result_dir/exit-code.txt"
cat "$afd_result_dir/pytest.log"
printf 'exit=%s; results=%s\n' "$afd_test_status" "$afd_result_dir"
```

缺少 torch、vLLM 或 Ascend 依赖时，CPU 主机可以正常收集并 skip；skip 不能记为
通过。测试中的 CPU 模型边界、输入准备和设备 kernel 替代不提供 NPU 算子精度、
模型接受率或设备 Graph 证据。
target Graph 输出由 CPU 夹具提供；draft KV 预计算和模型计算使用 CPU 边界，
不验收设备 cache 写入或 Graph 捕获。
接受/拒绝用例采用 greedy CPU 叶子，不覆盖随机采样、设备 rejection kernel
或真实 Engram gather kernel。

2026-11-11 A3 实际依赖回归中，完整 Attention runner 模块在第三轮为 **33 项通过**，
完整 Graph coordinator 模块在第四轮为 **117 项通过**，两轮均核对 57 项安装源码。
第三轮的另两项 profile/dummy 断言问题在第四轮修正；各轮独立记录，不合称一次
150 项全绿。运行基底使用已交付源码加本轮测试，排除了暂停的 MTP 本地改动。

随后 U2 增量的一轮完整回归为 **152 项通过、0 失败、0 跳过**（runner 35、
coordinator 117），64 项安装源码一致。它使用真实原生 U2 prepare，将三请求
q3 target 切为 4/5 两段，仍在父级做一次采样与历史推进；draft 期间无新增 F
事务。它只覆盖一次 target 到下一批 proposal 和下一轮 lookback 准备，连续
模型解码、真实设备 capture/replay 和 KV 写入仍需独立验收。

连续周期增量的单轮完整回归为 **158 项通过、0 失败、0 跳过**（runner 39、
coordinator 119），64 项安装源码一致。U1/U2 用例沿用同一请求状态，以首轮真实
draft、bonus 和 computed frontier 构造第二轮输入，并经过原生 cached-request
更新和采样。两轮采用不同 target logits，使全接受、首步拒绝和中途拒绝的结果
发生变化；检查最终 token、完整历史、下一轮 lookback 和 KV 收尾。第二轮
proposal 异常时，保留已更新的历史、上一批 draft 和尚未收尾的 KV 绑定。

另一组 U1/U2 组件用例执行 Graph 命中、miss 后完成 eager、再次命中的三个父
事务，每次结束后调用原生 DSpark proposal。中间路径使用新输入和独立算术
输出参考；三次 draft 均无新增 F 事务，保留的 Graph 输出存储和值不变。
F 的 ready/missing/ready 是脚本化可用性回复，不代表真实 F 进程重新捕获，
也不证明运行失败后的恢复。该组三步未连接完整 execute/sample/history 链；
连续采样由前一组两轮用例分别覆盖。

需要检查的接口行为包括：原生 proposal 的 token 和 Markov 依赖、target/draft
上下文隔离、异常后的上下文恢复，以及全接受、首步拒绝和中途拒绝后的输出、
history、computed frontier、下一轮 Engram lookback 和 KV 生命周期。

## 后续 A5 统一验收

完整模型使用实际部署的依赖版本，并记录与锁定版本的差异。先完成
[原生 eager 控制组](../../openspec/changes/afd-v41-hccl-text-eager/evidence/A5_NATIVE_EAGER_ZH.md)
的请求与正常退出，再对比同 checkpoint、tokenizer 和采样参数下的原生 DSpark
与 AFD DSpark。

| 验收项 | 必须取得的实际结果 |
| --- | --- |
| Proposal 与验证 | 实际 proposed/accepted token、拒绝位置、fallback 和输出一致性；不强制接受 |
| 请求状态 | 不同长度、连续/并发请求、取消后的 KV、Engram history 和随后请求 |
| Target-only | draft 期间没有新增 F recv 或远程 MoE 工作 |
| U2 / Graph | 实际两阶段执行、capture/replay/miss/fallback，以及同样的接受/拒绝与历史检查 |
| 退出与资源 | STOP、失败退出、新进程重启，以及显存和进程组回收 |
| 性能 | 同负载的吞吐、TTFT、TPOT、HBM 和实际使用卡数 |

现有 [A5 通信与 Graph 组件批量验证](DSV41_A5_BATCH_VALIDATION_ZH.md) 可以单独执行，
但其中的 q>1 通信/算术探针不能代替上述原生 DSpark 验收。当前完整模型门禁保留；
这份说明不提供绕过门禁的服务命令。PD 与 DSpark/Graph 的组合仍需独立验收。

回传 CPU 问题时优先提供 `git-head.txt`、`exit-code.txt`、`pytest.xml` 和
`pytest.log`。完整模型问题另附启动命令、实际依赖、请求参数及匹配的服务日志。
验证结果保留在运行环境；交付分支只提交代码、测试和操作说明。
