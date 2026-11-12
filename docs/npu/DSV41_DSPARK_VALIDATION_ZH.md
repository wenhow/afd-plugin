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

在具有匹配依赖的环境中，于仓库根目录运行以下八个完整模块。A5 容器的 Python
路径沿用 `/usr/local/python3.11.10/bin/python`；不要求 venv 或 `atb-env`。
这些测试不启动推理服务，不加载 checkpoint 权重，也不使用 NPU。

```bash
afd_python=/usr/local/python3.11.10/bin/python
afd_result_dir=$(mktemp -d /tmp/afd-dspark-cpu.XXXXXX)
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

git rev-parse HEAD > "$afd_result_dir/git-head.txt"
"$afd_python" -B -m pytest -q \
  tests/unit/compat/npu/test_dspark_pd.py \
  tests/unit/compat/npu/test_kv_layout_connector.py \
  tests/unit/compat/npu/test_runtime_config.py \
  tests/unit/compat/patches/test_signatures.py \
  tests/unit/model_executor/models/npu/test_deepseek_v41.py \
  tests/unit/model_executor/models/npu/test_deepseek_v41_staged.py \
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
target Graph 使用预供 CPU 输出或原生 capture 工厂产生的 CPU 输出；底层捕获和
回放仍由 CPU 边界替代。draft KV 预计算和模型计算使用 CPU 边界，不验收设备
cache 写入或 Graph 捕获。
接受/拒绝用例采用 greedy CPU 叶子，不覆盖随机采样、设备 rejection kernel
或真实 Engram gather kernel。

此前 A3 原生 capture 工厂组合的独立单轮回归为 **162 项通过、0 失败、0 跳过**
（完整 runner 43、coordinator 119），66 项安装源码一致。运行快照使用已交付
源码加四个测试文件，排除了暂停的普通 MTP 工作区改动。

U1/U2 组合用例沿用同一 runner、ModelState、请求状态和 Graph 缓存，连续执行
FULL 命中、miss 后完成 eager、FULL 再次命中。每轮通过实际 execute、原生
采样和 DSpark proposal、KV post-forward；后续输入由上一轮生成的 draft、
bonus 和 computed frontier 派生，并经过原生 cached-request 更新。U2 的
三请求 q3 target 按 4/5 两段准备，采样与历史推进仍在父级完成。

三轮采用不同 target logits，改变全接受、首步拒绝和中途拒绝的结果，检查
最终 token、完整 history、computed frontier、下一轮 lookback 和 KV 收尾。
execute 返回后先检查 Engram 执行资源已退役、history 尚未推进且 KV 仍绑定；
sample 后再检查新历史和 draft。中间 eager 的实际输出与独立算术参考逐值
比较后，由原生 sampler 继续消费；draft 无 target sidecar，也无新增 F 事务。
另有第二轮 proposal 异常用例，确认保留已更新历史、上一批 draft 和未收尾 KV。

F 的 ready/missing/ready 是脚本化可用性回复。第一、三轮 FULL 使用预供 CPU
输出；中间 eager 的 child metadata 和无 Engram 模型边界使用 CPU 夹具。
结果不证明真实 F 重新捕获、设备 Graph 随新输入重算、Engram producer/kernel、
实际 KV 写入或运行失败后的恢复。这些仍需 A5 完整模型验收。

新增 capture 组合保留原生 GPU runner、Ascend ACL manager 和 model manager 的
capture 调用链，执行原生 dummy 输入准备、U2 准备、模型上下文及持久输出存储。
q3 target 按 4/5 两段运行，使用 `stage_serial`、compute stream 开、hybrid 关的
组合。捕获阶段的全 padding 行由 CPU F 边界返回零；hidden 和有序 auxiliary
输出与独立未分段 CPU 参考比较，再由同一 runner 执行、采样和生成 DSpark draft。
成功和 draft 异常路径均检查 history、KV 绑定及捕获输出的保留。

该组合保留真实工厂和控制流程，但基础捕获、stream/event、远端 F、metadata
设备任务及模型算子使用 CPU 边界。回放句柄不重新执行 Python 模型，不能据此
认定设备 Graph 已捕获、随新输入重算或产生真实模型接受率。

需要检查的接口行为包括：原生 proposal 的 token 和 Markov 依赖、target/draft
上下文隔离、异常后的上下文恢复，以及全接受、首步拒绝和中途拒绝后的输出、
history、computed frontier、下一轮 Engram lookback 和 KV 生命周期。

## DSpark 与 Mooncake PD 的首个开发切片

外部连接器模块为 `afd_plugin.compat.npu.kv_layout_connector`，连接器名保持
`MooncakeHybridConnector`。配置检查仅接纳 target/draft eager、U1、固定验证长度、
CP/PP/DP 均为 1、关闭 prefix cache 和 sparse offload、P/D TP 相同的组合。
完整模型门禁仍保留；配置检查通过不表示可以启动完整模型服务。

A 端在原生加载后绑定真实 target/draft，KV 初始化时保留全部 draft 组并绑定实际
缓存视图。P 的每个分块在原生 execute 收尾后，使用原生下层 context writer 投影
auxiliary 特征并等待写入完成；所有 TP rank 交换结果成功后才发布请求 Ready。
原始 prompt 摘要与 P 实际计算的前缀长度分别记录；长度大于 1 的 prompt 沿用
原生 P 端少算最后一个 token 的行为。

D 在读取前校验请求代次、prompt 摘要、所有缓存组的块映射、schema 和 TP rank，
取得 P 发放的读取租约。读取完成且匹配的 DONE 得到 ACK 后才进入释放流程。
若读取已开始而无法确认结束，两端保留缓存并报错，需要由外层监督器重建进程；
本切片不提供超时强制释放或失败后自动恢复。P 的中途抢占明确拒绝。
正常退出也经过同一连接器检查：P 关闭新的 Ready 读取并确认没有活跃请求，
D 等待已接纳的读取及其 DONE/ACK 结束。未完成或退出失败会阻断后续原生 KV
释放；不能以 A/F STOP 已完成代替 P/D 读取结束。

CPU 用例保留原生 draft combine/project/store、组映射、连接器调度及请求就绪协议，
以 CPU 边界替代 RoPE、设备缓存写入、target 计算、socket 和 transfer engine。
TP 1/2/4/8 的报告检查不代表对应卡数的实际 collective 验收。A5 后续仍需检查
真实分块/滑窗传输、接受率、取消、故障退出和缓存生命周期，再扩展到 U2/Graph。

2026-11-11 请求就绪协议联合回归在 A3 实际依赖下单轮 **451 项通过、0 失败、0 跳过**：
prefix 39、connector 89、配置 117、签名 38、runner 49、coordinator 119。
85 项安装源码与锁定版本一致；快照使用已交付基线加本轮 13 个文件，排除了普通
MTP 工作区改动。此前失败轮次分别保留，没有合并为通过结果。此轮没有运行设备
探针、服务或完整模型，也没有调整容器依赖版本。

后续独立缓存回归单轮 **140 项通过、0 失败、0 跳过**（完整 connector 91、runner 49），
85 项安装源码一致，138 个生产 Python 文件保持已交付版本。新增用例给 P、D 分配独立
CPU 存储，使用真实接收器生成的地址批次，经有范围检查的 CPU 复制边界搬运；逐字节检查
target/draft 整页、不同 D 容量和逆序块映射、未写区域及 P 页不变，并验证 typed DONE/ACK。
它不构造 D 模型，也不运行设备传输或 attention kernel；远端读取完成到 D 新请求调度、
首步执行、采样及下一批 draft 的组合仍需接通。两轮回归分别记录，不合并计数。

随后原生调度回归单轮 **143 项通过、0 失败、0 跳过**（完整 connector 94、runner 49），
103 项安装源码一致，138 个生产 Python 文件保持已交付版本。新增用例保留真实 Scheduler、
连接器工厂与 KV 分配，检查接收完成通知前不执行、恢复后原始 prompt 和全部缓存组保留，
以及取消请求的缓存保持到接收完成后再释放。worker 的完成通知在此作为明确的外部输入；
本轮尚未串联实际接收器、D 模型执行、采样和 proposal。

首步逻辑上重算 prompt 尾 token，但物理调度长度应以原生输出为准：单 token prompt
在全命中后回退到 computed=0；普通 prompt 从原生计算前缀末尾继续，可能补成
`1 + num_speculative_tokens` 行。占位行不表示已经产生真实 draft proposal，不能据此
推断接受率或放宽 Graph/PD 准入。

新增组合保留同一原生 Scheduler、ActiveKV 和实际接收器：第一次零 token execute
接纳读取，完整 target/draft 页经 CPU 复制边界进入独立 D 存储；匹配的 typed DONE/ACK
完成后，下一次零 token execute 自然返回 `finished_recving`，再由同一 Scheduler
恢复真实 `NewRequestData`。单 token 与普通 prompt 分别覆盖，空 KV 结果沿原生行为
折叠为 `None`。整页 oracle 从源的绝对分配、计算前缀长度、块大小和滑窗规格独立推导
D 必需的范围；P 的保守滑窗导出可以多于 D 实际保留的块，不要求两端发布列表等长。
读取查询已修正原先逐组等长的限制，要求组数一致且 D 每组不超过 P 的发布范围。
完整 Ready 和 destination generation 保持在租约中，实际读取仍先检查尾部映射、
页布局及全部已加载 draft 组；不能用该结构校验代替真实传输校验。

D 的原生 target/draft 身份、consumer 与缓存视图在注册前固定；target 在这组用例中
只有模型身份，未执行计算。接收线程启动、socket 和 transfer engine 是明确的 CPU
边界，没有初始化 F 传输，也没有执行恢复后的 target、采样或 proposal。

该组合最终单轮 **189 项通过、0 失败、0 跳过**（完整 prefix 44、connector 94、runner 51），
103 项安装源码与锁定 Git 一致；138 个生产 Python 文件仅包含上述查询约束修正。
前三轮失败及诊断分别保留，未合并计数。没有运行设备、服务或依赖安装。

后续接入组合继续消费上述真实 `NewRequestData`，保留原生 RequestState、block table、
`add_requests`、`update_requests` 和 Ascend 批次分类。检查原始 prompt、history、computed
frontier、零值 draft 初态，以及含 SWA null 坐标的完整块表。原生 KVBlockZeroer 对新分配
物理块执行实际 CPU 字节清零；已接收前缀、跳过的 tuple 缓存及页内空隙保持不变。
该组合独立单轮 **189 项通过、0 失败、0 跳过**，116 项安装源码一致，138 个生产文件
保持前轮已交付版本；首次 fixture 缺少 zeroer 的失败单独保留。此轮到请求接入与
BatchReqState 为止，首步输入准备、target 计算、采样及 proposal 仍需后续组合。

新增首步组合继续沿用同一 P/D 注册、consumer、target/draft 模型和缓存。微型 CPU
模型在 schema 和绑定之前配置一致：target 三层、三路 auxiliary、两个 draft 层、
N=5；target 各层 SWA 独立，压缩 KV/index 保留原生共享关系。输入准备沿用实际
Ascend `prepare_inputs`、attention 状态选择、绝对块表 gather 和 slot wrapper；
仅复制、RoPE 缓存更新及 Triton kernel 使用明确的 CPU 叶子。完整块表保留 SWA null
位置，单 token prompt 与普通 prompt 分别检查原始尾 token 和初始 draft=0 的查询。

新的 target 用例从真实 Scheduler 完成接收后返回的 `NewRequestData` 开始，调用实际
`execute_model`。保留真实 EngramModelState 构造、`prepare_attn`、TargetDeviceMetadata
的 `run_build/finish` 和 NONE manager 的 eager dispatch。该微型模型关闭 Engram；
metadata builder、HC、Attention、MoE、norm、权重和词表 head 是 CPU 算术边界，
不构造远端 F。独立参考从 D 页快照和绝对块号推导三层 HC/pre_mix、hidden、三路 auxiliary、
pre-head residual 及整页写入；所有物理候选行先写 KV，再由采样决定接受前沿。

新的采样用例保留已注册的原生 Sampler、RejectionSampler、history 更新、原生 DSpark
context projection、draft forward 和顺序 Markov head。greedy kernel、计数、history
store、draft 输入准备和输出复制使用 CPU 叶子；不预置接受数量或替换 target logits。
用例检查 prompt/history、computed frontier、下一批 draft、接受后的新 draft context、
target 页和 P 源页。固定词表 head 的输出本身不能替代前述 hidden/整页数值检查。

2026-11-12 A3 恢复连接后，首步组合的三个完整模块独立单轮 **193 项通过、0 失败、
0 跳过**，125 项实际安装源码与锁定 Git 一致。修正仅涉及测试夹具：新增 SWA 层的
逐层规格及带页内空隙的存储、三层压缩配置的保留，以及 eager manager、AFD 元数据
和 draft 装饰器的正常初始化字段。此前失败轮次分别保留，不拼接为通过结果；138 个
生产 Python 文件保持交付基线，没有执行设备探针、推理服务或依赖安装。

后续两轮组合继续沿用同一请求、模型和缓存，通过原生 Scheduler 消费首轮结果，
再由 `take_draft_token_ids` 回传无 grammar 的占位列表供下一轮调度。真实 draft
仍留在 RequestState，实际输入准备读取其 token；占位值不能作为模型输入。
第二轮独立参考使用完整绝对块表及当轮清零列表，检查 hidden、auxiliary、整个 D 页，
以及实际首步拒绝后的 computed frontier、历史和新 draft context。该组合在 A3
实际依赖下八个完整模块独立单轮 **561 项通过、0 失败、0 跳过**，127 项安装源码
与锁定 Git 一致；138 个生产 Python 文件保持交付基线。它不覆盖随机采样或第二轮
全接受、中途拒绝；draft 的固定 head/Markov 输出也不能代替 draft Attention/HC
数值验收。该轮未开放完整模型与 PD Graph/U2。

后续增加同一 runner 的生命周期组合：原生 D 请求执行两轮后自然达到长度上限或
由 Scheduler 取消，再以同一请求 ID、原生 slot 和 owner 接收本地新 prefill，
检查新输入、历史、draft、整页缓存与输出。另在 target 调用入口注入 Python 异常，
检查原异常、terminal 拒绝重试及注册缓存保留。新请求不进行第二次远端读取；
Engram 关闭，不代表在途 DMA 取消、真实 Engram 或新进程故障恢复验收。
默认 penalties 关闭时，原生采样器不重建未使用的 histogram；本组不把该缓冲区
清零作为请求隔离条件。

上述增量与资格配置在 A3 实际依赖下十个完整模块独立单轮 **892 项通过、0 失败、
0 跳过**，133 项安装源码与锁定 Git 一致。此前 561 通过/4 失败及 890 通过/2 失败
轮次分别保留；前者修正未启用 penalties 的统计缓冲区预期，后者补齐故障用例已有的
sampling CPU 初始化。生产模型执行、协议和上游源码未改；本轮唯一生产变化是下述
显式资格配置校验。CPU 结果不代表完整 AFD、真实 Engram/HCCL 或设备故障恢复。

完整 AFD 仅新增[受控 eager/U1 资格配置](DSV41_AFD_EAGER_QUALIFICATION_ZH.md)，
默认仍拒绝，显式候选也只允许 DSpark 关闭。DSpark/PD Graph/U2 继续不准入。

后续补齐了受控 AFD 启动器和独立 F 本地握手入口，以及 STOP/engine/各 rank 的
清理诊断。A3 现有真实依赖下十四个完整模块单轮 **1015 项通过、0 失败、0 跳过**，
138 项安装源码与开发锁一致；覆盖原有 DSpark/PD、A/F worker、engine 生命周期和
F 入口。另有 13 项纯文件预检及 8 项 Linux 真实进程/HTTP 监督行为检查通过。
这些结果不包含完整模型、真实 HCCL 或 A5 冷启动验收；下一步只新增一个 AFD 短请求。

## 后续 A5 统一验收

2026-11-12 原生 eager 一个短请求、DSpark N=1 两个顺序短请求已通过；输出 token IDs
相同，两个 DSpark 请求各接受 13/19 个 draft，且由不同 DP worker 执行。
无需重跑三条请求，按[当前补充说明](DSV41_A5_PARALLEL_MINIMAL_ZH.md)核对退出记录；
补充包四份安装源码已全部确认。该结果限定于实际 A5 原生环境，完整 AFD 仍待验证。

完整模型使用实际部署的依赖版本，并记录与锁定版本的差异。先完成
[原生 eager 控制组](../../openspec/changes/afd-v41-hccl-text-eager/evidence/A5_NATIVE_EAGER_ZH.md)
的请求与正常退出，再对比同 checkpoint、tokenizer 和采样参数下的原生 DSpark
与 AFD DSpark。

[原生 DSpark eager 操作说明](DSV41_A5_NATIVE_DSPARK_ZH.md)提供服务启动后的统一
请求采集命令，保存每轮 token IDs、原生详细接受/拒绝指标和 HTTP 失败现场。
先完成 N=1 单请求，再进行同服务的不同长度/重复请求；A5 结果后续集中回传。

| 验收项 | 必须取得的实际结果 |
| --- | --- |
| Proposal 与验证 | 实际 proposed/accepted token、拒绝位置、fallback 和输出一致性；不强制接受 |
| 请求状态 | 不同长度、连续/并发请求、取消后的 KV、Engram history 和随后请求 |
| Target-only | draft 期间没有新增 F recv 或远程 MoE 工作 |
| U2 / Graph | 实际两阶段执行、capture/replay/miss/fallback，以及同样的接受/拒绝与历史检查 |
| 退出与资源 | STOP、失败退出、新进程重启，以及显存和进程组回收 |
| 性能 | 同负载的吞吐、TTFT、TPOT、HBM 和实际使用卡数 |

现有 [A5 通信与 Graph 组件批量验证](DSV41_A5_BATCH_VALIDATION_ZH.md) 可以单独执行，
但其中的 q>1 通信/算术探针不能代替上述原生 DSpark 验收。完整模型默认门禁保留；
现已提供[受控 AFD eager/U1 启动器](DSV41_AFD_EAGER_QUALIFICATION_ZH.md)，
仅显式候选、DSpark 关闭、一个短请求；默认预检，`--run` 才启动。
PD 与 DSpark/Graph 的组合仍需独立验收。

回传 CPU 问题时优先提供 `git-head.txt`、`exit-code.txt`、`pytest.xml` 和
`pytest.log`。完整模型问题另附启动命令、实际依赖、请求参数及匹配的服务日志。
验证结果保留在运行环境；交付分支只提交代码、测试和操作说明。
