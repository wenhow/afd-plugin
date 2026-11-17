# DSV4.1 非 Engram 功能：A5 最小集中验证

本轮先收口非 Engram 功能代码，再由 A5 运行下面这一批。**一条命令、7 个顺序场景**，
包含原生对照、正常 A/F Worker 四模式，以及 DSpark/PD 两个代表组合。
不需要重新执行旧的 17 场景列表，也不安排 profiler、性能或服务启停专项。

本批的正常模型使用原生三层 `[0,0,2]` C2 Attention/HC、原生完整 BF16 MoE、
真实 Scheduler、KV 规划和设备算子；参数为受控 dummy，Engram 关闭。
DSpark/PD 使用真实原生 DSpark draft、设备缓存写入和 Mooncake 传输，target/FFN
仍是有明确输出预期的受控组件。因此结果用于**功能组合验证**，不代表训练检查点
精度、完整 Engram 模型或所有拓扑/长运行已经验收。

## 1. 更新并执行

使用此前能运行 DSV4.1 的 A5 容器和 Python 环境，保留其 CANN/ATB/OPP 环境。
代码锁为 vLLM `ced6857a`、vLLM-Ascend `7e37e29a`。入口会核对原生 Python
源码，以及实际 A5 类型、Mooncake 和已安装原生扩展；验证包不携带 A3 二进制，
不安装包、不替换共享运行库。

```bash
git switch feat/dsv41-flash-hccl-v030
git pull --ff-only origin feat/dsv41-flash-hccl-v030

python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /mnt/workspace/models/DeepSeek-V4.1-Flash \
  --devices 0,1,2,3 \
  --output-dir /mnt/workspace/tmp/dsv41-a5-functional-20261117 \
  --run
```

`--model` 改为现有模型目录；这里只读原始 `config.json`，受控配置写入结果目录。
`--output-dir` 必须不存在，重跑时换一个目录。默认使用当前 `python`；也可显式传
`--python /实际虚拟环境/bin/python`。`--devices` 是当前进程可见的四个不同设备序号；
正常 A/F 用前两个，DSpark/PD 四角色用四个。执行前后检查整机八卡空闲。

默认单场景上限 480 秒；超时、任一角色非零退出、被强制清理、报告失败、残留进程或
设备未空闲都算失败。必要的每轮清理保留。遇到失败即停止，保留首次原始日志，
不会把已写出的成功字段覆盖进程异常。

可选的源码/参数自检去掉 `--run` 即可；该路径不导入设备运行库，**不计为 A5 验证通过**。

## 2. 七个场景及通过条件

| 场景 | 要验证的功能 | 关键判定 |
| --- | --- | --- |
| 1 原生 eager/U1 | A5 原生基线 | 真实 Worker、加载、KV、warmup；A/B 两轮后取消 A，B 结束，追加 C 再结束 |
| 2 AFD eager/U1 | 正常 A/F 接入和连续轮次 | 同一正常 Worker 生命周期，真实远程 MoE；与原生实际参数和采样结果比较 |
| 3 AFD eager/U2 多流 | 原生 U2 切分和 eager 多流接入 | 实际两阶段与 `eager-streamed` 记录；取消后剩余请求继续，C 可接入 |
| 4 AFD Graph/U1 | 正常捕获、回放及 prefill eager 路径 | 实际 capture/replay 返回记录，保留同一模型/KV owner，随后请求可继续 |
| 5 AFD Graph/U2 多流 | 两阶段 Graph 正常入口 | 两请求 decode 的真实 U2 replay、实际 Graph 多流执行记录，随后单请求及新请求继续 |
| 6 DSpark/PD eager/U1 | 拒绝后的 KV/历史与 Ready | prefix/辅助状态实际传输，拒绝后继续原生 proposal/verification，Ready 前沿和 typed DONE |
| 7 DSpark/PD Graph/U2 多流/跨层 | 代表性组合与请求生命周期 | `cross_layer_v1`、全部 Graph policy 轴显式开启；实际 capture/replay、接受、取消、追加 C、缓存复用与 DONE |

前五项使用同一模型声明与参数初始化，自动核对原生 79 份参数与 A58+F21 的
canonical 名字、shape、dtype、字节 SHA，并比较 A/B/C 的真实贪心采样序列。
不会预先写死正常模型的 token ID。DSpark 组件的受控 target 则检查事先声明的
接受/拒绝结果，不能把这些 token 当作完整模型精度证据。

多流场景检查实际执行路径和协议完成；没有设备 timeline，不能据此宣布物理重叠或性能收益。
Graph 场景同时经过 prefill 的 eager 路径；本批不额外扩成所有 bucket、miss 原因或拓扑组合。
P 端中途内存压力抢占仍按现有协议终止，不声称支持断点恢复；请求取消、正常退役和随后请求
是本批覆盖的生命周期。

## 3. 返回结果

首先看进程退出码和结果目录的 `summary.json`：

```bash
python -c 'import json,sys; r=json.load(open(sys.argv[1])); print({k:r[k] for k in ("success","npu_acceptance","runtime_source_sha256")})' \
  /mnt/workspace/tmp/dsv41-a5-functional-20261117/summary.json
```

本批通过需要退出码 `0`、`success=true`、`npu_acceptance=component_suite_passed`，
且七场景均完成。正常模型比较位于 `native_afd_comparison`；每轮命令、角色退出码、
实际阶段、参数、采样历史、KV 描述、原始 hidden 和清理结果都保留在各场景目录。

请回传整个结果目录的压缩包，无需手工筛日志：

```bash
tar -czf dsv41-a5-functional-20261117-results.tar.gz \
  -C /mnt/workspace/tmp dsv41-a5-functional-20261117
```

如果仍在 `SparseFlashMlaMetadata` 失败，保留这次结果，不先改模型几何或替换算子库。
A3 已知是首轮与原生一致、第二轮 decode C2 metadata 失败；本批 A5 原生/AFD 同参数
对照用于区分这个差异。**目前不能确认 A3 环境就是根因。**

## 4. 源码与实施边界

交付源码见 [准备记录](../../reports/dsv41-a5-functional/preparation.json) 和
[逐文件锁](../../reports/dsv41-a5-functional/bundle-lock.json)。入口解压并运行冻结源码，
各场景共享同一套 146 个生产模块，避免与本地其他开发改动混用。可直接在更新后的仓库执行，
不要求环境具备本地历史证据目录，也无需在 A5 重建验证包。

本轮补齐默认关闭的 `AFD_DSV41_FUNCTIONAL_QUALIFICATION=1` 正常 Worker 组合入口，
保留固定 BF16/dummy/C2/TP1/两请求上限、原生 DSpark、attested PD 等约束。
旧 eager/U1 资格入口和默认拒绝不变；脚本在首 EngineArgs 声明前设置所需值，
不通过构造后的改配置开启 U2/Graph。包含已有 Graph/U2 跨层 PD 准入和 FFN
后台异常向公开 poll/shutdown 传播的修复；未纳入暂停的旧 V4 MTP 改动。

完整检查点首 token 等价仍需与此前
[同 TP4 原生/AFD-F 单层 MoE 对照](DSV41_A5_AFD_MOE_PARITY_ZH.md)的结果一起分析。
该对照不加入本批默认七场景；原生 MoE 已有结果，不必重复原生单独对照。
Engram 专项、完整五种 PD 模型矩阵、代表性拓扑/长运行及性能/服务启停验收继续独立登记，
本批不会提前将这些任务勾选完成。
