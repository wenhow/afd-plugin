# V4.1 AFD eager/U1：受控资格配置

此入口用于准备首次完整 A/F 硬件资格验证。**它是候选配置准入，尚未经过 A5
完整 AFD 冷启动、文本推理或退出验收，也没有可交付执行的服务 recipe。**
默认仍拒绝完整 V4.1 AFD；本页不宣称正式支持，不提供 monkeypatch 或跳过校验的命令。

## 显式候选范围

仅在 worker 环境中显式设置 `AFD_DSV41_EAGER_QUALIFICATION=1` 时，现有
`fail_if_unsupported_npu_afd_features` 才检查下列候选。变量未设置或为 `0`
仍走原来的拒绝路径；其他值报错。调用方继续执行原有 worker/runtime 校验。

| 项目 | 第一批限定 |
| --- | --- |
| 模型执行 | V2、纯文本、BF16、eager、逻辑 U1 |
| A/F 通信 | `P2pHcclAFDConnector`、F 侧完整 gate、每个 native rank 与对端配对 |
| 设备候选 | A4/F4，两侧 native TP4；A、F 使用独立四卡及原生 TP world |
| 其他并行 | DP/PP/PCP/DCP=1；EP/SP/EPLB/DBO/Graph 关闭 |
| 载入 | 原始完整 checkpoint，显式 `deepseek_v4_fp8`，native auto/safetensors loader |
| A 侧 Engram | `cpu_offload=true`、`dp_shared_memory=false`、`embedding_across_dp=false` |
| F 侧状态 | 原生 worker 构造前隔离 Engram；speculation、PD 不进入 F |
| 请求 | max_num_seqs=1，同步调度；model/token 上限均不超过 1024 |
| 本批关闭 | DSpark、PD、prefix cache、稀疏 KV offload、LoRA、sleep、权重更新、细粒度 TP、shared-expert DP、Async CAM layered GMM |

候选直接复用现有 HCCL 校验，包括实际 `parallel_config.rank`，不以固定 rank=0
替代真实执行 rank。A 的原生 HC/pre_mix、Engram ModelState、shared cache 和采样
继续由原生 runner 持有。F 的私有配置经 `get_npu_ffn_config` 生成；不能直接修改
原始目标配置来删除 A 的 Engram。门禁不改变 loader、模型执行或通信协议。

DP1/PCP1 不允许沿用已成功原生 DP8/EP8 的 `dp_shared_memory=true`：锁定的
Ascend 配置与 embedding 构造都要求共享组大于一个 rank。此候选中四个 A rank
分别持有其 native TP head shard，F 不持有 Engram。

## 容量与对照边界

依据既有 [checkpoint 预算](../../openspec/changes/afd-v41-hccl-text-eager/evidence/a5-topology-budget-20261109.json)
重算 CPU offload 候选，数值为容量估计，未测加载峰值：

| 项目 | GiB |
| --- | ---: |
| Engram host 表总量 | 188.833133 |
| 每 A rank 的 host 表均值 | 47.208283 |
| 每 A rank HBM 参考（含旧 20 GiB RoPE 假设） | 21.926063 |
| 每 F rank HBM 参考 | 68.005807 |
| 同 TP4 原生完整模型 CPU offload HBM 参考 | 89.931871 |

以上未计实际复制参数、NZ/layout、加载临时分配、KV/index/history、量化与 MoE
workspace、HCCL 和进程开销；host 还需额外加载内存。A 暂定卡 0–3、F 卡 4–7，
最终映射须核查机器拓扑和逻辑/物理卡号，不在门禁中硬编码物理卡号。

同 TP4 原生参考距离 96 GiB 仅余约 6.068 GiB，且超过 `0.90` 对应的 86.4 GiB
预算，因此不能宣称该控制组可运行。已通过的原生 DP8/TP1/EP8 是功能参考；其 TP、
EP 和 Engram 共享方式与本候选不同，不能用于固定并行条件的性能归因。

## 源码身份与下一次最小 A5 验证

开发锁仍是 vLLM `ced6857a` / Ascend `7e37e29a`。用户已经跑通的原生环境为
Ascend `4ec6dc77`。后者的量化和 DSpark speculator 文件与开发锁不同；其依赖还包括
`ascend_forward_context.py` 和 `ops/fused_moe/dataclass/fused_experts.py`。
须核查实际安装的这四个文件，不能只替换其中两份源码拼成混合安装。

首批 EP 关闭时 MegaMoE 不参与该 AFD 候选。量化方法新增的 `shared_weights`
入参仍会经过普通 MoE 路径，其 dataclass/consumer 契约必须一并核查。
DSpark 的 draft MoE 私有配置和 current-config 上下文差异须在后续 N=1 准入前审查。
本次候选变量不开放 DSpark。

完成实际依赖 CPU 回归、源码闭包审查和启动器审查后，另行交付最小 A5 recipe：

1. AFD eager/U1、DSpark 关闭，一个固定 32-token 请求；采集冷加载、输出及内存。
2. 请求完成后先由 A 发出 STOP，等待 F 接收循环退出，再完成两侧服务退出；
   保存带时间的退出码、进程和设备快照。
3. 该门通过并单独开放 DSpark 候选后，再做 N=1 两个连续短请求及退出；确认
   由同一 worker 处理，并保留真实 proposal/acceptance/rejection。

Graph/U2、PD、并发和性能另行验收。原生已完成的三个短请求无需因新增门禁重跑；
其停服清理证据仍按[最小原生说明](DSV41_A5_PARALLEL_MINIMAL_ZH.md)收尾。
