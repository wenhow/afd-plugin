# 在 HCCL 文本路径上接入 V4.1 DSpark：技术设计

状态：2026-11-09 按用户要求并行准备源码和 CPU 契约，数值/硬件验收继续等待 C2。
本轮实现边界见 [并行开发记录](evidence/parallel-development-20261109.md)。
源码标号见 [锁定阅读索引](../../../.agents/dsv41/sources.lock.json)。
## 源码事实与阅读入口

阅读 U02 的 aux_hidden_states 收集，U04 的 draft，U08 的 speculative 配置补丁，以及 A04 的全局 context hook。
原生 target 在配置目标层入口、Engram 注入前收集 hidden_states.mean(dim=1)，使用 one-based layer IDs。
RFC 注册 DeepseekV41DSparkModel；不能 alias 为普通 MTP。

## 决策

- 保持 C2 的 target HCCL FFN 路径，draft 全部留 A，F 不构造或加载 DSpark draft。
- 在真正的 target remote-forward 边界建立 metadata；验证 draft/context/dummy 不发送无对应 MoE 的控制消息。
  若复用通用 hook，必须能精确识别本次是否有 remote target FFN；不得按 dspark_block_size 猜 F 接收次数。
- 已核对固定 V2：provider 只包围 execute，draft 在 sample_tokens 或 dummy 的 execute
  返回后调用；复用此 scope。loader 按独立 ModelConfig 参数区分 draft 与 target，
  F 在 native worker 构造前清除 speculative/Engram 绑定。保留总 guard，不新增 sampler 补丁。
- 保留原生 aux 顺序、HC mean/concat 和 SP gather 的裁剪语义，不改成 final normalized hidden。
- 区分 block size、draft stage count、proposal count 和 target verification token 数。
- 使用真实 sampler 的 acceptance/rejection，核对 KV advance、request history、fallback；
  不依赖强制 acceptance 或修改 sampler 来证明支持。

## 成本和回退

draft 全留 A 增加 A 权重与 workspace，需要重新预算卡数/HBM。
C3 可关闭 speculative_config 回退到 C2；不能为了节省 A 内存偷偷把 draft FFN 远程化。
C3 先形成 eager/U1 基线；Graph/DBO/PD 由本项目已列出的 C4–C7 实现和组合验收，不属于延期范围。

## 验证

先验证关闭 DSpark 时与 C2 一致，再验证启用时 target-only 协议、多请求接受/拒绝和退出。
采集实际 proposed/accepted token 数、acceptance length、TTFT/TPOT 与总吞吐、总卡数及每卡效率。
无硬件统计时只记录待验收，不能套用 V4 历史结果。

## 2026-11-09 完整迁移范围修订

本 change 保留原有实现/验收范围和进度。最终目标是 [HCCL 全能力迁移](../../HCCL_CAPABILITY_MIGRATION_ZH.md)，
新增 C4 microbatch/拓扑、C5 Graph、C6 多流、C7 DSpark/PD 组合均为本项目必交付。
新增任务见 [tasks.md](tasks.md) 第 2 节；本次不把既有通过扩大为完整迁移通过。
